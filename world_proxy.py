# standard library imports
import math
import itertools
import time
import numpy
import threading
import heapq
import logutil
from collections import deque
import queue
import concurrent.futures

# pyglet imports
import pyglet
image = pyglet.image
from pyglet.graphics import TextureGroup
from pyglet.math import Mat4
import pyglet.gl as gl


class _MeshJobResult(object):
    __slots__ = ("pos", "gen", "priority", "start_time", "_result", "_exc")

    def __init__(self, pos, gen, priority, start_time):
        self.pos = pos
        self.gen = gen
        self.priority = priority
        self.start_time = start_time
        self._result = None
        self._exc = None

    def set_result(self, value):
        self._result = value

    def set_exception(self, exc):
        self._exc = exc

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._result


class TransparentWaterGroup(TextureGroup):
    """Texture group that disables depth writes so water stays see-through."""
    def set_state(self):
        super().set_state()
        gl.glDepthMask(gl.GL_FALSE)

    def unset_state(self):
        gl.glDepthMask(gl.GL_TRUE)
        super().unset_state()

# local imports
import world_loader
import server_connection
import config
from config import SECTOR_SIZE, SECTOR_HEIGHT, LOADED_SECTORS
from util import normalize, sectorize, FACES, cube_v, cube_v2, compute_vertex_ao
from blocks import (
    BLOCK_VERTICES,
    BLOCK_COLORS,
    BLOCK_NORMALS,
    BLOCK_TEXTURES_FLIPPED,
    BLOCK_ID,
    BLOCK_SOLID,
    BLOCK_OCCLUDES,
    BLOCK_OCCLUDES_SAME,
    BLOCK_RENDER_ALL,
    BLOCK_GLOW,
    TEXTURE_PATH,
    BLOCK_LIGHT_LEVELS,
    DOOR_LOWER_IDS,
    DOOR_UPPER_IDS,
    BLOCK_COLLIDES,
    BLOCK_COLLISION_MIN,
    BLOCK_COLLISION_MAX,
    BLOCK_RENDER_OFFSET,
    BLOCK_FACE_COUNT,
    BLOCK_FACE_DIR,
    STAIR_COLLISION_BOXES,
    STAIR_UPSIDE_IDS,
)
import mapgen
import numpy
import entity as entity_codec

WATER = BLOCK_ID['Water']
MAX_LIGHT = getattr(config, 'MAX_LIGHT', 15)
EMPTY_LIGHT_LIST = numpy.zeros((0, 4), dtype=numpy.uint8)
NEIGHBOR_OFFSETS_8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def _light_list_equal(left, right):
    if left is right:
        return True
    if left is None or right is None:
        return left is None and right is None
    if len(left) != len(right):
        return False
    if len(left) == 0:
        return True
    return numpy.array_equal(left, right)
BLOCK_LIGHT_LUT = numpy.zeros(len(BLOCK_COLORS), dtype=numpy.uint8)
for _bid, _lvl in (BLOCK_LIGHT_LEVELS or {}).items():
    _scaled = int(round(float(_lvl) * MAX_LIGHT))
    if _scaled < 0:
        _scaled = 0
    elif _scaled > MAX_LIGHT:
        _scaled = MAX_LIGHT
    if _bid < len(BLOCK_LIGHT_LUT):
        BLOCK_LIGHT_LUT[_bid] = _scaled
#import logging
#logging.basicConfig(level = logging.INFO)
#def world_log(msg, *args):
#    logging.log(logging.INFO, 'WORLD: '+msg, *args)

class SectorProxy(object):
    def __init__(self, position, batch_solid, batch_water, group, water_group, model, shown=True):
        self.position = position[0],0,position[2]
        self.bposition = position[0],0,position[2]
        self.group = group
        self.water_group = water_group
        self.model = model
        # A Batch is a collection of vertex lists for batched rendering.
        self.batch = batch_solid
        self.batch_water = batch_water
        self.blocks = numpy.zeros((SECTOR_SIZE, SECTOR_HEIGHT, SECTOR_SIZE), dtype='u2')
        self.sky_floor = numpy.zeros((SECTOR_SIZE, SECTOR_SIZE), dtype=numpy.uint16)
        self.sky_side = EMPTY_LIGHT_LIST
        self.torch_side = EMPTY_LIGHT_LIST
        self.incoming_sky = {offset: EMPTY_LIGHT_LIST for offset in NEIGHBOR_OFFSETS_8}
        self.incoming_torch = {offset: EMPTY_LIGHT_LIST for offset in NEIGHBOR_OFFSETS_8}
        self.incoming_sky_updates = 0
        self.incoming_torch_updates = 0
        self.edge_sky_counts = (0, 0, 0, 0)
        self.edge_torch_counts = (0, 0, 0, 0)
        self.defer_upload = False
        self.edit_group_id = None
        self.light_dirty_from_edit = False
        self.light_initialized = False
        self.light_dirty_internal = True
        self.light_dirty_incoming = True
        self.light_neighbors_ready = False
        self.stat_load_count_total = 0
        self.stat_load_ms_total = 0.0
        self.stat_light_count_total = 0
        self.stat_light_ms_total = 0.0
        self.stat_mesh_count_total = 0
        self.stat_mesh_ms_total = 0.0
        self.stat_upload_count_total = 0
        self.stat_upload_ms_total = 0.0
        self.stat_load_count_prev = 0
        self.stat_load_ms_prev = 0.0
        self.stat_light_count_prev = 0
        self.stat_light_ms_prev = 0.0
        self.stat_mesh_count_prev = 0
        self.stat_mesh_ms_prev = 0.0
        self.stat_upload_count_prev = 0
        self.stat_upload_ms_prev = 0.0
        self.stat_load_count_last = 0
        self.stat_load_ms_last = 0.0
        self.stat_light_count_last = 0
        self.stat_light_ms_last = 0.0
        self.stat_mesh_count_last = 0
        self.stat_mesh_ms_last = 0.0
        self.stat_upload_count_last = 0
        self.stat_upload_ms_last = 0.0
        self.shown = shown
        # Mapping from position to a pyglet `VertextList` for all shown blocks.
        self.vt = []
        self.vt_water = []
        self.pending_vt = []
        self.pending_vt_water = []
        self.vt_data = None
        self.vt_upload_prepared = False
        self.vt_upload_token = 0
        self.vt_upload_active_token = None
        self.vt_upload_use_pending = False
        self.vt_upload_target = None
        self.vt_uploaded_ranges = {
            "current": {"solid": [], "water": []},
            "pending": {"solid": [], "water": []},
        }
        self.vt_clear_pending = False
        self.vt_upload_solid = 0
        self.vt_upload_water = 0
        self.vt_solid_quads = 0
        self.vt_water_quads = 0
        self._last_draw_detail_frame = -1
        self.force_full_upload = False
        self.invalidate_vt = False
        self.mesh_built = False
        self.mesh_job_pending = False
        self.mesh_job_dirty = False
        self.mesh_job_priority = False
        self.mesh_gen = 0
        self.pending_batch = None
        self.pending_batch_water = None
        self.patch_vt = []
        self.edit_token = 0
        self.edit_inflight = False

    def _note_new_vt_data(self):
        """Advance the upload token for new mesh data."""
        self.vt_upload_token += 1
        self.vt_upload_active_token = None
        self.vt_upload_use_pending = False
        self.vt_upload_target = None
        self.vt_uploaded_ranges["current"]["solid"] = []
        self.vt_uploaded_ranges["current"]["water"] = []
        self.vt_uploaded_ranges["pending"]["solid"] = []
        self.vt_uploaded_ranges["pending"]["water"] = []
        self.vt_upload_prepared = False

    def draw(self, draw_invalid = True, allow_upload = True):
        if self.batch is None or self.batch_water is None:
            self.batch, self.batch_water = self.model.get_batches()
        if allow_upload and draw_invalid and self.invalidate_vt:
            self.check_show()
            self.invalidate_vt = False
            draw_invalid = False
        self.batch.draw()
        return draw_invalid

    def draw_water(self):
        """Draw only the water batch for this sector."""
        self.batch_water.draw()

    def __getitem__(self, position):
        pos = position - numpy.array(self.bposition)
        if len(pos.shape)>1:
            pos = pos.T
            return self.blocks[pos[0],pos[1],pos[2]]
        return self.blocks[pos[0],pos[1],pos[2]]

    def invalidate(self):
        self.invalidate_vt = True

    def _clear_vt_lists(self, reset_upload_state=True):
        for vt in self.vt:
            vt.delete()
        for vt in self.vt_water:
            vt.delete()
        self.vt = []
        self.vt_water = []
        self.vt_clear_pending = False
        self.vt_uploaded_ranges["current"]["solid"] = []
        self.vt_uploaded_ranges["current"]["water"] = []
        if reset_upload_state:
            self.vt_upload_prepared = False
            self.vt_upload_solid = 0
            self.vt_upload_water = 0
            self.vt_solid_quads = 0
            self.vt_water_quads = 0

    def _clear_pending_vt(self, recycle_batches=True):
        for vt in self.pending_vt:
            vt.delete()
        for vt in self.pending_vt_water:
            vt.delete()
        self.pending_vt = []
        self.pending_vt_water = []
        self.vt_uploaded_ranges["pending"]["solid"] = []
        self.vt_uploaded_ranges["pending"]["water"] = []
        if recycle_batches and self.pending_batch is not None:
            self.model.unused_batches.append((self.pending_batch, self.pending_batch_water))
        self.pending_batch = None
        self.pending_batch_water = None

    def _vt_quad_count(self, vt_data, key):
        if vt_data is None:
            return 0
        entry = vt_data if not isinstance(vt_data, dict) else vt_data.get(key)
        if not entry or entry[0] <= 0:
            return 0
        return int(entry[0] // 4)

    def _prepare_upload_state(self, vt_data):
        self.vt_upload_prepared = True
        self.vt_upload_solid = 0
        self.vt_upload_water = 0
        self.vt_solid_quads = self._vt_quad_count(vt_data, 'solid')
        self.vt_water_quads = self._vt_quad_count(vt_data, 'water')

    def check_show(self,add_to_batch = True):
        if add_to_batch and self.vt_data is not None:
            if self.vt_upload_active_token != self.vt_upload_token:
                use_pending = self.vt_clear_pending and (self.vt or self.vt_water)
                self.vt_upload_use_pending = use_pending
                target = "pending" if use_pending else "current"
                if self.vt_upload_target != target:
                    self.vt_uploaded_ranges[target]["solid"] = []
                    self.vt_uploaded_ranges[target]["water"] = []
                    self.vt_upload_target = target
                if use_pending and self.pending_vt and not self.vt_upload_prepared:
                    # Discard pending geometry only when a newer mesh arrives.
                    self._clear_pending_vt()
                self.vt_upload_active_token = self.vt_upload_token
            else:
                use_pending = self.vt_upload_use_pending
                target = self.vt_upload_target or ("pending" if use_pending else "current")
            if use_pending and self.pending_batch is None:
                self.pending_batch, self.pending_batch_water = self.model.get_batches()
            if not self.vt_upload_prepared:
                self._prepare_upload_state(self.vt_data)

            def _build_list_chunk(vt_tuple, group, batch, start_quad, max_quads):
                if not vt_tuple or vt_tuple[0] <= 0:
                    return None, 0, 0
                (count, v, t, n, c, l) = vt_tuple
                quad_total = int(count // 4)
                if start_quad >= quad_total:
                    return None, 0, quad_total
                end_quad = min(quad_total, start_quad + max_quads)
                v_arr = numpy.asarray(v, dtype='f4')
                t_arr = numpy.asarray(t, dtype='f4')
                n_arr = numpy.asarray(n, dtype='f4')
                channels = int(len(c) / (quad_total * 4)) if quad_total else 4
                c_arr = numpy.asarray(c, dtype='f4')
                light_channels = int(len(l) / (quad_total * 4)) if quad_total else 2
                l_arr = numpy.asarray(l, dtype='f4')
                v_slice = v_arr[start_quad * 12:end_quad * 12].reshape(-1, 4, 3)
                t_slice = t_arr[start_quad * 8:end_quad * 8].reshape(-1, 4, 2)
                n_slice = n_arr[start_quad * 12:end_quad * 12].reshape(-1, 4, 3)
                c_slice = c_arr[start_quad * 4 * channels:end_quad * 4 * channels].reshape(-1, 4, channels)
                l_slice = l_arr[start_quad * 4 * light_channels:end_quad * 4 * light_channels].reshape(-1, 4, light_channels)
                order = [0, 1, 2, 0, 2, 3]
                tri_verts = v_slice[:, order, :].reshape(-1, 3)
                tri_tex = t_slice[:, order, :].reshape(-1, 2)
                tri_norm = n_slice[:, order, :].reshape(-1, 3)
                tri_col = c_slice[:, order, :].reshape(-1, channels)
                tri_light = l_slice[:, order, :].reshape(-1, light_channels)
                tri_count = len(tri_verts)
                vt = self.model.program.vertex_list(
                    tri_count,
                    gl.GL_TRIANGLES,
                    batch=batch,
                    group=group,
                    position=('f', tri_verts.ravel().astype('f4')),
                    tex_coords=('f', tri_tex.ravel().astype('f4')),
                    normal=('f', tri_norm.ravel().astype('f4')),
                    color=('f', tri_col.ravel().astype('f4')),
                    light=('f', tri_light.ravel().astype('f4')),
                )
                return vt, end_quad - start_quad, quad_total

            tri_chunk = getattr(config, 'UPLOAD_TRIANGLE_CHUNK', None)
            max_quads = None
            if not self.force_full_upload and tri_chunk is not None and tri_chunk > 0:
                max_quads = max(1, int(tri_chunk // 2))

            t_upload_start = time.perf_counter()
            solid_data = self.model._get_vt_entry(self.vt_data, 'solid')
            water_data = self.model._get_vt_entry(self.vt_data, 'water')
            solid_list = self.pending_vt if use_pending else self.vt
            water_list = self.pending_vt_water if use_pending else self.vt_water
            solid_batch = self.pending_batch if use_pending else self.batch
            water_batch = self.pending_batch_water if use_pending else self.batch_water
            total_tris = 0
            if max_quads is None:
                vt, uploaded, _ = _build_list_chunk(
                    solid_data, self.group, solid_batch, self.vt_upload_solid, self.vt_solid_quads
                )
                if vt is not None:
                    solid_list.append(vt)
                if uploaded:
                    self.vt_uploaded_ranges[target]["solid"].append(
                        (self.vt_upload_solid, self.vt_upload_solid + uploaded)
                    )
                self.vt_upload_solid += uploaded
                total_tris += uploaded * 2
                vt, uploaded, _ = _build_list_chunk(
                    water_data, self.water_group, water_batch, self.vt_upload_water, self.vt_water_quads
                )
                if vt is not None:
                    water_list.append(vt)
                if uploaded:
                    self.vt_uploaded_ranges[target]["water"].append(
                        (self.vt_upload_water, self.vt_upload_water + uploaded)
                    )
                self.vt_upload_water += uploaded
                total_tris += uploaded * 2
            else:
                budget = max_quads
                if self.vt_upload_water < self.vt_water_quads:
                    vt, uploaded, _ = _build_list_chunk(
                        water_data, self.water_group, water_batch, self.vt_upload_water, budget
                    )
                    if vt is not None:
                        water_list.append(vt)
                    if uploaded:
                        self.vt_uploaded_ranges[target]["water"].append(
                            (self.vt_upload_water, self.vt_upload_water + uploaded)
                        )
                    self.vt_upload_water += uploaded
                    total_tris += uploaded * 2
                    budget = max(0, budget - uploaded)
                if budget and self.vt_upload_solid < self.vt_solid_quads:
                    vt, uploaded, _ = _build_list_chunk(
                        solid_data, self.group, solid_batch, self.vt_upload_solid, budget
                    )
                    if vt is not None:
                        solid_list.append(vt)
                    if uploaded:
                        self.vt_uploaded_ranges[target]["solid"].append(
                            (self.vt_upload_solid, self.vt_upload_solid + uploaded)
                        )
                    self.vt_upload_solid += uploaded
                    total_tris += uploaded * 2

            elapsed = (time.perf_counter() - t_upload_start) * 1000.0
            if total_tris:
                self.model._mesh_log(
                    f"upload sector={self.position} tris={total_tris} ms={elapsed:.1f}"
                )
            if getattr(config, 'MESH_LOG', False):
                try:
                    solid_entry = self.model._get_vt_entry(self.vt_data, 'solid')
                    water_entry = self.model._get_vt_entry(self.vt_data, 'water')
                    solid_quads = 0 if not solid_entry else int(solid_entry[0] // 4)
                    water_quads = 0 if not water_entry else int(water_entry[0] // 4)
                    vt_kind = 'dict' if isinstance(self.vt_data, dict) else 'tuple'
                    self.model._mesh_log(
                        f"upload_detail sector={self.position} vt={vt_kind} solid_quads={solid_quads} water_quads={water_quads}"
                    )
                except Exception:
                    pass

            if self.vt_upload_solid >= self.vt_solid_quads and self.vt_upload_water >= self.vt_water_quads:
                def _ranges_cover(total_quads, ranges):
                    if total_quads <= 0:
                        return True
                    if not ranges:
                        return False
                    ranges = sorted(ranges, key=lambda r: r[0])
                    cur = 0
                    for start, end in ranges:
                        if start > cur:
                            return False
                        if end > cur:
                            cur = end
                        if cur >= total_quads:
                            return True
                    return cur >= total_quads

                solid_ok = _ranges_cover(self.vt_solid_quads, self.vt_uploaded_ranges[target]["solid"])
                water_ok = _ranges_cover(self.vt_water_quads, self.vt_uploaded_ranges[target]["water"])
                target_vt = self.pending_vt if target == "pending" else self.vt
                target_vt_water = self.pending_vt_water if target == "pending" else self.vt_water
                expected_solid_verts = self.vt_solid_quads * 6
                expected_water_verts = self.vt_water_quads * 6
                actual_solid_verts = sum(getattr(vt, 'count', 0) for vt in target_vt)
                actual_water_verts = sum(getattr(vt, 'count', 0) for vt in target_vt_water)
                verts_ok = True
                if expected_solid_verts and actual_solid_verts < expected_solid_verts:
                    verts_ok = False
                if expected_water_verts and actual_water_verts < expected_water_verts:
                    verts_ok = False
                if not (solid_ok and water_ok and verts_ok):
                    if self.pending_vt or self.pending_vt_water:
                        self._clear_pending_vt()
                    self.force_full_upload = True
                    self.vt_upload_prepared = False
                    self.vt_upload_active_token = None
                    self.vt_upload_use_pending = False
                    self.vt_upload_target = None
                    self.vt_clear_pending = True
                    self.model._queue_upload(self, priority=self.edit_inflight)
                    return
                if use_pending:
                    if self.pending_batch is not None and self.pending_batch_water is not None:
                        old_vt = self.vt
                        old_vt_water = self.vt_water
                        old_batch = self.batch
                        old_batch_water = self.batch_water
                        self.vt = self.pending_vt
                        self.vt_water = self.pending_vt_water
                        self.batch = self.pending_batch
                        self.batch_water = self.pending_batch_water
                        self.pending_vt = []
                        self.pending_vt_water = []
                        self.pending_batch = None
                        self.pending_batch_water = None
                        for vt in old_vt:
                            vt.delete()
                        for vt in old_vt_water:
                            vt.delete()
                        self.model.unused_batches.append((old_batch, old_batch_water))
                    else:
                        self._clear_pending_vt()
                # Clear any temporary patch geometry once full mesh is uploaded.
                for pv in self.patch_vt:
                    pv.delete()
                self.patch_vt.clear()
                self.vt_data = None
                self.vt_upload_prepared = False
                self.vt_upload_active_token = None
                self.vt_upload_use_pending = False
                self.vt_upload_target = None
                self.vt_uploaded_ranges["current"]["solid"] = []
                self.vt_uploaded_ranges["current"]["water"] = []
                self.vt_uploaded_ranges["pending"]["solid"] = []
                self.vt_uploaded_ranges["pending"]["water"] = []
                self.vt_clear_pending = False
                self.force_full_upload = False
                self.edit_inflight = False
                self.invalidate_vt = False
                self.mesh_built = True
        elif add_to_batch and self.vt_data is None and self.invalidate_vt:
            # Lazy rebuild when vt data was cleared for rebuild.
            self.invalidate_vt = False
            self.model._submit_mesh_job(self)
            return False
        elif add_to_batch and self.vt_data is None and self.shown and self.mesh_built:
            pass


class ModelProxy(object):

    def __init__(self, program):

        # A TextureGroup manages an OpenGL texture.
        texture = image.load(TEXTURE_PATH).get_texture()
        self.program = program
        shader_group = pyglet.graphics.ShaderGroup(self.program)
        # Bind texture via TextureGroup under the shader group.
        self.group = TextureGroup(texture, parent=shader_group, order=0)
        # Water draws after opaque with depth writes disabled so the seafloor stays visible.
        self.water_group = TransparentWaterGroup(texture, parent=shader_group, order=1)
        self.unused_batches = []
        self.pending_uploads = deque()
        self.pending_upload_set = set()
        self.load_radius = getattr(config, 'LOAD_RADIUS', LOADED_SECTORS + 1)  # prefetch a wider ring
        self.keep_radius = getattr(config, 'KEEP_RADIUS', self.load_radius + 1)  # hysteresis before unloading
        self.sector_edit_tokens = {}
        self.mesh_single_worker = getattr(config, 'MESH_SINGLE_WORKER', False)
        self.mesh_results = queue.SimpleQueue()
        self.mesh_interrupt = threading.Event()
        self.pending_priority_jobs = 0
        self.mesh_jobs_submitted_total = 0
        self.mesh_jobs_completed_total = 0
        self.mesh_active_jobs = 0
        self.mesh_active_cap = getattr(config, 'MESH_ACTIVE_CAP', 1)
        self._mesh_submit_frame = -1
        self._priority_submitted_frame = -1
        self.loader_sectors_received_total = 0
        self.loader_recent = deque(maxlen=5)
        self.mesh_recent = deque(maxlen=5)
        if self.mesh_single_worker:
            self.mesh_executor = None
            self.mesh_executor_hi = None
            self._mesh_worker_stop = threading.Event()
            self._mesh_job_cv = threading.Condition()
            self._mesh_job_queue = deque()
            self._mesh_worker_thread = threading.Thread(
                target=self._mesh_worker_loop,
                name="MeshWorker",
                daemon=True,
            )
            self._mesh_worker_thread.start()
        else:
            self.mesh_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=getattr(config, 'MESH_WORKERS', 2)
            )
            self.mesh_executor_hi = concurrent.futures.ThreadPoolExecutor(
                max_workers=getattr(config, 'MESH_EDIT_WORKERS', 1)
            )
        self.frame_id = 0
        self.frame_start = None
        self.mesh_budget_deadline = None
        self.player_sector = None
        self.player_pos = None
        self.player_look = None
        self._last_queue_log_frame = -1
        self._last_missing_log_frame = -1
        self.loader_messages = queue.SimpleQueue()
        self._loader_stop = threading.Event()
        self._loader_thread = None
        self._defer_work_frame = -1
        self._deferred_mesh_jobs = {}
        self._priority_light_queue = deque()
        self._priority_light_set = set()
        self._priority_mesh_queue = deque()
        self._priority_mesh_set = set()
        self._edit_group_id = 0
        self._edit_group_members = {}
        self._edit_group_pending = {}
        self.stats_shown_sectors = 0
        self.stats_total_sectors = 0
        self.stats_drawn_tris_solid = 0
        self.stats_drawn_tris_water = 0
        self.stats_cull_ms = 0.0
        self.stats_batches_solid = 0
        self.stats_batches_water = 0
        self.stats_vlists_solid = 0
        self.stats_vlists_water = 0
        self.stats_draw_calls = 0
        self.mesh_backlog_last = 0
        self.stat_load_count_total = 0
        self.stat_load_ms_total = 0.0
        self.stat_light_count_total = 0
        self.stat_light_ms_total = 0.0
        self.stat_mesh_count_total = 0
        self.stat_mesh_ms_total = 0.0
        self.stat_upload_count_total = 0
        self.stat_upload_ms_total = 0.0
        self.stat_load_count_prev = 0
        self.stat_load_ms_prev = 0.0
        self.stat_light_count_prev = 0
        self.stat_light_ms_prev = 0.0
        self.stat_mesh_count_prev = 0
        self.stat_mesh_ms_prev = 0.0
        self.stat_upload_count_prev = 0
        self.stat_upload_ms_prev = 0.0
        self.stat_load_count_last = 0
        self.stat_load_ms_last = 0.0
        self.stat_light_count_last = 0
        self.stat_light_ms_last = 0.0
        self.stat_mesh_count_last = 0
        self.stat_mesh_ms_last = 0.0
        self.stat_upload_count_last = 0
        self.stat_upload_ms_last = 0.0
        self.stat_loader_block_defer_total = 0
        self.stat_loader_block_mesh_total = 0
        self.stat_loader_block_inflight_total = 0
        self.stat_loader_sent_total = 0
        self.stat_load_refresh_total = 0
        self.stat_load_candidates_total = 0
        self.stat_loader_block_defer_prev = 0
        self.stat_loader_block_mesh_prev = 0
        self.stat_loader_block_inflight_prev = 0
        self.stat_loader_sent_prev = 0
        self.stat_load_refresh_prev = 0
        self.stat_load_candidates_prev = 0
        self.stat_loader_block_defer_last = 0
        self.stat_loader_block_mesh_last = 0
        self.stat_loader_block_inflight_last = 0
        self.stat_loader_sent_last = 0
        self.stat_load_refresh_last = 0
        self.stat_load_candidates_last = 0

        # The world is stored in sector chunks.
        self.sectors = {}
        self.update_sectors_pos = []
        self.update_ref_pos = None
        self._load_candidates_time = 0.0
        self._load_candidates_look = None
        self._mesh_candidates = []
        self._mesh_candidates_time = 0.0
        self._mesh_candidates_ref = None
        self._mesh_candidates_look = None

        self.loader_requests = []
        self.active_loader_request = [None, None]
        self.n_requests = 0
        self.n_responses = 0
        self.world_seed = None
        self.player = None
        self.players = []
        self.host_id = None
        self._server_outbox = deque()
        self._entity_history = {}
        self._entity_interp_delay = float(getattr(config, "ENTITY_INTERP_DELAY", 0.15))
        self._entity_interp_extrap = float(getattr(config, "ENTITY_INTERP_EXTRAP", 0.25))
        self.remote_players = {}
        self.accepted_player_name = None
        self.pending_spawn_state = None
        self.entity_snapshot_requested = False
        self.name_request_pending = False
        self.name_sent_once = False
        self.pending_entity_seed = None
        self.player_state_request = False

        loader_server_pipe = None
        self.server = None
        if config.DEBUG_SINGLE_BLOCK:
            # Create a single sector with one visible block for debugging.
            batch_solid, batch_water = self.get_batches()
            s = SectorProxy((0,0,0), batch_solid, batch_water, self.group, self.water_group, self, shown=True)
            s.blocks[:] = 0
            cx, cy, cz = SECTOR_SIZE//2, SECTOR_HEIGHT//2, SECTOR_SIZE//2
            block_id = 1  # dirt
            s.blocks[cx, cy, cz] = block_id
            vt_data = self._build_block_vt(block_id, numpy.array([cx, cy, cz], dtype=numpy.float32))
            s.vt_data = vt_data
            s._note_new_vt_data()
            s.check_show()
            self.sectors[s.position] = s
            self.loader = None
            logutil.log("WORLD", f"single block spawned at world coords {(cx, cy, cz)}", level="DEBUG")
        else:
            if config.SERVER_IP is not None:
                logutil.log("MAIN", f"Starting server on {config.SERVER_IP}")
                self.server = server_connection.start_server_connection(config.SERVER_IP)
                if self.server.proc is not None and not self.server.proc.is_alive():
                    print(f"CLIENT: server connection failed for {config.SERVER_IP}:{config.SERVER_PORT}")
                    self.server = None
                if self.server is not None:
                    loader_server_pipe = self.server.loader_pipe
            logutil.log("MAIN", "Starting sector loader")
            self.loader = world_loader.start_loader(loader_server_pipe)
            self.loader_requests.append(['get_seed', []])
            self._loader_thread = threading.Thread(
                target=self._loader_recv_loop,
                name="LoaderRecv",
                daemon=True,
            )
            self._loader_thread.start()

    def _mesh_worker_loop(self):
        while not self._mesh_worker_stop.is_set():
            with self._mesh_job_cv:
                while not self._mesh_job_queue and not self._mesh_worker_stop.is_set():
                    self._mesh_job_cv.wait()
                if self._mesh_worker_stop.is_set():
                    break
                (job, blocks, light_blocks, pos, ao_strength, ao_enabled, ambient,
                 incoming_sky, incoming_torch, internal_sky_floor, internal_sky_side,
                 internal_torch, recompute_internal, interrupt_event, build_mesh) = self._mesh_job_queue.popleft()
            try:
                vt_data = self._build_mesh_job(
                    blocks,
                    light_blocks,
                    pos,
                    ao_strength,
                    ao_enabled,
                    ambient,
                    incoming_sky,
                    incoming_torch,
                    internal_sky_floor,
                    internal_sky_side,
                    internal_torch,
                    recompute_internal,
                    interrupt_event,
                    build_mesh,
                )
            except Exception as e:
                job.set_exception(e)
            else:
                job.set_result(vt_data)
            self.mesh_results.put(job)

    def _enqueue_mesh_job(
        self,
        job,
        blocks,
        light_blocks,
        pos,
        ao_strength,
        ao_enabled,
        ambient,
        incoming_sky,
        incoming_torch,
        internal_sky_floor,
        internal_sky_side,
        internal_torch,
        recompute_internal,
        interrupt_event,
        build_mesh,
        priority=False,
    ):
        with self._mesh_job_cv:
            if priority:
                self._mesh_job_queue.appendleft((
                    job,
                    blocks,
                    light_blocks,
                    pos,
                    ao_strength,
                    ao_enabled,
                    ambient,
                    incoming_sky,
                    incoming_torch,
                    internal_sky_floor,
                    internal_sky_side,
                    internal_torch,
                    recompute_internal,
                    interrupt_event,
                    build_mesh,
                ))
            else:
                self._mesh_job_queue.append((
                    job,
                    blocks,
                    light_blocks,
                    pos,
                    ao_strength,
                    ao_enabled,
                    ambient,
                    incoming_sky,
                    incoming_torch,
                    internal_sky_floor,
                    internal_sky_side,
                    internal_torch,
                    recompute_internal,
                    interrupt_event,
                    build_mesh,
                ))
            self._mesh_job_cv.notify()

    def _loader_recv_loop(self):
        """Continuously drain loader responses so the loader never blocks on send."""
        while not self._loader_stop.is_set():
            try:
                raw = self.loader.recv()
            except EOFError:
                self.loader_messages.put(('__eof__', None))
                return
            except Exception as e:
                self.loader_messages.put(('__error__', e))
                return
            self.loader_messages.put(raw)

    def _has_budget(self, frame_start, upload_budget):
        if frame_start is None or upload_budget is None:
            return True
        return (time.perf_counter() - frame_start) < upload_budget

    def _mesh_log(self, msg):
        if not getattr(config, 'MESH_LOG', False):
            return
        logutil.log("MESH", msg)

    def _note_sector_load(self, sector_pos, ms):
        self.stat_load_count_total += 1
        self.stat_load_ms_total += ms
        sector = self.sectors.get(sector_pos)
        if sector is None:
            return
        sector.stat_load_count_total += 1
        sector.stat_load_ms_total += ms

    def _note_sector_light(self, sector_pos, ms):
        self.stat_light_count_total += 1
        self.stat_light_ms_total += ms
        sector = self.sectors.get(sector_pos)
        if sector is None:
            return
        sector.stat_light_count_total += 1
        sector.stat_light_ms_total += ms

    def _note_sector_mesh(self, sector_pos, ms):
        self.stat_mesh_count_total += 1
        self.stat_mesh_ms_total += ms
        sector = self.sectors.get(sector_pos)
        if sector is None:
            return
        sector.stat_mesh_count_total += 1
        sector.stat_mesh_ms_total += ms

    def _note_sector_upload(self, sector, ms):
        self.stat_upload_count_total += 1
        self.stat_upload_ms_total += ms
        sector.stat_upload_count_total += 1
        sector.stat_upload_ms_total += ms

    def _defer_mesh_job(self, sector, priority=False):
        """Queue a mesh job to submit on a later frame."""
        if sector is None:
            return
        prev = self._deferred_mesh_jobs.get(sector, False)
        self._deferred_mesh_jobs[sector] = prev or bool(priority)

    def _remove_from_edit_group(self, sector):
        gid = getattr(sector, "edit_group_id", None)
        if gid is None:
            return
        pending = self._edit_group_pending.get(gid)
        if pending is not None:
            pending.discard(sector)
        members = self._edit_group_members.get(gid)
        if members is not None:
            members.discard(sector)
        sector.edit_group_id = None

    def _start_edge_edit_group(self, sectors):
        self._edit_group_id += 1
        gid = self._edit_group_id
        members = set()
        for sector in sectors:
            if sector is None:
                continue
            self._remove_from_edit_group(sector)
            sector.defer_upload = True
            sector.edit_group_id = gid
            members.add(sector)
        if members:
            self._edit_group_members[gid] = set(members)
            self._edit_group_pending[gid] = set(members)
        return gid

    def _mark_edit_group_complete(self, sector):
        gid = getattr(sector, "edit_group_id", None)
        if gid is None:
            sector.defer_upload = False
            return
        pending = self._edit_group_pending.get(gid)
        if pending is None:
            sector.defer_upload = False
            sector.edit_group_id = None
            return
        pending.discard(sector)
        if pending:
            return
        members = self._edit_group_members.pop(gid, set())
        self._edit_group_pending.pop(gid, None)
        for member in members:
            if member is None:
                continue
            member.defer_upload = False
            member.edit_group_id = None
            if member.shown:
                self._queue_upload(member, priority=True)

    def _queue_priority_light(self, sector):
        if sector is None:
            return
        if sector in self._priority_light_set:
            return
        self._priority_light_set.add(sector)
        self._priority_light_queue.append(sector)

    def _queue_priority_mesh(self, sector):
        if sector is None:
            return
        if sector in self._priority_mesh_set:
            return
        self._priority_mesh_set.add(sector)
        self._priority_mesh_queue.append(sector)

    def _process_priority_work(self):
        """Run priority light/mesh steps on the main thread within a small budget."""
        start = time.perf_counter()
        budget_ms = float(getattr(config, 'PRIORITY_WORK_BUDGET_MS', 1.5))
        did_work = False
        while True:
            if (time.perf_counter() - start) * 1000.0 >= budget_ms:
                return did_work
            if self._priority_light_queue:
                sector = self._priority_light_queue.popleft()
                self._priority_light_set.discard(sector)
                if sector.mesh_job_pending:
                    self._queue_priority_light(sector)
                    did_work = True
                    continue
                if not self._neighbors_ready(sector, require_diagonals=True):
                    self._queue_priority_light(sector)
                    did_work = True
                    continue
                light_blocks = self._gather_blocks_tile_3x3(sector, allow_missing=False)
                if light_blocks is None:
                    self._queue_priority_light(sector)
                    did_work = True
                    continue
                incoming_sky, incoming_torch = self._build_incoming_from_neighbors(sector)
                ao_strength = getattr(config, 'AO_STRENGTH', 0.0)
                ao_enabled = getattr(config, 'AO_ENABLED', True)
                ambient = getattr(config, 'AMBIENT_LIGHT', 0.0)
                result = self._build_mesh_job(
                    None,
                    light_blocks,
                    sector.position,
                    ao_strength,
                    ao_enabled,
                    ambient,
                    incoming_sky,
                    incoming_torch,
                    sector.sky_floor,
                    sector.sky_side,
                    sector.torch_side,
                    True,
                    None,
                    False,
                )
                light_ms = float(result.get("light_ms") or 0.0)
                if light_ms > 0.0:
                    self._note_sector_light(sector.position, light_ms)
                sector.sky_floor = result.get("sky_floor", sector.sky_floor)
                sector.sky_side = result.get("sky_side", sector.sky_side)
                sector.torch_side = result.get("torch_side", sector.torch_side)
                sector.edge_sky_counts = result.get("edge_sky_counts", sector.edge_sky_counts)
                sector.edge_torch_counts = result.get("edge_torch_counts", sector.edge_torch_counts)
                sector.light_dirty_internal = False
                sector.light_dirty_incoming = False
                sector.light_neighbors_ready = True
                outgoing_sky = result.get("outgoing_sky") or {}
                outgoing_torch = result.get("outgoing_torch") or {}
                if outgoing_sky or outgoing_torch:
                    propagate_dirty = sector.light_dirty_from_edit or sector.edit_inflight
                    self._apply_outgoing_light(sector, outgoing_sky, outgoing_torch, propagate_dirty=propagate_dirty)
                sector.light_dirty_from_edit = False
                sector.light_initialized = True
                self._queue_priority_mesh(sector)
                did_work = True
                continue
            if self._priority_mesh_queue:
                sector = self._priority_mesh_queue.popleft()
                self._priority_mesh_set.discard(sector)
                if sector.mesh_job_pending:
                    self._queue_priority_mesh(sector)
                    did_work = True
                    continue
                if not self._mesh_ready(sector):
                    self._queue_priority_mesh(sector)
                    did_work = True
                    continue
                ignore_light_dirty = getattr(config, 'PRIORITY_MESH_IGNORE_LIGHT_DIRTY', True)
                if sector.light_dirty_internal and not ignore_light_dirty:
                    self._queue_priority_mesh(sector)
                    did_work = True
                    continue
                self._rebuild_sector_now_fast(sector, priority=True)
                if getattr(sector, "defer_upload", False):
                    self._mark_edit_group_complete(sector)
                did_work = True
                continue
            return did_work

    def _maybe_log_queue_state(self):
        if not getattr(config, 'LOG_QUEUE_STATE', False):
            return
        if self.frame_id == self._last_queue_log_frame:
            return
        self._last_queue_log_frame = self.frame_id
        inflight = self.n_requests - self.n_responses
        pending_sector_reqs = sum(1 for r in self.loader_requests if r[0] == 'sector_blocks')
        pending_mesh = 0
        for sector in self.sectors.values():
            if sector.mesh_job_pending:
                continue
            if sector.vt_data is None:
                pending_mesh += 1
                continue
            if self._needs_light(sector) and self._lighting_ready(sector):
                pending_mesh += 1
        logutil.log(
            "QUEUE",
            f"inflight={inflight} loader_requests={len(self.loader_requests)} "
            f"pending_sector_reqs={pending_sector_reqs} update_queue={len(self.update_sectors_pos)} "
            f"pending_mesh={pending_mesh} pending_uploads={len(self.pending_uploads)}",
        )

    def _maybe_log_missing_sectors(self, center):
        if not getattr(config, 'LOG_MISSING_SECTORS', False):
            return
        every = getattr(config, 'LOG_MISSING_SECTORS_EVERY_N_FRAMES', 30)
        if self.frame_id - self._last_missing_log_frame < every:
            return
        self._last_missing_log_frame = self.frame_id
        missing = []
        for dx, dz in itertools.product((-1, 0, 1), repeat=2):
            pos = (center[0] + dx * SECTOR_SIZE, 0, center[2] + dz * SECTOR_SIZE)
            if pos not in self.sectors:
                missing.append(pos)
        logutil.log("QUEUE", f"missing_3x3={missing}")

    def _should_refresh_load_candidates(self, ref_sector, look_vec):
        if self.update_ref_pos != ref_sector:
            return True
        if self._load_candidates_look is None or look_vec is None:
            if self._load_candidates_look != look_vec:
                return True
        else:
            lx, lz = look_vec[0], look_vec[2]
            ox, oz = self._load_candidates_look[0], self._load_candidates_look[2]
            llen = math.hypot(lx, lz)
            olen = math.hypot(ox, oz)
            if llen < 1e-6 or olen < 1e-6:
                return True
            dot = (lx / llen) * (ox / olen) + (lz / llen) * (oz / olen)
            if dot < 0.98:
                return True
        refresh_ms = getattr(config, 'LOAD_CANDIDATE_REFRESH_MS', None)
        if refresh_ms is None:
            return False
        return (time.perf_counter() - self._load_candidates_time) >= (refresh_ms / 1000.0)

    def _refresh_load_candidates(self, ref_sector, player_pos, look_vec, frustum_circle=None):
        if not self._should_refresh_load_candidates(ref_sector, look_vec):
            return False
        self.update_ref_pos = ref_sector
        self._load_candidates_look = look_vec
        self.update_sectors_pos = self._compute_load_candidates(
            ref_sector, player_pos, look_vec, frustum_circle
        )
        self._load_candidates_time = time.perf_counter()
        return True

    def _should_refresh_mesh_candidates(self, ref_sector, look_vec):
        if self._mesh_candidates_ref != ref_sector:
            return True
        if self._mesh_candidates_look is None or look_vec is None:
            if self._mesh_candidates_look != look_vec:
                return True
        else:
            lx, lz = look_vec[0], look_vec[2]
            ox, oz = self._mesh_candidates_look[0], self._mesh_candidates_look[2]
            llen = math.hypot(lx, lz)
            olen = math.hypot(ox, oz)
            if llen < 1e-6 or olen < 1e-6:
                return True
            dot = (lx / llen) * (ox / olen) + (lz / olen) * (oz / olen)
            if dot < 0.98:
                return True
        refresh_ms = getattr(config, 'MESH_CANDIDATE_REFRESH_MS', None)
        if refresh_ms is None:
            return False
        return (time.perf_counter() - self._mesh_candidates_time) >= (refresh_ms / 1000.0)

    def _refresh_mesh_candidates(self, ref_sector, player_pos, look_vec, frustum_circle=None):
        if not self._should_refresh_mesh_candidates(ref_sector, look_vec):
            return False
        self._mesh_candidates_ref = ref_sector
        self._mesh_candidates_look = look_vec
        self._mesh_candidates = self._compute_mesh_candidates(
            ref_sector, player_pos, look_vec, frustum_circle
        )
        self._mesh_candidates_time = time.perf_counter()
        return True

    def _needs_light(self, sector):
        if sector.light_initialized and not (sector.light_dirty_from_edit or sector.edit_inflight or sector.light_dirty_incoming):
            return False
        return sector.light_dirty_internal or sector.light_dirty_incoming

    def _mesh_ready(self, sector):
        return self._neighbors_ready(sector, require_diagonals=False)

    def _lighting_ready(self, sector):
        return self._neighbors_ready(sector, require_diagonals=sector.light_dirty_internal)

    def _merge_neighbor_lighting(self, sector):
        return

    def _mesh_backlog_count(self):
        """Count mesh work ready for visible or nearby sectors."""
        count = 0
        for sector in self.sectors.values():
            if not (sector.shown or self._in_priority_neighborhood(sector.position, radius=1)):
                continue
            if not self._neighbors_ready(sector, require_diagonals=sector.light_dirty_internal):
                continue
            if sector.mesh_job_pending:
                count += 1
                continue
            needs_mesh = (sector.vt_data is None and not sector.mesh_built)
            if needs_mesh:
                if self._mesh_ready(sector):
                    count += 1
                continue
            if sector.shown and self._needs_light(sector):
                count += 1
        self.mesh_backlog_last = count
        return count

    def _has_mesh_backlog(self):
        """Return True when there is mesh work ready to run on loaded sectors."""
        return self._mesh_backlog_count() > 0

    def _neighbors_have_light(self, sector):
        x0, _, z0 = sector.position
        for dx, dz in NEIGHBOR_OFFSETS_8:
            pos = (x0 + dx * SECTOR_SIZE, 0, z0 + dz * SECTOR_SIZE)
            n = self.sectors.get(pos)
            if n is None:
                return False
            if n.light_dirty_internal:
                return False
        return True

    def _neighbors_light_initialized(self, sector):
        x0, _, z0 = sector.position
        for dx, dz in NEIGHBOR_OFFSETS_8:
            pos = (x0 + dx * SECTOR_SIZE, 0, z0 + dz * SECTOR_SIZE)
            n = self.sectors.get(pos)
            if n is None or not n.light_initialized:
                return False
        return True

    def _neighbors_missing(self, sector):
        return False

    def _in_priority_neighborhood(self, sector_pos, radius=1):
        if self.player_sector is None:
            return False
        px, _, pz = self.player_sector
        return (abs(sector_pos[0] - px) <= SECTOR_SIZE * radius
                and abs(sector_pos[2] - pz) <= SECTOR_SIZE * radius)

    def process_pending_mesh_jobs(self, frustum_circle=None, allow_submit=True):
        """Submit mesh jobs using current sector state (unmeshed -> unlit)."""
        if not allow_submit:
            return
        if self._defer_work_frame == self.frame_id:
            return
        if self.player_sector is None:
            return
        self._refresh_mesh_candidates(
            self.player_sector,
            self.player_pos,
            self.player_look,
            frustum_circle,
        )
        candidates = getattr(self, "_mesh_candidates", None) or []
        if not candidates:
            return
        submit_budget = max(0, self.mesh_active_cap - self.mesh_active_jobs)
        submitted = 0
        for _, sector in candidates:
            if submitted >= submit_budget:
                break
            if sector.mesh_job_pending:
                continue
            if not self._neighbors_ready(sector, require_diagonals=sector.light_dirty_internal):
                continue
            needs_mesh = (sector.vt_data is None and not sector.mesh_built)
            if needs_mesh:
                if not self._mesh_ready(sector):
                    continue
                reason = "unmeshed"
            else:
                if not self._needs_light(sector):
                    continue
                reason = "light"
            is_priority = self._in_priority_neighborhood(sector.position)
            self._mesh_log(f"queue sector={sector.position} reason={reason}")
            before = self.mesh_active_jobs
            self._submit_mesh_job(sector, priority=is_priority)
            if self.mesh_active_jobs > before:
                submitted += 1

    def _get_vt_entry(self, vt_data, key=None):
        """Return a single vt tuple from a dict or bare tuple."""
        if vt_data is None:
            return None
        if isinstance(vt_data, dict):
            if key is None:
                return vt_data.get('solid') or vt_data.get('water')
            return vt_data.get(key)
        return vt_data

    def _triangles_in_vt(self, vt_data):
        """Return triangle count for a vt_data tuple (quads -> tris)."""
        if not vt_data:
            return 0

        def _tri_count(entry):
            if not entry:
                return 0
            quad_count = entry[0] / 4.0
            return int(quad_count * 2)

        if isinstance(vt_data, dict):
            return _tri_count(vt_data.get('solid')) + _tri_count(vt_data.get('water'))
        return _tri_count(vt_data)

    def _upload_chunk_quads(self):
        tri_chunk = getattr(config, 'UPLOAD_TRIANGLE_CHUNK', None)
        if tri_chunk is None or tri_chunk <= 0:
            return None
        return max(1, int(tri_chunk // 2))

    def _triangles_in_vt_chunk(self, sector):
        """Return triangle count for the next upload chunk for a sector."""
        if sector.vt_data is None:
            return 0
        if sector.force_full_upload:
            return self._triangles_in_vt(sector.vt_data)
        chunk_quads = self._upload_chunk_quads()
        if chunk_quads is None:
            return self._triangles_in_vt(sector.vt_data)
        if not sector.vt_upload_prepared:
            sector._prepare_upload_state(sector.vt_data)
        if sector.vt_upload_solid < sector.vt_solid_quads:
            remaining = sector.vt_solid_quads - sector.vt_upload_solid
        elif sector.vt_upload_water < sector.vt_water_quads:
            remaining = sector.vt_water_quads - sector.vt_upload_water
        else:
            return 0
        return min(chunk_quads, remaining) * 2

    @staticmethod
    def _build_mesh_job(
        blocks,
        light_blocks,
        position,
        ao_strength,
        ao_enabled,
        ambient,
        incoming_sky,
        incoming_torch,
        internal_sky_floor,
        internal_sky_side,
        internal_torch,
        recompute_internal,
        interrupt_event=None,
        build_mesh=True,
        debug_light_grids=False,
    ):
        """Build vt_data for a sector snapshot; runs off the main thread."""
        sidefill_enabled = getattr(config, 'SKY_SIDEFILL_ENABLED', True)
        torch_fill_enabled = getattr(config, 'TORCH_FILL_ENABLED', True)
        outgoing_boundary_only = getattr(config, 'LIGHT_OUTGOING_BOUNDARY_ONLY', True)
        def _pack_list(light_grid):
            coords = numpy.argwhere(light_grid > 0)
            if coords.size == 0:
                return EMPTY_LIGHT_LIST
            values = light_grid[coords[:, 0], coords[:, 1], coords[:, 2]]
            out = numpy.empty((len(coords), 4), dtype=numpy.uint8)
            out[:, :3] = coords.astype(numpy.uint8, copy=False)
            out[:, 3] = values.astype(numpy.uint8, copy=False)
            return out

        def _pack_boundary(light_grid, dx, dz):
            sx = light_grid.shape[0]
            if dx != 0 and dz != 0:
                x = 0 if dx < 0 else sx - 1
                z = 0 if dz < 0 else sx - 1
                col = light_grid[x, :, z]
                ys = numpy.nonzero(col)[0]
                if ys.size == 0:
                    return EMPTY_LIGHT_LIST
                out = numpy.empty((len(ys), 4), dtype=numpy.uint8)
                out[:, 0] = x
                out[:, 1] = ys.astype(numpy.uint8, copy=False)
                out[:, 2] = z
                out[:, 3] = col[ys].astype(numpy.uint8, copy=False)
                return out
            if dx != 0:
                x = 0 if dx < 0 else sx - 1
                plane = light_grid[x, :, :]
                coords = numpy.argwhere(plane > 0)
                if coords.size == 0:
                    return EMPTY_LIGHT_LIST
                out = numpy.empty((len(coords), 4), dtype=numpy.uint8)
                out[:, 0] = x
                out[:, 1] = coords[:, 0].astype(numpy.uint8, copy=False)
                out[:, 2] = coords[:, 1].astype(numpy.uint8, copy=False)
                out[:, 3] = plane[coords[:, 0], coords[:, 1]].astype(numpy.uint8, copy=False)
                return out
            z = 0 if dz < 0 else sx - 1
            plane = light_grid[:, :, z]
            coords = numpy.argwhere(plane > 0)
            if coords.size == 0:
                return EMPTY_LIGHT_LIST
            out = numpy.empty((len(coords), 4), dtype=numpy.uint8)
            out[:, 0] = coords[:, 0].astype(numpy.uint8, copy=False)
            out[:, 1] = coords[:, 1].astype(numpy.uint8, copy=False)
            out[:, 2] = z
            out[:, 3] = plane[coords[:, 0], coords[:, 1]].astype(numpy.uint8, copy=False)
            return out

        def _pack_outgoing(light_grid):
            if not outgoing_boundary_only:
                outgoing_chunk = _pack_list(light_grid)
                return {offset: outgoing_chunk for offset in NEIGHBOR_OFFSETS_8}
            outgoing = {}
            for dx, dz in NEIGHBOR_OFFSETS_8:
                outgoing[(dx, dz)] = _pack_boundary(light_grid, dx, dz)
            return outgoing

        def _apply_list(light_grid, entries, offset=(0, 0, 0)):
            if entries is None or len(entries) == 0:
                return
            coords = entries[:, :3].astype(numpy.intp, copy=False)
            if offset != (0, 0, 0):
                coords = coords + numpy.array(offset, dtype=numpy.intp)
            vals = entries[:, 3].astype(light_grid.dtype, copy=False)
            numpy.maximum.at(light_grid, (coords[:, 0], coords[:, 1], coords[:, 2]), vals)

        def _relax_4way(light_grid, air_mask):
            for _ in range(MAX_LIGHT - 1):
                neighbor_max = numpy.zeros_like(light_grid)
                neighbor_max[1:, :, :] = numpy.maximum(neighbor_max[1:, :, :], light_grid[:-1, :, :] - 1)
                neighbor_max[:-1, :, :] = numpy.maximum(neighbor_max[:-1, :, :], light_grid[1:, :, :] - 1)
                neighbor_max[:, :, 1:] = numpy.maximum(neighbor_max[:, :, 1:], light_grid[:, :, :-1] - 1)
                neighbor_max[:, :, :-1] = numpy.maximum(neighbor_max[:, :, :-1], light_grid[:, :, 1:] - 1)
                neighbor_max = numpy.maximum(neighbor_max, 0)
                new_light = numpy.where(air_mask, numpy.maximum(light_grid, neighbor_max), 0)
                if numpy.array_equal(new_light, light_grid):
                    break
                light_grid = new_light
            return light_grid

        def _relax_4way_sloped(light_grid, air_mask):
            """Relax sideways, allowing 1-block vertical steps per lateral move."""
            for _ in range(MAX_LIGHT - 1):
                neighbor_max = numpy.zeros_like(light_grid)
                neighbor_max[1:, :, :] = numpy.maximum(neighbor_max[1:, :, :], light_grid[:-1, :, :] - 1)
                neighbor_max[:-1, :, :] = numpy.maximum(neighbor_max[:-1, :, :], light_grid[1:, :, :] - 1)
                neighbor_max[:, :, 1:] = numpy.maximum(neighbor_max[:, :, 1:], light_grid[:, :, :-1] - 1)
                neighbor_max[:, :, :-1] = numpy.maximum(neighbor_max[:, :, :-1], light_grid[:, :, 1:] - 1)
                neighbor_max[1:, 1:, :] = numpy.maximum(neighbor_max[1:, 1:, :], light_grid[:-1, :-1, :] - 1)
                neighbor_max[1:, :-1, :] = numpy.maximum(neighbor_max[1:, :-1, :], light_grid[:-1, 1:, :] - 1)
                neighbor_max[:-1, 1:, :] = numpy.maximum(neighbor_max[:-1, 1:, :], light_grid[1:, :-1, :] - 1)
                neighbor_max[:-1, :-1, :] = numpy.maximum(neighbor_max[:-1, :-1, :], light_grid[1:, 1:, :] - 1)
                neighbor_max[:, 1:, 1:] = numpy.maximum(neighbor_max[:, 1:, 1:], light_grid[:, :-1, :-1] - 1)
                neighbor_max[:, 1:, :-1] = numpy.maximum(neighbor_max[:, 1:, :-1], light_grid[:, :-1, 1:] - 1)
                neighbor_max[:, :-1, 1:] = numpy.maximum(neighbor_max[:, :-1, 1:], light_grid[:, 1:, :-1] - 1)
                neighbor_max[:, :-1, :-1] = numpy.maximum(neighbor_max[:, :-1, :-1], light_grid[:, 1:, 1:] - 1)
                neighbor_max = numpy.maximum(neighbor_max, 0)
                new_light = numpy.where(air_mask, numpy.maximum(light_grid, neighbor_max), 0)
                if numpy.array_equal(new_light, light_grid):
                    break
                light_grid = new_light
            return light_grid

        def _relax_6way(light_grid, air_mask):
            for _ in range(MAX_LIGHT - 1):
                neighbor_max = numpy.zeros_like(light_grid)
                neighbor_max[1:, :, :] = numpy.maximum(neighbor_max[1:, :, :], light_grid[:-1, :, :] - 1)
                neighbor_max[:-1, :, :] = numpy.maximum(neighbor_max[:-1, :, :], light_grid[1:, :, :] - 1)
                neighbor_max[:, 1:, :] = numpy.maximum(neighbor_max[:, 1:, :], light_grid[:, :-1, :] - 1)
                neighbor_max[:, :-1, :] = numpy.maximum(neighbor_max[:, :-1, :], light_grid[:, 1:, :] - 1)
                neighbor_max[:, :, 1:] = numpy.maximum(neighbor_max[:, :, 1:], light_grid[:, :, :-1] - 1)
                neighbor_max[:, :, :-1] = numpy.maximum(neighbor_max[:, :, :-1], light_grid[:, :, 1:] - 1)
                neighbor_max = numpy.maximum(neighbor_max, 0)
                new_light = numpy.where(air_mask, numpy.maximum(light_grid, neighbor_max), 0)
                if numpy.array_equal(new_light, light_grid):
                    break
                light_grid = new_light
            return light_grid

        def _unique_frontier(coords, levels, shape):
            if coords.size == 0:
                return coords, levels
            area = shape[1] * shape[2]
            flat = coords[:, 0] * area + coords[:, 1] * shape[2] + coords[:, 2]
            order = numpy.argsort(flat)
            flat = flat[order]
            levels = levels[order]
            unique, idx = numpy.unique(flat, return_index=True)
            max_levels = numpy.maximum.reduceat(levels, idx)
            coords_u = numpy.empty((len(unique), 3), dtype=numpy.intp)
            coords_u[:, 0] = unique // area
            rem = unique % area
            coords_u[:, 1] = rem // shape[2]
            coords_u[:, 2] = rem % shape[2]
            return coords_u, max_levels

        def _relax_bfs(light_grid, air_mask, offsets, debug_updates=None):
            coords = numpy.argwhere(light_grid > 0)
            if coords.size == 0:
                return light_grid
            levels = light_grid[coords[:, 0], coords[:, 1], coords[:, 2]].astype(light_grid.dtype, copy=False)
            shape = light_grid.shape
            offsets_arr = numpy.array(offsets, dtype=numpy.intp)
            # Prefilter initial frontier: keep only sources that can improve a neighbor.
            keep = levels > 1
            if offsets_arr.size > 0 and numpy.any(keep):
                keep_any = numpy.zeros(len(coords), dtype=bool)
                for off in offsets_arr:
                    n_coords = coords + off
                    mask = (
                        (n_coords[:, 0] >= 0) & (n_coords[:, 0] < shape[0])
                        & (n_coords[:, 1] >= 0) & (n_coords[:, 1] < shape[1])
                        & (n_coords[:, 2] >= 0) & (n_coords[:, 2] < shape[2])
                    )
                    if not numpy.any(mask):
                        continue
                    idx = numpy.nonzero(mask)[0]
                    n_coords_m = n_coords[mask]
                    n_levels = levels[mask] - 1
                    air = air_mask[n_coords_m[:, 0], n_coords_m[:, 1], n_coords_m[:, 2]]
                    if not numpy.any(air):
                        continue
                    idx_air = idx[air]
                    n_coords_a = n_coords_m[air]
                    n_levels_a = n_levels[air]
                    existing = light_grid[n_coords_a[:, 0], n_coords_a[:, 1], n_coords_a[:, 2]]
                    better = n_levels_a > existing
                    if numpy.any(better):
                        keep_any[idx_air[better]] = True
                keep &= keep_any
            if numpy.any(keep):
                coords = coords[keep]
                levels = levels[keep]
            else:
                return light_grid
            while True:
                active = levels > 1
                if not numpy.any(active):
                    break
                coords = coords[active]
                levels = levels[active] - 1
                n_coords = numpy.repeat(coords, len(offsets_arr), axis=0)
                n_coords += numpy.tile(offsets_arr, (len(coords), 1))
                n_levels = numpy.repeat(levels, len(offsets_arr))
                mask = (
                    (n_coords[:, 0] >= 0) & (n_coords[:, 0] < shape[0])
                    & (n_coords[:, 1] >= 0) & (n_coords[:, 1] < shape[1])
                    & (n_coords[:, 2] >= 0) & (n_coords[:, 2] < shape[2])
                )
                if not numpy.any(mask):
                    break
                n_coords = n_coords[mask]
                n_levels = n_levels[mask]
                if n_coords.size == 0:
                    break
                air = air_mask[n_coords[:, 0], n_coords[:, 1], n_coords[:, 2]]
                if not numpy.any(air):
                    break
                n_coords = n_coords[air]
                n_levels = n_levels[air]
                if n_coords.size == 0:
                    break
                existing = light_grid[n_coords[:, 0], n_coords[:, 1], n_coords[:, 2]]
                better = n_levels > existing
                if not numpy.any(better):
                    break
                n_coords = n_coords[better]
                n_levels = n_levels[better]
                if n_coords.size == 0:
                    break
                n_coords, n_levels = _unique_frontier(n_coords, n_levels, shape)
                light_grid[n_coords[:, 0], n_coords[:, 1], n_coords[:, 2]] = n_levels
                if debug_updates is not None:
                    debug_updates[n_coords[:, 0], n_coords[:, 1], n_coords[:, 2]] = True
                coords = n_coords
                levels = n_levels
            return light_grid

        if build_mesh:
            # Compute exposed faces.
            solid = (blocks > 0) & (blocks != WATER)
            exposed_faces = numpy.zeros(blocks.shape + (6,), dtype=bool)
            neighbor = blocks[:, 1:, :]
            neighbor_occ = BLOCK_OCCLUDES[neighbor].astype(bool) | (BLOCK_OCCLUDES_SAME[neighbor].astype(bool) & (neighbor == blocks[:, :-1, :]))
            exposed_faces[:, :-1, :, 0] = ~neighbor_occ
            neighbor = blocks[:, :-1, :]
            neighbor_occ = BLOCK_OCCLUDES[neighbor].astype(bool) | (BLOCK_OCCLUDES_SAME[neighbor].astype(bool) & (neighbor == blocks[:, 1:, :]))
            exposed_faces[:, 1:, :, 1] = ~neighbor_occ
            neighbor = blocks[:-1, :, :]
            neighbor_occ = BLOCK_OCCLUDES[neighbor].astype(bool) | (BLOCK_OCCLUDES_SAME[neighbor].astype(bool) & (neighbor == blocks[1:, :, :]))
            exposed_faces[1:, :, :, 2] = ~neighbor_occ
            neighbor = blocks[1:, :, :]
            neighbor_occ = BLOCK_OCCLUDES[neighbor].astype(bool) | (BLOCK_OCCLUDES_SAME[neighbor].astype(bool) & (neighbor == blocks[:-1, :, :]))
            exposed_faces[:-1, :, :, 3] = ~neighbor_occ
            neighbor = blocks[:, :, 1:]
            neighbor_occ = BLOCK_OCCLUDES[neighbor].astype(bool) | (BLOCK_OCCLUDES_SAME[neighbor].astype(bool) & (neighbor == blocks[:, :, :-1]))
            exposed_faces[:, :, :-1, 4] = ~neighbor_occ
            neighbor = blocks[:, :, :-1]
            neighbor_occ = BLOCK_OCCLUDES[neighbor].astype(bool) | (BLOCK_OCCLUDES_SAME[neighbor].astype(bool) & (neighbor == blocks[:, :, 1:]))
            exposed_faces[:, :, 1:, 5] = ~neighbor_occ
            render_all = BLOCK_RENDER_ALL[blocks] != 0
            exposed_faces = (exposed_faces | render_all[..., None]) & solid[..., None]

        outgoing_sky = {}
        outgoing_torch = {}
        sky_floor = internal_sky_floor
        sky_side = internal_sky_side if sidefill_enabled else EMPTY_LIGHT_LIST
        torch_side = internal_torch if torch_fill_enabled else EMPTY_LIGHT_LIST
        edge_sky_counts = (0, 0, 0, 0)
        edge_torch_counts = (0, 0, 0, 0)
        light_ms = 0.0
        light_torch_ms = 0.0
        light_sky_ms = 0.0
        debug_torch = None
        debug_sky = None
        debug_sky_direct = None
        debug_torch_sources = None
        debug_torch_updates = None
        debug_sky_updates = None
        if recompute_internal:
            light_start = time.perf_counter()
            tile = light_blocks
            if tile is None:
                tile = numpy.zeros((SECTOR_SIZE * 3, SECTOR_HEIGHT, SECTOR_SIZE * 3), dtype=blocks.dtype)
                tile[SECTOR_SIZE:SECTOR_SIZE * 2, :, SECTOR_SIZE:SECTOR_SIZE * 2] = blocks[1:-1, :, 1:-1]
            sx = SECTOR_SIZE
            air_tile = (BLOCK_OCCLUDES[tile] == 0)
            tile_occ = (BLOCK_OCCLUDES[tile] != 0)
            tile_occ_rev = tile_occ[:, ::-1, :]
            tile_has_occ = tile_occ_rev.any(axis=1)
            tile_first_occ_rev = tile_occ_rev.argmax(axis=1)
            sky_floor_tile = numpy.where(tile_has_occ, SECTOR_HEIGHT - tile_first_occ_rev, 0).astype(numpy.uint16)
            y_grid = numpy.arange(SECTOR_HEIGHT)[None, :, None]
            center = tile[sx:2 * sx, :, sx:2 * sx]
            use_bfs = bool(getattr(config, "LIGHT_PROPAGATION_BFS", False))
            bfs_source_cap = getattr(config, "LIGHT_BFS_SOURCE_CAP", 4096)
            if torch_fill_enabled:
                torch_start = time.perf_counter()
                torch = numpy.zeros(tile.shape, dtype=numpy.int16)
                center_torch_sources = BLOCK_LIGHT_LUT[center]
                torch[sx:2 * sx, :, sx:2 * sx] = center_torch_sources
                if incoming_torch:
                    for (dx, dz), entries in incoming_torch.items():
                        if entries is None or len(entries) == 0:
                            continue
                        _apply_list(
                            torch,
                            entries,
                            offset=((dx + 1) * sx, 0, (dz + 1) * sx),
                        )
                torch_sources = int(numpy.count_nonzero(torch))
                if torch_sources > 0:
                    use_bfs_torch = use_bfs and (bfs_source_cap is None or torch_sources <= bfs_source_cap)
                    if use_bfs_torch:
                        if debug_light_grids:
                            debug_torch_updates = numpy.zeros(tile.shape, dtype=bool)
                        torch = _relax_bfs(
                            torch,
                            air_tile,
                            ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
                            debug_updates=debug_torch_updates,
                        )
                    else:
                        torch = _relax_6way(torch, air_tile)
                light_torch_ms = (time.perf_counter() - torch_start) * 1000.0
                if debug_light_grids:
                    debug_torch = torch
                    debug_torch_sources = numpy.zeros(tile.shape, dtype=bool)
                    debug_torch_sources[sx:2 * sx, :, sx:2 * sx] = (center_torch_sources > 0)
            else:
                torch = None
            sky_start = time.perf_counter()
            center_occ = (BLOCK_OCCLUDES[center] != 0)
            occ_rev = center_occ[:, ::-1, :]
            has_occ = occ_rev.any(axis=1)
            first_occ_rev = occ_rev.argmax(axis=1)
            sky_floor = numpy.where(has_occ, SECTOR_HEIGHT - first_occ_rev, 0).astype(numpy.uint16)
            if sidefill_enabled:
                sky_air_mask = air_tile & (y_grid < sky_floor_tile[:, None, :])
                if incoming_sky or numpy.any(sky_air_mask):
                    sky = numpy.zeros(tile.shape, dtype=numpy.int16)
                    direct_mask = (y_grid >= sky_floor[:, None, :])
                    sky[sx:2 * sx, :, sx:2 * sx] = numpy.where(direct_mask, MAX_LIGHT, 0)
                    if incoming_sky:
                        for (dx, dz), entries in incoming_sky.items():
                            if entries is None or len(entries) == 0:
                                continue
                            _apply_list(
                                sky,
                                entries,
                                offset=((dx + 1) * sx, 0, (dz + 1) * sx),
                            )
                    sky_sources = int(numpy.count_nonzero(sky))
                    if sky_sources > 0:
                        use_bfs_sky = use_bfs and (bfs_source_cap is None or sky_sources <= bfs_source_cap)
                        if use_bfs_sky:
                            if debug_light_grids:
                                debug_sky_updates = numpy.zeros(tile.shape, dtype=bool)
                            sky = _relax_bfs(
                                sky,
                                sky_air_mask,
                                (
                                    (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),
                                    (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                                    (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
                                ),
                                debug_updates=debug_sky_updates,
                            )
                        else:
                            sky = _relax_4way_sloped(sky, sky_air_mask)
                    if debug_light_grids:
                        debug_sky = sky
                        debug_sky_direct = numpy.zeros(tile.shape, dtype=bool)
                        debug_sky_direct[sx:2 * sx, :, sx:2 * sx] = direct_mask
                    center_sky = sky[sx:2 * sx, :, sx:2 * sx]
                    sky_side = _pack_list(numpy.where(direct_mask, 0, center_sky))
                    outgoing_sky = _pack_outgoing(center_sky)
                else:
                    sky_side = EMPTY_LIGHT_LIST
                    outgoing_sky = {}
            else:
                sky_side = EMPTY_LIGHT_LIST
            light_sky_ms = (time.perf_counter() - sky_start) * 1000.0
            if torch_fill_enabled:
                center_torch = torch[sx:2 * sx, :, sx:2 * sx]
                torch_side = _pack_list(center_torch)
                outgoing_torch = _pack_outgoing(center_torch)
            else:
                torch_side = EMPTY_LIGHT_LIST
            if not sidefill_enabled and not torch_fill_enabled:
                outgoing_sky = {offset: EMPTY_LIGHT_LIST for offset in NEIGHBOR_OFFSETS_8}
            light_ms = (time.perf_counter() - light_start) * 1000.0

        if not build_mesh:
            return {
                "vt_data": None,
                "sky_floor": sky_floor,
                "sky_side": sky_side,
                "torch_side": torch_side,
                "outgoing_sky": outgoing_sky,
                "outgoing_torch": outgoing_torch,
                "edge_sky_counts": edge_sky_counts,
                "edge_torch_counts": edge_torch_counts,
                "light_ms": light_ms,
                "light_torch_ms": light_torch_ms,
                "light_sky_ms": light_sky_ms,
                **(
                    {
                        "debug_torch_grid": debug_torch,
                        "debug_sky_grid": debug_sky,
                        "debug_sky_direct_mask": debug_sky_direct,
                        "debug_torch_sources": debug_torch_sources,
                        "debug_torch_updates": debug_torch_updates,
                        "debug_sky_updates": debug_sky_updates,
                    }
                    if debug_light_grids
                    else {}
                ),
            }

        # Assemble combined light for the center sector.
        torch_internal = numpy.zeros((SECTOR_SIZE, SECTOR_HEIGHT, SECTOR_SIZE), dtype=numpy.uint8)
        if torch_fill_enabled:
            _apply_list(torch_internal, torch_side)

        sky_internal = numpy.zeros((SECTOR_SIZE, SECTOR_HEIGHT, SECTOR_SIZE), dtype=numpy.uint8)
        if sky_floor is not None:
            y_grid = numpy.arange(SECTOR_HEIGHT)[None, :, None]
            direct_mask = (y_grid >= sky_floor[:, None, :])
            sky_internal[direct_mask] = MAX_LIGHT
        if sidefill_enabled:
            _apply_list(sky_internal, sky_side)

        tile_shape = (SECTOR_SIZE * 3, SECTOR_HEIGHT, SECTOR_SIZE * 3)
        torch_tile = numpy.zeros(tile_shape, dtype=numpy.uint8)
        sky_tile = numpy.zeros(tile_shape, dtype=numpy.uint8)
        center_off = (SECTOR_SIZE, 0, SECTOR_SIZE)
        if torch_fill_enabled:
            _apply_list(torch_tile, torch_side, offset=center_off)
        if sidefill_enabled:
            _apply_list(sky_tile, sky_side, offset=center_off)
        if sky_floor is not None:
            sky_tile[SECTOR_SIZE:SECTOR_SIZE * 2, :, SECTOR_SIZE:SECTOR_SIZE * 2] = sky_internal
        if torch_fill_enabled and incoming_torch:
            for (dx, dz), entries in incoming_torch.items():
                if entries is None or len(entries) == 0:
                    continue
                _apply_list(torch_tile, entries, offset=((dx + 1) * SECTOR_SIZE, 0, (dz + 1) * SECTOR_SIZE))
        if sidefill_enabled and incoming_sky:
            for (dx, dz), entries in incoming_sky.items():
                if entries is None or len(entries) == 0:
                    continue
                _apply_list(sky_tile, entries, offset=((dx + 1) * SECTOR_SIZE, 0, (dz + 1) * SECTOR_SIZE))

        tile_slice = (slice(SECTOR_SIZE - 1, SECTOR_SIZE * 2 + 1), slice(None), slice(SECTOR_SIZE - 1, SECTOR_SIZE * 2 + 1))
        torch_full = torch_tile[tile_slice].astype(numpy.float32)
        sky_full = sky_tile[tile_slice].astype(numpy.float32)
        if MAX_LIGHT > 0:
            torch_full /= float(MAX_LIGHT)
            sky_full /= float(MAX_LIGHT)
        edge_torch_counts = (
            int(numpy.count_nonzero(torch_full[0, :, :])),
            int(numpy.count_nonzero(torch_full[-1, :, :])),
            int(numpy.count_nonzero(torch_full[:, :, 0])),
            int(numpy.count_nonzero(torch_full[:, :, -1])),
        )
        edge_sky_counts = (
            int(numpy.count_nonzero(sky_full[0, :, :])),
            int(numpy.count_nonzero(sky_full[-1, :, :])),
            int(numpy.count_nonzero(sky_full[:, :, 0])),
            int(numpy.count_nonzero(sky_full[:, :, -1])),
        )

        exposed_torch = numpy.zeros(blocks.shape + (6,), dtype=numpy.float32)
        exposed_sky = numpy.zeros(blocks.shape + (6,), dtype=numpy.float32)
        exposed_torch[:, :-1, :, 0] = torch_full[:, 1:, :]
        exposed_torch[:, 1:, :, 1] = torch_full[:, :-1, :]
        exposed_torch[1:, :, :, 2] = torch_full[:-1, :, :]
        exposed_torch[:-1, :, :, 3] = torch_full[1:, :, :]
        exposed_torch[:, :, :-1, 4] = torch_full[:, :, 1:]
        exposed_torch[:, :, 1:, 5] = torch_full[:, :, :-1]
        exposed_sky[:, :-1, :, 0] = sky_full[:, 1:, :]
        exposed_sky[:, 1:, :, 1] = sky_full[:, :-1, :]
        exposed_sky[1:, :, :, 2] = sky_full[:-1, :, :]
        exposed_sky[:-1, :, :, 3] = sky_full[1:, :, :]
        exposed_sky[:, :, :-1, 4] = sky_full[:, :, 1:]
        exposed_sky[:, :, 1:, 5] = sky_full[:, :, :-1]

        exposed_faces = exposed_faces[1:-1, :, 1:-1]
        exposed_torch = exposed_torch[1:-1, :, 1:-1]
        exposed_sky = exposed_sky[1:-1, :, 1:-1]

        sx, sy, sz, _ = exposed_faces.shape
        face_mask = exposed_faces.reshape(sx*sy*sz, 6)
        torch_flat_all = exposed_torch.reshape(sx*sy*sz, 6)
        sky_flat_all = exposed_sky.reshape(sx*sy*sz, 6)
        block_mask = face_mask.any(axis=1)
        ao = None
        if ao_enabled and block_mask.any():
            ao_solid = (BLOCK_SOLID[blocks] != 0) & (BLOCK_RENDER_ALL[blocks] == 0)
            ao = compute_vertex_ao(
                ao_solid,
                (sx, sy, sz),
                ao_strength,
                block_mask=block_mask,
            )
        v = numpy.array([], dtype=numpy.float32)
        t = numpy.array([], dtype=numpy.float32)
        n = numpy.array([], dtype=numpy.float32)
        c = numpy.array([], dtype=numpy.float32)
        l = numpy.array([], dtype=numpy.float32)
        count = 0
        sector_grid = numpy.indices((SECTOR_SIZE, SECTOR_HEIGHT, SECTOR_SIZE)).transpose(1,2,3,0).reshape((SECTOR_SIZE*SECTOR_HEIGHT*SECTOR_SIZE,3))
        if block_mask.any():
            max_faces = BLOCK_VERTICES.shape[1]
            pos = sector_grid[block_mask] + numpy.array(position)
            face_mask = face_mask[block_mask]
            torch_flat = torch_flat_all[block_mask]
            sky_flat = sky_flat_all[block_mask]
            b = blocks[1:-1,:,1:-1].reshape(sx*sy*sz)[block_mask]
            face_dirs = BLOCK_FACE_DIR[b]
            face_counts = BLOCK_FACE_COUNT[b]
            face_idx = numpy.arange(max_faces, dtype=face_counts.dtype)[None, :]
            face_exists = face_idx < face_counts[:, None]
            face_mask = numpy.take_along_axis(face_mask, face_dirs, axis=1) & face_exists
            verts = (0.5*BLOCK_VERTICES[b].reshape(len(b),max_faces,4,3)
                     + pos[:,None,None,:] + BLOCK_RENDER_OFFSET).astype(numpy.float32)
            tex_base = BLOCK_TEXTURES_FLIPPED[b][:,:6].reshape(len(b),6,4,2).astype(numpy.float32)
            tex = numpy.take_along_axis(tex_base, face_dirs[:, :, None, None], axis=1)
            normals_base = BLOCK_NORMALS[face_dirs].astype(numpy.float32)
            normals = numpy.broadcast_to(normals_base[:, :, None, :], (len(b), max_faces, 4, 3))
            colors_base = BLOCK_COLORS[b][:,:6].reshape(len(b),6,4,3).astype(numpy.float32)
            colors_base = numpy.take_along_axis(colors_base, face_dirs[:, :, None, None], axis=1)
            torch_faces = numpy.take_along_axis(torch_flat, face_dirs, axis=1)
            sky_faces = numpy.take_along_axis(sky_flat, face_dirs, axis=1)
            torch_light = torch_faces[:, :, None, None]  # (N,max_faces,1,1)
            sky_light = sky_faces[:, :, None, None]
            try:
                light_gamma = float(getattr(config, 'LIGHT_GAMMA', 1.0))
            except Exception:
                light_gamma = 1.0
            if light_gamma < 0.01:
                light_gamma = 0.01
            if abs(light_gamma - 1.0) > 1e-6:
                torch_light = torch_light ** light_gamma
                sky_light = sky_light ** light_gamma
            if ao is not None:
                ao_faces = numpy.take_along_axis(ao, face_dirs[:, :, None], axis=1)
                ao_flat = numpy.where(face_mask[..., None], ao_faces, 1.0)
                torch_light = torch_light * ao_flat[..., None]
                sky_light = sky_light * ao_flat[..., None]
            # Glow acts as a minimum brightness floor after AO.
            emissive = BLOCK_GLOW[b][:, None, None, None] * 255.0
            emissive = numpy.broadcast_to(emissive, colors_base.shape[:-1] + (1,))
            colors_rgba = numpy.concatenate([colors_base, emissive], axis=3)
            colors = numpy.clip(colors_rgba, 0, 255)
            torch_light = numpy.broadcast_to(torch_light, colors_base.shape[:-1] + (1,))
            sky_light = numpy.broadcast_to(sky_light, colors_base.shape[:-1] + (1,))
            light_pair = numpy.concatenate([torch_light, sky_light], axis=3)
            v = verts[face_mask].reshape(-1,3).ravel()
            t = tex[face_mask].reshape(-1,2).ravel()
            n = normals[face_mask].reshape(-1,3).ravel()
            c = colors[face_mask].reshape(-1,4).ravel()
            l = light_pair[face_mask].reshape(-1,2).ravel()
            count = len(v)//3
        water_blocks = (blocks == WATER)
        water_exposed = numpy.zeros(blocks.shape + (6,), dtype=bool)
        neighbor = blocks[:,1:,:]
        air = (neighbor != WATER) & (BLOCK_SOLID[neighbor] == 0)
        water_exposed[:,:-1,:,0] = water_blocks[:,:-1,:] & air
        neighbor = blocks[:,:-1,:]
        air = (neighbor != WATER) & (BLOCK_SOLID[neighbor] == 0)
        water_exposed[:,1:,:,1] = water_blocks[:,1:,:] & air
        neighbor = blocks[:-1,:,:]
        air = (neighbor != WATER) & (BLOCK_SOLID[neighbor] == 0)
        water_exposed[1:,:,:,2] = water_blocks[1:,:,:] & air
        neighbor = blocks[1:,:,:]
        air = (neighbor != WATER) & (BLOCK_SOLID[neighbor] == 0)
        water_exposed[:-1,:,:,3] = water_blocks[:-1,:,:] & air
        neighbor = blocks[:,:,1:]
        air = (neighbor != WATER) & (BLOCK_SOLID[neighbor] == 0)
        water_exposed[:,:,:-1,4] = water_blocks[:,:,:-1] & air
        neighbor = blocks[:,:,:-1]
        air = (neighbor != WATER) & (BLOCK_SOLID[neighbor] == 0)
        water_exposed[:,:,1:,5] = water_blocks[:,:,1:] & air

        water_exposed = water_exposed[1:-1,:,1:-1]
        w_face_mask = water_exposed.reshape(sx*sy*sz, 6)
        water_mask = w_face_mask.any(axis=1)
        if water_mask.any():
            pos_w = sector_grid[water_mask] + numpy.array(position)
            face_mask_w = w_face_mask[water_mask]
            torch_w = torch_flat_all[water_mask]
            sky_w = sky_flat_all[water_mask]
            b = numpy.full(len(pos_w), WATER, dtype=numpy.int32)
            verts = (0.5*BLOCK_VERTICES[b][:,:6].reshape(len(b),6,4,3)
                     + pos_w[:,None,None,:] + BLOCK_RENDER_OFFSET).astype(numpy.float32)
            tex = BLOCK_TEXTURES_FLIPPED[b][:,:6].reshape(len(b),6,4,2).astype(numpy.float32)
            normals = numpy.broadcast_to(BLOCK_NORMALS[None,:,None,:], (len(b),6,4,3)).astype(numpy.float32)
            colors = BLOCK_COLORS[b][:,:6].reshape(len(b),6,4,3).astype(numpy.float32)

            face_verts = verts[face_mask_w].reshape(-1,4,3)
            face_tex = tex[face_mask_w].reshape(-1,4,2)
            face_norm = normals[face_mask_w].reshape(-1,4,3)
            face_col = colors[face_mask_w].reshape(-1,4,3)
            emissive_w = numpy.zeros(face_col.shape[:-1] + (1,), dtype=face_col.dtype)
            face_col = numpy.concatenate([face_col, emissive_w], axis=2)
            torch_light = torch_w[:, :, None, None]
            sky_light = sky_w[:, :, None, None]
            light_pair = numpy.concatenate([torch_light, sky_light], axis=3)
            light_pair = light_pair[face_mask_w].reshape(-1, 1, 2)
            light_pair = numpy.broadcast_to(light_pair, face_col.shape[:-1] + (2,))

            wv = face_verts.reshape(-1,3).ravel()
            wtcoords = face_tex.reshape(-1,2).ravel()
            wn = face_norm.reshape(-1,3).ravel()
            wc = face_col.reshape(-1,4).ravel()
            wl = light_pair.reshape(-1,2).ravel()
            water_count = len(wv) // 3
            water_data = (water_count, wv, wtcoords, wn, wc, wl)
        else:
            water_data = None
        solid_data = (count, v, t, n, c, l)
        vt_data = {'solid': solid_data, 'water': water_data}
        return {
            "vt_data": vt_data,
            "sky_floor": sky_floor,
            "sky_side": sky_side,
            "torch_side": torch_side,
            "outgoing_sky": outgoing_sky,
            "outgoing_torch": outgoing_torch,
            "edge_sky_counts": edge_sky_counts,
            "edge_torch_counts": edge_torch_counts,
            "light_ms": light_ms,
            "light_torch_ms": light_torch_ms,
            "light_sky_ms": light_sky_ms,
            **(
                {
                    "debug_torch_grid": debug_torch,
                    "debug_sky_grid": debug_sky,
                    "debug_sky_direct_mask": debug_sky_direct,
                    "debug_torch_sources": debug_torch_sources,
                    "debug_torch_updates": debug_torch_updates,
                    "debug_sky_updates": debug_sky_updates,
                }
                if debug_light_grids
                else {}
            ),
        }

    def _queue_upload(self, sector, priority=False):
        """Queue a sector for upload/rebuild; avoid reshuffling if already queued."""
        if getattr(sector, "defer_upload", False):
            return
        if sector in self.pending_upload_set:
            return
        self.pending_upload_set.add(sector)
        if priority:
            self.pending_uploads.appendleft(sector)
        else:
            self.pending_uploads.append(sector)

    def _dequeue_upload(self, sector):
        """Remove a sector from the upload queue if present."""
        if sector in self.pending_upload_set:
            self.pending_upload_set.discard(sector)
            try:
                self.pending_uploads.remove(sector)
            except ValueError:
                pass

    def _submit_mesh_job(self, sector, priority=False):
        """Kick off an async mesh build for this sector."""
        if priority:
            sector.mesh_job_priority = True
            self.mesh_interrupt.set()
        if sector.mesh_job_pending:
            sector.mesh_job_dirty = True
            return
        if sector.vt_data is not None and not sector.invalidate_vt and not self._needs_light(sector):
            return
        use_priority = sector.mesh_job_priority or priority
        if self.mesh_active_jobs >= self.mesh_active_cap:
            self._defer_mesh_job(sector, priority=use_priority)
            return
        sector.mesh_job_pending = True
        sector.mesh_job_dirty = False
        sector.mesh_gen += 1
        recompute_internal = sector.light_dirty_internal
        build_mesh = sector.shown
        if build_mesh and sector.light_dirty_internal and not self._neighbors_have_light(sector):
            build_mesh = False
        if build_mesh and sector.vt_data is None and not (sector.edit_inflight or sector.light_initialized):
            build_mesh = False
        if build_mesh and sector.vt_data is None and not self._neighbors_light_initialized(sector):
            build_mesh = False
        require_diagonals = recompute_internal
        if build_mesh or recompute_internal:
            if not self._neighbors_ready(sector, require_diagonals=require_diagonals):
                sector.mesh_job_pending = False
                self._defer_mesh_job(sector, priority=use_priority)
                return
        if recompute_internal and not self._neighbors_ready(sector, require_diagonals=True):
            recompute_internal = False
            sector.light_dirty_internal = True
            sector.light_neighbors_ready = False
        if build_mesh and sector.light_dirty_internal:
            build_mesh = False
        if not build_mesh and not recompute_internal:
            sector.mesh_job_pending = False
            if sector.invalidate_vt or sector.vt_data is None:
                self._defer_mesh_job(sector, priority=use_priority)
            return
        light_blocks = None
        if recompute_internal:
            light_blocks = self._gather_blocks_tile_3x3(sector, allow_missing=False)
            sector.light_neighbors_ready = True
        blocks = None
        incoming_sky = sector.incoming_sky
        incoming_torch = sector.incoming_torch
        if build_mesh or recompute_internal:
            incoming_sky, incoming_torch = self._build_incoming_from_neighbors(sector)
        if build_mesh:
            blocks = self._gather_blocks_halo(sector, require_diagonals=False)
            if blocks is None:
                if recompute_internal and light_blocks is not None:
                    build_mesh = False
                    blocks = None
                else:
                    sector.mesh_job_pending = False
                    self._defer_mesh_job(sector, priority=use_priority)
                    return
        self.mesh_jobs_submitted_total += 1
        ao_strength = getattr(config, 'AO_STRENGTH', 0.0)
        ao_enabled = getattr(config, 'AO_ENABLED', True)
        ambient = getattr(config, 'AMBIENT_LIGHT', 0.0)
        gen = sector.mesh_gen
        pos = sector.position

        sector.mesh_job_priority = False
        if self.mesh_single_worker:
            job = _MeshJobResult(pos, gen, use_priority, time.perf_counter())
            self._enqueue_mesh_job(
                job,
                blocks,
                light_blocks,
                pos,
                ao_strength,
                ao_enabled,
                ambient,
                incoming_sky,
                incoming_torch,
                sector.sky_floor,
                sector.sky_side,
                sector.torch_side,
                recompute_internal,
                None if use_priority else self.mesh_interrupt,
                build_mesh,
                priority=use_priority,
            )
            self.mesh_active_jobs += 1
        else:
            def _done(fut):
                self.mesh_results.put(fut)
            executor = self.mesh_executor_hi if use_priority else self.mesh_executor
            future = executor.submit(
                  self._build_mesh_job,
                  blocks,
                  light_blocks,
                  pos,
                  ao_strength,
                ao_enabled,
                ambient,
                incoming_sky,
                incoming_torch,
                sector.sky_floor,
                  sector.sky_side,
                sector.torch_side,
                recompute_internal,
                None if use_priority else self.mesh_interrupt,
                build_mesh,
            )
            future.gen = gen
            future.pos = pos
            future.priority = use_priority
            future.start_time = time.perf_counter()
            future.add_done_callback(_done)
            self.mesh_active_jobs += 1
        self._mesh_submit_frame = self.frame_id
        if use_priority:
            self._priority_submitted_frame = self.frame_id
        if use_priority:
            self.pending_priority_jobs += 1
        self._mesh_log(
            f"submit sector={pos} gen={gen} priority={use_priority}"
        )

    def _drain_mesh_results(self):
        """Consume finished mesh jobs and enqueue uploads if still valid."""
        while not self.mesh_results.empty():
            fut = self.mesh_results.get_nowait()
            try:
                result = fut.result()
            except Exception as e:
                logutil.log("MESH", f"job failed {e}", level="ERROR")
                self.mesh_active_jobs = max(0, self.mesh_active_jobs - 1)
                continue
            self.mesh_active_jobs = max(0, self.mesh_active_jobs - 1)
            pos = getattr(fut, 'pos', None)
            gen = getattr(fut, 'gen', None)
            was_priority = getattr(fut, 'priority', False)
            start_time = getattr(fut, 'start_time', None)
            if start_time is not None:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            else:
                elapsed_ms = None
            if was_priority:
                self.pending_priority_jobs = max(0, self.pending_priority_jobs - 1)
            if pos is None:
                continue
            sector = self.sectors.get(pos)
            if sector is None:
                continue
            sector.mesh_job_pending = False
            if sector.mesh_job_dirty:
                sector.mesh_job_dirty = False
                self._submit_mesh_job(sector, priority=sector.mesh_job_priority)
                continue
            if gen is not None and sector.mesh_gen != gen:
                # stale result
                continue
            if not isinstance(result, dict) or "vt_data" not in result:
                logutil.log("MESH", f"job returned unexpected result for {pos}", level="WARN")
                continue
            self.mesh_jobs_completed_total += 1
            self.mesh_recent.append(pos)
            light_ms = float(result.get("light_ms") or 0.0)
            if light_ms > 0.0:
                self._note_sector_light(pos, light_ms)
            sector.sky_floor = result.get("sky_floor", sector.sky_floor)
            sector.sky_side = result.get("sky_side", sector.sky_side)
            sector.torch_side = result.get("torch_side", sector.torch_side)
            sector.edge_sky_counts = result.get("edge_sky_counts", sector.edge_sky_counts)
            sector.edge_torch_counts = result.get("edge_torch_counts", sector.edge_torch_counts)
            sector.light_dirty_internal = False
            sector.light_dirty_incoming = False
            outgoing_sky = result.get("outgoing_sky") or {}
            outgoing_torch = result.get("outgoing_torch") or {}
            if outgoing_sky or outgoing_torch:
                propagate_dirty = sector.light_dirty_from_edit or sector.edit_inflight
                self._apply_outgoing_light(sector, outgoing_sky, outgoing_torch, propagate_dirty=propagate_dirty)
            sector.light_dirty_from_edit = False
            if light_ms > 0.0:
                sector.light_initialized = True
            if light_ms > 0.0 and getattr(config, "LIGHT_PULL_BOUNDARY_FROM_NEIGHBORS", True):
                x0, _, z0 = sector.position
                for dx, dz in NEIGHBOR_OFFSETS_8:
                    npos = (x0 + dx * SECTOR_SIZE, 0, z0 + dz * SECTOR_SIZE)
                    neighbor = self.sectors.get(npos)
                    if neighbor is None or not neighbor.shown:
                        continue
                    # Onion rule: only re-mesh when neighbor light ring is complete.
                    if neighbor.light_dirty_internal or not self._neighbors_have_light(neighbor):
                        continue
                    neighbor.invalidate()
                    self._submit_mesh_job(neighbor, priority=neighbor.edit_inflight)
            if result["vt_data"] is None:
                if sector.shown and self._neighbors_have_light(sector):
                    self._submit_mesh_job(sector, priority=sector.edit_inflight)
                continue
            if elapsed_ms is not None:
                mesh_ms = max(0.0, elapsed_ms - light_ms)
                self._note_sector_mesh(pos, mesh_ms)
            sector.vt_data = result["vt_data"]
            sector._note_new_vt_data()
            sector.vt_clear_pending = True
            sector.mesh_built = True
            sector.invalidate()
            if getattr(config, 'MESH_LOG', False):
                try:
                    water_blocks = (sector.blocks == WATER)
                    water_entry = self._get_vt_entry(sector.vt_data, 'water')
                    water_quads = 0 if not water_entry else int(water_entry[0] // 4)
                    if water_blocks.any() and water_quads == 0:
                        above = sector.blocks[:, 1:, :] != WATER
                        surface = water_blocks[:, :-1, :] & above
                        surface_count = int(surface.sum())
                        self._mesh_log(
                            f"water_missing sector={pos} blocks={int(water_blocks.sum())} surface={surface_count}"
                        )
                except Exception:
                    pass
            if sector.shown:
                self._queue_upload(sector, priority=sector.edit_inflight)
            if elapsed_ms is not None:
                self._mesh_log(
                    f"done sector={pos} gen={gen} priority={was_priority} ms={elapsed_ms:.1f}"
                )
        if self.pending_priority_jobs == 0 and self.mesh_interrupt.is_set():
            self.mesh_interrupt.clear()

    def _apply_outgoing_light(self, sector, outgoing_sky, outgoing_torch, propagate_dirty=True):
        """Apply sender-model light buffers to neighbors and queue rebuilds."""
        x, _, z = sector.position
        for dx, dz in NEIGHBOR_OFFSETS_8:
            npos = (x + dx * SECTOR_SIZE, 0, z + dz * SECTOR_SIZE)
            neighbor = self.sectors.get(npos)
            if neighbor is None:
                continue
            key = (-dx, -dz)
            changed = False
            if key in neighbor.incoming_sky:
                new_sky = outgoing_sky.get((dx, dz), EMPTY_LIGHT_LIST)
                old_sky = neighbor.incoming_sky[key]
                if not _light_list_equal(old_sky, new_sky):
                    neighbor.incoming_sky[key] = new_sky
                    neighbor.incoming_sky_updates += 1
                    changed = True
            if key in neighbor.incoming_torch:
                new_torch = outgoing_torch.get((dx, dz), EMPTY_LIGHT_LIST)
                old_torch = neighbor.incoming_torch[key]
                if not _light_list_equal(old_torch, new_torch):
                    neighbor.incoming_torch[key] = new_torch
                    neighbor.incoming_torch_updates += 1
                    changed = True
            if not changed:
                continue
            if not propagate_dirty:
                continue
            neighbor.light_dirty_incoming = True
            can_relight = self._neighbors_ready(neighbor, require_diagonals=True)
            if not neighbor.light_dirty_internal and can_relight:
                neighbor.light_dirty_internal = True
                neighbor.light_neighbors_ready = False
            if can_relight:
                neighbor.invalidate()
                if neighbor.shown:
                    self._submit_mesh_job(neighbor, priority=neighbor.edit_inflight)

    def _mark_neighbor_outgoing_dirty(self, sector_pos):
        """Force neighbors to recompute outgoing light when a sector becomes available."""
        x, _, z = sector_pos
        for dx, dz in NEIGHBOR_OFFSETS_8:
            npos = (x + dx * SECTOR_SIZE, 0, z + dz * SECTOR_SIZE)
            neighbor = self.sectors.get(npos)
            if neighbor is None:
                continue
            if not self._neighbors_ready(neighbor, require_diagonals=True):
                neighbor.light_neighbors_ready = False
                continue
            neighbor.light_neighbors_ready = True
            if not neighbor.light_dirty_incoming:
                continue
            if neighbor.light_dirty_internal:
                continue
            neighbor.light_dirty_internal = True
            if neighbor.shown or self._has_shown_neighbor(neighbor):
                self._submit_mesh_job(neighbor, priority=neighbor.edit_inflight)

    def _has_shown_neighbor(self, sector):
        x, _, z = sector.position
        for dx, dz in NEIGHBOR_OFFSETS_8:
            pos = (x + dx * SECTOR_SIZE, 0, z + dz * SECTOR_SIZE)
            n = self.sectors.get(pos)
            if n is not None and n.shown:
                return True
        return False

    def process_pending_uploads(self, frame_start=None, upload_budget=None, uploaded_tris=0, tri_budget=None):
        """Upload queued sector vertex data while budget remains."""
        while self.pending_uploads and self._has_budget(frame_start, upload_budget):
            if self.player_sector is None or len(self.pending_uploads) == 1:
                s = self.pending_uploads.popleft()
                self.pending_upload_set.discard(s)
            else:
                best_idx = 0
                best_prio = None
                for idx, sector in enumerate(self.pending_uploads):
                    prio = self._sector_priority(
                        self.player_sector,
                        sector.position,
                        self.player_pos,
                        self.player_look,
                        None,
                    )
                    if best_prio is None or prio < best_prio:
                        best_prio = prio
                        best_idx = idx
                if best_idx:
                    self.pending_uploads.rotate(-best_idx)
                s = self.pending_uploads.popleft()
                if best_idx:
                    self.pending_uploads.rotate(best_idx)
                self.pending_upload_set.discard(s)
            if s.vt_data is not None:
                tri_count = self._triangles_in_vt_chunk(s)
                over_time = not self._has_budget(frame_start, upload_budget)
                over_tris = tri_budget is not None and (uploaded_tris + tri_count) > tri_budget
                if not s.force_full_upload and (over_time or over_tris):
                    # Not enough budget left; try next frame.
                    self._queue_upload(s)
                    break
                upload_start = time.perf_counter()
                s.check_show(add_to_batch=True)
                upload_ms = (time.perf_counter() - upload_start) * 1000.0
                self._note_sector_upload(s, upload_ms)
                uploaded_tris += tri_count
            elif s.invalidate_vt:
                upload_start = time.perf_counter()
                s.check_show(add_to_batch=True)
                upload_ms = (time.perf_counter() - upload_start) * 1000.0
                self._note_sector_upload(s, upload_ms)
                s.invalidate_vt = False
        return uploaded_tris

    def _rebuild_sector_now_fast(self, sector, priority=False):
        """Synchronously rebuild a sector mesh using existing light buffers."""
        t0 = time.perf_counter()
        if not self._mesh_ready(sector):
            return
        blocks = self._gather_blocks_halo(sector, require_diagonals=False)
        if blocks is None:
            return
        incoming_sky = sector.incoming_sky
        incoming_torch = sector.incoming_torch
        if getattr(config, "LIGHT_PULL_BOUNDARY_FROM_NEIGHBORS", True):
            incoming_sky, incoming_torch = self._build_incoming_from_neighbors(sector)
            sector.incoming_sky = incoming_sky
            sector.incoming_torch = incoming_torch
        ao_strength = getattr(config, 'AO_STRENGTH', 0.0)
        ao_enabled = getattr(config, 'AO_ENABLED', True)
        ambient = getattr(config, 'AMBIENT_LIGHT', 0.0)
        result = self._build_mesh_job(
            blocks,
            None,
            sector.position,
            ao_strength,
            ao_enabled,
            ambient,
            incoming_sky,
            incoming_torch,
            sector.sky_floor,
            sector.sky_side,
            sector.torch_side,
            False,
            None,
        )
        sector.vt_data = result["vt_data"]
        sector._note_new_vt_data()
        sector.vt_upload_prepared = False
        sector.vt_clear_pending = True
        sector.force_full_upload = bool(priority)
        sector.mesh_built = True
        sector.light_dirty_incoming = False
        if sector.shown:
            if priority:
                self._dequeue_upload(sector)
                if not getattr(sector, "defer_upload", False):
                    sector.check_show(add_to_batch=True)
            else:
                self._queue_upload(sector, priority=priority)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._note_sector_mesh(sector.position, elapsed_ms)
        self._mesh_log(
            f"sync_fast sector={sector.position} priority={priority} ms={elapsed_ms:.1f}"
        )

    def _invalidate_and_rebuild(self, sector, sync=False, priority=False):
        """Invalidate the sector for rebuild and schedule lighting recompute."""
        sector.light_dirty_internal = True
        if sync:
            self._rebuild_sector_now_fast(sector, priority=priority)
        else:
            sector.invalidate()
            if sector.shown:
                self._submit_mesh_job(sector, priority=priority)

    def set_matrices(self, projection, view, camera_pos):
        # Convert pyglet Mat4 to column-major numpy arrays
        proj = numpy.array(list(projection), dtype='f4').reshape((4, 4), order='F')
        view_mat = numpy.array(list(view), dtype='f4').reshape((4, 4), order='F')

        # The view matrix from look_at already has the translation.
        # The shader also applies translation via u_camera_pos.
        # To avoid double-correction, we remove the translation from the view matrix
        # and rely on the shader's relative positioning.
        view_mat[3, :3] = 0.0
        
        # Shader expects flat (1D) arrays
        self.program['u_projection'] = proj.ravel(order='F')
        self.program['u_view'] = view_mat.ravel(order='F')
        self.program['u_camera_pos'] = tuple(camera_pos) # Pass eye position
        
        # Set model matrix to identity for world rendering
        self.program['u_model'] = Mat4()

    def get_batches(self):
        while self.unused_batches:
            batch = self.unused_batches.pop(0)
            if batch[0] is not None and batch[1] is not None:
                return batch
        return (pyglet.graphics.Batch(), pyglet.graphics.Batch())

    def release_sector(self, sector):
        # Drop any pending uploads for this sector before freeing GPU buffers.
        self.pending_upload_set.discard(sector)
        try:
            self.pending_uploads.remove(sector)
        except ValueError:
            pass
        self._remove_from_edit_group(sector)
        sector._clear_vt_lists()
        sector._clear_pending_vt()
        sector.vt_data = None
        sector.mesh_job_pending = False
        sector.mesh_job_dirty = False
        sector.patch_vt = []
        self.unused_batches.append((sector.batch, sector.batch_water))
        del self.sectors[sector.position]

    def __getitem__(self, position):
        """
        retrieves the block at the (x,y,z) coordinate tuple `position`
        """
        try:
            return self.sectors[sectorize(position)][position]
        except:
            return None

    def get_vertical_column(self, x, z):
        """
        Return the full column of blocks at integer (x,z) coordinates, or None
        if the corresponding sector is not loaded.
        """
        sx = sectorize((x, 0, z))
        sector = self.sectors.get(sx)
        if sector is None:
            return None
        ix = int(round(x)) - sector.position[0]
        iz = int(round(z)) - sector.position[2]
        if not (0 <= ix < config.SECTOR_SIZE and 0 <= iz < config.SECTOR_SIZE):
            return None
        return sector.blocks[ix, :, iz]

    def find_surface_y(self, x, z):
        """
        Find the y-coordinate of the ground at the given (x, z) position.
        """
        column = self.get_vertical_column(x, z)
        if column is None or column.size == 0:
            return None
        non_air = column != 0
        if not non_air.any():
            return None
        y = int(numpy.nonzero(non_air)[0][-1])
        return float(y + 1)

    def add_block(self, position, block, notify_server = True, keep_patches=False, priority=False):
        spos = sectorize(position)
        if spos in self.sectors:
            s = self.sectors[spos]
            # Apply locally for instant feedback
            rel = numpy.array(position) - numpy.array(s.position)
            try:
                prev_block = int(s.blocks[rel[0], rel[1], rel[2]])
                s.blocks[rel[0], rel[1], rel[2]] = block
            except Exception:
                prev_block = None
            s.invalidate_vt = True
            s.light_dirty_internal = True
            on_edge = rel[0] in (0, SECTOR_SIZE - 1) or rel[2] in (0, SECTOR_SIZE - 1)
            if priority or on_edge:
                self._invalidate_for_edit(
                    s,
                    rel,
                    priority=True,
                )
            else:
                self._submit_mesh_job(s, priority=True)
            s.edit_inflight = True
            s.edit_token += 1
            self.sector_edit_tokens[spos] = s.edit_token
            self.loader_requests.insert(0, ['set_block', [notify_server, position, block, s.edit_token]])
            # Immediate visual patch
            if getattr(config, 'USE_PATCH_MESH', False):
                if not keep_patches:
                    for pv in s.patch_vt:
                        pv.delete()
                    s.patch_vt = []
                if block != 0:
                    world_pos = numpy.array(position, dtype=float)
                    vt = self._build_block_vt(block, world_pos)
                    key = 'water' if block == WATER else 'solid'
                    tri_verts, tri_tex, tri_norm, tri_col, tri_light = self._triangulate_vt(vt, key)
                    if len(tri_verts) > 0:
                        group = s.water_group if block == WATER else s.group
                        batch = s.batch_water if block == WATER else s.batch
                        patch = self.program.vertex_list(
                            len(tri_verts),
                            gl.GL_TRIANGLES,
                            batch=batch,
                            group=group,
                            position=('f', tri_verts.ravel().astype('f4')),
                            tex_coords=('f', tri_tex.ravel().astype('f4')),
                            normal=('f', tri_norm.ravel().astype('f4')),
                            color=('f', tri_col.ravel().astype('f4')),
                            light=('f', tri_light.ravel().astype('f4')),
                          )
                        s.patch_vt.append(patch)

    def add_blocks(self, updates, notify_server=True, priority=False):
        """Batch apply multiple block updates in one loader request."""
        if not updates:
            return
        sector_updates = {}
        edge_sectors = set()
        for position, block in updates:
            spos = sectorize(position)
            if spos not in self.sectors:
                continue
            s = self.sectors[spos]
            rel = numpy.array(position) - numpy.array(s.position)
            try:
                prev_block = int(s.blocks[rel[0], rel[1], rel[2]])
                s.blocks[rel[0], rel[1], rel[2]] = block
            except Exception:
                continue
            s.invalidate_vt = True
            s.light_dirty_internal = True
            on_edge = rel[0] in (0, SECTOR_SIZE - 1) or rel[2] in (0, SECTOR_SIZE - 1)
            if on_edge:
                edge_sectors.add(spos)
            s.edit_inflight = True
            sector_updates.setdefault(spos, s)
        if not sector_updates:
            return
        for spos, s in sector_updates.items():
            if priority or spos in edge_sectors:
                self._invalidate_for_edit(
                    s,
                    None,
                    priority=True,
                )
            else:
                self._submit_mesh_job(s, priority=True)
        token_map = {}
        for spos, sector in sector_updates.items():
            sector.edit_token += 1
            self.sector_edit_tokens[spos] = sector.edit_token
            token_map[spos] = sector.edit_token
        self.loader_requests.insert(0, ['set_blocks', [notify_server, updates, token_map]])
        # Immediate visual patch geometry.
        if getattr(config, 'USE_PATCH_MESH', False):
            for position, block in updates:
                spos = sectorize(position)
                if spos not in self.sectors:
                    continue
                s = self.sectors[spos]
                world_pos = numpy.array(position, dtype=float)
                vt = self._build_block_vt(block, world_pos)
                key = 'water' if block == WATER else 'solid'
                tri_verts, tri_tex, tri_norm, tri_col, tri_light = self._triangulate_vt(vt, key)
                if len(tri_verts) == 0:
                    continue
                group = s.water_group if block == WATER else s.group
                batch = s.batch_water if block == WATER else s.batch
                patch = self.program.vertex_list(
                    len(tri_verts),
                    gl.GL_TRIANGLES,
                    batch=batch,
                    group=group,
                    position=('f', tri_verts.ravel().astype('f4')),
                    tex_coords=('f', tri_tex.ravel().astype('f4')),
                    normal=('f', tri_norm.ravel().astype('f4')),
                    color=('f', tri_col.ravel().astype('f4')),
                    light=('f', tri_light.ravel().astype('f4')),
                )
                s.patch_vt.append(patch)

    def remove_block(self, position, notify_server = True, priority=False):
        pos = normalize(position)
        existing = self[pos]
        if existing == WATER:
            if not self._can_remove_water(pos):
                return
        if existing in DOOR_LOWER_IDS:
            upper_pos = (pos[0], pos[1] + 1, pos[2])
            if self[upper_pos] in DOOR_UPPER_IDS:
                self.add_block(upper_pos, 0, notify_server, priority=priority)
        elif existing in DOOR_UPPER_IDS:
            lower_pos = (pos[0], pos[1] - 1, pos[2])
            if self[lower_pos] in DOOR_LOWER_IDS:
                self.add_block(lower_pos, 0, notify_server, priority=priority)
        self.add_block(pos, 0, notify_server, priority=priority)
        # If we removed terrain below the waterline and there's adjacent water, flow in.
        if existing != WATER and pos[1] < self._water_level():
            if any(self[normalize((pos[0]+dx, pos[1]+dy, pos[2]+dz))] == WATER for dx, dy, dz in FACES):
                self.add_block(pos, WATER, notify_server, priority=priority)

    def _can_remove_water(self, pos):
        """Allow removal only if water pocket is smaller than 4 contiguous blocks."""
        seen = set()
        q = [pos]
        water_count = 0
        while q:
            cur = q.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if self[cur] != WATER:
                continue
            water_count += 1
            if water_count >= 4:
                return False
            cx, cy, cz = cur
            for dx, dy, dz in FACES:
                npos = (cx + dx, cy + dy, cz + dz)
                if npos not in seen:
                    q.append(npos)
        return True

    def _water_level(self):
        return getattr(mapgen, 'GLOBAL_WATER_LEVEL', getattr(mapgen, 'WATER_LEVEL', 70))

    def draw(self, position, frustum_circle, frame_start=None, upload_budget=None):
        """Draw only sectors intersecting the current view frustum projection."""
        self.frame_start = frame_start if frame_start is not None else time.perf_counter()
        draw_invalid = True
        # Default to opaque pass.
        self.program['u_water_pass'] = False
        # Keep alpha synced with config in case it changes.
        self.program['u_water_alpha'] = getattr(config, 'WATER_ALPHA', 0.8)

        stats_enabled = True
        shown = 0
        drawn_tris_solid = 0
        drawn_tris_water = 0
        cull_ms = 0.0
        solid_batches = set()
        water_batches = set()
        vlists_solid = 0
        vlists_water = 0
        for s in self.sectors.values():
            t0 = time.perf_counter()
            visible = self._sector_overlaps_frustum(s.position, frustum_circle)
            if stats_enabled:
                cull_ms += (time.perf_counter() - t0) * 1000.0
            was_shown = s.shown
            s.shown = visible
            if visible and not was_shown:
                self._mark_neighbor_outgoing_dirty(s.position)
            if visible and self._needs_light(s) and not s.mesh_job_pending:
                if self._lighting_ready(s):
                    self._submit_mesh_job(s)
            if not visible:
                continue
            if s.batch is not None:
                solid_batches.add(s.batch)
            if s.batch_water is not None:
                water_batches.add(s.batch_water)
            if stats_enabled:
                shown += 1
                if s.vt_solid_quads:
                    drawn_tris_solid += int(s.vt_solid_quads * 2)
                if s.vt_water_quads:
                    drawn_tris_water += int(s.vt_water_quads * 2)
                vlists_solid += len(s.vt)
                vlists_water += len(s.vt_water)

            if getattr(config, 'MESH_LOG', False):
                if self.frame_id - s._last_draw_detail_frame >= 30:
                    s._last_draw_detail_frame = self.frame_id
                    solid_verts = sum(getattr(vt, 'count', 0) for vt in s.vt)
                    water_verts = sum(getattr(vt, 'count', 0) for vt in s.vt_water)
                    exp_solid = s.vt_solid_quads * 6
                    exp_water = s.vt_water_quads * 6
                    self._mesh_log(
                        f"draw_detail sector={s.position} shown={s.shown} "
                        f"vt_solid={len(s.vt)} vt_water={len(s.vt_water)} "
                        f"solid_verts={solid_verts}/{exp_solid} "
                        f"water_verts={water_verts}/{exp_water} "
                        f"upload_solid={s.vt_upload_solid}/{s.vt_solid_quads} "
                        f"upload_water={s.vt_upload_water}/{s.vt_water_quads}"
                    )

            if s.vt_data is not None or (draw_invalid and s.invalidate_vt):
                self._queue_upload(s)

        for batch in solid_batches:
            batch.draw()

        # Drain async mesh results before uploads.
        self._drain_mesh_results()
        if stats_enabled:
            self.stats_shown_sectors = shown
            self.stats_total_sectors = len(self.sectors)
            self.stats_drawn_tris_solid = drawn_tris_solid
            self.stats_drawn_tris_water = drawn_tris_water
            self.stats_cull_ms = cull_ms
            self.stats_batches_solid = len(solid_batches)
            self.stats_batches_water = len(water_batches)
            self.stats_vlists_solid = vlists_solid
            self.stats_vlists_water = vlists_water
            self.stats_draw_calls = len(solid_batches) + len(water_batches)

    def draw_water_pass(self):
        """Draw transparent water after opaque passes so depth writes stay intact."""
        # Ensure block shading path is active for water.
        self.program['u_use_texture'] = True
        self.program['u_use_vertex_color'] = True
        # Reset model transform so water is rendered in world space.
        self.program['u_model'] = Mat4()
        # Keep alpha synced with config in case it changes.
        self.program['u_water_alpha'] = getattr(config, 'WATER_ALPHA', 0.8)
        self.program['u_water_pass'] = True
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        cull_enabled = gl.glIsEnabled(gl.GL_CULL_FACE)
        if cull_enabled:
            gl.glDisable(gl.GL_CULL_FACE)
        gl.glDepthMask(gl.GL_FALSE)  # depth test stays on; just stop writing so opaque depth survives
        water_batches = set()
        for s in self.sectors.values():
            if s.shown and s.batch_water is not None:
                water_batches.add(s.batch_water)
        for batch in water_batches:
            batch.draw()
        gl.glDepthMask(gl.GL_TRUE)
        if cull_enabled:
            gl.glEnable(gl.GL_CULL_FACE)
        self.program['u_water_pass'] = False

    def neighbor_sectors(self, pos):
        """
        return a tuple (dx, dz, sector) of currently loaded neighbors to the sector at pos
        """
        pos = sectorize(pos)
        for x in ((-1,0),(1,0),(0,-1),(0,1)):
            npos = (pos[0]+x[0]*SECTOR_SIZE,0,pos[2]+x[1]*SECTOR_SIZE)
            if npos in self.sectors:
                yield x[0],x[1],self.sectors[npos]

    def _sector_overlaps_frustum(self, sector_pos, frustum_circle):
        """Return True if the sector AABB intersects the 2D frustum circle."""
        if not frustum_circle:
            return True
        (center, rad) = frustum_circle
        cx, cz = center
        min_x = sector_pos[0]
        max_x = sector_pos[0] + SECTOR_SIZE
        min_z = sector_pos[2]
        max_z = sector_pos[2] + SECTOR_SIZE
        # Clamp circle center to sector bounds to find closest point
        nearest_x = min(max(cx, min_x), max_x)
        nearest_z = min(max(cz, min_z), max_z)
        dx = cx - nearest_x
        dz = cz - nearest_z
        return (dx * dx + dz * dz) <= (rad * rad)

    def _sector_priority(self, ref_sector, sector_pos, player_pos, look_vec, frustum_circle=None):
        """Priority is distance-first, then frustum membership and alignment."""
        dx = (sector_pos[0] - ref_sector[0]) / float(SECTOR_SIZE)
        dz = (sector_pos[2] - ref_sector[2]) / float(SECTOR_SIZE)
        dist = math.hypot(dx, dz)
        dist2 = (sector_pos[0] - ref_sector[0]) ** 2 + (sector_pos[2] - ref_sector[2]) ** 2
        outside_frustum = (
            frustum_circle is not None
            and not self._sector_overlaps_frustum(sector_pos, frustum_circle)
        )
        align_key = 0.0
        if (frustum_circle is not None and player_pos is not None and look_vec is not None):
            dx = (sector_pos[0] + SECTOR_SIZE * 0.5) - player_pos[0]
            dz = (sector_pos[2] + SECTOR_SIZE * 0.5) - player_pos[2]
            dist_len = math.hypot(dx, dz)
            look_len = math.hypot(look_vec[0], look_vec[2])
            if dist_len > 1e-6 and look_len > 1e-6:
                dot = (dx / dist_len) * (look_vec[0] / look_len) + (dz / dist_len) * (look_vec[2] / look_len)
                align_key = -dot
        view_scale = getattr(config, 'VIEW_PRIORITY_SCALE', 0.0)
        if view_scale:
            align_key *= float(view_scale)
        else:
            align_key = 0.0
        return (dist, dist2, 1 if outside_frustum else 0, align_key)

    def _compute_load_candidates(self, ref_sector, player_pos, look_vec, frustum_circle=None):
        candidates = []
        load_radius = max(self.load_radius, LOADED_SECTORS)
        G = range(-load_radius, load_radius + 1)
        for dx, dy, dz in itertools.product(G, (0,), G):
            pos = numpy.array([ref_sector[0], ref_sector[1], ref_sector[2]]) + numpy.array(
                [dx * SECTOR_SIZE, dy, dz * SECTOR_SIZE]
            )
            pos = sectorize(pos)
            if pos in self.sectors:
                continue
            prio = self._sector_priority(ref_sector, pos, player_pos, look_vec, frustum_circle)
            candidates.append((prio, pos))
        candidates.sort()
        if getattr(config, 'LOG_LOAD_CANDIDATES', False):
            every = getattr(config, 'LOG_LOAD_CANDIDATES_EVERY_N_FRAMES', 30)
            if self.frame_id % max(1, every) == 0:
                sample = [pos for _, pos in candidates[:8]]
                logutil.log("QUEUE", f"load_candidates ref={ref_sector} sample={sample}")
        return candidates

    def _compute_mesh_candidates(self, ref_sector, player_pos, look_vec, frustum_circle=None):
        candidates = []
        for sector in self.sectors.values():
            if sector.mesh_job_pending:
                continue
            needs_mesh = sector.vt_data is None or sector.invalidate_vt
            if needs_mesh:
                if not self._mesh_ready(sector):
                    continue
                if sector.vt_data is None and not (sector.edit_inflight or sector.light_initialized):
                    continue
                if sector.vt_data is None and not self._neighbors_light_initialized(sector):
                    continue
                if sector.light_dirty_internal and not self._neighbors_have_light(sector):
                    continue
            else:
                if not (self._needs_light(sector) and self._lighting_ready(sector)):
                    continue
            prio = self._sector_priority(
                ref_sector,
                sector.position,
                player_pos,
                look_vec,
                frustum_circle,
            )
            candidates.append((prio, sector))
        candidates.sort(key=lambda item: item[0])
        return candidates

    def pick_frame_work(self, ref_sector, player_pos, look_vec, frustum_circle=None):
        if self.update_ref_pos != ref_sector or not self.update_sectors_pos:
            self._refresh_load_candidates(ref_sector, player_pos, look_vec, frustum_circle)
        self._refresh_mesh_candidates(ref_sector, player_pos, look_vec, frustum_circle)
        load_candidates = self.update_sectors_pos
        mesh_candidates = self._mesh_candidates
        any_mesh_pending = any(s.mesh_job_pending for s in self.sectors.values())
        missing_near = False
        for _, pos in load_candidates:
            if (abs(pos[0] - ref_sector[0]) <= SECTOR_SIZE
                    and abs(pos[2] - ref_sector[2]) <= SECTOR_SIZE):
                missing_near = True
                break
        if not load_candidates and not mesh_candidates:
            return "none"
        if load_candidates and not mesh_candidates:
            return "load"
        if mesh_candidates and not load_candidates:
            return "mesh"
        if missing_near:
            return "load"
        if any_mesh_pending and load_candidates:
            return "load"
        load_prio, _ = load_candidates[0]
        mesh_prio, _ = mesh_candidates[0]
        return "load" if load_prio <= mesh_prio else "mesh"

    def _neighbors_ready(self, sector, require_diagonals=False):
        """Return True if the cardinal (and optionally diagonal) neighbors are loaded."""
        x0, _, z0 = sector.position
        card = [
            (x0 + SECTOR_SIZE, 0, z0),
            (x0 - SECTOR_SIZE, 0, z0),
            (x0, 0, z0 + SECTOR_SIZE),
            (x0, 0, z0 - SECTOR_SIZE),
        ]
        for p in card:
            if p not in self.sectors:
                return False
        if not require_diagonals:
            return True
        diag = [
            (x0 + SECTOR_SIZE, 0, z0 + SECTOR_SIZE),
            (x0 + SECTOR_SIZE, 0, z0 - SECTOR_SIZE),
            (x0 - SECTOR_SIZE, 0, z0 + SECTOR_SIZE),
            (x0 - SECTOR_SIZE, 0, z0 - SECTOR_SIZE),
        ]
        return all(p in self.sectors for p in diag)

    def _pack_plane_entries(self, plane, x_fixed=None, z_fixed=None):
        coords = numpy.argwhere(plane > 0)
        if coords.size == 0:
            return EMPTY_LIGHT_LIST
        out = numpy.empty((len(coords), 4), dtype=numpy.uint8)
        if x_fixed is not None:
            out[:, 0] = x_fixed
            out[:, 1] = coords[:, 0].astype(numpy.uint8, copy=False)
            out[:, 2] = coords[:, 1].astype(numpy.uint8, copy=False)
            out[:, 3] = plane[coords[:, 0], coords[:, 1]].astype(numpy.uint8, copy=False)
        else:
            out[:, 0] = coords[:, 1].astype(numpy.uint8, copy=False)
            out[:, 1] = coords[:, 0].astype(numpy.uint8, copy=False)
            out[:, 2] = z_fixed
            out[:, 3] = plane[coords[:, 0], coords[:, 1]].astype(numpy.uint8, copy=False)
        return out

    def _pack_line_entries(self, line, x_fixed, z_fixed):
        ys = numpy.nonzero(line)[0]
        if ys.size == 0:
            return EMPTY_LIGHT_LIST
        out = numpy.empty((len(ys), 4), dtype=numpy.uint8)
        out[:, 0] = x_fixed
        out[:, 1] = ys.astype(numpy.uint8, copy=False)
        out[:, 2] = z_fixed
        out[:, 3] = line[ys].astype(numpy.uint8, copy=False)
        return out

    def _boundary_sky_entries(self, neighbor, dx, dz):
        sx = SECTOR_SIZE
        sy = SECTOR_HEIGHT
        sky_floor = neighbor.sky_floor
        sky_side = neighbor.sky_side
        y_grid = numpy.arange(sy)[:, None]
        if dx != 0 and dz != 0:
            x = 0 if dx > 0 else sx - 1
            z = 0 if dz > 0 else sx - 1
            floor_val = sky_floor[x, z]
            line = numpy.where(y_grid[:, 0] >= floor_val, MAX_LIGHT, 0).astype(numpy.uint8)
            if sky_side is not None and len(sky_side):
                mask = (sky_side[:, 0] == x) & (sky_side[:, 2] == z)
                if mask.any():
                    entries = sky_side[mask]
                    line[entries[:, 1]] = numpy.maximum(line[entries[:, 1]], entries[:, 3])
            return self._pack_line_entries(line, x, z)
        if dx != 0:
            x = 0 if dx > 0 else sx - 1
            floor_col = sky_floor[x, :]
            plane = numpy.where(y_grid >= floor_col[None, :], MAX_LIGHT, 0).astype(numpy.uint8)
            if sky_side is not None and len(sky_side):
                mask = (sky_side[:, 0] == x)
                if mask.any():
                    entries = sky_side[mask]
                    plane[entries[:, 1], entries[:, 2]] = numpy.maximum(
                        plane[entries[:, 1], entries[:, 2]], entries[:, 3]
                    )
            return self._pack_plane_entries(plane, x_fixed=x)
        z = 0 if dz > 0 else sx - 1
        floor_row = sky_floor[:, z]
        plane = numpy.where(y_grid >= floor_row[None, :], MAX_LIGHT, 0).astype(numpy.uint8)
        if sky_side is not None and len(sky_side):
            mask = (sky_side[:, 2] == z)
            if mask.any():
                entries = sky_side[mask]
                plane[entries[:, 1], entries[:, 0]] = numpy.maximum(
                    plane[entries[:, 1], entries[:, 0]], entries[:, 3]
                )
        return self._pack_plane_entries(plane, z_fixed=z)

    def _boundary_torch_entries(self, neighbor, dx, dz):
        sx = SECTOR_SIZE
        sy = SECTOR_HEIGHT
        torch_side = neighbor.torch_side
        if torch_side is None or len(torch_side) == 0:
            return EMPTY_LIGHT_LIST
        if dx != 0 and dz != 0:
            x = 0 if dx > 0 else sx - 1
            z = 0 if dz > 0 else sx - 1
            mask = (torch_side[:, 0] == x) & (torch_side[:, 2] == z)
            if not mask.any():
                return EMPTY_LIGHT_LIST
            entries = torch_side[mask]
            line = numpy.zeros((sy,), dtype=numpy.uint8)
            line[entries[:, 1]] = numpy.maximum(line[entries[:, 1]], entries[:, 3])
            return self._pack_line_entries(line, x, z)
        if dx != 0:
            x = 0 if dx > 0 else sx - 1
            mask = (torch_side[:, 0] == x)
            if not mask.any():
                return EMPTY_LIGHT_LIST
            entries = torch_side[mask]
            plane = numpy.zeros((sy, sx), dtype=numpy.uint8)
            plane[entries[:, 1], entries[:, 2]] = numpy.maximum(
                plane[entries[:, 1], entries[:, 2]], entries[:, 3]
            )
            return self._pack_plane_entries(plane, x_fixed=x)
        z = 0 if dz > 0 else sx - 1
        mask = (torch_side[:, 2] == z)
        if not mask.any():
            return EMPTY_LIGHT_LIST
        entries = torch_side[mask]
        plane = numpy.zeros((sy, sx), dtype=numpy.uint8)
        plane[entries[:, 1], entries[:, 0]] = numpy.maximum(
            plane[entries[:, 1], entries[:, 0]], entries[:, 3]
        )
        return self._pack_plane_entries(plane, z_fixed=z)

    def _build_incoming_from_neighbors(self, sector):
        if not getattr(config, 'LIGHT_PULL_BOUNDARY_FROM_NEIGHBORS', True):
            return sector.incoming_sky, sector.incoming_torch
        incoming_sky = dict(sector.incoming_sky)
        incoming_torch = dict(sector.incoming_torch)
        x0, _, z0 = sector.position
        for dx, dz in NEIGHBOR_OFFSETS_8:
            npos = (x0 + dx * SECTOR_SIZE, 0, z0 + dz * SECTOR_SIZE)
            neighbor = self.sectors.get(npos)
            if neighbor is None or neighbor.light_dirty_internal:
                continue
            incoming_sky[(dx, dz)] = self._boundary_sky_entries(neighbor, dx, dz)
            incoming_torch[(dx, dz)] = self._boundary_torch_entries(neighbor, dx, dz)
        return incoming_sky, incoming_torch

    def _gather_blocks_halo(self, sector, require_diagonals=True):
        """Assemble a temporary padded blocks array from neighbors."""
        if not self._neighbors_ready(sector, require_diagonals=require_diagonals):
            return None
        sx = SECTOR_SIZE
        halo = numpy.zeros((sx + 2, SECTOR_HEIGHT, sx + 2), dtype=sector.blocks.dtype)
        halo[1:-1, :, 1:-1] = sector.blocks
        x0, _, z0 = sector.position

        east = self.sectors.get((x0 + SECTOR_SIZE, 0, z0))
        west = self.sectors.get((x0 - SECTOR_SIZE, 0, z0))
        south = self.sectors.get((x0, 0, z0 + SECTOR_SIZE))
        north = self.sectors.get((x0, 0, z0 - SECTOR_SIZE))
        if east is None or west is None or south is None or north is None:
            return None
        halo[sx + 1, :, 1:-1] = east.blocks[0, :, :]
        halo[0, :, 1:-1] = west.blocks[sx - 1, :, :]
        halo[1:-1, :, sx + 1] = south.blocks[:, :, 0]
        halo[1:-1, :, 0] = north.blocks[:, :, sx - 1]

        if require_diagonals:
            ne = self.sectors.get((x0 + SECTOR_SIZE, 0, z0 - SECTOR_SIZE))
            nw = self.sectors.get((x0 - SECTOR_SIZE, 0, z0 - SECTOR_SIZE))
            se = self.sectors.get((x0 + SECTOR_SIZE, 0, z0 + SECTOR_SIZE))
            sw = self.sectors.get((x0 - SECTOR_SIZE, 0, z0 + SECTOR_SIZE))
            if ne is None or nw is None or se is None or sw is None:
                return None
            halo[sx + 1, :, 0] = ne.blocks[0, :, sx - 1]
            halo[0, :, 0] = nw.blocks[sx - 1, :, sx - 1]
            halo[sx + 1, :, sx + 1] = se.blocks[0, :, 0]
            halo[0, :, sx + 1] = sw.blocks[sx - 1, :, 0]
        return halo

    def _gather_blocks_tile_3x3(self, sector, allow_missing=False):
        """Assemble a 3x3 sector tile for lighting (48xH x48)."""
        sx = SECTOR_SIZE
        tile = numpy.zeros((sx * 3, SECTOR_HEIGHT, sx * 3), dtype=sector.blocks.dtype)
        x0, _, z0 = sector.position
        for ox in (-1, 0, 1):
            for oz in (-1, 0, 1):
                pos = (x0 + ox * sx, 0, z0 + oz * sx)
                s = self.sectors.get(pos)
                if s is None:
                    if allow_missing:
                        continue
                    return None
                xs = (ox + 1) * sx
                zs = (oz + 1) * sx
                tile[xs:xs + sx, :, zs:zs + sx] = s.blocks
        return tile

    def _invalidate_for_edit(
        self,
        sector,
        rel,
        priority=False,
    ):
        """Invalidate a sector and touched neighbors after a local edit."""
        sector.invalidate()
        sector.light_dirty_internal = True
        sector.light_dirty_from_edit = True
        if sector.shown:
            if priority:
                self._queue_priority_light(sector)
            else:
                self._submit_mesh_job(sector, priority=priority)
        on_edge = True
        if rel is not None:
            on_edge = rel[0] in (0, SECTOR_SIZE - 1) or rel[2] in (0, SECTOR_SIZE - 1)
        if not on_edge:
            return
        if rel is None:
            neighbor_offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
        else:
            neighbor_offsets = []
            if rel[0] == 0:
                neighbor_offsets.append((-1, 0))
            elif rel[0] == SECTOR_SIZE - 1:
                neighbor_offsets.append((1, 0))
            if rel[2] == 0:
                neighbor_offsets.append((0, -1))
            elif rel[2] == SECTOR_SIZE - 1:
                neighbor_offsets.append((0, 1))
        neighbor_offsets = tuple(neighbor_offsets)
        if priority and on_edge:
            group_sectors = [sector]
            for dx, dz in neighbor_offsets:
                npos = (sector.position[0] + dx * SECTOR_SIZE, 0, sector.position[2] + dz * SECTOR_SIZE)
                n = self.sectors.get(npos)
                if n is not None:
                    group_sectors.append(n)
            self._start_edge_edit_group(group_sectors)
        for dx, dz in neighbor_offsets:
            if dx == 0 and dz == 0:
                continue
            npos = (sector.position[0] + dx * SECTOR_SIZE, 0, sector.position[2] + dz * SECTOR_SIZE)
            n = self.sectors.get(npos)
            if n is None:
                continue
            n.invalidate()
            if n.shown:
                if priority:
                    self._queue_priority_mesh(n)
                else:
                    self._submit_mesh_job(n, priority=priority)

    def is_sector_ready(self, position, radius=1, require_diagonals=False):
        """Return True when the player's sector and neighbors within radius are loaded."""
        spos = sectorize(position)
        sector = self.sectors.get(spos)
        if sector is None:
            return False
        if radius <= 0:
            return True
        if radius <= 1:
            return self._neighbors_ready(sector, require_diagonals)
        x0, _, z0 = sector.position
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                pos = (x0 + dx * SECTOR_SIZE, 0, z0 + dz * SECTOR_SIZE)
                if pos not in self.sectors:
                    return False
        return True

    def is_host(self):
        return self.server is not None and self.player is not None and self.host_id == self.player.id

    def queue_server_message(self, message, data):
        if self.server is None:
            return
        if data is None:
            data = []
        elif not isinstance(data, (list, tuple)):
            data = [data]
        self._server_outbox.append([message, data])

    def _process_server_outbox(self):
        if not self.server:
            return
        while self._server_outbox:
            payload = self._server_outbox.popleft()
            self.server.send(payload)

    def _handle_entity_batch(self, payload, snapshot=False):
        states = entity_codec.unpack_entity_batch(payload)
        now = time.perf_counter()
        active_ids = set()
        for state in states:
            entity_id = state.get("id")
            if entity_id is None:
                continue
            active_ids.add(entity_id)
            history = self._entity_history.get(entity_id)
            if history is None:
                history = deque(maxlen=3)
                self._entity_history[entity_id] = history
            history.append((now, state))
        if snapshot:
            for existing_id in list(self._entity_history.keys()):
                if existing_id not in active_ids:
                    self._entity_history.pop(existing_id, None)

    def _handle_entity_despawn(self, payload):
        ids = payload.get("ids", payload) if isinstance(payload, dict) else payload
        if ids is None:
            return
        for entity_id in list(ids):
            self._entity_history.pop(int(entity_id), None)

    def _handle_server_message(self, msg, data):
        if msg == 'connected':
            self.player, self.players = data
            self._sync_remote_players_from_list(self.players)
            if self.player is not None:
                desired = getattr(config, "PLAYER_NAME", None)
                if desired and desired != self.player.name:
                    self.player.name = desired
                    self.queue_server_message("set_name", desired)
                    self.accepted_player_name = None
                    self.name_request_pending = True
                    self.name_sent_once = True
            others = [p for p in self.players if self.player is None or p.id != self.player.id]
            if others:
                roster = ", ".join(f"{p.name}({p.id})" for p in others)
            else:
                roster = "none"
            print(
                f"CLIENT: connected to {config.SERVER_IP}:{config.SERVER_PORT} "
                f"as {self.player.name}({self.player.id}); others={roster}"
            )
            return
        if msg == 'host_assign':
            if isinstance(data, (list, tuple)) and data:
                self.host_id = data[0]
            else:
                self.host_id = data
            if self.is_host():
                self.entity_snapshot_requested = True
            return
        if msg == 'other_player_join':
            if isinstance(data, (list, tuple)) and data:
                player = data[0]
            else:
                player = data
            if player is not None:
                self.players.append(player)
                self._sync_remote_players_from_list([player])
            return
        if msg == 'other_player_leave':
            player_id = None
            if isinstance(data, (list, tuple)) and data:
                player_id = data[0]
            else:
                player_id = data
            if player_id is not None:
                try:
                    pid = int(player_id)
                except (TypeError, ValueError):
                    pid = None
                if pid is not None:
                    self.remote_players.pop(pid, None)
                    self.players = [p for p in self.players if p.id != pid]
            return
        if msg == 'player_set_name':
            player_id = None
            name = None
            if isinstance(data, (list, tuple)):
                if len(data) >= 2:
                    player_id, name = data[0], data[1]
                elif len(data) == 1:
                    name = data[0]
            if player_id is not None:
                if self.player is not None and player_id == self.player.id:
                    self.accepted_player_name = name
                    self.name_request_pending = False
                if self.player is not None:
                    for p in self.players:
                        if p.id == player_id:
                            p.name = name
                            break
                self._update_remote_player_name(player_id, name)
            return
        if msg == 'player_set_position':
            player_id = None
            position = None
            rotation = None
            if isinstance(data, (list, tuple)):
                if len(data) >= 3:
                    player_id, position, rotation = data[0], data[1], data[2]
                elif len(data) == 2:
                    player_id, position = data[0], data[1]
                elif len(data) == 1:
                    position = data[0]
            if player_id is not None and position is not None:
                self._update_remote_player_position(player_id, position, rotation)
            return
        if msg == 'player_spawn_state':
            state = None
            if isinstance(data, (list, tuple)) and data:
                state = data[0]
            elif isinstance(data, dict):
                state = data
            if state:
                self.pending_spawn_state = state
            return
        if msg == 'player_set_block':
            logutil.log("SERVER", f"player_set_block {data}")
            pos, block = data
            if self[pos] != block:
                self.add_block(pos, block, False, priority=False)
            return
        if msg == 'entity_snapshot':
            self._handle_entity_batch(data, snapshot=True)
            return
        if msg == 'entity_update':
            self._handle_entity_batch(data, snapshot=False)
            return
        if msg == 'entity_spawn':
            self._handle_entity_batch(data, snapshot=False)
            return
        if msg == 'entity_despawn':
            self._handle_entity_despawn(data)
            return
        if msg == 'entity_request_snapshot':
            self.entity_snapshot_requested = True
            return
        if msg == 'entity_seed_snapshot':
            seed = None
            if isinstance(data, (list, tuple)) and data:
                seed = data[0]
            elif isinstance(data, dict):
                seed = data
            if seed:
                self.pending_entity_seed = seed
            return
        if msg == 'player_state_request':
            self.player_state_request = True
            return

    def _sync_remote_players_from_list(self, players):
        if not players:
            return
        for player in players:
            if self.player is not None and player.id == self.player.id:
                continue
            info = self.remote_players.get(player.id)
            if info is None:
                info = {}
                self.remote_players[player.id] = info
            info["name"] = getattr(player, "name", "Player")
            info["position"] = getattr(player, "position", (0.0, 0.0, 0.0))
            info["rotation"] = getattr(player, "rotation", (0.0, 0.0))

    def _update_remote_player_name(self, player_id, name):
        try:
            pid = int(player_id)
        except (TypeError, ValueError):
            return
        info = self.remote_players.get(pid)
        if info is None:
            info = {}
            self.remote_players[pid] = info
        if name is not None:
            info["name"] = name

    def _update_remote_player_position(self, player_id, position, rotation=None):
        try:
            pid = int(player_id)
        except (TypeError, ValueError):
            return
        info = self.remote_players.get(pid)
        if info is None:
            info = {}
            self.remote_players[pid] = info
        now = time.perf_counter()
        prev_pos = info.get("position")
        prev_time = info.get("last_time")
        info["position"] = position
        info["last_time"] = now
        if prev_pos is not None and prev_time is not None:
            dt = now - prev_time
            if dt > 1e-6:
                p0 = numpy.asarray(prev_pos, dtype=numpy.float32)
                p1 = numpy.asarray(position, dtype=numpy.float32)
                info["velocity"] = ((p1 - p0) / dt).tolist()
        if rotation is not None:
            info["rotation"] = rotation

    def _process_server_messages(self, ipc_ok=None):
        if not self.server:
            return
        self._process_server_outbox()
        if ipc_ok is None:
            ipc_ok = True
        while ipc_ok and self.server.poll():
            try:
                msg, data = self.server.recv()
                self._handle_server_message(msg, data)
            except EOFError:
                logutil.log("SERVER", "server returned EOF", level="WARN")
                break

    def get_interpolated_entities(self, now=None):
        if now is None:
            now = time.perf_counter()
        render_time = now - self._entity_interp_delay
        result = {}
        for entity_id, history in self._entity_history.items():
            if not history:
                continue
            if len(history) == 1:
                result[entity_id] = history[-1][1]
                continue
            prev = None
            next_state = None
            for sample_time, sample_state in history:
                if sample_time <= render_time:
                    prev = (sample_time, sample_state)
                if sample_time >= render_time:
                    next_state = (sample_time, sample_state)
                    break
            if prev is None:
                result[entity_id] = history[0][1]
                continue
            if next_state is None:
                last_time, last_state = history[-1]
                dt = render_time - last_time
                if dt > self._entity_interp_extrap:
                    result[entity_id] = last_state
                    continue
                vel = numpy.asarray(last_state.get("vel", (0.0, 0.0, 0.0)), dtype=numpy.float32)
                pos = numpy.asarray(last_state.get("pos", (0.0, 0.0, 0.0)), dtype=numpy.float32)
                state = dict(last_state)
                state["pos"] = pos + vel * dt
                result[entity_id] = state
                continue
            t0, state0 = prev
            t1, state1 = next_state
            if t1 <= t0:
                result[entity_id] = state1
                continue
            t = (render_time - t0) / (t1 - t0)
            pos0 = numpy.asarray(state0.get("pos", (0.0, 0.0, 0.0)), dtype=numpy.float32)
            pos1 = numpy.asarray(state1.get("pos", (0.0, 0.0, 0.0)), dtype=numpy.float32)
            rot0 = numpy.asarray(state0.get("rot", (0.0, 0.0)), dtype=numpy.float32)
            rot1 = numpy.asarray(state1.get("rot", (0.0, 0.0)), dtype=numpy.float32)
            pos = pos0 + (pos1 - pos0) * t
            rot = rot0 + (rot1 - rot0) * t
            state = dict(state0)
            state["pos"] = pos
            state["rot"] = rot
            seg0 = state0.get("segment_positions")
            seg1 = state1.get("segment_positions")
            if seg0 is not None and seg1 is not None and len(seg0) == len(seg1):
                a0 = numpy.asarray(seg0, dtype=numpy.float32)
                a1 = numpy.asarray(seg1, dtype=numpy.float32)
                state["segment_positions"] = (a0 + (a1 - a0) * t).tolist()
            result[entity_id] = state
        return result

    def update_sectors(self, old, new, player_pos=None, look_vec=None, frustum_circle=None, ipc_budget_ms=None, allow_send=True):
        """
        the observer has moved from sector old to new
        """
        deadline = None
        if ipc_budget_ms is not None:
            if ipc_budget_ms <= 0.0:
                deadline = 0.0
            else:
                deadline = time.perf_counter() + ipc_budget_ms / 1000.0

        def _ipc_ok():
            if deadline is None:
                return True
            if deadline == 0.0:
                return False
            return time.perf_counter() <= deadline

        new = sectorize(new)
        self.player_sector = new
        self.player_pos = player_pos
        self.player_look = look_vec
        if self._defer_work_frame != self.frame_id and self._deferred_mesh_jobs:
            for sector, priority in list(self._deferred_mesh_jobs.items()):
                if sector in self.sectors.values() and not sector.mesh_job_pending:
                    self._submit_mesh_job(sector, priority=priority)
            self._deferred_mesh_jobs.clear()
        self._maybe_log_queue_state()
        self._maybe_log_missing_sectors(new)
        inflight = self.n_requests - self.n_responses
        max_inflight = getattr(config, 'LOADER_MAX_INFLIGHT', 1)
        strict_inflight = getattr(config, 'LOADER_STRICT_INFLIGHT', False)
        # Always allow IPC while the current sector is missing to bootstrap terrain.
        if new not in self.sectors:
            deadline = None

        prev_ref = self.update_ref_pos

        self._process_server_messages(ipc_ok=_ipc_ok())

        # Process any queued loader messages without blocking.
        processed_msgs = 0
        while not self.loader_messages.empty():
            if processed_msgs > 0 and not _ipc_ok():
                break
            raw = self.loader_messages.get_nowait()
            if isinstance(raw, tuple) and raw[0] == '__eof__':
                logutil.log("CLIENT", "loader returned EOF", level="WARN")
                break
            if isinstance(raw, tuple) and raw[0] == '__error__':
                logutil.log("CLIENT", f"loader recv error {raw[1]}", level="WARN")
                break
            recv_start = time.perf_counter()
            # Allow either (msg, data) tuples or variable-length lists.
            msg, data = None, None
            if isinstance(raw, (list, tuple)) and len(raw) > 0:
                msg = raw[0]
                if len(raw) == 2:
                    data = raw[1]
                else:
                    data = raw[1:]
            else:
                logutil.log("CLIENT", f"recv unexpected payload {raw}", level="WARN")
            recv_ms = (time.perf_counter() - recv_start) * 1000.0
            logutil.log("CLIENT", f"received {msg} in {recv_ms:.1f}ms")
            if msg == 'sector_blocks':
                spos1, b1, v1, light1 = data
                self.n_responses += 1
                self.active_loader_request = [None, None]
                response_ms = (time.time() - self.loader_time) * 1000.0
                logutil.log("CLIENT", f"loader response ms={response_ms:.1f}")
                if getattr(config, 'LOG_LOADER_FLOW', False):
                    logutil.log("CLIENT", f"loader_resp sector={spos1}")
                self._update_sector(spos1, b1, v1, light1)
                self._note_sector_load(spos1, response_ms)
                self.loader_sectors_received_total += 1
                self.loader_recent.append(spos1)
            if msg == 'sector_blocks2':
                self.n_responses += 1
                self.active_loader_request = [None, None]
                response_ms = (time.time() - self.loader_time) * 1000.0
                logutil.log("CLIENT", f"loader response ms={response_ms:.1f}")
                # sector_blocks2 payload may be [sector_results, token] or just sector_results
                token = None
                sector_results = data
                if isinstance(data, (list, tuple)) and len(data) == 2:
                    sector_results, token = data
                per_sector_ms = response_ms / max(1, len(sector_results))
                for item in sector_results:
                    spos, b, v, light = item
                    if token is not None:
                        # Drop stale responses superseded by newer edits.
                        if isinstance(token, dict):
                            if self.sector_edit_tokens.get(spos, 0) != token.get(spos, 0):
                                continue
                        elif isinstance(token, int):
                            if self.sector_edit_tokens.get(spos, 0) != token:
                                continue
                        # For local edits, we already applied blocks; avoid full reload.
                        if spos in self.sectors:
                            s = self.sectors[spos]
                            s.edit_inflight = False
                            continue
                    if getattr(config, 'LOG_LOADER_FLOW', False):
                        logutil.log("CLIENT", f"loader_resp sector={spos}")
                    self._update_sector(spos, b, v, light)
                    self._note_sector_load(spos, per_sector_ms)
                    self.loader_sectors_received_total += 1
                    self.loader_recent.append(spos)
            if msg == 'set_block_ack':
                self.n_responses += 1
                self.active_loader_request = [None, None]
                if isinstance(data, (list, tuple)) and len(data) == 3:
                    pos, block_id, token = data
                else:
                    pos = data[0] if data else None
                    token = None
                if pos is not None:
                    spos = sectorize(pos)
                    if token is not None and self.sector_edit_tokens.get(spos, 0) != token:
                        continue
                    s = self.sectors.get(spos)
                    if s is not None:
                        s.edit_inflight = False
                continue
            if msg == 'set_blocks_ack':
                self.n_responses += 1
                self.active_loader_request = [None, None]
                updates = []
                token_map = None
                if isinstance(data, (list, tuple)):
                    if len(data) == 2:
                        updates, token_map = data
                    elif len(data) == 1:
                        updates = data[0]
                if token_map is None:
                    token_map = {}
                for pos, _block in updates:
                    spos = sectorize(pos)
                    token = token_map.get(spos)
                    if token is not None and self.sector_edit_tokens.get(spos, 0) != token:
                        continue
                    s = self.sectors.get(spos)
                    if s is not None:
                        s.edit_inflight = False
                continue
            if msg == 'seed':
                self.n_responses += 1
                self.world_seed = data
                logutil.log("CLIENT", f"loader seed={data}")
            processed_msgs += 1

        refreshed = self._refresh_load_candidates(new, player_pos, look_vec, frustum_circle)
        if refreshed:
            self.stat_load_refresh_total += 1
            self.stat_load_candidates_total += len(self.update_sectors_pos)
        if prev_ref != new:
            keep_radius = max(self.keep_radius, max(self.load_radius, LOADED_SECTORS))
            for s in list(self.sectors):
                if (abs(new[0] - s[0]) > keep_radius * SECTOR_SIZE
                        or abs(new[2] - s[2]) > keep_radius * SECTOR_SIZE):
                    logutil.log("WORLD", f"dropping sector={s} loaded={len(self.sectors)}")
                    self.release_sector(self.sectors[s])

        missing_any = False
        missing_near = False
        for _, pos in self.update_sectors_pos:
            if pos in self.sectors:
                continue
            missing_any = True
            if (abs(pos[0] - new[0]) <= SECTOR_SIZE
                    and abs(pos[2] - new[2]) <= SECTOR_SIZE):
                missing_near = True
            if missing_near:
                break

        priority_processed = self._process_priority_work()
        priority_work = priority_processed or bool(self._priority_light_queue) or bool(self._priority_mesh_queue)

        if not missing_any and inflight == 0 and not self.loader_requests and processed_msgs == 0 and not priority_work:
            self._process_server_messages(ipc_ok=_ipc_ok())
            return

        priority_work = (priority_work
                         or self.pending_priority_jobs > 0
                         or bool(self._deferred_mesh_jobs))
        if not priority_work:
            for sector in self.sectors.values():
                if not sector.edit_inflight:
                    continue
                if sector.mesh_job_pending or sector.invalidate_vt or sector in self.pending_upload_set:
                    priority_work = True
                    break

        allowed_load = None
        if not priority_work:
            allowed_load = {new}
            for sector in self.sectors.values():
                needs_pipeline = ((sector.vt_data is None and not sector.mesh_built)
                                  or sector.light_dirty_internal)
                if not needs_pipeline:
                    continue
                x, _, z = sector.position
                for dx, dz in NEIGHBOR_OFFSETS_8:
                    allowed_load.add((x + dx * SECTOR_SIZE, 0, z + dz * SECTOR_SIZE))

        requested = set()
        if self.active_loader_request[0] == 'sector_blocks':
            requested.add(self.active_loader_request[1])
        for req in self.loader_requests:
            if req[0] == 'sector_blocks':
                requested.add(req[1][0])

        if refreshed or not self.loader_requests:
            non_sector = [r for r in self.loader_requests if r[0] != 'sector_blocks']
            candidates = []
            if not priority_work:
                for priority, pos in self.update_sectors_pos:
                    if pos in requested or pos in self.sectors:
                        continue
                    if allowed_load is not None and pos not in allowed_load:
                        continue
                    in_3x3 = (abs(pos[0] - new[0]) <= SECTOR_SIZE
                              and abs(pos[2] - new[2]) <= SECTOR_SIZE)
                    candidates.append(((0 if in_3x3 else 1), priority, pos))
            candidates.sort()
            queue_limit = max_inflight if max_inflight > 0 else 1
            sector_reqs = []
            for _, _, pos in candidates[:queue_limit]:
                logutil.log("WORLD", f"queueing sector={pos}")
                sector_reqs.append(['sector_blocks', [pos]])
            self.loader_requests = non_sector + sector_reqs

        if self._defer_work_frame == self.frame_id:
            allow_send = False
            self.stat_loader_block_defer_total += 1

        if priority_work:
            allow_send = False
            if self.loader_requests:
                req_type = self.loader_requests[0][0]
                if req_type in ('set_block', 'set_blocks'):
                    allow_send = True

        block_on_mesh = getattr(config, 'LOADER_BLOCK_ON_MESH_BACKLOG', False)
        if block_on_mesh and allow_send and self._has_mesh_backlog():
            backlog_min = max(1, int(getattr(config, 'LOADER_BLOCK_MESH_BACKLOG_MIN', 8)))
            min_loaded = max(1, int(getattr(config, 'LOADER_BLOCK_MESH_MIN_LOADED', 36)))
            allow_near = missing_near and len(self.sectors) < min_loaded
            if not allow_near and self._mesh_backlog_count() >= backlog_min:
                if self.mesh_active_jobs >= self.mesh_active_cap:
                    if not self.loader_requests or self.loader_requests[0][0] not in ('set_block', 'set_blocks'):
                        allow_send = False
                self.stat_loader_block_mesh_total += 1

        if allow_send and len(self.loader_requests) > 0:
            if strict_inflight and inflight > 0:
                self.stat_loader_block_inflight_total += 1
            else:
                req_type = self.loader_requests[0][0]
                if req_type != 'sector_blocks' or inflight < max_inflight:
                    self.loader_time = time.time()
                    self.n_requests += 1
                    if req_type == 'sector_blocks':
                        self.active_loader_request = ['sector_blocks', self.loader_requests[0][1][0]]
                        if getattr(config, 'LOG_LOADER_FLOW', False):
                            logutil.log("CLIENT", f"loader_send sector={self.loader_requests[0][1][0]}")
                    logutil.log("CLIENT", f"sending request to loader {self.loader_requests[0][0]}")
                    self.loader.send(self.loader_requests.pop(0))
                    self.stat_loader_sent_total += 1

        self._process_server_messages(ipc_ok=_ipc_ok())

    def _build_block_vt(self, block_id, pos):
        face_count = int(BLOCK_FACE_COUNT[block_id])
        if face_count <= 0:
            return {'solid': None, 'water': None}
        dirs = BLOCK_FACE_DIR[block_id][:face_count]
        verts = (0.5 * BLOCK_VERTICES[block_id][:face_count].reshape(face_count, 4, 3)
                 + pos[None, None, :] + BLOCK_RENDER_OFFSET).astype(numpy.float32)
        tex_base = BLOCK_TEXTURES_FLIPPED[block_id][:6].reshape(6, 4, 2).astype(numpy.float32)
        tex = numpy.take(tex_base, dirs, axis=0).astype(numpy.float32)
        normals_base = BLOCK_NORMALS[dirs].astype(numpy.float32)
        normals = numpy.broadcast_to(normals_base[:, None, :], (face_count, 4, 3))
        colors_base = BLOCK_COLORS[block_id][:6].reshape(6, 4, 3).astype(numpy.float32)
        colors_rgb = numpy.take(colors_base, dirs, axis=0)
        emissive = numpy.full((face_count, 4, 1), BLOCK_GLOW[block_id]*255.0, dtype=numpy.float32)
        colors = numpy.concatenate([colors_rgb, emissive], axis=2)
        light = numpy.ones((6,4,2), dtype=numpy.float32)
        face_mask = numpy.ones((6,4), dtype=bool)
        v = verts[face_mask].ravel().astype('f4')
        t = tex[face_mask].ravel().astype('f4')
        n = normals[face_mask].ravel().astype('f4')
        c = colors[face_mask].ravel().astype('f4')
        l = light[face_mask].ravel().astype('f4')
        count = len(v)//3
        if config.DEBUG_SINGLE_BLOCK:
            logutil.log("DEBUG", f"block vertex sample {v[:18]} tex {t[:8]}")
        solid = (count, v, t, n, c, l)
        return {'solid': solid, 'water': None}

    def _triangulate_vt(self, vt_data, key=None):
        """Convert quad vt_data to triangle arrays."""
        vt_tuple = self._get_vt_entry(vt_data, key)
        if not vt_tuple or vt_tuple[0] <= 0:
            return (numpy.array([], dtype='f4').reshape(0, 3),
                    numpy.array([], dtype='f4').reshape(0, 2),
                    numpy.array([], dtype='f4').reshape(0, 3),
                    numpy.array([], dtype='f4').reshape(0, 3),
                    numpy.array([], dtype='f4').reshape(0, 2))
        count, v, t, n, c, l = vt_tuple
        quad_verts = numpy.array(v, dtype='f4').reshape(-1, 4, 3)
        quad_tex = numpy.array(t, dtype='f4').reshape(-1, 4, 2)
        quad_norm = numpy.array(n, dtype='f4').reshape(-1, 4, 3)
        channels = int(len(c) // len(quad_verts) // 4) if len(quad_verts) else 3
        quad_col = numpy.array(c, dtype='f4').reshape(-1, 4, channels)
        light_channels = int(len(l) // len(quad_verts) // 4) if len(quad_verts) else 2
        quad_light = numpy.array(l, dtype='f4').reshape(-1, 4, light_channels)
        order = [0, 1, 2, 0, 2, 3]
        tri_verts = quad_verts[:, order, :].reshape(-1, 3)
        tri_tex = quad_tex[:, order, :].reshape(-1, 2)
        tri_norm = quad_norm[:, order, :].reshape(-1, 3)
        tri_col = quad_col[:, order, :].reshape(-1, channels)
        tri_light = quad_light[:, order, :].reshape(-1, light_channels)
        return tri_verts, tri_tex, tri_norm, tri_col, tri_light

    def _update_sector(self, spos, b, v, light):
        if b is not None:
            if spos in self.sectors:
                logutil.log("WORLD", f"updating existing sector data {spos}")
                s = self.sectors[spos]
                s.blocks[:,:,:] = b
                s.sky_floor[:] = 0
                s.sky_side = EMPTY_LIGHT_LIST
                s.torch_side = EMPTY_LIGHT_LIST
                s.incoming_sky_updates = 0
                s.incoming_torch_updates = 0
                s.edge_sky_counts = (0, 0, 0, 0)
                s.edge_torch_counts = (0, 0, 0, 0)
                s.defer_upload = False
                s.edit_group_id = None
                s.light_dirty_from_edit = False
                s.light_initialized = False
                s.light_dirty_internal = True
                s.light_dirty_incoming = True
                s.light_neighbors_ready = False
                s.vt_data = v
                s.mesh_built = v is not None
                if v is not None:
                    s._note_new_vt_data()
                else:
                    s.vt_upload_active_token = None
                    s.vt_upload_prepared = False
                s.vt_clear_pending = True
                self._invalidate_and_rebuild(s)
                self._mark_neighbor_outgoing_dirty(s.position)
            else:
                logutil.log("WORLD", f"setting new sector data {spos}")
                batch_solid, batch_water = self.get_batches()
                s = SectorProxy(spos, batch_solid, batch_water, self.group, self.water_group, self)
                s.blocks[:,:,:] = b
                s.vt_data = v
                s.mesh_built = v is not None
                if v is not None:
                    s._note_new_vt_data()
                else:
                    s.vt_upload_active_token = None
                    s.vt_upload_prepared = False
                s.vt_clear_pending = True
                s.light_neighbors_ready = False
                s.incoming_sky_updates = 0
                s.incoming_torch_updates = 0
                s.edge_sky_counts = (0, 0, 0, 0)
                s.edge_torch_counts = (0, 0, 0, 0)
                s.defer_upload = False
                s.edit_group_id = None
                s.light_dirty_from_edit = False
                s.light_initialized = False
                self.sectors[sectorize(spos)] = s
                self._invalidate_and_rebuild(s)
                self._mark_neighbor_outgoing_dirty(s.position)

    def hit_test(self, position, vector, max_distance=8):
        """ Line of sight search from current position. If a block is
        intersected it is returned, along with the block previously in the line
        of sight. If no block is found, return None, None.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position to check visibility from.
        vector : tuple of len 3
            The line of sight vector.
        max_distance : int
            How many blocks away to search for a hit.

        """
        m = 8
        x, y, z = position
        dx, dy, dz = vector
        previous = None
        for _ in range(max_distance * m):
            key = normalize((x, y, z))
            if key != previous:
                b = self[key]
                # Treat water as transparent for interaction; keep searching for solid terrain.
                if b != 0 and b is not None and b != WATER:
                    return key, previous
            previous = key
            x, y, z = x + dx / m, y + dy / m, z + dz / m
        return None, None

    def measure_void_distance(self, position, vector, max_distance=64):
        """Return the number of solid blocks after the first hit until air along a ray.

        Starts from the current sight line, finds the first solid block, then steps
        block-by-block along the ray direction until the first air/void cell or until
        max_distance is exceeded. Returns None when no initial hit is found.
        """
        hit, _ = self.hit_test(position, vector, max_distance=max_distance)
        if not hit:
            return None

        hx, hy, hz = hit
        dx, dy, dz = vector
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length == 0:
            return None
        dirx, diry, dirz = dx / length, dy / length, dz / length

        # Start at center of hit block to avoid immediately re-hitting it.
        cellx, celly, cellz = hx, hy, hz
        px, py, pz = hx + 0.5, hy + 0.5, hz + 0.5
        step_x = 1 if dirx >= 0 else -1
        step_y = 1 if diry >= 0 else -1
        step_z = 1 if dirz >= 0 else -1

        invx = 1.0 / dirx if dirx != 0 else float("inf")
        invy = 1.0 / diry if diry != 0 else float("inf")
        invz = 1.0 / dirz if dirz != 0 else float("inf")

        t_max_x = ((cellx + (1 if step_x > 0 else 0)) - px) * invx if invx != float("inf") else float("inf")
        t_max_y = ((celly + (1 if step_y > 0 else 0)) - py) * invy if invy != float("inf") else float("inf")
        t_max_z = ((cellz + (1 if step_z > 0 else 0)) - pz) * invz if invz != float("inf") else float("inf")

        t_delta_x = abs(invx)
        t_delta_y = abs(invy)
        t_delta_z = abs(invz)

        traveled_blocks = 0
        t = 0.0
        max_steps = int(math.ceil(max_distance)) + 2

        for _ in range(max_steps):
            # advance to next voxel boundary
            if t_max_x < t_max_y:
                if t_max_x < t_max_z:
                    cellx += step_x
                    t = t_max_x
                    t_max_x += t_delta_x
                else:
                    cellz += step_z
                    t = t_max_z
                    t_max_z += t_delta_z
            else:
                if t_max_y < t_max_z:
                    celly += step_y
                    t = t_max_y
                    t_max_y += t_delta_y
                else:
                    cellz += step_z
                    t = t_max_z
                    t_max_z += t_delta_z

            if t > max_distance:
                break
            block = self[(cellx, celly, cellz)]
            if block == 0 or block is None:
                return traveled_blocks
            traveled_blocks += 1

        return traveled_blocks

    def nearest_mushroom_in_sector(self, sector_pos, player_pos=None):
        """Return nearest mushroom world coords inside a loaded sector, or None."""
        mush_id = BLOCK_ID.get('Mushroom')
        if mush_id is None:
            return None
        sector = self.sectors.get(sector_pos)
        if sector is None or sector.blocks is None:
            return None
        coords = numpy.argwhere(sector.blocks == mush_id)
        if coords.size == 0:
            return None
        if player_pos is None:
            player_pos = (sector_pos[0], 0, sector_pos[2])
        px, py, pz = player_pos
        best = None
        best_d2 = None
        for cx, cy, cz in coords:
            wx = sector_pos[0] + int(cx) - 1
            wy = int(cy)
            wz = sector_pos[2] + int(cz) - 1
            dx = wx - px
            dy = wy - py
            dz = wz - pz
            d2 = dx*dx + dy*dy + dz*dz
            if best is None or d2 < best_d2:
                best = (wx, wy, wz)
                best_d2 = d2
        return best

    def exposed(self, position):
        """ Returns False is given `position` is surrounded on all 6 sides by
        blocks, True otherwise.

        """
        x, y, z = position
        for dx, dy, dz in FACES:
            b = self[normalize((x + dx, y + dy, z + dz))]
            if not BLOCK_SOLID[b]:
                return True
        return False

    def collide(self, position, bounding_box, velocity=None, prev_position=None):
        """
        Checks to see if an entity at the given `position` with the given
        `bounding_box` is colliding with any blocks in the world.

        Axis-aligned resolution against block AABBs.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position to check for collisions at.
        bounding_box : tuple of len 3
            The (width, height, depth) of the entity.

        Returns
        -------
        position : tuple of len 3
            The new position of the entity taking into account collisions.
        vertical_collision : bool
            True if the entity collided with the ground or ceiling.
        """
        width, height, depth = bounding_box
        prev = prev_position if prev_position is not None else position
        p = [position[0], position[1], position[2]]
        vertical_collision = False
        horizontal_collision = False

        def axis_bounds(pos):
            min_x = pos[0] - width / 2
            max_x = pos[0] + width / 2
            min_y = pos[1]
            max_y = pos[1] + height
            min_z = pos[2] - depth / 2
            max_z = pos[2] + depth / 2
            return min_x, max_x, min_y, max_y, min_z, max_z

        def block_range(min_v, max_v, centered):
            eps = 1e-6
            if centered:
                lo = int(math.floor(min_v - 0.5 + eps))
                hi = int(math.floor(max_v + 0.5 - eps))
            else:
                lo = int(math.floor(min_v + eps))
                hi = int(math.floor(max_v - eps))
            return range(lo, hi + 1)

        def iter_block_boxes(block_id, bx, by, bz):
            if block_id in STAIR_COLLISION_BOXES:
                for min_v, max_v in STAIR_COLLISION_BOXES[block_id]:
                    yield (
                        bx + min_v[0], bx + max_v[0],
                        by + min_v[1], by + max_v[1],
                        bz + min_v[2], bz + max_v[2],
                    )
            else:
                yield (
                    bx + BLOCK_COLLISION_MIN[block_id][0], bx + BLOCK_COLLISION_MAX[block_id][0],
                    by + BLOCK_COLLISION_MIN[block_id][1], by + BLOCK_COLLISION_MAX[block_id][1],
                    bz + BLOCK_COLLISION_MIN[block_id][2], bz + BLOCK_COLLISION_MAX[block_id][2],
                )

        def resolve_axis(axis, base_pos):
            nonlocal vertical_collision, horizontal_collision
            delta = p[axis] - prev[axis]
            if abs(delta) < 1e-6:
                return

            pos = [base_pos[0], base_pos[1], base_pos[2]]
            pos[axis] = prev[axis] + delta
            min_x, max_x, min_y, max_y, min_z, max_z = axis_bounds(pos)
            min_bx = block_range(min_x, max_x, centered=True)
            min_by = block_range(min_y, max_y, centered=False)
            min_bz = block_range(min_z, max_z, centered=True)

            prev_min_x, prev_max_x, prev_min_y, prev_max_y, prev_min_z, prev_max_z = axis_bounds(prev)

            for bx in min_bx:
                for by in min_by:
                    for bz in min_bz:
                        block_id = self[normalize((bx, by, bz))]
                        if not block_id or not BLOCK_COLLIDES[block_id]:
                            continue
                        for (block_min_x, block_max_x,
                             block_min_y, block_max_y,
                             block_min_z, block_max_z) in iter_block_boxes(block_id, bx, by, bz):
                            if axis in (0, 2) and block_id in STAIR_COLLISION_BOXES and block_id not in STAIR_UPSIDE_IDS:
                                foot_y = prev_min_y - by
                                if -0.05 <= foot_y <= 1.0:
                                    if not (max_x <= block_min_x or min_x >= block_max_x or max_z <= block_min_z or min_z >= block_max_z):
                                        low_half = block_max_y <= by + 0.5 + 1e-6
                                        high_half = block_min_y >= by + 0.5 - 1e-6
                                        if low_half:
                                            target_y = by + 0.5
                                            if velocity is not None:
                                                if pos[1] >= target_y - 1e-4:
                                                    velocity[1] = min(velocity[1], 0.0)
                                                else:
                                                    velocity[1] = max(velocity[1], 3.0)
                                            continue
                                        if high_half and foot_y >= 0.5 - 1e-4:
                                            target_y = by + 1.0
                                            if velocity is not None:
                                                if pos[1] >= target_y - 1e-4:
                                                    velocity[1] = min(velocity[1], 0.0)
                                                else:
                                                    velocity[1] = max(velocity[1], 3.0)
                                            continue
                            if axis == 0:
                                if max_y <= block_min_y or min_y >= block_max_y:
                                    continue
                                if max_z <= block_min_z or min_z >= block_max_z:
                                    continue
                                if max_x <= block_min_x or min_x >= block_max_x:
                                    continue
                                if delta > 0:
                                    pos[0] = block_min_x - width / 2
                                else:
                                    pos[0] = block_max_x + width / 2
                                horizontal_collision = True
                                min_x, max_x, min_y, max_y, min_z, max_z = axis_bounds(pos)
                            elif axis == 2:
                                if max_y <= block_min_y or min_y >= block_max_y:
                                    continue
                                if max_x <= block_min_x or min_x >= block_max_x:
                                    continue
                                if max_z <= block_min_z or min_z >= block_max_z:
                                    continue
                                if delta > 0:
                                    pos[2] = block_min_z - depth / 2
                                else:
                                    pos[2] = block_max_z + depth / 2
                                horizontal_collision = True
                                min_x, max_x, min_y, max_y, min_z, max_z = axis_bounds(pos)
                            else:
                                if max_x <= block_min_x or min_x >= block_max_x:
                                    continue
                                if max_z <= block_min_z or min_z >= block_max_z:
                                    continue
                                if delta < 0 and prev_min_y >= block_max_y and min_y < block_max_y:
                                    pos[1] = block_max_y
                                    vertical_collision = True
                                elif delta > 0 and prev_max_y <= block_min_y and max_y > block_min_y:
                                    pos[1] = block_min_y - height
                                min_x, max_x, min_y, max_y, min_z, max_z = axis_bounds(pos)

            p[axis] = pos[axis]

        # Resolve X with Y/Z at previous position.
        resolve_axis(0, prev)
        # Resolve Z with X resolved, Y at previous position.
        resolve_axis(2, (p[0], prev[1], prev[2]))
        # Resolve Y with X/Z resolved.
        resolve_axis(1, (p[0], prev[1], p[2]))

        # Snap to ground when settling to prevent jitter below surface.
        if p[1] <= prev[1] + 1e-6 and (velocity is None or velocity[1] <= 0):
            min_x, max_x, min_y, max_y, min_z, max_z = axis_bounds(p)
            prev_min_y = axis_bounds(prev)[2]
            eps = 1e-4
            snap_pad = 1e-4
            for bx in block_range(min_x, max_x, centered=True):
                for bz in block_range(min_z, max_z, centered=True):
                    by = int(math.floor(min_y - eps))
                    block_id = self[normalize((bx, by, bz))]
                    if not block_id or not BLOCK_COLLIDES[block_id]:
                        continue
                    for (block_min_x, block_max_x,
                         block_min_y, block_max_y,
                         block_min_z, block_max_z) in iter_block_boxes(block_id, bx, by, bz):
                        overlap_x = min(max_x, block_max_x) - max(min_x, block_min_x)
                        overlap_z = min(max_z, block_max_z) - max(min_z, block_min_z)
                        if overlap_x <= snap_pad or overlap_z <= snap_pad:
                            continue
                        if min_y >= block_max_y - eps and min_y <= block_max_y + eps:
                            if horizontal_collision and prev_min_y < block_max_y - eps:
                                continue
                            p[1] = block_max_y
                            vertical_collision = True
                            break
                    if vertical_collision:
                        break
                if vertical_collision:
                    break
        return tuple(p), vertical_collision

    def quit(self,kill_server=True):
        if self.n_requests > self.n_responses:
            logutil.log(
                "WORLD",
                f"draining loader responses inflight={self.n_requests - self.n_responses}",
            )
            try:
                while not self.loader_messages.empty():
                    self.loader_messages.get_nowait()
                    self.n_responses += 1
            except Exception as e:
                logutil.log("WORLD", f"drain loader error {e}", level="WARN")
        if getattr(self, 'mesh_single_worker', False):
            self._mesh_worker_stop.set()
            if hasattr(self, '_mesh_job_cv'):
                with self._mesh_job_cv:
                    self._mesh_job_cv.notify()
            if getattr(self, '_mesh_worker_thread', None) is not None:
                self._mesh_worker_thread.join(timeout=0.2)
        logutil.log("WORLD", "shutting down loader")
        self._loader_stop.set()
        self.loader.send(['quit',0])
        if self.server is not None:
            logutil.log("WORLD", "closing server connection")
            self.server.send(['quit',0])
