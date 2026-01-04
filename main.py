import math
import random
import time
import sys
from datetime import datetime

# pyglet imports
import pyglet
pyglet.options['debug_gl'] = False
pyglet.options['debug_media'] = False

image = pyglet.image
from pyglet.window import key, mouse
import pyglet.gl as gl
import pyglet.shapes as shapes
from pyglet.math import Mat4, Vec3

GLfloat3 = gl.GLfloat*3
GLfloat4 = gl.GLfloat*4

# standard lib imports
from collections import deque
import numpy as np
import itertools

# local module imports
import world_proxy as world
import util
import config
import shaders
import renderer
import world_entity_store
import logutil
import mapgen
from entities.player import HUMANOID_MODEL, Player
from entities.snake import SNAKE_MODEL, SnakeEntity
from entities.snail import SNAIL_MODEL, SnailEntity
from entities.seagull import SEAGULL_MODEL, SeagullEntity
from entities.dog import DOG_MODEL, Dog
from blocks import TEXTURE_PATH
from config import DIST, TICKS_PER_SEC, FLYING_SPEED, GRAVITY, JUMP_SPEED, \
        MAX_JUMP_HEIGHT, PLAYER_HEIGHT, TERMINAL_VELOCITY, TICKS_PER_SEC, \
        WALKING_SPEED, LOADED_SECTORS
from blocks import (
    BLOCK_ID,
    BLOCK_TEXTURES,
    BLOCK_VERTICES,
    BLOCK_COLORS,
    BLOCK_SOLID,
    BLOCK_COLLIDES,
    BLOCK_PICKER_FACE,
    BLOCK_INVENTORY,
    ORIENTED_BLOCK_IDS,
    WALL_MOUNTED_BLOCK_IDS,
    DOOR_BASE_IDS,
    DOOR_LOWER_IDS,
    DOOR_UPPER_IDS,
    DOOR_LOWER_TO_UPPER,
    DOOR_UPPER_TO_LOWER,
    DOOR_LOWER_TOGGLE,
    DOOR_UPPER_TOGGLE,
    ORIENT_SOUTH,
    ORIENT_WEST,
    ORIENT_NORTH,
    ORIENT_EAST,
    STAIR_BASE_IDS,
    STAIR_ORIENTED_IDS,
    STAIR_ORIENTED_UP_IDS,
)
WATER = BLOCK_ID['Water']


def pad_header(title, width=40):
    title = title.strip()
    pad = (width-len(title)-2)//2
    if pad <= 0:
        return title
    return '='*pad + ' ' + title + ' ' + '='*pad

class Window(pyglet.window.Window):

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)


        # Whether or not the window exclusively captures the mouse.
        self.exclusive = False

        # When flying gravity has no effect and speed is increased.
        self.flying = True
        self.fly_climb = 0

        # Strafing is moving lateral to the direction you are facing,
        # e.g. moving to the left or right while continuing to face forward.
        #
        # First element is -1 when moving forward, 1 when moving back, and 0
        # otherwise. The second element is -1 when moving left, 1 when moving
        # right, and 0 otherwise.
        self.strafe = [0, 0]

        # Current (x, y, z) position in the world, specified with floats. Note
        # that, perhaps unlike in math class, the y-axis is the vertical axis.
        if config.DEBUG_SINGLE_BLOCK:
            bx = config.SECTOR_SIZE//2
            by = config.SECTOR_HEIGHT//2
            bz = config.SECTOR_SIZE//2 + 2
            self.position = (bx, by, bz)
            logutil.log("MAIN", f"starting at {self.position} for single-block debug", level="DEBUG")
        else:
            self.position = (0, 160, 0)

        # First element is rotation of the player in the x-z plane (ground
        # plane) measured from the z-axis down. The second is the rotation
        # angle from the ground plane up. Rotation is in degrees.
        #
        # The vertical plane rotation ranges from -90 (looking straight down) to
        # 90 (looking straight up). The horizontal rotation range is unbounded.
        self.rotation = (0, 0)
        # Which sector the player is currently in.
        self.sector = None

        # The crosshairs at the center of the screen.
        self.reticle = []

        self.camera_mode = 'first_person'
        self.third_person_distance = 5
        self._camera_yaw_elapsed = 0.0
        self._camera_yaw_start = float(self.rotation[0])
        self._camera_yaw_active_target = None
        self._camera_pitch_elapsed = 0.0
        self._camera_pitch_start = float(self.rotation[1])
        self._camera_pitch_active_target = None


        self.inventory_item = None

        # Debug flags
        self._printed_mats = False

        # A list of blocks the player can place. Hit num keys to cycle.
        self.inventory = list(BLOCK_INVENTORY)

        # The current block the user can place. Hit num keys to cycle.
        self.block = self.inventory[0]

        # Convenience list of num keys.
        self.num_keys = [
            key._1, key._2, key._3, key._4, key._5,
            key._6, key._7, key._8, key._9, key._0]
        self.inv_prev_keys = [key.MINUS]
        self.inv_next_keys = [key.EQUAL]

        # Shader program used for world rendering.
        self.block_program = shaders.create_block_shader()
        self.block_program['u_texture'] = 0
        self.block_program['u_light_dir'] = getattr(config, 'SUN_LIGHT_DIR', (0.35, 1.0, 0.65))
        self.block_program['u_fog_color'] = getattr(config, 'DAY_FOG_COLOR', (0.5, 0.69, 1.0))
        self.block_program['u_water_pass'] = False
        self.block_program['u_water_alpha'] = getattr(config, 'WATER_ALPHA', 0.8)
        self.block_program['u_ambient_light'] = getattr(config, 'AMBIENT_LIGHT', 0.0)
        self.block_program['u_sky_intensity'] = getattr(config, 'SKY_INTENSITY', 1.0)
        if config.DEBUG_SINGLE_BLOCK:
            self.block_program['u_fog_start'] = 1e6
            self.block_program['u_fog_end'] = 2e6
        else:
            self.block_program['u_fog_start'] = 0.75 * DIST
            self.block_program['u_fog_end'] = DIST

        self.day_night_enabled = getattr(config, "DAY_NIGHT_CYCLE_ENABLED", False)
        self.day_length = float(getattr(config, "DAY_LENGTH_SECONDS", 1200.0))
        if self.day_length <= 1e-6:
            self.day_length = 1200.0
        self.day_phase = float(getattr(config, "DAY_START_PHASE", 0.0)) % 1.0
        sun_dir = getattr(config, "SUN_LIGHT_DIR", (0.35, 1.0, 0.65))
        self._sun_dir_xz = self._normalize_xz_dir(sun_dir)
        self._day_light_dir = self._normalize_light_dir(sun_dir[0], sun_dir[1], sun_dir[2])
        self._day_ambient = getattr(config, "AMBIENT_LIGHT", 0.0)
        self._day_sky_intensity = getattr(config, "SKY_INTENSITY", 1.0)
        self._day_fog_color = getattr(config, "DAY_FOG_COLOR", (0.5, 0.69, 1.0))
        if self.day_night_enabled:
            self._update_day_night(0.0)

        # Instance of the model that handles the world.
        self.model = world.ModelProxy(self.block_program)

        # Entity rendering setup
        saved_player_state = world_entity_store.load_entity_state("player")
        has_saved_player = saved_player_state is not None
        if saved_player_state:
            self.position = tuple(saved_player_state.get("pos", self.position))
            self.rotation = tuple(saved_player_state.get("rot", self.rotation))
            self.flying = saved_player_state.get("flying", self.flying)
        self.player_entity = Player(self.model, HUMANOID_MODEL, position=self.position)
        self.player_entity.id = 0
        self.player_entity.position = np.array(self.position, dtype=float)
        self.player_entity.rotation = np.array(self.rotation, dtype=float)
        if saved_player_state:
            self.player_entity.on_ground = saved_player_state.get("on_ground", False)
            if self.player_entity.on_ground:
                self.player_entity.velocity[1] = 0.0
        if not has_saved_player:
            # New player spawns should snap to ground; saved positions are restored verbatim.
            grounded_pos, _ = self.model.collide(
                tuple(self.player_entity.position), self.player_entity.bounding_box
            )
            self.player_entity.position = np.array(grounded_pos, dtype=float)
        self.position = tuple(self.player_entity.position)

        self.entity_renderers = {
            'player': renderer.AnimatedEntityRenderer(self.block_program, HUMANOID_MODEL),
            'snake': renderer.SnakeRenderer(self.block_program, SNAKE_MODEL),
            'snail': renderer.AnimatedEntityRenderer(self.block_program, SNAIL_MODEL),
            'seagull': renderer.AnimatedEntityRenderer(self.block_program, SEAGULL_MODEL),
            'dog': renderer.AnimatedEntityRenderer(self.block_program, DOG_MODEL),
        }
        self.entity_objects = {
            self.player_entity.id: self.player_entity,
        }
        self._next_entity_id = 1
        self._sector_entity_state = {}
        self._entity_sector_map = {}
        self._last_loaded_sectors = set()
        self._biome_generator = None
        self._entity_spawn_radius = int(getattr(config, "ENTITY_SPAWN_RADIUS", 3))
        self._entity_max = int(getattr(config, "ENTITY_MAX", 6))
        ready_radius = getattr(config, 'READY_SECTOR_RADIUS', None)
        if ready_radius is None:
            ready_fraction = getattr(config, 'READY_SECTOR_FRACTION', 1.0)
            self.ready_sector_radius = max(1, int(round(config.LOADED_SECTORS * ready_fraction)))
        else:
            self.ready_sector_radius = int(ready_radius)
        self.snake_enabled = True
        self.snail_enabled = True
        self.seagull_enabled = True
        self.dog_enabled = True
        self.entities = {
            eid: entity.to_network_dict()
            for eid, entity in self.entity_objects.items()
            if self._entity_is_enabled(entity)
        }
        self._entity_persist_interval = 5.0
        self._entity_persist_timer = self._entity_persist_interval
        self._persist_entity_states()
        self.frame_id = 0


        # Texture atlas for UI previews.
        self.texture_atlas = image.load(TEXTURE_PATH)

        # The label that is displayed in the top left of the canvas.
        self.label = pyglet.text.Label('', font_name='Consolas', font_size=18,
            x=10, y=self.height - 10, anchor_x='left', anchor_y='top',
            color=(0, 0, 0, 255))
        self.hud_info_label = pyglet.text.Label('', font_name='Consolas', font_size=14,
            x=10, y=self.height - 10 - self.label.content_height - 4,
            anchor_x='left', anchor_y='top',
            color=(0, 0, 0, 255), multiline=True, width=self.width - 20)
        self.sector_label = pyglet.text.Label('', font_name='Consolas', font_size=14,
            x=10, y=self.height - 10 - self.label.content_height - self.hud_info_label.content_height - 8,
            anchor_x='left', anchor_y='top',
            color=(0, 0, 0, 255), multiline=True, width=self.width - 20)
        self.sector_debug_label = pyglet.text.Label('', font_name='Consolas', font_size=14,
            x=10, y=self.height - 10 - self.label.content_height - self.hud_info_label.content_height - self.sector_label.content_height - 12,
            anchor_x='left', anchor_y='top',
            color=(0, 0, 0, 255), multiline=True, width=self.width - 20)
        self.entity_label = pyglet.text.Label('', font_name='Consolas', font_size=14,
            x=10, y=self.height - 10 - self.label.content_height - self.hud_info_label.content_height - self.sector_label.content_height - self.sector_debug_label.content_height - 16,
            anchor_x='left', anchor_y='top',
            color=(0, 0, 0, 255), multiline=True, width=self.width - 20)
        self.keybind_label = pyglet.text.Label('', font_name='Consolas', font_size=14,
            x=10, y=self.height - 10 - self.label.content_height - self.hud_info_label.content_height - self.sector_label.content_height - self.sector_debug_label.content_height - self.entity_label.content_height - 20,
            anchor_x='left', anchor_y='top',
            color=(0, 0, 0, 255), multiline=True, width=self.width - 20)
        self._label_bg = shapes.Rectangle(0, 0, 1, 1, color=(255, 255, 255))
        self._label_bg.opacity = 80  # more transparent
        self._underwater_overlay = shapes.Rectangle(0, 0, 1, 1, color=config.UNDERWATER_COLOR)
        self.last_draw_ms = 0.0
        self.last_update_ms = 0.0
        self._hud_probe_frame = 0
        self._hud_probe_sector = None
        self._hud_probe_void = 'N/A'
        self._hud_probe_mush = 'NA'
        self.hud_visible = True
        self.hud_details_visible = False
        self.vsync_enabled = False
        self._hud_stats_start = time.perf_counter()
        self._hud_stats_frames = 0
        self._hud_stats_dt_sum = 0.0
        self._hud_stats_dt_min = None
        self._hud_stats_dt_max = None
        self._hud_stats_draw_sum = 0.0
        self._hud_stats_draw_min = None
        self._hud_stats_draw_max = None
        self._hud_stats_update_sum = 0.0
        self._hud_stats_update_min = None
        self._hud_stats_update_max = None
        self._hud_stats_sector_sum = 0.0
        self._hud_stats_sector_min = None
        self._hud_stats_sector_max = None
        self._hud_stats_update_sectors_sum = 0.0
        self._hud_stats_update_sectors_min = None
        self._hud_stats_update_sectors_max = None
        self._hud_stats_mesh_jobs_sum = 0.0
        self._hud_stats_mesh_jobs_min = None
        self._hud_stats_mesh_jobs_max = None
        self._hud_stats_physics_sum = 0.0
        self._hud_stats_physics_min = None
        self._hud_stats_physics_max = None
        self._hud_stats_anim_sum = 0.0
        self._hud_stats_anim_min = None
        self._hud_stats_anim_max = None
        self._hud_stats_world_sum = 0.0
        self._hud_stats_world_min = None
        self._hud_stats_world_max = None
        self._hud_stats_entities_sum = 0.0
        self._hud_stats_entities_min = None
        self._hud_stats_entities_max = None
        self._hud_stats_water_sum = 0.0
        self._hud_stats_water_min = None
        self._hud_stats_water_max = None
        self._hud_stats_overlay_sum = 0.0
        self._hud_stats_overlay_min = None
        self._hud_stats_overlay_max = None
        self._hud_stats_hud_sum = 0.0
        self._hud_stats_hud_min = None
        self._hud_stats_hud_max = None
        self._hud_stats_upload_sum = 0.0
        self._hud_stats_upload_min = None
        self._hud_stats_upload_max = None
        self._hud_stats_slow_count = 0
        self._hud_stats_slow_max = 0.0
        self._hud_stats_detail_text = ""
        self._hud_stats_draw_detail_text = ""
        self._last_sector_ms = 0.0
        self._last_update_sectors_ms = 0.0
        self._last_mesh_jobs_ms = 0.0
        self._last_physics_ms = 0.0
        self._last_anim_ms = 0.0
        self._last_update_total_ms = 0.0
        self._hud_stats_text = ""
        self._hud_block1_text = ""
        self._hud_block2_text = ""
        self._hud_block3_text = ""
        self._hud_block4_text = ""
        self._hud_block5_text = ""
        self._hud_block_last_update = 0.0
        self._hud_detail_last_update = 0.0
        self._hud_detail_dirty = False
        self._hud_last_loader_count = 0
        self._hud_last_mesh_done = 0
        self._hud_upload_budget_ms = 0.0
        self._hud_upload_skipped = 0
        self._hud_last_upload_skipped = 0
        self._hud_tri_budget = 0
        self._hud_tri_uploaded = 0
        self._hud_upload_pending = 0
        self._hud_layout_dirty = True
        self._last_hud_ms = 0.0
        self._hud_profile_last_log = 0.0
        self._hud_profile_accum = 0.0
        self._hud_profile_count = 0
        self._hud_profile_max = 0.0
        self._hud_profile_segments = {}

        # Target frame pacing and local FPS tracking (not dependent on pyglet internals).
        desired_fps = getattr(config, 'TARGET_FPS', None)
        self.target_fps = desired_fps or self._detect_refresh_rate() or 60
        self._frame_times = deque(maxlen=120)
        self._last_frame_time = time.perf_counter()
        # Use pyglet's clock-based limiter; avoid double-limiting with manual sleeps.
        try:
            pyglet.clock.set_fps_limit(self.target_fps)
        except Exception:
            pass

        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)

        # This call schedules the `update()` method to be called
        # TICKS_PER_SEC. This is the main game event loop.
        pyglet.clock.schedule_interval(self.update, 1.0 / TICKS_PER_SEC)

    def set_exclusive_mouse(self, exclusive):
        """ If `exclusive` is True, the game will capture the mouse, if False
        the game will ignore the mouse.

        """
        super(Window, self).set_exclusive_mouse(exclusive)
        self.exclusive = exclusive

    def get_sight_vector(self):
        """ Returns the current line of sight vector indicating the direction
        the player is looking.

        """
        x, y = self.rotation
        # y ranges from -90 to 90, or -pi/2 to pi/2, so m ranges from 0 to 1 and
        # is 1 when looking ahead parallel to the ground and 0 when looking
        # straight up or down.
        m = math.cos(math.radians(y))
        # dy ranges from -1 to 1 and is -1 when looking straight down and 1 when
        # looking straight up.
        dy = math.sin(math.radians(y))
        dx = math.cos(math.radians(x - 90)) * m
        dz = math.sin(math.radians(x - 90)) * m
        return (dx, dy, dz)

    def _hud_text_for_clipboard(self):
        lines = [
            self.label.text,
            self.hud_info_label.text,
            self.sector_label.text,
            self.sector_debug_label.text,
            self.entity_label.text,
            self.keybind_label.text,
        ]
        return "\n".join(line for line in lines if line)

    def _copy_hud_to_clipboard(self):
        """Copy HUD text without external dependencies."""
        text = self._hud_text_for_clipboard()
        if not text:
            return False
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            root.clipboard_clear()
            root.clipboard_append(text)
            root.update()
            root.destroy()
            return True
        except Exception:
            return False

    def _yaw_lerp(self, start, end, t):
        delta = ((end - start + 180.0) % 360.0) - 180.0
        return start + delta * t

    def _normalize_light_dir(self, x, y, z):
        mag = math.sqrt(x * x + y * y + z * z)
        if mag <= 1e-6:
            return (0.0, 1.0, 0.0)
        return (x / mag, y / mag, z / mag)

    def _normalize_xz_dir(self, sun_dir):
        x = float(sun_dir[0])
        z = float(sun_dir[2])
        mag = math.sqrt(x * x + z * z)
        if mag <= 1e-6:
            return (0.35, 0.65)
        return (x / mag, z / mag)

    def _update_day_night(self, dt):
        if not self.day_night_enabled:
            return
        if self.day_length > 1e-6:
            self.day_phase = (self.day_phase + dt / self.day_length) % 1.0
        angle = self.day_phase * 2.0 * math.pi
        sun_elev = math.cos(angle)
        day_factor = 0.5 + 0.5 * sun_elev
        curve = getattr(config, "DAY_LIGHT_CURVE", 1.0)
        if curve != 1.0:
            day_factor = day_factor ** curve

        day_ambient = getattr(config, "DAY_AMBIENT_LIGHT", getattr(config, "AMBIENT_LIGHT", 0.1))
        night_ambient = getattr(config, "NIGHT_AMBIENT_LIGHT", day_ambient)
        self._day_ambient = night_ambient + (day_ambient - night_ambient) * day_factor

        day_sky = getattr(config, "DAY_SKY_INTENSITY", getattr(config, "SKY_INTENSITY", 1.0))
        night_sky = getattr(config, "NIGHT_SKY_INTENSITY", day_sky)
        self._day_sky_intensity = night_sky + (day_sky - night_sky) * day_factor

        day_fog = getattr(config, "DAY_FOG_COLOR", (0.5, 0.69, 1.0))
        night_fog = getattr(config, "NIGHT_FOG_COLOR", day_fog)
        self._day_fog_color = (
            night_fog[0] + (day_fog[0] - night_fog[0]) * day_factor,
            night_fog[1] + (day_fog[1] - night_fog[1]) * day_factor,
            night_fog[2] + (day_fog[2] - night_fog[2]) * day_factor,
        )

        xz = self._sun_dir_xz
        self._day_light_dir = self._normalize_light_dir(xz[0], sun_elev, xz[1])

    def _player_back_orient(self):
        """Return orientation for wall-attached blocks based on player facing."""
        dx, _, dz = self.get_sight_vector()
        if abs(dx) > abs(dz):
            return ORIENT_WEST if dx > 0 else ORIENT_EAST
        return ORIENT_NORTH if dz > 0 else ORIENT_SOUTH

    def _face_orient(self, face):
        """Return orientation from a hit face (block - empty)."""
        dx, dy, dz = face
        if dy != 0:
            return None
        if dx == 1:
            return ORIENT_EAST
        if dx == -1:
            return ORIENT_WEST
        if dz == 1:
            return ORIENT_SOUTH
        if dz == -1:
            return ORIENT_NORTH
        return None

    def _hit_face_point(self, origin, vector, block, face):
        """Return the hit point on the block face, or None if it can't be computed."""
        ox, oy, oz = origin
        vx, vy, vz = vector
        dx, dy, dz = face
        if dx != 0:
            plane = block[0] - 0.5 * dx
            if abs(vx) < 1e-6:
                return None
            t = (plane - ox) / vx
        elif dy != 0:
            plane = block[1] + (0.0 if dy > 0 else 1.0)
            if abs(vy) < 1e-6:
                return None
            t = (plane - oy) / vy
        else:
            plane = block[2] - 0.5 * dz
            if abs(vz) < 1e-6:
                return None
            t = (plane - oz) / vz
        if t < 0:
            return None
        return (ox + vx * t, oy + vy * t, oz + vz * t)

    def get_motion_vector(self):
        """ Returns the current motion vector indicating the velocity of the
        player.

        Returns
        -------
        vector : tuple of len 3
            Tuple containing the velocity in x, y, and z respectively.

        """
        if any(self.strafe):
            x, y = self.rotation
            strafe = math.degrees(math.atan2(*self.strafe))
            y_angle = math.radians(y)*any(self.strafe)
            x_angle = math.radians(x + strafe)*any(self.strafe)
            if self.flying:
                m = math.cos(y_angle)
                dy = math.sin(y_angle)
                if self.strafe[1]:
                    # Moving left or right.
                    dy = 0.0
                    m = 1
                if self.strafe[0] > 0:
                    # Moving backwards.
                    dy *= -1
                # When you are flying up or down, you have less left and right
                # motion.
                dx = math.cos(x_angle) * m
                dz = math.sin(x_angle) * m
            else:
                dy = 0.0
                dx = math.cos(x_angle)
                dz = math.sin(x_angle)
        else:
            dy = 0.0
            dx = 0.0
            dz = 0.0
        if self.flying and self.fly_climb!=0:
            dy = self.fly_climb
        return (dx, dy, dz)

    def update(self, dt):
        """ This method is scheduled to be called repeatedly by the pyglet
        clock.

        Parameters
        ----------
        dt : float
            The change in time since the last call.

        """
        update_start = time.perf_counter()
        self._update_day_night(dt)
        sector = util.sectorize(self.position)
        t0 = time.perf_counter()
        frustum_circle = None
        update_sectors_ms = 0.0
        mesh_jobs_ms = 0.0
        if self.model.loader is not None:
            look_vec = self.get_sight_vector()
            frustum_circle = self.get_frustum_circle()
            t1 = time.perf_counter()
            self.model.update_sectors(
                self.sector,
                sector,
                self.position,
                look_vec,
                frustum_circle=frustum_circle,
                allow_send=True,
            )
            update_sectors_ms = (time.perf_counter() - t1) * 1000.0
            self.sector = sector
        self._sync_sector_entities(sector)
        self.model.mesh_budget_deadline = None
        t2 = time.perf_counter()
        self.model.process_pending_mesh_jobs(frustum_circle=frustum_circle, allow_submit=True)
        mesh_jobs_ms = (time.perf_counter() - t2) * 1000.0
        sector_ms = (time.perf_counter() - t0) * 1000.0
        m = getattr(config, 'PHYSICS_SUBSTEPS_MAX', 2)
        m = max(1, int(m))
        dt = min(dt, 0.2)
        entity_ms_total = 0.0
        entity_updates_total = 0
        substeps = 0
        t0 = time.perf_counter()
        for _ in range(m):
            step_ms, step_updates = self._update(dt / m)
            entity_ms_total += step_ms
            entity_updates_total += step_updates
            substeps += 1
        physics_ms = (time.perf_counter() - t0) * 1000.0
        enabled_entities = sum(
            1 for entity in self.entity_objects.values() if self._entity_is_enabled(entity)
        )
        
        # Update entity animations
        t0 = time.perf_counter()
        for entity_renderer in self.entity_renderers.values():
            entity_renderer.update(dt)
        anim_ms = (time.perf_counter() - t0) * 1000.0
        total_ms = (time.perf_counter() - update_start) * 1000.0
        self._last_sector_ms = sector_ms
        self._last_update_sectors_ms = update_sectors_ms
        self._last_mesh_jobs_ms = mesh_jobs_ms
        self._last_physics_ms = physics_ms
        self._last_anim_ms = anim_ms
        self._last_update_total_ms = total_ms
        logutil.log(
            "MAINLOOP",
            f"update sector_ms={sector_ms:.2f} update_sectors_ms={update_sectors_ms:.2f} "
            f"mesh_jobs_ms={mesh_jobs_ms:.2f} physics_ms={physics_ms:.2f} anim_ms={anim_ms:.2f} "
            f"total_ms={total_ms:.2f}",
        )
        slow_thresh = getattr(config, "UPDATE_SLOW_LOG_MS", None)
        if slow_thresh is not None and total_ms >= slow_thresh:
            self._hud_stats_slow_count += 1
            if total_ms > self._hud_stats_slow_max:
                self._hud_stats_slow_max = total_ms
            logutil.log(
                "MAINLOOP",
                f"update_slow total_ms={total_ms:.2f} sector_ms={sector_ms:.2f} "
                f"update_sectors_ms={update_sectors_ms:.2f} mesh_jobs_ms={mesh_jobs_ms:.2f} "
                f"physics_ms={physics_ms:.2f} anim_ms={anim_ms:.2f} "
                f"entities={enabled_entities} updates={entity_updates_total} substeps={substeps}",
            )
        logutil.log(
            "MAINLOOP",
            f"entities enabled={enabled_entities} updates={entity_updates_total} substeps={substeps} ms={entity_ms_total:.2f}",
        )
        self.last_update_ms = total_ms

    def _ensure_biome_generator(self):
        if self._biome_generator is not None:
            return self._biome_generator
        seed = getattr(self.model, "world_seed", None)
        if seed is None:
            return None
        mapgen.initialize_biome_map_generator(seed=seed)
        self._biome_generator = mapgen.biome_generator
        return self._biome_generator

    def _entity_type_enabled(self, entity_type):
        if entity_type == "snake":
            return self.snake_enabled
        if entity_type == "snail":
            return self.snail_enabled
        if entity_type == "seagull":
            return self.seagull_enabled
        if entity_type == "dog":
            return self.dog_enabled
        return True

    def _sector_distance(self, a, b):
        dx = abs((a[0] - b[0]) // config.SECTOR_SIZE)
        dz = abs((a[2] - b[2]) // config.SECTOR_SIZE)
        return max(dx, dz)

    def _sector_in_range(self, ref_sector, sector_pos, radius):
        if ref_sector is None:
            return False
        return self._sector_distance(ref_sector, sector_pos) <= radius

    def _spawn_entity_for_plan(self, sector_pos, plan):
        if plan is None:
            return None
        sector = self.model.sectors.get(sector_pos)
        if sector is None:
            return None
        entity_type = plan.get("type")
        local_pos = plan.get("local_pos")
        if entity_type is None or local_pos is None:
            return None
        local_x = local_pos[0]
        local_y = None
        local_z = local_pos[1]
        if len(local_pos) >= 3:
            local_y = local_pos[1]
            local_z = local_pos[2]
        spawn_pos = None
        if entity_type == "seagull":
            spawn_pos = self._find_water_spawn(sector, local_x, local_z)
        elif entity_type == "snail":
            spawn_pos = self._find_cave_spawn(sector, local_x, local_z)
        else:
            if local_y is not None:
                spawn_pos = self._spawn_from_surface_hint(sector, local_x, local_y, local_z)
            if spawn_pos is None:
                spawn_pos = self._find_surface_spawn(sector, local_x, local_z)
        if spawn_pos is None:
            return None

        entity_id = self._next_entity_id
        self._next_entity_id += 1
        if entity_type == "snake":
            entity = SnakeEntity(self.model, player_position=spawn_pos, entity_id=entity_id)
        elif entity_type == "snail":
            entity = SnailEntity(
                self.model,
                player_position=spawn_pos,
                entity_id=entity_id,
                saved_state={"pos": spawn_pos, "rot": (0.0, 0.0)},
            )
        elif entity_type == "seagull":
            entity = SeagullEntity(
                self.model,
                player_position=spawn_pos,
                entity_id=entity_id,
                saved_state={"pos": spawn_pos, "rot": (0.0, 0.0)},
            )
        elif entity_type == "dog":
            entity = Dog(self.model, entity_id=entity_id, saved_state={"pos": spawn_pos, "rot": (0.0, 0.0)})
            entity.snap_to_ground()
        else:
            return None
        self.entity_objects[entity_id] = entity
        return entity

    def _spawn_from_surface_hint(self, sector, local_x, local_y, local_z):
        if (
            local_x < 0
            or local_x >= config.SECTOR_SIZE
            or local_z < 0
            or local_z >= config.SECTOR_SIZE
        ):
            return None
        column = sector.blocks[local_x, :, local_z]
        if local_y <= 0 or local_y >= column.shape[0]:
            return None
        top = local_y - 1
        if not BLOCK_COLLIDES[column[top]] or column[top] == WATER:
            return None
        if local_y < column.shape[0] and BLOCK_COLLIDES[column[local_y]]:
            return None
        wx = sector.position[0] + local_x
        wz = sector.position[2] + local_z
        return (float(wx), float(local_y), float(wz))

    def _find_surface_spawn(self, sector, local_x, local_z):
        offsets = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, 1), (1, -1), (-1, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2),
        ]
        for dx, dz in offsets:
            lx = local_x + dx
            lz = local_z + dz
            if lx < 0 or lx >= config.SECTOR_SIZE or lz < 0 or lz >= config.SECTOR_SIZE:
                continue
            column = sector.blocks[lx, :, lz]
            collides = BLOCK_COLLIDES[column]
            if not collides.any():
                continue
            top = int(np.nonzero(collides)[0][-1])
            if column[top] == WATER:
                continue
            if top + 1 < column.shape[0] and BLOCK_COLLIDES[column[top + 1]]:
                continue
            wx = sector.position[0] + lx
            wz = sector.position[2] + lz
            return (float(wx), float(top + 1), float(wz))
        return None

    def _find_water_spawn(self, sector, local_x, local_z):
        offsets = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, 1), (1, -1), (-1, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2),
        ]
        for dx, dz in offsets:
            lx = local_x + dx
            lz = local_z + dz
            if lx < 0 or lx >= config.SECTOR_SIZE or lz < 0 or lz >= config.SECTOR_SIZE:
                continue
            column = sector.blocks[lx, :, lz]
            water = column == WATER
            if not water.any():
                continue
            top = int(np.nonzero(water)[0][-1])
            wx = sector.position[0] + lx
            wz = sector.position[2] + lz
            y = float(top + 1 + getattr(SeagullEntity, "ALTITUDE_OFFSET", 12.0))
            return (float(wx), y, float(wz))
        return None

    def _find_cave_spawn(self, sector, local_x, local_z):
        offsets = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, 1), (1, -1), (-1, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2),
            (3, 0), (-3, 0), (0, 3), (0, -3),
        ]
        for dx, dz in offsets:
            lx = local_x + dx
            lz = local_z + dz
            if lx < 0 or lx >= config.SECTOR_SIZE or lz < 0 or lz >= config.SECTOR_SIZE:
                continue
            column = sector.blocks[lx, :, lz]
            collides = BLOCK_COLLIDES[column]
            if not collides.any():
                continue
            top = int(np.nonzero(collides)[0][-1])
            if top <= 4:
                continue
            for y in range(2, top - 2):
                if column[y] != 0:
                    continue
                if not BLOCK_COLLIDES[column[y - 1]] or column[y - 1] == WATER:
                    continue
                if column[y + 1] != 0:
                    continue
                ceiling_band = column[y + 2:min(top, y + 12)]
                if ceiling_band.size == 0 or not BLOCK_COLLIDES[ceiling_band].any():
                    continue
                wx = sector.position[0] + lx
                wz = sector.position[2] + lz
                return (float(wx), float(y), float(wz))
        return None

    def _despawn_entity(self, entity_id, reset_sector_spawn=False):
        entity = self.entity_objects.pop(entity_id, None)
        if entity is None:
            return
        sector_pos = self._entity_sector_map.pop(entity_id, None)
        if sector_pos is None:
            return
        state = self._sector_entity_state.get(sector_pos)
        if state is None:
            return
        state["entity_id"] = None
        if reset_sector_spawn:
            state["spawned"] = False

    def _enforce_entity_cap(self):
        max_entities = int(getattr(config, "ENTITY_MAX", self._entity_max))
        extras = [
            entity for entity in self.entity_objects.values()
            if entity is not self.player_entity
        ]
        if max_entities <= 0:
            for entity in extras:
                self._despawn_entity(entity.id, reset_sector_spawn=False)
            return
        if len(extras) <= max_entities:
            return
        px, py, pz = self.player_entity.position
        extras.sort(
            key=lambda ent: (ent.position[0] - px) ** 2
            + (ent.position[1] - py) ** 2
            + (ent.position[2] - pz) ** 2,
            reverse=True,
        )
        remove_count = len(extras) - max_entities
        for entity in extras[:remove_count]:
            self._despawn_entity(entity.id, reset_sector_spawn=False)

    def _sync_sector_entities(self, player_sector):
        generator = self._ensure_biome_generator()
        if generator is None:
            return
        loaded_sectors = set(self.model.sectors.keys())
        if self._last_loaded_sectors:
            for dropped in self._last_loaded_sectors - loaded_sectors:
                state = self._sector_entity_state.get(dropped)
                if state and state.get("entity_id") is not None:
                    self._despawn_entity(state["entity_id"], reset_sector_spawn=True)
                elif state:
                    state["spawned"] = False
        self._last_loaded_sectors = loaded_sectors
        if player_sector is None:
            return

        for sector_pos in loaded_sectors:
            if not self._sector_in_range(player_sector, sector_pos, self._entity_spawn_radius):
                continue
            state = self._sector_entity_state.get(sector_pos)
            if state is None:
                plan = generator.sector_entity_plan(sector_pos)
                state = {"plan": plan, "spawned": False, "entity_id": None}
                self._sector_entity_state[sector_pos] = state
            plan = state.get("plan")
            if plan is None or state.get("spawned"):
                continue
            if not self._entity_type_enabled(plan.get("type")):
                continue
            entity = self._spawn_entity_for_plan(sector_pos, plan)
            if entity is None:
                continue
            state["spawned"] = True
            state["entity_id"] = entity.id
            self._entity_sector_map[entity.id] = sector_pos

        self._enforce_entity_cap()


    def _update(self, dt):
        """ Private implementation of the `update()` method. This is where most
        of the motion logic lives, along with gravity and collision detection.

        Parameters
        ----------
        dt : float
            The change in time since the last call.

        """
        motion_vector = self.get_motion_vector()
        ladder_aligning = (
            getattr(self.player_entity, "_ladder_mount_timer", 0.0) > 1e-6
            or getattr(self.player_entity, "_ladder_dismount_yaw_timer", 0.0) > 1e-6
        )
        if ladder_aligning:
            motion_vector = (0.0, 0.0, 0.0)
        dx, _, dz = motion_vector
        is_moving = abs(dx) > 1e-6 or abs(dz) > 1e-6
        target_animation = 'walk' if is_moving else 'idle'
        self.player_entity.current_animation = target_animation

        if not self.model.is_sector_ready(self.player_entity.position, radius=self.ready_sector_radius):
            return 0.0, 0

        camera_rot = (-self.rotation[0], self.rotation[1])
        context = {
            "motion_vector": motion_vector,
            "flying": self.flying,
            "world_model": self.model,
            "camera_rotation": camera_rot,
            "apply_camera_rotation": is_moving and not ladder_aligning,
            "player_position": self.player_entity.position.copy(),
            "camera_mode": self.camera_mode,
            "strafe": (0, 0) if ladder_aligning else tuple(self.strafe),
            "jump": False if ladder_aligning else bool(self.keys[key.SPACE]),
        }
        t0 = time.perf_counter()
        update_count = 0
        updated_entities = {}

        for entity in self.entity_objects.values():
            if not self._entity_is_enabled(entity):
                continue
            if not self.model.is_sector_ready(entity.position, radius=0):
                continue
            entity.update(dt, context)
            update_count += 1
            
            # Update the renderer's current animation based on the entity's state
            entity_renderer = self.entity_renderers.get(entity.type)
            if entity_renderer:
                if isinstance(entity_renderer, renderer.SnakeRenderer):
                    # SnakeRenderer has its own update for segments, and AnimatedEntityRenderer for the head.
                    # The head's animation is set here.
                    entity_renderer.head_renderer.set_animation(entity.current_animation)
                else:
                    entity_renderer.set_animation(entity.current_animation)

            updated_entities[entity.id] = entity.to_network_dict()
            if entity is self.player_entity:
                context["player_position"] = entity.position.copy()

        self.entities = updated_entities
        entity_ms = (time.perf_counter() - t0) * 1000.0
        self.position = tuple(self.player_entity.position)
        if getattr(self.player_entity, "camera_yaw_follow", False):
            target = getattr(self.player_entity, "camera_yaw_target", None)
            duration = getattr(self.player_entity, "camera_yaw_duration", 0.0)
            if target is not None and duration > 1e-6:
                if self._camera_yaw_active_target is None or abs(target - self._camera_yaw_active_target) > 1e-6:
                    self._camera_yaw_active_target = target
                    self._camera_yaw_start = float(self.rotation[0])
                    self._camera_yaw_elapsed = 0.0
                self._camera_yaw_elapsed = min(duration, self._camera_yaw_elapsed + dt)
                t = min(1.0, self._camera_yaw_elapsed / duration)
                self.rotation = (self._yaw_lerp(self._camera_yaw_start, target, t), self.rotation[1])
        else:
            self._camera_yaw_active_target = None
            self._camera_yaw_elapsed = 0.0
        if getattr(self.player_entity, "camera_pitch_follow", False):
            target = getattr(self.player_entity, "camera_pitch_target", None)
            duration = getattr(self.player_entity, "camera_pitch_duration", 0.0)
            if target is not None and duration > 1e-6:
                if self._camera_pitch_active_target is None or abs(target - self._camera_pitch_active_target) > 1e-6:
                    self._camera_pitch_active_target = target
                    self._camera_pitch_start = float(self.rotation[1])
                    self._camera_pitch_elapsed = 0.0
                self._camera_pitch_elapsed = min(duration, self._camera_pitch_elapsed + dt)
                t = min(1.0, self._camera_pitch_elapsed / duration)
                pitch = self._camera_pitch_start + (target - self._camera_pitch_start) * t
                self.rotation = (self.rotation[0], pitch)
        else:
            self._camera_pitch_active_target = None
            self._camera_pitch_elapsed = 0.0
        self._entity_persist_timer -= dt
        if self._entity_persist_timer <= 0:
            self._persist_entity_states()
            self._entity_persist_timer = self._entity_persist_interval
        return entity_ms, update_count

    def _entity_is_enabled(self, entity):
        if entity.type == "snake":
            return self.snake_enabled
        if entity.type == "snail":
            return self.snail_enabled
        if entity.type == "seagull":
            return self.seagull_enabled
        if entity.type == "dog":
            return self.dog_enabled
        return True

    def _persist_entity_states(self):
        state = self.player_entity.serialize_state()
        if state is not None:
            world_entity_store.save_entity_state(self.player_entity.type, state)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        ctrl = self.keys[key.LCTRL] or self.keys[key.RCTRL]

        # Ctrl + wheel => zoom (only in third-person)
        if ctrl and self.camera_mode == 'third_person':
            self.third_person_distance -= scroll_y
            self.third_person_distance = max(2, min(self.third_person_distance, 20))
            return

        # Plain wheel => inventory cycle (in BOTH camera modes)
        ind = self.inventory.index(self.block)

        step = int(scroll_y)
        if step == 0:
            step = 1 if scroll_y > 0 else -1

        ind = (ind + step) % len(self.inventory)
        self.block = self.inventory[ind]
        self.update_inventory_item_batch()

    def on_mouse_press(self, x, y, button, modifiers):
        """ Called when a mouse button is pressed. See pyglet docs for button
        amd modifier mappings.

        Parameters
        ----------
        x, y : int
            The coordinates of the mouse click. Always center of the screen if
            the mouse is captured.
        button : int
            Number representing mouse button that was clicked. 1 = left button,
            4 = right button.
        modifiers : int
            Number representing any modifying keys that were pressed when the
            mouse button was clicked.

        """
        if self.exclusive:
            vector = self.get_sight_vector()
            _, _, eye = self.get_view_projection()  # eye is Vec3
            hit_origin = (eye.x, eye.y, eye.z)
            block, previous = self.model.hit_test(hit_origin, vector)
            if (button == mouse.RIGHT) or \
                    ((button == mouse.LEFT) and (modifiers & key.MOD_CTRL)):
                # ON OSX, control + left click = right click.
                if block:
                    block_id = self.model[block]
                    if block_id in DOOR_LOWER_IDS or block_id in DOOR_UPPER_IDS:
                        if block_id in DOOR_UPPER_IDS:
                            lower_pos = (block[0], block[1] - 1, block[2])
                            lower_id = self.model[lower_pos]
                        else:
                            lower_pos = block
                            lower_id = block_id
                        if lower_id in DOOR_LOWER_TOGGLE:
                            upper_pos = (lower_pos[0], lower_pos[1] + 1, lower_pos[2])
                            new_lower = DOOR_LOWER_TOGGLE[lower_id]
                            new_upper = DOOR_LOWER_TO_UPPER[new_lower]
                            updates = [(lower_pos, new_lower)]
                            if self.model[upper_pos] in DOOR_UPPER_TO_LOWER:
                                updates.append((upper_pos, new_upper))
                            self.model.add_blocks(updates, priority=True)
                            return
                if previous:
                    px, py, pz = util.normalize(self.position)
                    if not (previous == (px, py, pz) or previous == (px, py-1, pz)):
                        block_id = BLOCK_ID[self.block]
                        if block_id in STAIR_BASE_IDS:
                            orient = (self._player_back_orient() + 2) % 4
                            upside = False
                            if block is not None:
                                face = (block[0] - previous[0], block[1] - previous[1], block[2] - previous[2])
                                if face[1] > 0:
                                    upside = True
                                elif face[1] == 0:
                                    hit = self._hit_face_point(hit_origin, vector, block, face)
                                    if hit is not None and (hit[1] - block[1]) > 0.5:
                                        upside = True
                            if upside:
                                block_id = STAIR_ORIENTED_UP_IDS[block_id][orient]
                            else:
                                block_id = STAIR_ORIENTED_IDS[block_id][orient]
                        else:
                            oriented = ORIENTED_BLOCK_IDS.get(block_id)
                            if oriented is not None:
                                if block_id in WALL_MOUNTED_BLOCK_IDS:
                                    if block is None:
                                        return
                                    face = (block[0] - previous[0], block[1] - previous[1], block[2] - previous[2])
                                    orient = self._face_orient(face)
                                    if orient is None:
                                        return
                                else:
                                    orient = self._player_back_orient()
                                block_id = oriented[orient]
                                if block_id in DOOR_LOWER_TO_UPPER:
                                    upper_pos = (previous[0], previous[1] + 1, previous[2])
                                    if self.model[upper_pos] not in (0, None):
                                        return
                                    upper_id = DOOR_LOWER_TO_UPPER[block_id]
                                    self.model.add_blocks(
                                        [(previous, block_id), (upper_pos, upper_id)],
                                        priority=True,
                                    )
                                    return
                        self.model.add_block(previous, block_id, priority=True)
            elif button == pyglet.window.mouse.LEFT and block:
                self.model.remove_block(block, priority=True)
        else:
            self.set_exclusive_mouse(True)

    def on_mouse_motion(self, x, y, dx, dy):
        """ Called when the player moves the mouse.

        Parameters
        ----------
        x, y : int
            The coordinates of the mouse click. Always center of the screen if
            the mouse is captured.
        dx, dy : float
            The movement of the mouse.

        """
        if self.exclusive:
            if (
                getattr(self.player_entity, "_ladder_mount_timer", 0.0) > 1e-6
                or getattr(self.player_entity, "_ladder_dismount_yaw_timer", 0.0) > 1e-6
            ):
                return
            m = 0.15
            x, y = self.rotation  # x = yaw (degrees), y = pitch (degrees)
            x = x + dx * m
            y = max(-90, min(90, y + dy * m))
            self.rotation = (x, y)
            if config.DEBUG_SINGLE_BLOCK:
                logutil.log("MAIN", f"rotation yaw={x:.2f} pitch={y:.2f}", level="DEBUG")

    def on_key_press(self, symbol, modifiers):
        """ Called when the player presses a key. See pyglet docs for key
        mappings.

        Parameters
        ----------
        symbol : int
            Number representing the key that was pressed.
        modifiers : int
            Number representing any modifying keys that were pressed.

        """
        if symbol == key.W:
            self.strafe[0] -= 1
        elif symbol == key.S:
            self.strafe[0] += 1
        elif symbol == key.A:
            self.strafe[1] -= 1
        elif symbol == key.D:
            self.strafe[1] += 1
        elif symbol == key.SPACE:
            if self.flying:
                self.fly_climb += 1
            if self.player_entity.on_ground:
                self.player_entity.velocity[1] = JUMP_SPEED
        elif symbol == key.LSHIFT:
            if self.flying:
                self.fly_climb -= 1
        elif symbol == key.ESCAPE:
            self.set_exclusive_mouse(False)
        elif symbol == key.TAB:
            self.flying = not self.flying
        elif symbol == key.F5:
            if self.camera_mode == 'first_person':
                self.camera_mode = 'third_person'
            else:
                self.camera_mode = 'first_person'
        elif symbol == key.F1:
            self.hud_visible = not self.hud_visible
            self._hud_layout_dirty = True
        elif symbol == key.F2:
            self.vsync_enabled = not self.vsync_enabled
            try:
                self.set_vsync(self.vsync_enabled)
            except Exception:
                pass
        elif symbol == key.F3:
            self.hud_details_visible = not self.hud_details_visible
            self._hud_layout_dirty = True
            self._hud_detail_dirty = True
        elif symbol == key.B:
            self._toggle_snail()
        elif symbol == key.M:
            self._toggle_seagull()
        elif symbol in self.num_keys:
            index = (symbol - self.num_keys[0]) % len(self.inventory)
            self.block = self.inventory[index]
            self.update_inventory_item_batch()
        elif symbol in self.inv_prev_keys:
            ind = self.inventory.index(self.block)
            self.block = self.inventory[(ind - 1) % len(self.inventory)]
            self.update_inventory_item_batch()
        elif symbol in self.inv_next_keys:
            ind = self.inventory.index(self.block)
            self.block = self.inventory[(ind + 1) % len(self.inventory)]
            self.update_inventory_item_batch()
        elif symbol == key.N:
            self._toggle_snake()
        elif symbol == key.V:
            self._toggle_dog()
        elif symbol == key.F8:
            copied = self._copy_hud_to_clipboard()
            if not copied:
                logutil.log("MAIN", "HUD clipboard copy failed", level="WARN")

    def _toggle_snake(self):
        self.snake_enabled = not self.snake_enabled
        status = "enabled" if self.snake_enabled else "disabled"
        logutil.log("MAIN", f"Snake {status}")

    def _toggle_snail(self):
        self.snail_enabled = not self.snail_enabled
        status = "enabled" if self.snail_enabled else "disabled"
        logutil.log("MAIN", f"Snail {status}")

    def _toggle_seagull(self):
        self.seagull_enabled = not self.seagull_enabled
        status = "enabled" if self.seagull_enabled else "disabled"
        logutil.log("MAIN", f"Seagull {status}")

    def _toggle_dog(self):
        self.dog_enabled = not self.dog_enabled
        status = "enabled" if self.dog_enabled else "disabled"
        logutil.log("MAIN", f"Dog {status}")

    def on_key_release(self, symbol, modifiers):
        """ Called when the player releases a key. See pyglet docs for key
        mappings.

        Parameters
        ----------
        symbol : int
            Number representing the key that was pressed.
        modifiers : int
            Number representing any modifying keys that were pressed.

        """
        if symbol == key.W:
            self.strafe[0] += 1
        elif symbol == key.S:
            self.strafe[0] -= 1
        elif symbol == key.A:
            self.strafe[1] += 1
        elif symbol == key.D:
            self.strafe[1] -= 1
        elif symbol == key.SPACE:
            self.fly_climb = 0
        elif symbol == key.LSHIFT:
            self.fly_climb = 0

    def on_resize(self, width, height):
        """ Called when the window is resized to a new `width` and `height`.

        """
        # label
        self.label.y = height - 10
        for label in (
            self.hud_info_label,
            self.sector_label,
            self.sector_debug_label,
            self.entity_label,
            self.keybind_label,
        ):
            label.width = width - 20
        self._hud_layout_dirty = True
        # reticle uses shader-based shapes instead of deprecated vertex_list
        self.reticle_batch = None
        cx, cy = self.width / 2, self.height / 2
        n = 10
        self.reticle = [
            shapes.Line(cx - n, cy, cx + n, cy, thickness=2, color=(255, 255, 255)),
            shapes.Line(cx, cy - n, cx, cy + n, thickness=2, color=(255, 255, 255)),
        ]
        #inventory item
        self.update_inventory_item_batch()

    def update_inventory_item_batch(self):
        if self.inventory_item is not None:
            self.inventory_item.delete()
        # Draw a textured preview of the selected block using the atlas.
        block_id = BLOCK_ID[self.block]
        picker_face = int(BLOCK_PICKER_FACE[block_id])
        t = BLOCK_TEXTURES[block_id][picker_face]  # configured face tex coords
        tex = self.texture_atlas.get_texture()
        x0, y0, x1, y1 = t[0]*tex.width, t[1]*tex.height, t[4]*tex.width, t[5]*tex.height
        region = tex.get_region(x=int(x0), y=int(y0), width=int(x1 - x0), height=int(y1 - y0))
        size = 64
        sprite = pyglet.sprite.Sprite(region, x=16, y=16)
        scale_x = size / sprite.width
        scale_y = size / sprite.height
        sprite.scale = min(scale_x, scale_y)
        # Tint the HUD icon with the block's vertex color (use the displayed face).
        face_colors = BLOCK_COLORS[block_id].reshape(6, 4, 3)
        picker_colors = face_colors[picker_face]
        avg_color = np.rint(picker_colors.mean(axis=0)).astype(int)
        avg_color = tuple(int(max(0, min(255, c))) for c in avg_color)
        sprite.color = avg_color
        self.inventory_item = sprite
        #outline
#        v = size/2+(size/2+0.1)*BLOCK_VERTICES[BLOCK_ID[self.block]] + numpy.tile(numpy.array([16,16+size/2,0]),4)
#        v = numpy.hstack((v[:,:3],v,v[:,-3:]))
#        v = v.ravel()
#        c = 1*numpy.array([1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1]).repeat(6)
#        self.inventory_item_outline = self.inventory_batch.add(len(v)/3, gl.GL_LINE_STRIP, self.inventory_outline_group,
#            ('v3f/static', v),
#            ('c3B/static', c),
#        )

    def on_close(self):
        self._persist_entity_states()
        self.model.quit()
        pyglet.window.Window.on_close(self)

    def set_2d(self):
        """ Configure OpenGL to draw in 2d.

        """
        width, height = self.get_size()
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glViewport(0, 0, width, height)

    def get_view_projection(self):
        width, height = self.get_size()
        aspect = width / float(height)
        projection = Mat4.perspective_projection(aspect, 0.1, 512.0, 65)

        camera_pos = self.player_entity.get_camera_position()
        player_head_pos = Vec3(camera_pos[0], camera_pos[1], camera_pos[2])

        dx, dy, dz = self.get_sight_vector()
        forward = Vec3(dx, dy, dz).normalize()  # :contentReference[oaicite:4]{index=4}

        eye_world = player_head_pos
        if self.camera_mode == 'third_person':
            eye_world = player_head_pos - forward * self.third_person_distance  # :contentReference[oaicite:5]{index=5}

        up = Vec3(0.0, 1.0, 0.0)

        # IMPORTANT: view is defined in camera-relative space (camera at origin)
        view = Mat4.look_at(Vec3(0.0, 0.0, 0.0), forward, up)

        return projection, view, eye_world

    def set_3d(self):
        """ Configure OpenGL to draw in 3d.

        """
        width, height = self.get_size()

        gl.glEnable(gl.GL_DEPTH_TEST)

        gl.glViewport(0, 0, width, height)
        projection, view, eye_pos = self.get_view_projection()
        # pyglet Mat4 supports direct upload; ensure contiguous float32 arrays
        self.model.set_matrices(projection, view, eye_pos)
        if config.DEBUG_SINGLE_BLOCK and not self._printed_mats:
            dx, dy, dz = self.get_sight_vector()
            logutil.log("MAIN", f"projection matrix {np.array(projection)}", level="DEBUG")
            logutil.log("MAIN", f"view matrix {np.array(view)}", level="DEBUG")
            logutil.log("MAIN", f"position {self.position} rotation {self.rotation} sight {(dx, dy, dz)}", level="DEBUG")
            self._printed_mats = True
##        gl.glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, GLfloat4(0.35,1.0,0.65,0.0))
        #gl.glLightfv(gl.GL_LIGHT0,gl.GL_SPECULAR, GLfloat4(1,1,1,1))


    def get_frustum_circle(self):
        x, pitch = self.rotation
        dx = math.cos(math.radians(x - 90))
        dz = math.sin(math.radians(x - 90))

        c = [0,2]
        vec = np.array([dx,dz])
        ovec = np.array([-dz,dx])
        pos = np.array([self.position[0], self.position[2]])
        # Pull the frustum center back when pitched to cover nearby terrain without disabling culling.
        forward_scale = max(0.25, math.cos(math.radians(pitch)))
        center = pos + vec * (DIST/2) * forward_scale
        far_corner = pos + vec*DIST + ovec*DIST*np.tan(65.0/180.0 * np.pi)/2
        rad = ((center-far_corner)**2).sum()**0.5/2
        # Inflate more as pitch increases (looking up/down) to reduce popping while staying bounded.
        tilt = min(1.0, abs(pitch) / 90.0)
        rad *= 1.05 + 0.45 * tilt
        return center, rad

    def on_draw(self):
        """ Called by pyglet to draw the canvas.

        """
        frame_start = time.perf_counter()
        dt = frame_start - self._last_frame_time
        self._last_frame_time = frame_start
        self._frame_times.append(dt)
        self.frame_id += 1
        logutil.set_frame(self.frame_id)
        logutil.log("FRAME", f"start dt_ms={dt*1000.0:.2f}")
        self.model.frame_id = self.frame_id
        # Allow a small slice of the frame for mesh uploads; keep rendering priority.
        frame_budget = 1.0 / self.target_fps if self.target_fps else 1.0 / 60.0
        # upload_budget = 0.3 * frame_budget
        upload_budget = max(0.5/self.target_fps, 0.5 * frame_budget)
        self.clear()
        self.set_3d()

        if self.day_night_enabled:
            self.block_program['u_ambient_light'] = self._day_ambient
            self.block_program['u_sky_intensity'] = self._day_sky_intensity
            self.block_program['u_light_dir'] = self._day_light_dir
            self.block_program['u_fog_color'] = self._day_fog_color
            gl.glClearColor(
                self._day_fog_color[0],
                self._day_fog_color[1],
                self._day_fog_color[2],
                1.0,
            )
        else:
            self.block_program['u_ambient_light'] = getattr(config, 'AMBIENT_LIGHT', 0.0)
            self.block_program['u_sky_intensity'] = getattr(config, 'SKY_INTENSITY', 1.0)
            self.block_program['u_light_dir'] = getattr(config, 'SUN_LIGHT_DIR', (0.35, 1.0, 0.65))
            fog_color = getattr(config, 'DAY_FOG_COLOR', (0.5, 0.69, 1.0))
            self.block_program['u_fog_color'] = fog_color
            gl.glClearColor(
                fog_color[0],
                fog_color[1],
                fog_color[2],
                1.0,
            )
        
        t0 = time.perf_counter()
        # Draw world (opaque pass only so water can overlay entities).
        self.model.draw(
            self.position,
            self.get_frustum_circle(),
            frame_start,
            upload_budget,
        )
        world_ms = (time.perf_counter() - t0) * 1000.0
        
        t0 = time.perf_counter()
        # Draw entities
        self.block_program.bind()
        self.block_program['u_use_texture'] = False
        # self.block_program['u_use_vertex_color'] = False
        for entity_id, entity_state in self.entities.items():
            if entity_state['type'] == 'player' and self.camera_mode == 'first_person':
                continue
            r = self.entity_renderers.get(entity_state['type'])
            if r:
                r.draw(entity_state)
        self.block_program['u_use_texture'] = True
        # self.block_program['u_use_vertex_color'] = True
        entity_draw_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        # Draw water after entities so it tints submerged parts.
        self.model.draw_water_pass()
        self.block_program.unbind()
        water_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        # 2D overlay
        self.set_2d()
        if self._is_underwater():
            self.draw_underwater_overlay()
        hud_start = time.perf_counter()
        self.draw_label()
        self._last_hud_ms = (time.perf_counter() - hud_start) * 1000.0
        if self.camera_mode == 'first_person':
            self.draw_reticle()
        self.draw_inventory_item()
        self.draw_focused_block()
        overlay_ms = (time.perf_counter() - t0) * 1000.0
        
        # Use leftover budget to upload meshes at the end of the frame.
        elapsed = time.perf_counter() - frame_start
        upload_start = time.perf_counter()
        remaining = frame_budget - (time.perf_counter() - frame_start)
        extra_budget = min(upload_budget, remaining) if remaining > 0.0 else 0.0
        min_budget = getattr(config, 'UPLOAD_MIN_BUDGET_MS', 1.0) / 1000.0
        budget = max(extra_budget, min_budget)
        self._hud_upload_budget_ms = budget * 1000.0
        before_pending = len(self.model.pending_uploads)
        tri_budget = getattr(config, 'UPLOAD_TRIANGLE_BUDGET', None)
        uploaded_tris = self.model.process_pending_uploads(upload_start, budget, 0, tri_budget)
        if uploaded_tris == 0 and before_pending > 0:
            self._hud_upload_skipped += 1
        self._hud_tri_budget = tri_budget if tri_budget is not None else 0
        self._hud_tri_uploaded = uploaded_tris
        self._hud_upload_pending = len(self.model.pending_uploads)
        upload_ms = (time.perf_counter() - upload_start) * 1000.0
        logutil.log(
            "MAINLOOP",
            f"draw world_ms={world_ms:.2f} entity_ms={entity_draw_ms:.2f} "
            f"water_ms={water_ms:.2f} overlay_ms={overlay_ms:.2f} upload_ms={upload_ms:.2f}",
        )
        self.last_draw_ms = (time.perf_counter()-frame_start)*1000.0
        dt_ms = dt * 1000.0
        self._hud_stats_frames += 1
        self._hud_stats_dt_sum += dt_ms
        self._hud_stats_draw_sum += self.last_draw_ms
        self._hud_stats_update_sum += self.last_update_ms
        sector_ms = getattr(self, "_last_sector_ms", 0.0)
        update_sectors_ms = getattr(self, "_last_update_sectors_ms", 0.0)
        mesh_jobs_ms = getattr(self, "_last_mesh_jobs_ms", 0.0)
        physics_ms = getattr(self, "_last_physics_ms", 0.0)
        anim_ms = getattr(self, "_last_anim_ms", 0.0)
        update_total_ms = getattr(self, "_last_update_total_ms", 0.0)
        self._hud_stats_sector_sum += sector_ms
        self._hud_stats_update_sectors_sum += update_sectors_ms
        self._hud_stats_mesh_jobs_sum += mesh_jobs_ms
        self._hud_stats_physics_sum += physics_ms
        self._hud_stats_anim_sum += anim_ms
        self._hud_stats_world_sum += world_ms
        self._hud_stats_entities_sum += entity_draw_ms
        self._hud_stats_water_sum += water_ms
        self._hud_stats_overlay_sum += overlay_ms
        self._hud_stats_hud_sum += self._last_hud_ms
        self._hud_stats_upload_sum += upload_ms
        if self._hud_stats_sector_min is None or sector_ms < self._hud_stats_sector_min:
            self._hud_stats_sector_min = sector_ms
        if self._hud_stats_sector_max is None or sector_ms > self._hud_stats_sector_max:
            self._hud_stats_sector_max = sector_ms
        if self._hud_stats_update_sectors_min is None or update_sectors_ms < self._hud_stats_update_sectors_min:
            self._hud_stats_update_sectors_min = update_sectors_ms
        if self._hud_stats_update_sectors_max is None or update_sectors_ms > self._hud_stats_update_sectors_max:
            self._hud_stats_update_sectors_max = update_sectors_ms
        if self._hud_stats_mesh_jobs_min is None or mesh_jobs_ms < self._hud_stats_mesh_jobs_min:
            self._hud_stats_mesh_jobs_min = mesh_jobs_ms
        if self._hud_stats_mesh_jobs_max is None or mesh_jobs_ms > self._hud_stats_mesh_jobs_max:
            self._hud_stats_mesh_jobs_max = mesh_jobs_ms
        if self._hud_stats_physics_min is None or physics_ms < self._hud_stats_physics_min:
            self._hud_stats_physics_min = physics_ms
        if self._hud_stats_physics_max is None or physics_ms > self._hud_stats_physics_max:
            self._hud_stats_physics_max = physics_ms
        if self._hud_stats_anim_min is None or anim_ms < self._hud_stats_anim_min:
            self._hud_stats_anim_min = anim_ms
        if self._hud_stats_anim_max is None or anim_ms > self._hud_stats_anim_max:
            self._hud_stats_anim_max = anim_ms
        if self._hud_stats_world_min is None or world_ms < self._hud_stats_world_min:
            self._hud_stats_world_min = world_ms
        if self._hud_stats_world_max is None or world_ms > self._hud_stats_world_max:
            self._hud_stats_world_max = world_ms
        if self._hud_stats_entities_min is None or entity_draw_ms < self._hud_stats_entities_min:
            self._hud_stats_entities_min = entity_draw_ms
        if self._hud_stats_entities_max is None or entity_draw_ms > self._hud_stats_entities_max:
            self._hud_stats_entities_max = entity_draw_ms
        if self._hud_stats_water_min is None or water_ms < self._hud_stats_water_min:
            self._hud_stats_water_min = water_ms
        if self._hud_stats_water_max is None or water_ms > self._hud_stats_water_max:
            self._hud_stats_water_max = water_ms
        if self._hud_stats_overlay_min is None or overlay_ms < self._hud_stats_overlay_min:
            self._hud_stats_overlay_min = overlay_ms
        if self._hud_stats_overlay_max is None or overlay_ms > self._hud_stats_overlay_max:
            self._hud_stats_overlay_max = overlay_ms
        if self._hud_stats_hud_min is None or self._last_hud_ms < self._hud_stats_hud_min:
            self._hud_stats_hud_min = self._last_hud_ms
        if self._hud_stats_hud_max is None or self._last_hud_ms > self._hud_stats_hud_max:
            self._hud_stats_hud_max = self._last_hud_ms
        if self._hud_stats_upload_min is None or upload_ms < self._hud_stats_upload_min:
            self._hud_stats_upload_min = upload_ms
        if self._hud_stats_upload_max is None or upload_ms > self._hud_stats_upload_max:
            self._hud_stats_upload_max = upload_ms
        if self._hud_stats_dt_min is None or dt_ms < self._hud_stats_dt_min:
            self._hud_stats_dt_min = dt_ms
        if self._hud_stats_dt_max is None or dt_ms > self._hud_stats_dt_max:
            self._hud_stats_dt_max = dt_ms
        if self._hud_stats_draw_min is None or self.last_draw_ms < self._hud_stats_draw_min:
            self._hud_stats_draw_min = self.last_draw_ms
        if self._hud_stats_draw_max is None or self.last_draw_ms > self._hud_stats_draw_max:
            self._hud_stats_draw_max = self.last_draw_ms
        if self._hud_stats_update_min is None or self.last_update_ms < self._hud_stats_update_min:
            self._hud_stats_update_min = self.last_update_ms
        if self._hud_stats_update_max is None or self.last_update_ms > self._hud_stats_update_max:
            self._hud_stats_update_max = self.last_update_ms
        elapsed = time.perf_counter() - self._hud_stats_start
        if elapsed >= 1.0:
            frames = max(1, self._hud_stats_frames)
            avg_dt = self._hud_stats_dt_sum / frames
            avg_draw = self._hud_stats_draw_sum / frames
            avg_update = self._hud_stats_update_sum / frames
            self._hud_stats_text = (
                f"dt avg={avg_dt:.2f} min={self._hud_stats_dt_min:.2f} max={self._hud_stats_dt_max:.2f} | "
                f"draw avg={avg_draw:.2f} min={self._hud_stats_draw_min:.2f} max={self._hud_stats_draw_max:.2f} | "
                f"update avg={avg_update:.2f} min={self._hud_stats_update_min:.2f} max={self._hud_stats_update_max:.2f} | "
                f"slow={self._hud_stats_slow_count} max={self._hud_stats_slow_max:.2f}"
            )
            avg_sector = self._hud_stats_sector_sum / frames
            avg_update_sectors = self._hud_stats_update_sectors_sum / frames
            avg_mesh_jobs = self._hud_stats_mesh_jobs_sum / frames
            avg_physics = self._hud_stats_physics_sum / frames
            avg_anim = self._hud_stats_anim_sum / frames
            avg_world = self._hud_stats_world_sum / frames
            avg_entities = self._hud_stats_entities_sum / frames
            avg_water = self._hud_stats_water_sum / frames
            avg_overlay = self._hud_stats_overlay_sum / frames
            avg_hud = self._hud_stats_hud_sum / frames
            avg_upload = self._hud_stats_upload_sum / frames
            self._hud_stats_detail_text = (
                f"sector avg={avg_sector:.2f} min={self._hud_stats_sector_min:.2f} max={self._hud_stats_sector_max:.2f} | "
                f"update_sectors avg={avg_update_sectors:.2f} min={self._hud_stats_update_sectors_min:.2f} max={self._hud_stats_update_sectors_max:.2f} | "
                f"mesh_jobs avg={avg_mesh_jobs:.2f} min={self._hud_stats_mesh_jobs_min:.2f} max={self._hud_stats_mesh_jobs_max:.2f} | "
                f"physics avg={avg_physics:.2f} min={self._hud_stats_physics_min:.2f} max={self._hud_stats_physics_max:.2f} | "
                f"anim avg={avg_anim:.2f} min={self._hud_stats_anim_min:.2f} max={self._hud_stats_anim_max:.2f}"
            )
            self._hud_stats_draw_detail_text = (
                f"world avg={avg_world:.2f} min={self._hud_stats_world_min:.2f} max={self._hud_stats_world_max:.2f} | "
                f"ent avg={avg_entities:.2f} min={self._hud_stats_entities_min:.2f} max={self._hud_stats_entities_max:.2f} | "
                f"water avg={avg_water:.2f} min={self._hud_stats_water_min:.2f} max={self._hud_stats_water_max:.2f} | "
                f"overlay avg={avg_overlay:.2f} min={self._hud_stats_overlay_min:.2f} max={self._hud_stats_overlay_max:.2f} | "
                f"HUD avg={avg_hud:.2f} min={self._hud_stats_hud_min:.2f} max={self._hud_stats_hud_max:.2f} | "
                f"upload avg={avg_upload:.2f} min={self._hud_stats_upload_min:.2f} max={self._hud_stats_upload_max:.2f}"
            )
            title = pad_header("Frame update timing")
            self._hud_block2_text = f"{title}\n{self._hud_stats_text}"
            if self._hud_stats_detail_text:
                self._hud_block2_text = (
                    f"{title}\n{self._hud_stats_text}\n{self._hud_stats_detail_text}"
                )
            if self._hud_stats_draw_detail_text:
                self._hud_block2_text = (
                    f"{self._hud_block2_text}\n{self._hud_stats_draw_detail_text}"
                )
            self._hud_stats_start = time.perf_counter()
            self._hud_stats_frames = 0
            self._hud_stats_dt_sum = 0.0
            self._hud_stats_dt_min = None
            self._hud_stats_dt_max = None
            self._hud_stats_draw_sum = 0.0
            self._hud_stats_draw_min = None
            self._hud_stats_draw_max = None
            self._hud_stats_update_sum = 0.0
            self._hud_stats_update_min = None
            self._hud_stats_update_max = None
            self._hud_stats_sector_sum = 0.0
            self._hud_stats_sector_min = None
            self._hud_stats_sector_max = None
            self._hud_stats_update_sectors_sum = 0.0
            self._hud_stats_update_sectors_min = None
            self._hud_stats_update_sectors_max = None
            self._hud_stats_mesh_jobs_sum = 0.0
            self._hud_stats_mesh_jobs_min = None
            self._hud_stats_mesh_jobs_max = None
            self._hud_stats_physics_sum = 0.0
            self._hud_stats_physics_min = None
            self._hud_stats_physics_max = None
            self._hud_stats_anim_sum = 0.0
            self._hud_stats_anim_min = None
            self._hud_stats_anim_max = None
            self._hud_stats_world_sum = 0.0
            self._hud_stats_world_min = None
            self._hud_stats_world_max = None
            self._hud_stats_entities_sum = 0.0
            self._hud_stats_entities_min = None
            self._hud_stats_entities_max = None
            self._hud_stats_water_sum = 0.0
            self._hud_stats_water_min = None
            self._hud_stats_water_max = None
            self._hud_stats_overlay_sum = 0.0
            self._hud_stats_overlay_min = None
            self._hud_stats_overlay_max = None
            self._hud_stats_hud_sum = 0.0
            self._hud_stats_hud_min = None
            self._hud_stats_hud_max = None
            self._hud_stats_upload_sum = 0.0
            self._hud_stats_upload_min = None
            self._hud_stats_upload_max = None
            self._hud_stats_slow_count = 0
            self._hud_stats_slow_max = 0.0
            self._hud_stats_draw_detail_text = ""
        logutil.log("FRAME", f"end ms={self.last_draw_ms:.2f}")

    def _set_label_text(self, label, text):
        if label.text == text:
            return False
        label.text = text
        return True

    def draw_label(self):
        """ Draw the label in the top left of the screen.

        """
        if not self.hud_visible:
            return
        hud_profile = getattr(config, 'HUD_PROFILE', False)
        if hud_profile:
            profile_start = time.perf_counter()
            profile_last = profile_start
            profile_segments = self._hud_profile_segments
            def _profile_mark(key):
                nonlocal profile_last
                now_mark = time.perf_counter()
                profile_segments[key] = profile_segments.get(key, 0.0) + (now_mark - profile_last) * 1000.0
                profile_last = now_mark
        x, y, z = self.position
        rx, ry = self.rotation
        sector = util.sectorize((x, y, z))
        if getattr(config, 'HUD_PROBE_ENABLED', False):
            self._hud_probe_frame += 1
            needs_refresh = (
                self._hud_probe_sector != sector
                or self._hud_probe_frame >= getattr(config, 'HUD_PROBE_EVERY_N_FRAMES', 15)
            )
            if needs_refresh:
                self._hud_probe_sector = sector
                self._hud_probe_frame = 0
                sight = self.get_sight_vector()
                void_dist = self.model.measure_void_distance(self.position, sight, max_distance=64)
                if void_dist is None:
                    self._hud_probe_void = 'N/A'
                elif void_dist >= 64:
                    self._hud_probe_void = '>=64'
                else:
                    self._hud_probe_void = str(void_dist)
                mush_pos = self.model.nearest_mushroom_in_sector(sector, self.position)
                if mush_pos is None:
                    self._hud_probe_mush = 'NA'
                else:
                    mx, my, mz = mush_pos
                    self._hud_probe_mush = f'{mx},{my},{mz}'
            void_text = self._hud_probe_void
            mush_text = self._hud_probe_mush
        else:
            void_text = 'N/A'
            mush_text = 'NA'
        if hud_profile:
            _profile_mark("probe")
        now = time.perf_counter()
        if self.hud_details_visible:
            refresh_s = float(getattr(config, "HUD_DETAIL_REFRESH_S", 1.0))
            if now - self._hud_detail_last_update >= refresh_s:
                self._hud_detail_last_update = now
                self._hud_detail_dirty = True
        if now - self._hud_block_last_update >= 1.0:
            self._hud_block_last_update = now
            fps = self._current_fps()
            seed_val = getattr(self.model, "world_seed", None)
            seed_text = str(seed_val) if seed_val is not None else "NA"
            self._hud_block1_text = (
                'FPS(%.1f), pos(%.2f, %.2f, %.2f) sector(%d, 0, %d) rot(%.1f, %.1f) seed %s void %s mush %s' % (
                    fps, x, y, z, sector[0], sector[2], rx, ry, seed_text, void_text, mush_text
                )
            )
            if hud_profile:
                _profile_mark("build_block1")
            if not self.hud_details_visible:
                self._hud_block2_text = ""
                self._hud_block3_text = ""
                self._hud_block4_text = ""
                self._hud_block5_text = ""
            else:
                sector_state = [f"Sector({sector[0]}, 0, {sector[2]})"]
                sector_debug = []
                s = self.model.sectors.get(sector)
                def _refresh_stat_last(obj):
                    obj.stat_load_count_last = obj.stat_load_count_total - obj.stat_load_count_prev
                    obj.stat_load_ms_last = obj.stat_load_ms_total - obj.stat_load_ms_prev
                    obj.stat_light_count_last = obj.stat_light_count_total - obj.stat_light_count_prev
                    obj.stat_light_ms_last = obj.stat_light_ms_total - obj.stat_light_ms_prev
                    obj.stat_mesh_count_last = obj.stat_mesh_count_total - obj.stat_mesh_count_prev
                    obj.stat_mesh_ms_last = obj.stat_mesh_ms_total - obj.stat_mesh_ms_prev
                    obj.stat_upload_count_last = obj.stat_upload_count_total - obj.stat_upload_count_prev
                    obj.stat_upload_ms_last = obj.stat_upload_ms_total - obj.stat_upload_ms_prev
                    obj.stat_load_count_prev = obj.stat_load_count_total
                    obj.stat_load_ms_prev = obj.stat_load_ms_total
                    obj.stat_light_count_prev = obj.stat_light_count_total
                    obj.stat_light_ms_prev = obj.stat_light_ms_total
                    obj.stat_mesh_count_prev = obj.stat_mesh_count_total
                    obj.stat_mesh_ms_prev = obj.stat_mesh_ms_total
                    obj.stat_upload_count_prev = obj.stat_upload_count_total
                    obj.stat_upload_ms_prev = obj.stat_upload_ms_total
                    if hasattr(obj, "stat_loader_block_defer_total"):
                        obj.stat_loader_block_defer_last = obj.stat_loader_block_defer_total - obj.stat_loader_block_defer_prev
                        obj.stat_loader_block_mesh_last = obj.stat_loader_block_mesh_total - obj.stat_loader_block_mesh_prev
                        obj.stat_loader_block_inflight_last = obj.stat_loader_block_inflight_total - obj.stat_loader_block_inflight_prev
                        obj.stat_loader_sent_last = obj.stat_loader_sent_total - obj.stat_loader_sent_prev
                        obj.stat_load_refresh_last = obj.stat_load_refresh_total - obj.stat_load_refresh_prev
                        obj.stat_load_candidates_last = obj.stat_load_candidates_total - obj.stat_load_candidates_prev
                        obj.stat_loader_block_defer_prev = obj.stat_loader_block_defer_total
                        obj.stat_loader_block_mesh_prev = obj.stat_loader_block_mesh_total
                        obj.stat_loader_block_inflight_prev = obj.stat_loader_block_inflight_total
                        obj.stat_loader_sent_prev = obj.stat_loader_sent_total
                        obj.stat_load_refresh_prev = obj.stat_load_refresh_total
                        obj.stat_load_candidates_prev = obj.stat_load_candidates_total

                _refresh_stat_last(self.model)
                if s is None:
                    sector_state.append("state=missing")
                    sector_debug.append("state=missing")
                else:
                    _refresh_stat_last(s)

                    light = 'Y' if (not s.light_dirty_internal and not s.light_dirty_incoming) else 'N'
                    mesh_ready = self.model._mesh_ready(s)
                    neighbors_missing = self.model._neighbors_missing(s)
                    needs_mesh = (s.vt_data is None and not s.mesh_built)
                    needs_light = self.model._needs_light(s)
                    if s.mesh_job_pending:
                        waiting = "mesh_job"
                    elif neighbors_missing:
                        waiting = "neighbors"
                    elif needs_mesh:
                        waiting = "mesh"
                    elif needs_light:
                        waiting = "light_recalc"
                    else:
                        waiting = "idle"
                    def _incoming_stats(incoming):
                        nonempty = 0
                        total_entries = 0
                        for entries in incoming.values():
                            if entries is None or len(entries) == 0:
                                continue
                            nonempty += 1
                            total_entries += len(entries)
                        return nonempty, total_entries

                    def _quad_count(entry):
                        if not entry or entry[0] <= 0:
                            return 0
                        return int(entry[0] // 4)

                    def _float_count(entry):
                        if not entry or entry[0] <= 0:
                            return 0
                        _, v, t, n, c, l = entry
                        return len(v) + len(t) + len(n) + len(c) + len(l)

                    vt_info = "vt=N"
                    if s.vt_data is not None:
                        if isinstance(s.vt_data, dict):
                            solid_entry = s.vt_data.get('solid')
                            water_entry = s.vt_data.get('water')
                        else:
                            solid_entry = s.vt_data
                            water_entry = None
                        solid_quads = _quad_count(solid_entry)
                        water_quads = _quad_count(water_entry)
                        total_floats = _float_count(solid_entry) + _float_count(water_entry)
                        kb = (total_floats * 4) / 1024.0 if total_floats else 0.0
                        vt_info = f"vt=Y quads={solid_quads}/{water_quads} kb={kb:.1f}"
                    else:
                        solid_quads = int(getattr(s, "vt_solid_quads", 0) or 0)
                        water_quads = int(getattr(s, "vt_water_quads", 0) or 0)
                        if solid_quads or water_quads:
                            vt_info = f"vt=N quads={solid_quads}/{water_quads}"

                    uploaded = "Y" if (s.vt or s.vt_water) else "N"
                    solid_verts = sum(getattr(vt, "count", 0) for vt in s.vt)
                    water_verts = sum(getattr(vt, "count", 0) for vt in s.vt_water)
                    upload_solid = f"{s.vt_upload_solid}/{s.vt_solid_quads}"
                    upload_water = f"{s.vt_upload_water}/{s.vt_water_quads}"
                    solid_tris_expected = int(s.vt_solid_quads * 2)
                    water_tris_expected = int(s.vt_water_quads * 2)
                    solid_tris_actual = int(solid_verts // 3) if solid_verts else 0
                    water_tris_actual = int(water_verts // 3) if water_verts else 0
                    pending_vt = len(getattr(s, "pending_vt", []))
                    pending_vt_water = len(getattr(s, "pending_vt_water", []))
                    use_pending = "Y" if getattr(s, "vt_upload_use_pending", False) else "N"
                    upload_prepared = "Y" if s.vt_upload_prepared else "N"
                    clear_pending = "Y" if s.vt_clear_pending else "N"
                    token = getattr(s, "vt_upload_token", 0)
                    active_token = getattr(s, "vt_upload_active_token", None)
                    active_token_text = "None" if active_token is None else str(active_token)
                    dirty = "Y" if s.mesh_job_dirty else "N"
                    inflight = "Y" if s.edit_inflight else "N"
                    sky_nonempty, sky_entries = _incoming_stats(s.incoming_sky)
                    torch_nonempty, torch_entries = _incoming_stats(s.incoming_torch)
                    sky_updates = getattr(s, "incoming_sky_updates", 0)
                    torch_updates = getattr(s, "incoming_torch_updates", 0)
                    edge_sky_counts = getattr(s, "edge_sky_counts", (0, 0, 0, 0))
                    edge_torch_counts = getattr(s, "edge_torch_counts", (0, 0, 0, 0))

                    sector_state.append(
                        f"light={light} mesh_ready={'Y' if mesh_ready else 'N'} "
                        f"pending={'Y' if s.mesh_job_pending else 'N'} "
                        f"built={'Y' if s.mesh_built else 'N'} "
                        f"uploaded={uploaded} vt_lists={len(s.vt)}/{len(s.vt_water)} "
                        f"verts={solid_verts}/{water_verts} "
                        f"upload={upload_solid}/{upload_water} "
                        f"{vt_info} wait={waiting}"
                    )
                    sector_state.append(
                        f"edge_sky={sky_nonempty}/8 e={sky_entries} u={sky_updates} "
                        f"edge_torch={torch_nonempty}/8 e={torch_entries} u={torch_updates}"
                    )
                    sector_state.append(
                        "edge_use sky(W/E/N/S)=%d/%d/%d/%d torch(W/E/N/S)=%d/%d/%d/%d"
                        % (
                            edge_sky_counts[0], edge_sky_counts[1], edge_sky_counts[2], edge_sky_counts[3],
                            edge_torch_counts[0], edge_torch_counts[1], edge_torch_counts[2], edge_torch_counts[3],
                        )
                    )
                    sector_debug.append(
                        f"tris={solid_tris_actual}/{solid_tris_expected} verts={solid_verts}/"
                        f"{solid_tris_expected * 3} water_tris={water_tris_actual}/{water_tris_expected} "
                        f"water_verts={water_verts}/{water_tris_expected * 3} "
                        f"pending_vt={pending_vt}/{pending_vt_water} use_pending={use_pending} "
                        f"prep={upload_prepared} clear_pending={clear_pending} token={token}/{active_token_text} "
                        f"gen={s.mesh_gen} dirty={dirty} inflight={inflight}"
                    )

                self._hud_block3_text = f"{pad_header("Current sector")}\n" + " | ".join(sector_state)
                if sector_debug:
                    self._hud_block3_text = f"{self._hud_block3_text}\n" + " | ".join(sector_debug)
                if s is not None:
                    table = [
                        "stage   total_ct total_ms 1s_ct 1s_ms",
                        f"load    {s.stat_load_count_total:8d} {s.stat_load_ms_total:8.1f} {s.stat_load_count_last:5d} {s.stat_load_ms_last:6.1f}",
                        f"light   {s.stat_light_count_total:8d} {s.stat_light_ms_total:8.1f} {s.stat_light_count_last:5d} {s.stat_light_ms_last:6.1f}",
                        f"mesh    {s.stat_mesh_count_total:8d} {s.stat_mesh_ms_total:8.1f} {s.stat_mesh_count_last:5d} {s.stat_mesh_ms_last:6.1f}",
                        f"upload  {s.stat_upload_count_total:8d} {s.stat_upload_ms_total:8.1f} {s.stat_upload_count_last:5d} {s.stat_upload_ms_last:6.1f}",
                    ]
                    self._hud_block3_text = f"{self._hud_block3_text}\n" + "\n".join(table)

                model = self.model
                inflight = model.n_requests - model.n_responses
                pending_load = len(getattr(model, "update_sectors_pos", []))
                loader_q = len(getattr(model, "loader_requests", []))
                mesh_pending = getattr(model, "mesh_active_jobs", 0)
                upload_pending = len(getattr(model, "pending_uploads", []))
                loader_total = getattr(model, "loader_sectors_received_total", 0)
                mesh_total = getattr(model, "mesh_jobs_completed_total", 0)
                mesh_submit_total = getattr(model, "mesh_jobs_submitted_total", 0)
                loader_delta = loader_total - self._hud_last_loader_count
                mesh_delta = mesh_total - self._hud_last_mesh_done
                mesh_submit_delta = mesh_submit_total - getattr(self, "_hud_last_mesh_submitted", 0)
                self._hud_last_loader_count = loader_total
                self._hud_last_mesh_done = mesh_total
                self._hud_last_mesh_submitted = mesh_submit_total
                upload_skipped_delta = self._hud_upload_skipped - self._hud_last_upload_skipped
                self._hud_last_upload_skipped = self._hud_upload_skipped
                recent_load = list(getattr(model, "loader_recent", []))
                recent_mesh = list(getattr(model, "mesh_recent", []))
                stats_enabled = True
                shown = getattr(model, "stats_shown_sectors", 0)
                total = getattr(model, "stats_total_sectors", len(model.sectors))
                drawn_solid = getattr(model, "stats_drawn_tris_solid", 0)
                drawn_water = getattr(model, "stats_drawn_tris_water", 0)
                cull_ms = getattr(model, "stats_cull_ms", 0.0)
                batches_solid = getattr(model, "stats_batches_solid", 0)
                batches_water = getattr(model, "stats_batches_water", 0)
                vlists_solid = getattr(model, "stats_vlists_solid", 0)
                vlists_water = getattr(model, "stats_vlists_water", 0)
                draw_calls = getattr(model, "stats_draw_calls", 0)
                if not stats_enabled:
                    shown_text = "shown=NA/NA"
                    tris_text = "tris_solid=NA tris_water=NA"
                    cull_text = "cull_ms=NA"
                    draw_text = "draw_calls=NA batches=NA/NA vlists=NA/NA"
                else:
                    shown_text = f"shown={shown}/{total}"
                    tris_text = f"tris_solid={drawn_solid} tris_water={drawn_water}"
                    cull_text = f"cull_ms={cull_ms:.2f}"
                    draw_text = f"draw_calls={draw_calls} batches={batches_solid}/{batches_water} vlists={vlists_solid}/{vlists_water}"

                self._hud_block4_text = (
                    f"{pad_header("Terrain info")}\n"
                    f"loaded={len(model.sectors)} pending_load={pending_load} inflight={inflight} "
                    f"loader_q={loader_q} mesh_pending={mesh_pending} upload_pending={upload_pending}\n"
                    f"{shown_text} {tris_text} {cull_text}\n"
                    f"{draw_text}"
                )
                terrain_table = [
                    "stage   total_ct total_ms 1s_ct 1s_ms",
                    f"load    {model.stat_load_count_total:8d} {model.stat_load_ms_total:8.1f} {model.stat_load_count_last:5d} {model.stat_load_ms_last:6.1f}",
                    f"light   {model.stat_light_count_total:8d} {model.stat_light_ms_total:8.1f} {model.stat_light_count_last:5d} {model.stat_light_ms_last:6.1f}",
                    f"mesh    {model.stat_mesh_count_total:8d} {model.stat_mesh_ms_total:8.1f} {model.stat_mesh_count_last:5d} {model.stat_mesh_ms_last:6.1f}",
                    f"upload  {model.stat_upload_count_total:8d} {model.stat_upload_ms_total:8.1f} {model.stat_upload_count_last:5d} {model.stat_upload_ms_last:6.1f}",
                ]
                self._hud_block4_text += "\n" + "\n".join(terrain_table)
                backlog_count = getattr(model, "mesh_backlog_last", 0)
                loader_table = [
                    "loader  total_ct 1s_ct",
                    f"send    {model.stat_loader_sent_total:8d} {model.stat_loader_sent_last:5d}",
                    f"block_defer {model.stat_loader_block_defer_total:6d} {model.stat_loader_block_defer_last:5d}",
                    f"block_mesh  {model.stat_loader_block_mesh_total:6d} {model.stat_loader_block_mesh_last:5d}",
                    f"block_inflight {model.stat_loader_block_inflight_total:3d} {model.stat_loader_block_inflight_last:5d}",
                    f"refresh {model.stat_load_refresh_total:6d} {model.stat_load_refresh_last:5d}",
                    f"candidates {model.stat_load_candidates_total:3d} {model.stat_load_candidates_last:5d}",
                    f"mesh_backlog {backlog_count:4d}",
                ]
                draw_stats_line = "draw_stats=on" if stats_enabled else "draw_stats=off"
                self._hud_block4_text += "\n" + "\n".join(loader_table)
                self._hud_block4_text += f"\n{draw_stats_line}"
                self._hud_block4_text += (
                    f"\nWork/s: loader={loader_delta} mesh_done={mesh_delta} mesh_submit={mesh_submit_delta}"
                )
                self._hud_block4_text += (
                    f"\nUpload: budget_ms={self._hud_upload_budget_ms:.2f} skipped={upload_skipped_delta} "
                    f"tri_budget={self._hud_tri_budget} tri_uploaded={self._hud_tri_uploaded} "
                    f"pending={self._hud_upload_pending}"
                )
                if recent_load or recent_mesh:
                    self._hud_block4_text += f"\nRecent: load={recent_load} mesh={recent_mesh}"

                entity_lines = []
                for entity_state in self.entities.values():
                    if entity_state['type'] == 'player':
                        continue
                    pos = entity_state.get('pos')
                    if pos is None:
                        continue
                    px, py, pz = pos
                    entity_lines.append(f"{entity_state['type']}({px:.1f},{py:.1f},{pz:.1f})")
                title = pad_header("Entities")
                if entity_lines:
                    self._hud_block5_text = f"{title}\n" + " | ".join(entity_lines)
                else:
                    self._hud_block5_text = f"{title}\nnone"
                if hud_profile:
                    _profile_mark("build_block2_5")
        if not self.hud_details_visible:
            self._hud_detail_dirty = False
        if hud_profile:
            _profile_mark("build")

        layout_dirty = False
        layout_dirty |= self._set_label_text(self.label, self._hud_block1_text)
        if hud_profile:
            _profile_mark("set_label_block1")
        info_text = f"{self._hud_block2_text}\n" if self._hud_block2_text else ""
        sector_text = f"{self._hud_block3_text}\n" if self._hud_block3_text else ""
        debug_text = f"{self._hud_block4_text}\n" if self._hud_block4_text else ""
        entity_text = f"{self._hud_block5_text}\n" if self._hud_block5_text else ""
        if self.hud_details_visible:
            detail_update = self._hud_detail_dirty or self._hud_layout_dirty
            if detail_update:
                layout_dirty |= self._set_label_text(self.hud_info_label, info_text)
                if hud_profile:
                    _profile_mark("set_label_block2")
                layout_dirty |= self._set_label_text(self.sector_label, sector_text)
                if hud_profile:
                    _profile_mark("set_label_block3")
                layout_dirty |= self._set_label_text(self.sector_debug_label, debug_text)
                if hud_profile:
                    _profile_mark("set_label_block4")
                layout_dirty |= self._set_label_text(self.entity_label, entity_text)
                if hud_profile:
                    _profile_mark("set_label_block5")
        else:
            layout_dirty |= self._set_label_text(self.hud_info_label, "")
            layout_dirty |= self._set_label_text(self.sector_label, "")
            layout_dirty |= self._set_label_text(self.sector_debug_label, "")
            layout_dirty |= self._set_label_text(self.entity_label, "")
        hud_state = "on" if self.hud_visible else "off"
        vsync_state = "on" if self.vsync_enabled else "off"
        camera_state = "1p" if self.camera_mode == 'first_person' else "3p"
        details_state = "on" if self.hud_details_visible else "off"
        snake_state = "on" if self.snake_enabled else "off"
        snail_state = "on" if self.snail_enabled else "off"
        seagull_state = "on" if self.seagull_enabled else "off"
        dog_state = "on" if self.dog_enabled else "off"
        keybind_text = (
            "Toggles: (F1)HUD=%s (F2)Vsync=%s (F3)Details=%s (F5)Cam=%s (F8)Copy | "
            "(V)Dog=%s (B)Snail=%s S(N)nake=%s (M)Seagull=%s"
            % (hud_state, vsync_state, details_state, camera_state, snake_state, snail_state, seagull_state, dog_state)
        )
        layout_dirty |= self._set_label_text(self.keybind_label, keybind_text)
        if hud_profile:
            _profile_mark("set_label_keybind")

        if layout_dirty:
            self._hud_layout_dirty = True

        if self._hud_layout_dirty:
            wrap_width = max(120, self.width - 20)
            for lbl in (self.hud_info_label, self.sector_label, self.sector_debug_label, self.entity_label, self.keybind_label):
                lbl.width = wrap_width
                lbl.multiline = True
            line_spacing = 4
            if self.hud_details_visible:
                self.hud_info_label.y = self.label.y - self.label.content_height - line_spacing
                self.sector_label.y = self.hud_info_label.y - self.hud_info_label.content_height - line_spacing
                self.sector_debug_label.y = self.sector_label.y - self.sector_label.content_height - line_spacing
                self.entity_label.y = self.sector_debug_label.y - self.sector_debug_label.content_height - line_spacing
                self.keybind_label.y = self.entity_label.y - self.entity_label.content_height - line_spacing
            else:
                self.hud_info_label.y = self.label.y
                self.sector_label.y = self.label.y
                self.sector_debug_label.y = self.label.y
                self.entity_label.y = self.label.y
                self.keybind_label.y = self.label.y - self.label.content_height - line_spacing

            # Light backdrop to keep text readable on bright backgrounds.
            pad_x = 6
            pad_y = 3
            top = self.label.y
            bottom = self.keybind_label.y - self.keybind_label.content_height
            if self.hud_details_visible:
                entity_width = max(
                    self.label.content_width,
                    self.hud_info_label.content_width,
                    self.sector_label.content_width,
                    self.sector_debug_label.content_width,
                    self.entity_label.content_width,
                    self.keybind_label.content_width,
                )
            else:
                entity_width = max(
                    self.label.content_width,
                    self.keybind_label.content_width,
                )
            bg_width = entity_width + pad_x * 2
            bg_height = (top - bottom) + pad_y * 2
            bg_x = self.label.x - pad_x
            bg_y = bottom - pad_y
            self._label_bg.x = bg_x
            self._label_bg.y = bg_y
            self._label_bg.width = bg_width
            self._label_bg.height = bg_height
            self._hud_layout_dirty = False
        if self._hud_detail_dirty and not self._hud_layout_dirty:
            self._hud_detail_dirty = False
        if hud_profile:
            _profile_mark("layout")

        self._label_bg.draw()
        if hud_profile:
            _profile_mark("draw_bg")
        self.label.draw()
        if hud_profile:
            _profile_mark("draw_block1")
        if self.hud_details_visible:
            self.hud_info_label.draw()
            if hud_profile:
                _profile_mark("draw_block2")
            self.sector_label.draw()
            if hud_profile:
                _profile_mark("draw_block3")
            self.sector_debug_label.draw()
            if hud_profile:
                _profile_mark("draw_block4")
            self.entity_label.draw()
            if hud_profile:
                _profile_mark("draw_block5")
        self.keybind_label.draw()
        if hud_profile:
            _profile_mark("draw_keybind")
            total_ms = (time.perf_counter() - profile_start) * 1000.0
            self._hud_profile_accum += total_ms
            self._hud_profile_count += 1
            if total_ms > self._hud_profile_max:
                self._hud_profile_max = total_ms
            log_interval = float(getattr(config, 'HUD_PROFILE_LOG_S', 1.0))
            spike_ms = getattr(config, 'HUD_PROFILE_SPIKE_MS', None)
            spike = spike_ms is not None and total_ms >= float(spike_ms)
            log_now = (now - self._hud_profile_last_log) >= log_interval
            if log_now or spike:
                avg_ms = self._hud_profile_accum / max(1, self._hud_profile_count)
                segs = self._hud_profile_segments
                logutil.log(
                    "HUD",
                    "profile total_ms=%.2f avg=%.2f max=%.2f probe=%.2f "
                    "build=%.2f build_b1=%.2f build_b2_5=%.2f "
                    "set_b1=%.2f set_b2=%.2f set_b3=%.2f set_b4=%.2f set_b5=%.2f set_key=%.2f "
                    "layout=%.2f draw_bg=%.2f draw_b1=%.2f draw_b2=%.2f draw_b3=%.2f "
                    "draw_b4=%.2f draw_b5=%.2f draw_key=%.2f%s"
                    % (
                        total_ms,
                        avg_ms,
                        self._hud_profile_max,
                        segs.get("probe", 0.0),
                        segs.get("build", 0.0),
                        segs.get("build_block1", 0.0),
                        segs.get("build_block2_5", 0.0),
                        segs.get("set_label_block1", 0.0),
                        segs.get("set_label_block2", 0.0),
                        segs.get("set_label_block3", 0.0),
                        segs.get("set_label_block4", 0.0),
                        segs.get("set_label_block5", 0.0),
                        segs.get("set_label_keybind", 0.0),
                        segs.get("layout", 0.0),
                        segs.get("draw_bg", 0.0),
                        segs.get("draw_block1", 0.0),
                        segs.get("draw_block2", 0.0),
                        segs.get("draw_block3", 0.0),
                        segs.get("draw_block4", 0.0),
                        segs.get("draw_block5", 0.0),
                        segs.get("draw_keybind", 0.0),
                        " spike=Y" if spike and not log_now else "",
                    ),
                )
            if log_now:
                self._hud_profile_last_log = now
                self._hud_profile_accum = 0.0
                self._hud_profile_count = 0
                self._hud_profile_max = 0.0
                self._hud_profile_segments = {}

    def _current_fps(self):
        """Return a smoothed FPS based on recent draw intervals."""
        if not self._frame_times:
            return 0.0
        total = sum(self._frame_times)
        return len(self._frame_times) / total if total > 0 else 0.0

    def _detect_refresh_rate(self):
        """Try to read the current monitor refresh rate for frame capping."""
        try:
            screen = self.display.get_default_screen()
            mode = screen.get_mode()
            rate = getattr(mode, 'refresh_rate', None)
            if rate:
                return rate
        except Exception:
            logutil.log("MAIN", "refresh rate not detected", level="WARN")
            pass
        return None

    def draw_focused_block(self):
        """ Draw edges around the block under the crosshairs for placement feedback. """
        return  # temporarily disabled due to pyglet draw API mismatch

    def draw_reticle(self):
        """ Draw the crosshairs in the center of the screen.

        """
        blend_enabled = bool(gl.glIsEnabled(gl.GL_BLEND))
        logic_enabled = bool(gl.glIsEnabled(gl.GL_COLOR_LOGIC_OP))
        if blend_enabled:
            gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_COLOR_LOGIC_OP)
        gl.glLogicOp(gl.GL_XOR)
        for line in self.reticle:
            line.draw()
        if not logic_enabled:
            gl.glDisable(gl.GL_COLOR_LOGIC_OP)
        if blend_enabled:
            gl.glEnable(gl.GL_BLEND)

    def draw_inventory_item(self):
        if self.inventory_item:
            self.inventory_item.draw()

    def _is_underwater(self):
        """Return True if the camera is currently inside a water block."""
        pos = util.normalize(self.position)
        if self.model[pos] == WATER:
            return True
        head_pos = (pos[0], pos[1] + 1, pos[2])
        return self.model[head_pos] == WATER

    def draw_underwater_overlay(self):
        """Render a full-viewport tint when submerged to avoid per-block transparency."""
        width, height = self.get_size()
        self._underwater_overlay.x = 0
        self._underwater_overlay.y = 0
        self._underwater_overlay.width = width
        self._underwater_overlay.height = height
        self._underwater_overlay.draw()



def setup_fog():
    """ Configure the OpenGL fog properties.

    """
    # Fixed-function fog isn't available on modern/core profiles; skip if missing.
    return



def setup():
    """ Basic OpenGL configuration.

    """
    # Set the color of "clear", i.e. the sky, in rgba.
    fog = getattr(config, 'DAY_FOG_COLOR', (0.5, 0.69, 1.0))
    gl.glClearColor(fog[0], fog[1], fog[2], 1)
    # Cull back faces for better fill-rate; geometry is built with consistent winding.
    gl.glEnable(gl.GL_CULL_FACE)
    gl.glCullFace(gl.GL_BACK)

    #gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_DST_ALPHA)
    #gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
#    gl.glBlendFunc(gl.GL_ZERO, gl.GL_SRC_COLOR)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    # Set the texture minification/magnification function to GL_NEAREST (nearest
    # in Manhattan distance) to the specified texture coordinates. GL_NEAREST
    # "is generally faster than GL_LINEAR, but it can produce textured images
    # with sharper edges because the transition between texture elements is not
    # as smooth."
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    # Fixed-function texture env not needed with shaders.
    setup_fog()


def main():
    logutil.log("MAIN", f"minepy2 start {datetime.now().isoformat(sep=' ', timespec='seconds')}")
    if len(sys.argv)>1:
        arg = sys.argv[1]
        if ':' in arg:
            host, port = arg.split(':', 1)
            config.SERVER_IP = host
            try:
                config.SERVER_PORT = int(port)
            except ValueError:
                pass
        else:
            config.SERVER_IP = arg
        logutil.log("MAIN", f"Using server IP address {config.SERVER_IP}:{config.SERVER_PORT}")
    window = Window(width=300, height=200, caption='Pyglet', resizable=True, vsync=False)
    # Hide the mouse cursor and prevent the mouse from leaving the window.
    window.set_exclusive_mouse(True)
    setup()
    try:
        pyglet.app.run()
    except:
        import traceback
        traceback.print_exc()
        logutil.log("MAIN", "terminating child processes")
        window.model.quit()
        window.set_exclusive_mouse(False)


if __name__ == '__main__':
    main()
