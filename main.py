import math
import random
import time
import sys
import multiprocessing
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
import entity as entity_codec
import logutil
import mapgen
import server as server_module
from entities.player import HUMANOID_MODEL, Player
from entities.snake import SNAKE_MODEL, SnakeEntity
from entities.snail import SNAIL_MODEL, SnailEntity
from entities.seagull import SEAGULL_MODEL, SeagullEntity
from entities.dog import DOG_MODEL, Dog
from entities.dinotrex import DINOTREX_MODEL, DinoTrexEntity
from entities.mosasaurus import MOSASAURUS_MODEL, MosasaurusEntity
from entities.fish_school import FISH_MODEL, FishSchoolEntity, FISH_SEGMENT_CONFIGS, FISH_TAIL_LENGTH
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
    _mesh_aabb_min as BLOCK_RENDER_AABB_MIN,
    _mesh_aabb_max as BLOCK_RENDER_AABB_MAX,
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

FOCUSED_BLOCK_EDGES = np.array(
    [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ],
    dtype=np.int32,
)
WATER = BLOCK_ID['Water']


def pad_header(title, width=40):
    title = title.strip()
    pad = (width-len(title)-2)//2
    if pad <= 0:
        return title
    return '='*pad + ' ' + title + ' ' + '='*pad

def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def _norm(v: np.ndarray, eps=1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)

def _dir_from_az_elev_deg(az_deg: float, elev_rad: float) -> np.ndarray:
    """
    Your convention:
      az=0 => north => (0,0,-1)
      az=90 => east  => (1,0, 0)
      az=180 => south => (0,0, 1)
    """
    az = math.radians(az_deg % 360.0)
    ce = math.cos(elev_rad)
    x = ce * math.sin(az)
    y = math.sin(elev_rad)
    z = ce * (-math.cos(az))
    return _norm(np.array([x, y, z], dtype=np.float32))

def _rodrigues_rotate(v: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotate vector v around unit axis by theta (radians).
    """
    k = axis
    ct = math.cos(theta)
    st = math.sin(theta)
    # v' = v ct + (k×v) st + k (k·v) (1-ct)
    return (v * ct
            + np.cross(k, v) * st
            + k * (np.dot(k, v)) * (1.0 - ct)).astype(np.float32)

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
        self.block_program['u_light_dir_exp'] = getattr(config, 'LIGHT_DIR_EXP', 1.0)
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
        self.time_mode = "1x"
        self.day_length = float(getattr(config, "DAY_LENGTH_SECONDS", 1200.0))
        if self.day_length <= 1e-6:
            self.day_length = 1200.0
        self.day_phase = float(getattr(config, "DAY_START_PHASE", 0.0)) % 1.0
        self._sun_max_elev_rad = math.radians(float(getattr(config, "SUN_MAX_ELEV_DEG", 75.0)))
        sun_dir = getattr(config, "SUN_LIGHT_DIR", (0.35, 1.0, 0.65))
        self._sun_dir_xz = self._normalize_xz_dir(sun_dir)
        path_azim = getattr(config, "SUN_PATH_AZIMUTH_DEG", None)
        if path_azim is not None:
            self._sun_path_azim_deg = float(path_azim) % 360.0
        else:
            az_rad = math.atan2(self._sun_dir_xz[0], -self._sun_dir_xz[1])
            self._sun_path_azim_deg = math.degrees(az_rad) % 360.0
        self._day_light_dir = self._normalize_light_dir(sun_dir[0], sun_dir[1], sun_dir[2])
        self._day_ambient = getattr(config, "AMBIENT_LIGHT", 0.0)
        self._day_sky_intensity = getattr(config, "SKY_INTENSITY", 1.0)
        self._day_fog_color = getattr(config, "DAY_FOG_COLOR", (0.5, 0.69, 1.0))
        self._sun_tint = self._pick_sun_tint()
        self._sun_screen_pos = None
        if self.day_night_enabled:
            self._update_day_night(0.0)

        # Instance of the model that handles the world.
        self.model = world.ModelProxy(self.block_program)
        self._focus_outline_vlist = None
        self._focus_face_vlist = None

        # Entity rendering setup
        saved_player_state = None
        if getattr(config, "SERVER_IP", None) is None:
            saved_player_state = world_entity_store.load_entity_state("player")
        has_saved_player = saved_player_state is not None
        if saved_player_state:
            self.position = tuple(saved_player_state.get("pos", self.position))
            camera_rot = saved_player_state.get("camera_rot")
            if camera_rot is not None:
                self.rotation = tuple(camera_rot)
            else:
                self.rotation = tuple(saved_player_state.get("rot", self.rotation))
            self.flying = saved_player_state.get("flying", self.flying)
            saved_vel = saved_player_state.get("vel")
            saved_camera_mode = saved_player_state.get("camera_mode")
            saved_camera_dist = saved_player_state.get("third_person_distance")
        player_name = getattr(config, "PLAYER_NAME", "Player")
        self.player_entity = Player(self.model, HUMANOID_MODEL, position=self.position)
        self.player_entity.name = player_name
        self.player_entity.id = 0
        self.player_entity.position = np.array(self.position, dtype=float)
        self.player_entity.rotation = np.array(self.rotation, dtype=float)
        if saved_player_state:
            self.player_entity.on_ground = saved_player_state.get("on_ground", False)
            if saved_vel is not None:
                self.player_entity.velocity = np.array(saved_vel, dtype=float)
            if self.player_entity.on_ground:
                self.player_entity.velocity[1] = 0.0
            saved_rot = saved_player_state.get("rot")
            if saved_rot is not None and camera_rot is not None:
                self.player_entity.rotation = np.array(saved_rot, dtype=float)
            if saved_camera_mode is not None:
                self.camera_mode = saved_camera_mode
            if saved_camera_dist is not None:
                self.third_person_distance = float(saved_camera_dist)
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
            'dinotrex': renderer.AnimatedEntityRenderer(self.block_program, DINOTREX_MODEL),
            'mosasaurus': renderer.AnimatedEntityRenderer(self.block_program, MOSASAURUS_MODEL),
            'fish_school': renderer.SnakeRenderer(
                self.block_program,
                FISH_MODEL,
                segment_configs=FISH_SEGMENT_CONFIGS,
                tail_length=FISH_TAIL_LENGTH,
                segment_capacity=FishSchoolEntity.SCHOOL_SIZE,
            ),
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
        self.mosasaurus_enabled = True
        self.fish_school_enabled = True
        self.entities = {
            eid: entity.to_network_dict()
            for eid, entity in self.entity_objects.items()
            if self._entity_is_enabled(entity)
        }
        self._entity_persist_interval = 5.0
        self._entity_persist_timer = self._entity_persist_interval
        self._persist_entity_states()
        self._entity_net_interval = float(getattr(config, "ENTITY_NET_INTERVAL", 0.1))
        self._entity_snapshot_interval = float(getattr(config, "ENTITY_SNAPSHOT_INTERVAL", 1.0))
        self._entity_net_timer = 0.0
        self._entity_snapshot_timer = 0.0
        self._player_net_interval = float(getattr(config, "PLAYER_NET_INTERVAL", 0.1))
        self._player_net_timer = 0.0
        self._player_name_sent = False
        self._player_name_labels = {}
        self.frame_id = 0


        # Texture atlas for UI previews.
        self.texture_atlas = image.load(TEXTURE_PATH)

        # The label that is displayed in the top left of the canvas.
        self.label = pyglet.text.Label('', font_name='Consolas', font_size=18,
            x=10, y=self.height - 10, anchor_x='left', anchor_y='top',
            color=(255, 255, 255, 255))
        self.hud_info_label = pyglet.text.Label('', font_name='Consolas', font_size=14,
            x=10, y=self.height - 10 - self.label.content_height - 4,
            anchor_x='left', anchor_y='top',
            color=(255, 255, 255, 255), multiline=True, width=self.width - 20)
        self.sector_label = pyglet.text.Label('', font_name='Consolas', font_size=14,
            x=10, y=self.height - 10 - self.label.content_height - self.hud_info_label.content_height - 8,
            anchor_x='left', anchor_y='top',
            color=(255, 255, 255, 255), multiline=True, width=self.width - 20)
        self.sector_debug_label = pyglet.text.Label('', font_name='Consolas', font_size=14,
            x=10, y=self.height - 10 - self.label.content_height - self.hud_info_label.content_height - self.sector_label.content_height - 12,
            anchor_x='left', anchor_y='top',
            color=(255, 255, 255, 255), multiline=True, width=self.width - 20)
        self.entity_label = pyglet.text.Label('', font_name='Consolas', font_size=14,
            x=10, y=self.height - 10 - self.label.content_height - self.hud_info_label.content_height - self.sector_label.content_height - self.sector_debug_label.content_height - 16,
            anchor_x='left', anchor_y='top',
            color=(255, 255, 255, 255), multiline=True, width=self.width - 20)
        self.keybind_label = pyglet.text.Label('', font_name='Consolas', font_size=14,
            x=10, y=self.height - 10 - self.label.content_height - self.hud_info_label.content_height - self.sector_label.content_height - self.sector_debug_label.content_height - self.entity_label.content_height - 20,
            anchor_x='left', anchor_y='top',
            color=(255, 255, 255, 255), multiline=True, width=self.width - 20)
        self._label_bg = shapes.Rectangle(0, 0, 1, 1, color=(0, 0, 0))
        self._label_bg.opacity = 140
        self._underwater_overlay = shapes.Rectangle(0, 0, 1, 1, color=config.UNDERWATER_COLOR)
        self.last_draw_ms = 0.0
        self.last_update_ms = 0.0
        self._hud_probe_frame = 0
        self._hud_probe_sector = None
        self._hud_probe_void = 'N/A'
        self._hud_probe_mush = 'NA'
        self.hud_visible = True
        self.hud_mode = "minimal"
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

    def _hud_time_of_day(self):
        if not getattr(self, "day_night_enabled", False):
            return "NA"
        phase = float(getattr(self, "day_phase", 0.0)) % 1.0
        hour = ((phase + 0.5) * 24.0) % 24.0
        if hour >= 23.0 or hour < 1.0:
            return "Midnight"
        if hour < 2.0:
            return "Wolf hour"
        if hour < 4.0:
            return "Owl hour"
        if hour < 6.0:
            return "Daybreak"
        if hour < 7.0:
            return "Sunrise"
        if hour < 10.0:
            return "Early morning"
        if hour < 12.0:
            return "Late morning"
        if hour < 14.0:
            return "Midday"
        if hour < 16.0:
            return "Early afternoon"
        if hour < 18.0:
            return "Late afternoon"
        if hour < 19.0:
            return "Sunset"
        if hour < 21.0:
            return "Early evening"
        return "Late evening"

    def _seed_word(self, seed):
        if seed is None:
            return "NA"
        try:
            seed_val = int(seed)
        except Exception:
            return "NA"
        consonants = [
            "b", "c", "d", "f", "g", "h", "j", "k",
            "l", "m", "n", "p", "r", "s", "t", "v",
            "w", "y", "z",
        ]
        vowels = ["a", "e", "i", "o", "u"]
        rng = random.Random(seed_val)
        syllables = []
        for _ in range(4):
            syllables.append(rng.choice(consonants) + rng.choice(vowels))
        return "".join(syllables).capitalize()

    def _hud_cardinal(self, yaw_deg):
        dirs = [
            "N", "NNE", "NE", "ENE",
            "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW",
            "W", "WNW", "NW", "NNW",
        ]
        yaw = float(yaw_deg) % 360.0
        idx = int(round(yaw / 22.5)) % 16
        return dirs[idx]

    def _hud_cardinal_text(self, abbrev):
        full = {
            "N": "North",
            "E": "East",
            "S": "South",
            "W": "West",
        }
        return full.get(abbrev, abbrev)

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

    def _pick_sun_tint(self):
        base = getattr(config, "SUN_TINT_BASE", (1.0, 0.55, 0.35))
        variance = getattr(config, "SUN_TINT_VARIANCE", (0.08, 0.08, 0.08))
        tint = []
        for idx in range(3):
            v = base[idx] + random.uniform(-variance[idx], variance[idx])
            tint.append(max(0.0, min(1.0, v)))
        return tuple(tint)

    def _boost_sun_color(self, color):
        boost = getattr(config, "SUN_TINT_BOOST", 1.6)
        return tuple(int(max(0, min(255, c * 255 * boost))) for c in color)

    def _compute_sun_world_dir(self, hour_angle: float) -> np.ndarray:
        """
        Returns unit vector (x,y,z) sun direction in world space using a planar orbit.

        Conventions preserved:
        - hour_angle in [-pi, +pi], with noon at 0
        - sunrise ~ -pi/2, sunset ~ +pi/2 for the stylized cycle
        """
        noon_azim = float(getattr(config, "SUN_PATH_AZIMUTH_DEG", 180.0)) % 360.0
        span = float(getattr(config, "SUN_PATH_SPAN_DEG", 90.0))

        if bool(getattr(config, "SUN_PATH_NOON_NORTH", False)):
            noon_azim = (noon_azim + 180.0) % 360.0

        max_elev_rad = self._sun_max_elev_rad  # use the already-initialized value

        # Anchor directions
        v_noon = _dir_from_az_elev_deg(noon_azim, max_elev_rad)
        v_rise = _dir_from_az_elev_deg(noon_azim - span, 0.0)
        v_set  = _dir_from_az_elev_deg(noon_azim + span, 0.0)

        # Orbit plane normal
        n = _norm(np.cross(v_noon, v_rise))

        # Angular distance from noon to sunrise along the orbit
        t0 = math.acos(_clamp(float(np.dot(v_noon, v_rise)), -1.0, 1.0))

        # Ensure axis orientation: positive theta should move toward sunset
        test = _rodrigues_rotate(v_noon, n, +t0)
        if float(np.dot(test, v_set)) < float(np.dot(v_noon, v_set)):
            n = -n

        # Map hour_angle to u in [-1, +1] across the daylight half, clamp at night.
        # sunrise (-pi/2) => u=-1 ; noon (0) => u=0 ; sunset (+pi/2) => u=+1
        u = (2.0 * hour_angle) / math.pi
        u_day = _clamp(u, -1.0, +1.0)

        theta = u_day * t0
        v = _rodrigues_rotate(v_noon, n, theta)
        return _norm(v)

    def _update_day_night(self, dt):
        if not self.day_night_enabled:
            return

        # Advance time
        if self.time_mode == "10x":
            speed = 10.0
        elif self.time_mode == "0x":
            speed = 0.0
        else:
            speed = 1.0

        if speed != 0.0 and self.day_length > 1e-6:
            self.day_phase = (self.day_phase + (dt * speed) / self.day_length) % 1.0

        # ------------------------------------------------------------------
        # Phase convention preserved: day_phase=0 => NOON.
        # hour_angle in [-pi, +pi], noon at 0, sunset +pi/2, midnight +/-pi, sunrise -pi/2.
        # ------------------------------------------------------------------
        hour_angle = (self.day_phase * 2.0 * math.pi + math.pi) % (2.0 * math.pi) - math.pi
        sun_elev_norm = math.cos(hour_angle)  # identical to previous cos(angle) behavior, but wrapped

        # ------------------------------------------------------------------
        # Visibility + twilight: keep your prior semantics so colors/tints stay tuned
        # NOTE: your SUN_TWILIGHT_RANGE is being treated as a threshold in "cos-space" (as before)
        # ------------------------------------------------------------------
        twilight = float(getattr(config, "SUN_TWILIGHT_RANGE", 0.12))
        twilight = max(twilight, 1e-6)

        twilight_factor = max(0.0, 1.0 - min(1.0, abs(sun_elev_norm) / twilight))
        self._sun_twilight_factor = twilight_factor

        self._sun_visibility = max(0.0, min(1.0, (sun_elev_norm + twilight) / (2.0 * twilight)))
        sun_visibility = self._sun_visibility

        # Diagnostic (keep it aligned to your legacy driver)
        self._sun_elev_raw = sun_elev_norm

        # ------------------------------------------------------------------
        # Sun direction (planar orbit), driven by THE SAME hour_angle
        # ------------------------------------------------------------------
        sun_vec = self._compute_sun_world_dir(hour_angle)
        if sun_vec is None:
            return
        sun_vec = np.asarray(sun_vec, dtype=np.float32)
        n = float(np.linalg.norm(sun_vec))
        if n <= 1e-6:
            return
        sun_vec /= n

        self._sun_world_dir = (float(sun_vec[0]), float(sun_vec[1]), float(sun_vec[2]))

        # ------------------------------------------------------------------
        # Day factor curve (keep legacy semantics if you want)
        # If you previously used 0.5+0.5*cos(...) this matches it.
        # ------------------------------------------------------------------
        day_factor = max(0.0, min(1.0, 0.5 + 0.5 * sun_elev_norm))
        curve = float(getattr(config, "DAY_LIGHT_CURVE", 1.0))
        if curve != 1.0:
            day_factor = day_factor ** curve

        # ------------------------------------------------------------------
        # Lighting & sky parameters: blend using sun_visibility (legacy semantics)
        # ------------------------------------------------------------------
        day_ambient = float(getattr(config, "DAY_AMBIENT_LIGHT", getattr(config, "AMBIENT_LIGHT", 0.1)))
        night_ambient = float(getattr(config, "NIGHT_AMBIENT_LIGHT", day_ambient))
        self._day_ambient = night_ambient + (day_ambient - night_ambient) * sun_visibility

        day_sky = float(getattr(config, "DAY_SKY_INTENSITY", getattr(config, "SKY_INTENSITY", 1.0)))
        night_sky = float(getattr(config, "NIGHT_SKY_INTENSITY", day_sky))
        self._day_sky_intensity = night_sky + (day_sky - night_sky) * sun_visibility

        # ------------------------------------------------------------------
        # Fog color with twilight tint (legacy timing)
        # ------------------------------------------------------------------
        day_fog = getattr(config, "DAY_FOG_COLOR", (0.5, 0.69, 1.0))
        night_fog = getattr(config, "NIGHT_FOG_COLOR", day_fog)
        tint = getattr(self, "_sun_tint", (1.0, 0.55, 0.35))

        base_fog = (
            night_fog[0] + (day_fog[0] - night_fog[0]) * sun_visibility,
            night_fog[1] + (day_fog[1] - night_fog[1]) * sun_visibility,
            night_fog[2] + (day_fog[2] - night_fog[2]) * sun_visibility,
        )

        self._day_fog_color = (
            base_fog[0] * (1.0 - twilight_factor) + tint[0] * twilight_factor,
            base_fog[1] * (1.0 - twilight_factor) + tint[1] * twilight_factor,
            base_fog[2] * (1.0 - twilight_factor) + tint[2] * twilight_factor,
        )

        # ------------------------------------------------------------------
        # Daylight direction for shading:
        # Follow sun direction above the horizon; at night blend to straight-up.
        # Use sun_visibility (legacy timing) for the blend.
        # ------------------------------------------------------------------
        day_dir = np.array([sun_vec[0], max(0.0, sun_vec[1]), sun_vec[2]], dtype=np.float32)
        dn = float(np.linalg.norm(day_dir))
        if dn > 1e-6:
            day_dir /= dn
        else:
            day_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        night_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        mix = night_dir * (1.0 - sun_visibility) + day_dir * sun_visibility
        mn = float(np.linalg.norm(mix))
        if mn > 1e-6:
            mix /= mn
        else:
            mix = night_dir

        self._day_light_dir = self._normalize_light_dir(float(mix[0]), float(mix[1]), float(mix[2]))

    def _player_back_orient(self):
        """Return orientation for wall-attached blocks based on player facing."""
        dx, _, dz = self.get_sight_vector()
        if abs(dx) > abs(dz):
            return ORIENT_WEST if dx > 0 else ORIENT_EAST
        return ORIENT_NORTH if dz > 0 else ORIENT_SOUTH

    def _multiplayer_active(self):
        return self.model is not None and self.model.server is not None

    def _is_multiplayer_host(self):
        return self._multiplayer_active() and self.model.is_host()

    def _is_multiplayer_client(self):
        return self._multiplayer_active() and not self.model.is_host()

    def _queue_entity_batch(self, message, states):
        if not self._is_multiplayer_host() or not states:
            return
        payload = entity_codec.pack_entity_batch(states)
        self.model.queue_server_message(message, payload)

    def _queue_entity_spawn(self, entity_state):
        self._queue_entity_batch("entity_spawn", [entity_state])

    def _queue_entity_despawn(self, entity_id):
        if not self._is_multiplayer_host():
            return
        self.model.queue_server_message("entity_despawn", {"ids": [int(entity_id)]})

    def _network_entity_tick(self, dt):
        if not self._is_multiplayer_host():
            return
        self._entity_net_timer += dt
        self._entity_snapshot_timer += dt
        if getattr(self.model, "entity_snapshot_requested", False):
            self.model.entity_snapshot_requested = False
            snapshot = [
                state for state in self.entities.values()
                if state.get("type") != "player"
            ]
            self._queue_entity_batch("entity_snapshot", snapshot)
        if self._entity_net_interval > 0 and self._entity_net_timer >= self._entity_net_interval:
            self._entity_net_timer %= self._entity_net_interval
            updates = [
                state for state in self.entities.values()
                if state.get("type") != "player"
            ]
            self._queue_entity_batch("entity_update", updates)
        if self._entity_snapshot_interval > 0 and self._entity_snapshot_timer >= self._entity_snapshot_interval:
            self._entity_snapshot_timer %= self._entity_snapshot_interval
            snapshot = [
                state for state in self.entities.values()
                if state.get("type") != "player"
            ]
            self._queue_entity_batch("entity_snapshot", snapshot)

    def _network_player_tick(self, dt):
        if not self._multiplayer_active():
            return
        accepted_name = getattr(self.model, "accepted_player_name", None)
        if accepted_name and accepted_name != self.player_entity.name:
            self.player_entity.name = accepted_name
            config.PLAYER_NAME = accepted_name
            self._player_name_sent = True
        if self.player_entity.name != getattr(config, "PLAYER_NAME", self.player_entity.name):
            self.player_entity.name = getattr(config, "PLAYER_NAME", self.player_entity.name)
            self._player_name_sent = False
            self.model.name_sent_once = False
        self._player_net_timer += dt
        if (not self._player_name_sent
                and self.model.player is not None
                and not self.model.name_request_pending
                and not self.model.name_sent_once):
            self.model.queue_server_message("set_name", self.player_entity.name)
            self._player_name_sent = True
            self.model.name_request_pending = True
            self.model.name_sent_once = True
        if getattr(self.model, "player_state_request", False):
            self.model.player_state_request = False
            state = self.player_entity.serialize_state()
            state["camera_rot"] = tuple(self.rotation)
            state["camera_mode"] = self.camera_mode
            state["third_person_distance"] = self.third_person_distance
            self.model.queue_server_message("set_position", [state.get("pos"), state.get("rot"), state])
        if self._player_net_interval <= 0:
            return
        if self._player_net_timer < self._player_net_interval:
            return
        self._player_net_timer %= self._player_net_interval
        state = self.player_entity.serialize_state()
        state["camera_rot"] = tuple(self.rotation)
        state["camera_mode"] = self.camera_mode
        state["third_person_distance"] = self.third_person_distance
        self.model.queue_server_message("set_position", [state.get("pos"), state.get("rot"), state])

    def _remote_player_states(self):
        if self.model is None:
            return {}
        states = {}
        now = time.perf_counter()
        interp = {}
        if hasattr(self.model, "get_interpolated_remote_players"):
            interp = self.model.get_interpolated_remote_players(now=now)
        for player_id, info in getattr(self.model, "remote_players", {}).items():
            pos = info.get("position", (0.0, 0.0, 0.0))
            rot = info.get("rotation", (0.0, 0.0))
            vel = info.get("velocity", (0.0, 0.0, 0.0))
            name = info.get("name", f"Player {player_id}")
            interp_info = interp.get(player_id)
            if interp_info:
                pos = interp_info.get("position", pos)
                rot = interp_info.get("rotation", rot)
                vel = interp_info.get("velocity", vel)
                name = interp_info.get("name", name)
            moving = np.linalg.norm(np.asarray(vel, dtype=float)) > 0.1
            states[player_id] = {
                "id": player_id,
                "type": "player",
                "pos": pos,
                "rot": rot,
                "vel": vel,
                "on_ground": False,
                "animation": "walk" if moving else "idle",
                "name": name,
            }
        return states

    def _sync_player_id_from_server(self):
        if self.model is None or self.model.player is None:
            return
        server_id = self.model.player.id
        if server_id is None:
            return
        if self.player_entity.id == server_id:
            return
        old_id = self.player_entity.id
        self.player_entity.id = server_id
        if old_id in self.entity_objects:
            self.entity_objects.pop(old_id, None)
        self.entity_objects[self.player_entity.id] = self.player_entity

    def _apply_pending_spawn_state(self):
        if self.model is None:
            return
        state = getattr(self.model, "pending_spawn_state", None)
        if not state:
            return
        pos = state.get("pos")
        rot = state.get("rot")
        camera_rot = state.get("camera_rot")
        vel = state.get("vel")
        flying = state.get("flying")
        on_ground = state.get("on_ground")
        camera_mode = state.get("camera_mode")
        camera_dist = state.get("third_person_distance")
        if pos is not None:
            self.player_entity.position = np.array(pos, dtype=float)
            self.position = tuple(self.player_entity.position)
        if rot is not None:
            self.player_entity.rotation = np.array(rot, dtype=float)
            if camera_rot is None:
                self.rotation = (-float(rot[0]), float(rot[1]))
        if camera_rot is not None:
            self.rotation = tuple(camera_rot)
        if camera_mode is not None:
            self.camera_mode = camera_mode
        if camera_dist is not None:
            self.third_person_distance = float(camera_dist)
        if vel is not None:
            self.player_entity.velocity = np.array(vel, dtype=float)
        else:
            self.player_entity.velocity[:] = 0.0
        if flying is not None:
            self.player_entity.flying = bool(flying)
            self.flying = bool(flying)
        if on_ground is not None:
            self.player_entity.on_ground = bool(on_ground)
        self.model.pending_spawn_state = None

    def _apply_entity_seed(self):
        if not self._is_multiplayer_host():
            return
        seed = getattr(self.model, "pending_entity_seed", None)
        if not seed:
            return
        states = entity_codec.unpack_entity_batch(seed)
        if not states:
            self.model.pending_entity_seed = None
            return
        self.entity_objects = {self.player_entity.id: self.player_entity}
        self._entity_sector_map = {}
        self._sector_entity_state = {}
        max_id = self.player_entity.id
        for state in states:
            etype = state.get("type")
            pos = state.get("pos", (0.0, 0.0, 0.0))
            rot = state.get("rot", (0.0, 0.0))
            ent_id = int(state.get("id", 0))
            entity = None
            if etype == "snake":
                entity = SnakeEntity(self.model, player_position=pos, entity_id=ent_id)
            elif etype == "snail":
                entity = SnailEntity(self.model, player_position=pos, entity_id=ent_id)
            elif etype == "seagull":
                entity = SeagullEntity(self.model, player_position=pos, entity_id=ent_id)
            elif etype == "dog":
                entity = Dog(self.model, entity_id=ent_id, saved_state={"pos": pos, "rot": rot})
            elif etype == "dinotrex":
                entity = DinoTrexEntity(self.model, entity_id=ent_id, saved_state={"pos": pos, "rot": rot})
            elif etype == "mosasaurus":
                entity = MosasaurusEntity(self.model, player_position=pos, entity_id=ent_id, saved_state={"pos": pos, "rot": rot})
            elif etype == "fish_school":
                entity = FishSchoolEntity(self.model, player_position=pos, entity_id=ent_id, saved_state={"pos": pos, "rot": rot})
            if entity is None:
                continue
            entity.id = ent_id
            entity.position = np.array(pos, dtype=float)
            entity.rotation = np.array(rot, dtype=float)
            entity.velocity = np.array(state.get("vel", (0.0, 0.0, 0.0)), dtype=float)
            entity.current_animation = state.get("animation", "idle")
            if etype == "snake" or etype == "fish_school":
                segs = state.get("segment_positions")
                if segs:
                    entity.segment_positions = np.array(segs, dtype=float)
            self.entity_objects[ent_id] = entity
            max_id = max(max_id, ent_id)
        self._next_entity_id = max_id + 1
        self.entities = {eid: ent.to_network_dict() for eid, ent in self.entity_objects.items()}
        self.model.pending_entity_seed = None

    def _world_to_screen(self, world_pos, projection, view, eye_world):
        width, height = self.get_size()
        rel = np.array(world_pos, dtype=float) - np.array(eye_world, dtype=float)
        vec = np.array([rel[0], rel[1], rel[2], 1.0], dtype=float)
        view_mat = np.array(view, dtype=float).reshape(4, 4)
        proj_mat = np.array(projection, dtype=float).reshape(4, 4)
        clip = proj_mat @ (view_mat @ vec)
        w = clip[3]
        if w <= 1e-6:
            return None
        ndc = clip[:3] / w
        if ndc[2] < -1.0 or ndc[2] > 1.0:
            return None
        sx = (ndc[0] * 0.5 + 0.5) * width
        sy = (ndc[1] * 0.5 + 0.5) * height
        if sx < 0 or sx > width or sy < 0 or sy > height:
            return None
        return sx, sy

    def _draw_player_name_labels(self):
        if not self._multiplayer_active():
            return
        projection, view, eye_world = self.get_view_projection()
        remote_states = self._remote_player_states()
        alive_ids = set()
        for player_id, state in remote_states.items():
            alive_ids.add(player_id)
            name = state.get("name", f"Player {player_id}")
            pos = state.get("pos", (0.0, 0.0, 0.0))
            label_pos = (pos[0], pos[1] + float(PLAYER_HEIGHT) + 0.4, pos[2])
            screen = self._world_to_screen(label_pos, projection, view, eye_world)
            if screen is None:
                continue
            label = self._player_name_labels.get(player_id)
            if label is None:
                label = pyglet.text.Label(
                    name,
                    font_name='Consolas',
                    font_size=10,
                    color=(255, 255, 255, 255),
                    anchor_x='center',
                    anchor_y='bottom',
                )
                self._player_name_labels[player_id] = label
            label.text = name
            label.x, label.y = screen
            label.draw()
        for stale_id in list(self._player_name_labels.keys()):
            if stale_id not in alive_ids:
                self._player_name_labels.pop(stale_id, None)

    def _compute_sun_screen_pos(self, projection, view, eye_world, allow_below=False):
        if not getattr(config, "SUN_ENABLED", True):
            return None
        sun_dir = getattr(self, "_sun_world_dir", None)
        if sun_dir is None:
            sun_dir = getattr(self, "_day_light_dir", getattr(config, "SUN_LIGHT_DIR", (0.35, 1.0, 0.65)))
            sun_dir = self._normalize_light_dir(sun_dir[0], sun_dir[1], sun_dir[2])
        if not allow_below and sun_dir[1] <= 0.0:
            return None
        distance = float(getattr(config, "SUN_DISTANCE", 400.0))
        sun_world = (
            eye_world[0] + sun_dir[0] * distance,
            eye_world[1] + sun_dir[1] * distance,
            eye_world[2] + sun_dir[2] * distance,
        )
        screen = self._world_to_screen(sun_world, projection, view, eye_world)
        if screen is not None:
            return (screen[0], screen[1], sun_dir[1])
        # Fallback: project view-space direction to screen.
        width, height = self.get_size()
        view_mat = np.array(list(view), dtype='f4').reshape((4, 4), order='F')
        proj_mat = np.array(list(projection), dtype='f4').reshape((4, 4), order='F')
        view_dir = view_mat @ np.array([sun_dir[0], sun_dir[1], sun_dir[2], 0.0], dtype='f4')
        if view_dir[2] >= -1e-6:
            return None
        clip = proj_mat @ np.array([view_dir[0], view_dir[1], view_dir[2], 1.0], dtype='f4')
        w = clip[3]
        if abs(w) <= 1e-6:
            return None
        ndc = clip[:3] / w
        sx = (ndc[0] * 0.5 + 0.5) * width
        sy = (ndc[1] * 0.5 + 0.5) * height
        if sx < 0 or sx > width or sy < 0 or sy > height:
            return None
        return (sx, sy, sun_dir[1])

    def _draw_sky_gradient(self):
        if not getattr(self, "day_night_enabled", False):
            return
        projection, view, eye_world = self.get_view_projection()
        pos = self._compute_sun_screen_pos(projection, view, eye_world, allow_below=True)
        if pos is None:
            return
        sx, sy, _ = pos
        width, height = self.get_size()
        base = getattr(self, "_day_fog_color", (0.5, 0.69, 1.0))
        tint = getattr(self, "_sun_tint", (1.0, 0.55, 0.35))
        strength = max(0.0, min(1.0, getattr(self, "_sun_twilight_factor", 0.0)))
        strength = math.sqrt(strength)
        strength *= float(getattr(config, "SUN_GRADIENT_STRENGTH", 1.0))
        strength = max(0.0, min(1.0, strength))
        center = (
            base[0] * (1.0 - strength) + tint[0] * strength,
            base[1] * (1.0 - strength) + tint[1] * strength,
            base[2] * (1.0 - strength) + tint[2] * strength,
        )
        center_rgb = [int(max(0, min(255, c * 255))) for c in center]
        outer_rgb = [int(max(0, min(255, c * 255))) for c in base]
        def _to_ndc(px, py):
            return (
                (px / float(width)) * 2.0 - 1.0,
                (py / float(height)) * 2.0 - 1.0,
            )
        cx, cy = _to_ndc(sx, sy)
        c0x, c0y = _to_ndc(0.0, 0.0)
        c1x, c1y = _to_ndc(width, 0.0)
        c2x, c2y = _to_ndc(width, height)
        c3x, c3y = _to_ndc(0.0, height)
        positions = [
            cx, cy, 0.0,
            c0x, c0y, 0.0,
            c1x, c1y, 0.0,
            c2x, c2y, 0.0,
            c3x, c3y, 0.0,
            c0x, c0y, 0.0,
        ]
        colors = center_rgb + [0]
        for _ in range(5):
            colors += outer_rgb + [0]
        normals = [0.0, 0.0, 1.0] * 6
        tex = [0.0, 0.0] * 6
        light = [1.0, 1.0] * 6
        prev_use_tex = self.block_program["u_use_texture"]
        prev_use_color = self.block_program["u_use_vertex_color"]
        prev_water_pass = self.block_program["u_water_pass"]
        prev_ambient = self.block_program["u_ambient_light"]
        prev_light_dir = self.block_program["u_light_dir"]
        prev_light_exp = self.block_program["u_light_dir_exp"]
        prev_projection = self.block_program["u_projection"]
        prev_view = self.block_program["u_view"]
        prev_camera = self.block_program["u_camera_pos"]
        prev_model = self.block_program["u_model"]
        self.block_program.bind()
        self.block_program["u_use_texture"] = False
        self.block_program["u_use_vertex_color"] = True
        self.block_program["u_water_pass"] = False
        self.block_program["u_ambient_light"] = 1.0
        self.block_program["u_light_dir"] = (0.0, 0.0, 1.0)
        self.block_program["u_light_dir_exp"] = 1.0
        self.block_program["u_projection"] = Mat4()
        self.block_program["u_view"] = Mat4()
        self.block_program["u_camera_pos"] = (0.0, 0.0, 0.0)
        self.block_program["u_model"] = Mat4()
        vl = self.block_program.vertex_list(
            6,
            gl.GL_TRIANGLE_FAN,
            position=('f', np.array(positions, dtype='f4')),
            tex_coords=('f', np.array(tex, dtype='f4')),
            normal=('f', np.array(normals, dtype='f4')),
            color=('f', np.array(colors, dtype='f4')),
            light=('f', np.array(light, dtype='f4')),
        )
        vl.draw(gl.GL_TRIANGLE_FAN)
        vl.delete()
        self.block_program["u_use_texture"] = prev_use_tex
        self.block_program["u_use_vertex_color"] = prev_use_color
        self.block_program["u_water_pass"] = prev_water_pass
        self.block_program["u_ambient_light"] = prev_ambient
        self.block_program["u_light_dir"] = prev_light_dir
        self.block_program["u_light_dir_exp"] = prev_light_exp
        self.block_program["u_projection"] = prev_projection
        self.block_program["u_view"] = prev_view
        self.block_program["u_camera_pos"] = prev_camera
        self.block_program["u_model"] = prev_model
        self.block_program.unbind()

    def _draw_sun_quad(self, center, right, up, size, color, alpha):
        half = size * 0.5
        p1 = center - right * half - up * half
        p2 = center + right * half - up * half
        p3 = center + right * half + up * half
        p4 = center - right * half + up * half
        positions = [
            p1.x, p1.y, p1.z,
            p2.x, p2.y, p2.z,
            p3.x, p3.y, p3.z,
            p1.x, p1.y, p1.z,
            p3.x, p3.y, p3.z,
            p4.x, p4.y, p4.z,
        ]
        tex = [0.0, 0.0] * 6
        forward = Vec3(*self.get_sight_vector()).normalize()
        normal = (-forward.x, -forward.y, -forward.z)
        normals = list(normal) * 6
        color_vals = [float(color[0]), float(color[1]), float(color[2]), 0.0]
        colors = color_vals * 6
        light = [1.0, 1.0] * 6
        vl = self.block_program.vertex_list(
            6,
            gl.GL_TRIANGLES,
            position=('f', np.array(positions, dtype='f4')),
            tex_coords=('f', np.array(tex, dtype='f4')),
            normal=('f', np.array(normals, dtype='f4')),
            color=('f', np.array(colors, dtype='f4')),
            light=('f', np.array(light, dtype='f4')),
        )
        prev_alpha = self.block_program["u_water_alpha"]
        self.block_program["u_water_alpha"] = alpha / 255.0
        vl.draw(gl.GL_TRIANGLES)
        vl.delete()
        self.block_program["u_water_alpha"] = prev_alpha

    def _draw_sun(self):
        if not getattr(config, "SUN_ENABLED", True):
            return
        visibility = float(getattr(self, "_sun_visibility", 1.0))
        if visibility <= 1e-6:
            return
        projection, view, eye_world = self.get_view_projection()
        sun_dir = getattr(self, "_sun_world_dir", None)
        if sun_dir is None:
            sun_dir = getattr(self, "_day_light_dir", getattr(config, "SUN_LIGHT_DIR", (0.35, 1.0, 0.65)))
        sun_dir = Vec3(*sun_dir).normalize()
        elev = max(0.0, min(1.0, sun_dir.y))
        size = float(getattr(config, "SUN_SIZE", 0.04))
        horizon_scale = float(getattr(config, "SUN_HORIZON_SCALE", 0.3))
        scale = 1.0 + horizon_scale * (1.0 - max(0.0, min(1.0, elev)))
        distance = float(getattr(config, "SUN_DISTANCE", 400.0))
        fov_deg = 65.0
        half_angle = math.radians(fov_deg * size * 0.5)
        base = distance * math.tan(half_angle) * 2.0
        sun_size = base * scale
        center = Vec3(eye_world[0], eye_world[1], eye_world[2]) + sun_dir * distance
        forward = Vec3(*self.get_sight_vector()).normalize()
        up = Vec3(0.0, 1.0, 0.0)
        right = forward.cross(up)
        right_mag = math.sqrt(right.x * right.x + right.y * right.y + right.z * right.z)
        if right_mag < 1e-6:
            up = Vec3(0.0, 0.0, 1.0)
            right = forward.cross(up)
            right_mag = math.sqrt(right.x * right.x + right.y * right.y + right.z * right.z)
            if right_mag < 1e-6:
                return
        right = right.normalize()
        up = right.cross(forward).normalize()
        glow_steps = int(getattr(config, "SUN_GLOW_STEPS", 4))
        glow_alpha = int(getattr(config, "SUN_GLOW_ALPHA", 120))
        glow_color = getattr(self, "_sun_tint", (1.0, 0.55, 0.35))
        prev_use_tex = self.block_program["u_use_texture"]
        prev_use_color = self.block_program["u_use_vertex_color"]
        prev_water_pass = self.block_program["u_water_pass"]
        prev_model = self.block_program["u_model"]
        prev_ambient = self.block_program["u_ambient_light"]
        prev_sky = self.block_program["u_sky_intensity"]
        prev_light_dir = self.block_program["u_light_dir"]
        prev_light_exp = self.block_program["u_light_dir_exp"]
        prev_fog_start = self.block_program["u_fog_start"]
        prev_fog_end = self.block_program["u_fog_end"]
        self.block_program["u_use_texture"] = False
        self.block_program["u_use_vertex_color"] = True
        self.block_program["u_water_pass"] = True
        self.block_program["u_model"] = Mat4()
        self.block_program["u_ambient_light"] = 1.0
        self.block_program["u_sky_intensity"] = 1.0
        self.block_program["u_light_dir"] = (0.0, 0.0, 1.0)
        self.block_program["u_light_dir_exp"] = 1.0
        self.block_program["u_fog_start"] = 1.0e9
        self.block_program["u_fog_end"] = 1.0e9 + 1.0
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDepthMask(gl.GL_FALSE)
        for idx in range(glow_steps, 0, -1):
            factor = idx / float(glow_steps)
            extent = sun_size * (1.0 + 2.2 * factor)
            alpha = int(glow_alpha * factor * factor * visibility)
            color = self._boost_sun_color(glow_color)
            self._draw_sun_quad(center, right, up, extent, color, alpha)
        color = getattr(self, "_sun_tint", (1.0, 0.55, 0.35))
        color = self._boost_sun_color(color)
        self._draw_sun_quad(center, right, up, sun_size, color, int(200 * visibility))
        self.block_program["u_use_texture"] = prev_use_tex
        self.block_program["u_use_vertex_color"] = prev_use_color
        self.block_program["u_water_pass"] = prev_water_pass
        self.block_program["u_model"] = prev_model
        self.block_program["u_ambient_light"] = prev_ambient
        self.block_program["u_sky_intensity"] = prev_sky
        self.block_program["u_light_dir"] = prev_light_dir
        self.block_program["u_light_dir_exp"] = prev_light_exp
        self.block_program["u_fog_start"] = prev_fog_start
        self.block_program["u_fog_end"] = prev_fog_end
        gl.glDisable(gl.GL_BLEND)
        gl.glDepthMask(gl.GL_TRUE)

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
        self._sync_player_id_from_server()
        self._apply_pending_spawn_state()
        self._apply_entity_seed()
        if not self._is_multiplayer_client():
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
        self._network_entity_tick(dt)
        self._network_player_tick(dt)
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
        if entity_type == "mosasaurus":
            return self.mosasaurus_enabled
        if entity_type == "fish_school":
            return self.fish_school_enabled
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
        elif entity_type == "dinotrex":
            entity = DinoTrexEntity(self.model, entity_id=entity_id, saved_state={"pos": spawn_pos, "rot": (0.0, 0.0)})
            entity.snap_to_ground()
        elif entity_type == "mosasaurus":
            entity = MosasaurusEntity(self.model, player_position=spawn_pos, entity_id=entity_id)
        elif entity_type == "fish_school":
            entity = FishSchoolEntity(self.model, player_position=spawn_pos, entity_id=entity_id)
        else:
            return None
        self.entity_objects[entity_id] = entity
        self._queue_entity_spawn(entity.to_network_dict())
        return entity

    def _spawn_entity_at(self, entity_type, spawn_pos, rotation=(0.0, 0.0), snap_to_ground=False):
        if entity_type is None or spawn_pos is None:
            return None
        entity_id = self._next_entity_id
        self._next_entity_id += 1
        spawn_pos = (float(spawn_pos[0]), float(spawn_pos[1]), float(spawn_pos[2]))
        rot = (float(rotation[0]), float(rotation[1]))
        entity = None
        if entity_type == "snake":
            entity = SnakeEntity(self.model, player_position=spawn_pos, entity_id=entity_id)
        elif entity_type == "snail":
            entity = SnailEntity(
                self.model,
                player_position=spawn_pos,
                entity_id=entity_id,
                saved_state={"pos": spawn_pos, "rot": rot},
            )
        elif entity_type == "seagull":
            entity = SeagullEntity(
                self.model,
                player_position=spawn_pos,
                entity_id=entity_id,
                saved_state={"pos": spawn_pos, "rot": rot},
            )
        elif entity_type == "dog":
            entity = Dog(self.model, entity_id=entity_id, saved_state={"pos": spawn_pos, "rot": rot})
        elif entity_type == "dinotrex":
            entity = DinoTrexEntity(self.model, entity_id=entity_id, saved_state={"pos": spawn_pos, "rot": rot})
        elif entity_type == "mosasaurus":
            entity = MosasaurusEntity(self.model, player_position=spawn_pos, entity_id=entity_id)
        elif entity_type == "fish_school":
            entity = FishSchoolEntity(self.model, player_position=spawn_pos, entity_id=entity_id)
        if entity is None:
            return None
        if snap_to_ground and entity_type not in ("seagull", "mosasaurus", "fish_school"):
            entity.snap_to_ground()
        self.entity_objects[entity_id] = entity
        self._queue_entity_spawn(entity.to_network_dict())
        return entity

    def _spawn_entity_shortcut(self, entity_type):
        if self.player_entity is None:
            return None
        dx, _, dz = self.get_sight_vector()
        forward = np.array([dx, dz], dtype=float)
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            yaw = float(self.rotation[0])
            forward = np.array(
                [math.cos(math.radians(yaw - 90.0)), math.sin(math.radians(yaw - 90.0))],
                dtype=float,
            )
            norm = np.linalg.norm(forward)
        if norm < 1e-6:
            return None
        forward /= norm
        base_pos = self.player_entity.position
        target_x = float(base_pos[0] + forward[0] * 10.0)
        target_z = float(base_pos[2] + forward[1] * 10.0)
        ground_y = None
        if hasattr(self.model, "find_surface_y"):
            ground_y = self.model.find_surface_y(target_x, target_z)
        if ground_y is None:
            ground_y = float(base_pos[1])
        spawn_y = float(ground_y + (10.0 if entity_type == "seagull" else 0.0))
        spawn_pos = (target_x, spawn_y, target_z)
        snap_to_ground = entity_type != "seagull"
        return self._spawn_entity_at(entity_type, spawn_pos, snap_to_ground=snap_to_ground)

    def _despawn_non_player_entities(self):
        ids = [eid for eid, ent in self.entity_objects.items() if ent is not self.player_entity]
        for eid in ids:
            self._despawn_entity(eid, reset_sector_spawn=False)

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
        self._queue_entity_despawn(entity_id)
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
        multiplayer_client = self._is_multiplayer_client()

        for entity in self.entity_objects.values():
            if not self._entity_is_enabled(entity):
                continue
            if multiplayer_client and entity is not self.player_entity:
                continue
            if not self.model.is_sector_ready(entity.position, radius=0):
                continue
            entity.update(dt, context)
            update_count += 1
            
            updated_entities[entity.id] = entity.to_network_dict()
            if entity is self.player_entity:
                context["player_position"] = entity.position.copy()

        if multiplayer_client:
            now = time.perf_counter()
            self.entities = self.model.get_interpolated_entities(now=now)
            self.entities[self.player_entity.id] = self.player_entity.to_network_dict()
        else:
            self.entities = updated_entities
        remote_players = self._remote_player_states()
        if remote_players:
            self.entities.update(remote_players)
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
        if entity.type == "mosasaurus":
            return self.mosasaurus_enabled
        if entity.type == "fish_school":
            return self.fish_school_enabled
        return True

    def _persist_entity_states(self):
        state = self.player_entity.serialize_state()
        state["camera_rot"] = tuple(self.rotation)
        state["camera_mode"] = self.camera_mode
        state["third_person_distance"] = self.third_person_distance
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
                    if block is None:
                        return
                    support_id = self.model[block]
                    if not BLOCK_SOLID[support_id]:
                        return
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
        if modifiers & key.MOD_CTRL:
            spawn_type = None
            if symbol == key.X:
                self._despawn_non_player_entities()
                return
            if symbol == key.N:
                spawn_type = "snake"
            elif symbol == key.B:
                spawn_type = "snail"
            elif symbol == key.M:
                spawn_type = "seagull"
            elif symbol == key.V:
                spawn_type = "dog"
            elif symbol == key.T:
                spawn_type = "dinotrex"
            elif symbol == key.Y:
                spawn_type = "mosasaurus"
            elif symbol == key.U:
                spawn_type = "fish_school"
            if spawn_type:
                self._spawn_entity_shortcut(spawn_type)
                return
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
        elif symbol == key.F4:
            if modifiers & key.MOD_SHIFT:
                self.day_phase = 0.75
            elif modifiers & key.MOD_CTRL:
                self.day_phase = 0.25
            else:
                if self.time_mode == "1x":
                    self.time_mode = "10x"
                elif self.time_mode == "10x":
                    self.time_mode = "0x"
                else:
                    self.time_mode = "1x"
        elif symbol == key.F5:
            if self.camera_mode == 'first_person':
                self.camera_mode = 'third_person'
            else:
                self.camera_mode = 'first_person'
        elif symbol == key.F6:
            self.model.collisions_enabled = not getattr(self.model, "collisions_enabled", True)
            status = "enabled" if self.model.collisions_enabled else "disabled"
            logutil.log("MAIN", f"Collisions {status}")
        elif symbol == key.F1:
            if not self.hud_visible:
                self.hud_visible = True
                self.hud_mode = "minimal"
            else:
                self.hud_mode = "full" if self.hud_mode == "minimal" else "minimal"
            if self.hud_mode == "minimal":
                self.hud_details_visible = False
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
        if self._multiplayer_active():
            state = self.player_entity.serialize_state()
            state["camera_rot"] = tuple(self.rotation)
            state["camera_mode"] = self.camera_mode
            state["third_person_distance"] = self.third_person_distance
            self.model.queue_server_message("set_position", [state.get("pos"), state.get("rot"), state])
        self.model.quit()
        if hasattr(self, "server_process") and self.server_process is not None:
            self.server_process.terminate()
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
        sun_far = float(getattr(config, "SUN_FAR_PLANE", 0.0))
        far_plane = max(512.0, sun_far)
        projection = Mat4.perspective_projection(aspect, 0.1, far_plane, 65)

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
        self._last_view_matrix = view
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

    def _light_dir_view(self, light_dir):
        if light_dir is None:
            return (0.0, 1.0, 0.0)
        view = getattr(self, "_last_view_matrix", None)
        if view is None:
            return light_dir
        view_mat = np.array(list(view), dtype='f4').reshape((4, 4), order='F')
        vec = np.array([light_dir[0], light_dir[1], light_dir[2], 0.0], dtype='f4')
        out = view_mat @ vec
        mag = math.sqrt(float(out[0] * out[0] + out[1] * out[1] + out[2] * out[2]))
        if mag <= 1e-6:
            return (0.0, 1.0, 0.0)
        return (float(out[0] / mag), float(out[1] / mag), float(out[2] / mag))


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
        self.set_2d()
        self._draw_sky_gradient()
        self.set_3d()

        if self.day_night_enabled:
            self.block_program['u_ambient_light'] = self._day_ambient
            self.block_program['u_sky_intensity'] = self._day_sky_intensity
            self.block_program['u_light_dir'] = self._light_dir_view(self._day_light_dir)
            self.block_program['u_fog_color'] = self._day_fog_color
            self.block_program['u_light_dir_exp'] = getattr(config, 'LIGHT_DIR_EXP', 1.0)
            gl.glClearColor(
                self._day_fog_color[0],
                self._day_fog_color[1],
                self._day_fog_color[2],
                1.0,
            )
        else:
            self.block_program['u_ambient_light'] = getattr(config, 'AMBIENT_LIGHT', 0.0)
            self.block_program['u_sky_intensity'] = getattr(config, 'SKY_INTENSITY', 1.0)
            self.block_program['u_light_dir'] = self._light_dir_view(
                getattr(config, 'SUN_LIGHT_DIR', (0.35, 1.0, 0.65))
            )
            fog_color = getattr(config, 'DAY_FOG_COLOR', (0.5, 0.69, 1.0))
            self.block_program['u_light_dir_exp'] = getattr(config, 'LIGHT_DIR_EXP', 1.0)
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
            if entity_id == self.player_entity.id:
                continue
            r = self.entity_renderers.get(entity_state['type'])
            if r:
                anim = entity_state.get('animation')
                if isinstance(r, renderer.SnakeRenderer):
                    if anim:
                        r.head_renderer.set_animation(anim)
                elif anim:
                    r.set_animation(anim)
                r.draw(entity_state)
        if self.camera_mode == 'third_person':
            local_state = self.player_entity.to_network_dict()
            local_state["pos"] = self.player_entity.position.copy()
            local_state["rot"] = self.player_entity.rotation.copy()
            local_state["animation"] = self.player_entity.current_animation
            r = self.entity_renderers.get(local_state['type'])
            if r:
                anim = local_state.get('animation')
                if anim:
                    r.set_animation(anim)
                r.draw(local_state)
        self.block_program['u_use_texture'] = True
        # self.block_program['u_use_vertex_color'] = True
        entity_draw_ms = (time.perf_counter() - t0) * 1000.0

        self._draw_sun()

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
        self._draw_player_name_labels()
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
        minimal_mode = getattr(self, "hud_mode", "minimal") == "minimal"
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
        if self.hud_details_visible and not minimal_mode:
            refresh_s = float(getattr(config, "HUD_DETAIL_REFRESH_S", 1.0))
            if now - self._hud_detail_last_update >= refresh_s:
                self._hud_detail_last_update = now
                self._hud_detail_dirty = True
        if now - self._hud_block_last_update >= 1.0:
            self._hud_block_last_update = now
            fps = self._current_fps()
            seed_val = getattr(self.model, "world_seed", None)
            seed_text = self._seed_word(seed_val)
            facing = self._hud_cardinal(rx)
            time_text = self._hud_time_of_day()
            rot_text = rx % 360.0
            if minimal_mode:
                facing_text = self._hud_cardinal_text(facing)
                self._hud_block1_text = f"{time_text} on {seed_text} facing {facing_text} (F1 for more)"
                self._hud_block2_text = ""
                self._hud_block3_text = ""
                self._hud_block4_text = ""
                self._hud_block5_text = ""
            else:
                self._hud_block1_text = (
                    'FPS(%.1f), pos(%.2f, %.2f, %.2f) sector(%d, 0, %d) rot (%s, %.1f, %.1f) '
                    '%s on %s' % (
                        fps, x, y, z, sector[0], sector[2], facing, rot_text, ry, time_text, seed_text
                    )
                )
            if hud_profile:
                _profile_mark("build_block1")
            if minimal_mode or not self.hud_details_visible:
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
        if self.hud_details_visible and not minimal_mode:
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
        coll_state = "on" if getattr(self.model, "collisions_enabled", True) else "off"
        time_mode_text = getattr(self, "time_mode", "1x")
        if minimal_mode:
            keybind_text = ""
        else:
            keybind_text = (
                "Toggles: (F1)HUD=%s (F2)Vsync=%s (F3)Debug=%s (F4)Time=%s (F5)Cam=%s (F6)Coll=%s (F8)Copy | "
                "(V)Dog=%s (B)Snail=%s S(N)nake=%s (M)Seagull=%s"
                % (
                    hud_state,
                    vsync_state,
                    details_state,
                    time_mode_text,
                    camera_state,
                    coll_state,
                    dog_state,
                    snail_state,
                    snake_state,
                    seagull_state,
                )
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
            bottom = self.keybind_label.y - self.keybind_label.content_height if not minimal_mode else (self.label.y - self.label.content_height)
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
                ) if not minimal_mode else self.label.content_width
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
        if not self.exclusive or self.camera_mode != 'first_person':
            return

        vector = self.get_sight_vector()
        _, _, eye = self.get_view_projection()  # eye is Vec3
        hit_origin = (eye.x, eye.y, eye.z)
        block, previous = self.model.hit_test(hit_origin, vector)
        if not block:
            return

        block_id = self.model[block]
        if block_id is None or block_id == 0:
            return

        self.set_3d()
        self._ensure_focus_vlists()

        block_pos = np.array(block, dtype='f4')
        render_min = BLOCK_RENDER_AABB_MIN[block_id]
        render_max = BLOCK_RENDER_AABB_MAX[block_id]
        outline_pad = getattr(config, "FOCUSED_BLOCK_OUTLINE_PAD", 0.02)
        outline_min = render_min - outline_pad
        outline_max = render_max + outline_pad
        outline_world_min = block_pos + outline_min
        outline_world_max = block_pos + outline_max
        outline_positions = self._build_focus_outline_positions(outline_world_min, outline_world_max)
        self._focus_outline_vlist.position[:] = outline_positions.ravel()

        self.block_program.bind()
        self.block_program['u_use_texture'] = False
        self.block_program['u_use_vertex_color'] = True
        self._focus_outline_vlist.draw(gl.GL_LINES)

        if previous is not None and BLOCK_SOLID[block_id]:
            face = (
                previous[0] - block[0],
                previous[1] - block[1],
                previous[2] - block[2],
            )
            if abs(face[0]) + abs(face[1]) + abs(face[2]) == 1:
                face_pad = getattr(config, "FOCUSED_BLOCK_FACE_PAD", 0.03)
                face_positions, face_normal = self._build_focus_face_cross_positions(
                    block_pos + render_min,
                    block_pos + render_max,
                    face,
                    face_pad,
                )
                self._focus_face_vlist.position[:] = face_positions.ravel()
                self._focus_face_vlist.normal[:] = np.tile(face_normal, (4, 1)).ravel()
                self._focus_face_vlist.draw(gl.GL_LINES)

        self.block_program['u_use_texture'] = True
        self.block_program.unbind()

    def _ensure_focus_vlists(self):
        if self._focus_outline_vlist is None:
            outline_pos = np.zeros((24, 3), dtype='f4')
            outline_tex = np.zeros((24, 2), dtype='f4')
            outline_norm = np.tile(np.array([0.0, 1.0, 0.0], dtype='f4'), (24, 1))
            outline_color = np.tile(
                np.array(getattr(config, "FOCUSED_BLOCK_OUTLINE_COLOR", (255, 255, 255, 255)), dtype='f4'),
                (24, 1),
            )
            outline_light = np.ones((24, 2), dtype='f4')
            self._focus_outline_vlist = self.block_program.vertex_list(
                24,
                gl.GL_LINES,
                position=('f', outline_pos.ravel()),
                tex_coords=('f', outline_tex.ravel()),
                normal=('f', outline_norm.ravel()),
                color=('f', outline_color.ravel()),
                light=('f', outline_light.ravel()),
            )
        if self._focus_face_vlist is None:
            face_pos = np.zeros((4, 3), dtype='f4')
            face_tex = np.zeros((4, 2), dtype='f4')
            face_norm = np.tile(np.array([0.0, 1.0, 0.0], dtype='f4'), (4, 1))
            face_color = np.tile(
                np.array(getattr(config, "FOCUSED_BLOCK_FACE_COLOR", (255, 210, 80, 255)), dtype='f4'),
                (4, 1),
            )
            face_light = np.ones((4, 2), dtype='f4')
            self._focus_face_vlist = self.block_program.vertex_list(
                4,
                gl.GL_LINES,
                position=('f', face_pos.ravel()),
                tex_coords=('f', face_tex.ravel()),
                normal=('f', face_norm.ravel()),
                color=('f', face_color.ravel()),
                light=('f', face_light.ravel()),
            )

    def _build_focus_outline_positions(self, min_v, max_v):
        x0, y0, z0 = min_v
        x1, y1, z1 = max_v
        corners = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ],
            dtype='f4',
        )
        return corners[FOCUSED_BLOCK_EDGES].reshape(-1, 3)

    def _build_focus_face_cross_positions(self, min_v, max_v, face, pad):
        dx, dy, dz = face
        x0, y0, z0 = min_v
        x1, y1, z1 = max_v
        if dx != 0:
            x = (x0 if dx < 0 else x1) + dx * pad
            ym = 0.5 * (y0 + y1)
            zm = 0.5 * (z0 + z1)
            positions = np.array(
                [[x, ym, z0], [x, ym, z1], [x, y0, zm], [x, y1, zm]],
                dtype='f4',
            )
        elif dy != 0:
            y = (y0 if dy < 0 else y1) + dy * pad
            xm = 0.5 * (x0 + x1)
            zm = 0.5 * (z0 + z1)
            positions = np.array(
                [[x0, y, zm], [x1, y, zm], [xm, y, z0], [xm, y, z1]],
                dtype='f4',
            )
        else:
            z = (z0 if dz < 0 else z1) + dz * pad
            xm = 0.5 * (x0 + x1)
            ym = 0.5 * (y0 + y1)
            positions = np.array(
                [[x0, ym, z], [x1, ym, z], [xm, y0, z], [xm, y1, z]],
                dtype='f4',
            )
        return positions, np.array([dx, dy, dz], dtype='f4')

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


def _parse_server_arg(arg):
    if arg == 'LAN':
        return server_module.get_network_ip(), config.SERVER_PORT
    if ':' in arg:
        host, port = arg.split(':', 1)
        try:
            port_val = int(port)
        except ValueError:
            port_val = config.SERVER_PORT
        return host, port_val
    return arg, config.SERVER_PORT


def main():
    logutil.log("MAIN", f"minepy2 start {datetime.now().isoformat(sep=' ', timespec='seconds')}")
    server_proc = None
    if len(sys.argv)>1:
        arg = sys.argv[1]
        if arg == '-serve':
            if len(sys.argv) < 3:
                print("Usage: python main.py -serve <LAN|address[:port]> <name>")
                return
            addr_arg = sys.argv[2]
            name_arg = sys.argv[3] if len(sys.argv) > 3 else None
            host, port = _parse_server_arg(addr_arg)
            config.SERVER_IP = host
            config.SERVER_PORT = port
            if name_arg:
                config.PLAYER_NAME = name_arg
            server_proc = multiprocessing.Process(
                target=server_module.start_server,
                args=(host, port),
                daemon=True,
            )
            server_proc.start()
            logutil.log("MAIN", f"Hosting server at {config.SERVER_IP}:{config.SERVER_PORT}")
        else:
            name_arg = sys.argv[2] if len(sys.argv) > 2 else None
            host, port = _parse_server_arg(arg)
            config.SERVER_IP = host
            config.SERVER_PORT = port
            if name_arg:
                config.PLAYER_NAME = name_arg
            logutil.log("MAIN", f"Using server IP address {config.SERVER_IP}:{config.SERVER_PORT}")
    window = Window(width=300, height=200, caption='Pyglet', resizable=True, vsync=False)
    if server_proc is not None:
        window.server_process = server_proc
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
        if server_proc is not None:
            server_proc.terminate()


if __name__ == '__main__':
    main()
