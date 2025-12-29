import math
import random
import time
import sys

# pyglet imports
import pyglet
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
)
WATER = BLOCK_ID['Water']


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
        self.block_program['u_light_dir'] = (0.35, 1.0, 0.65)
        self.block_program['u_fog_color'] = (0.5, 0.69, 1.0)
        self.block_program['u_water_pass'] = False
        self.block_program['u_water_alpha'] = getattr(config, 'WATER_ALPHA', 0.8)
        if config.DEBUG_SINGLE_BLOCK:
            self.block_program['u_fog_start'] = 1e6
            self.block_program['u_fog_end'] = 2e6
        else:
            self.block_program['u_fog_start'] = 0.75 * DIST
            self.block_program['u_fog_end'] = DIST

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

        saved_snake_state = world_entity_store.load_entity_state("snake")
        self.snake_entity = SnakeEntity(
            self.model, player_position=self.position, entity_id=1, saved_state=saved_snake_state
        )
        saved_snail_state = world_entity_store.load_entity_state("snail")
        self.snail_entity = SnailEntity(
            self.model, player_position=self.position, entity_id=2, saved_state=saved_snail_state
        )
        saved_seagull_state = world_entity_store.load_entity_state("seagull")
        self.seagull_entity = SeagullEntity(
            self.model, player_position=self.position, entity_id=3, saved_state=saved_seagull_state
        )
        saved_dog_state = world_entity_store.load_entity_state("dog")
        self.dog_entity = Dog(self.model, entity_id=4, saved_state=saved_dog_state)
        self.dog_entity.snap_to_ground()
        self.entity_renderers = {
            'player': renderer.AnimatedEntityRenderer(self.block_program, HUMANOID_MODEL),
            'snake': renderer.SnakeRenderer(self.block_program, SNAKE_MODEL),
            'snail': renderer.AnimatedEntityRenderer(self.block_program, SNAIL_MODEL),
            'seagull': renderer.AnimatedEntityRenderer(self.block_program, SEAGULL_MODEL),
            'dog': renderer.AnimatedEntityRenderer(self.block_program, DOG_MODEL),
        }
        self.entity_objects = {
            self.player_entity.id: self.player_entity,
            self.snake_entity.id: self.snake_entity,
            self.snail_entity.id: self.snail_entity,
            self.seagull_entity.id: self.seagull_entity,
            self.dog_entity.id: self.dog_entity,
        }
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
        self.label = pyglet.text.Label('', font_name='Arial', font_size=18,
            x=10, y=self.height - 10, anchor_x='left', anchor_y='top',
            color=(0, 0, 0, 255))
        self.sector_label = pyglet.text.Label('', font_name='Arial', font_size=14,
            x=10, y=self.height - 10 - self.label.content_height - 4,
            anchor_x='left', anchor_y='top',
            color=(0, 0, 0, 255))
        self.sector_debug_label = pyglet.text.Label('', font_name='Arial', font_size=14,
            x=10, y=self.height - 10 - self.label.content_height - self.sector_label.content_height - 8,
            anchor_x='left', anchor_y='top',
            color=(0, 0, 0, 255))
        self.entity_label = pyglet.text.Label('', font_name='Arial', font_size=14,
            x=10, y=self.height - 10 - self.label.content_height - self.sector_label.content_height - self.sector_debug_label.content_height - 12,
            anchor_x='left', anchor_y='top',
            color=(0, 0, 0, 255))
        self.keybind_label = pyglet.text.Label('', font_name='Arial', font_size=14,
            x=10, y=self.height - 10 - self.label.content_height - self.sector_label.content_height - self.sector_debug_label.content_height - self.entity_label.content_height - 16,
            anchor_x='left', anchor_y='top',
            color=(0, 0, 0, 255))
        self._label_bg = shapes.Rectangle(0, 0, 1, 1, color=(255, 255, 255))
        self._label_bg.opacity = 120  # semi-transparent
        self._underwater_overlay = shapes.Rectangle(0, 0, 1, 1, color=config.UNDERWATER_COLOR)
        self.last_draw_ms = 0.0
        self.last_update_ms = 0.0
        self._hud_probe_frame = 0
        self._hud_probe_sector = None
        self._hud_probe_void = 'N/A'
        self._hud_probe_mush = 'NA'

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
            self.sector_label.text,
            self.sector_debug_label.text,
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
        self.model.mesh_budget_deadline = None
        t2 = time.perf_counter()
        self.model.process_pending_mesh_jobs(frustum_circle=frustum_circle, allow_submit=True)
        mesh_jobs_ms = (time.perf_counter() - t2) * 1000.0
        sector_ms = (time.perf_counter() - t0) * 1000.0
        m = 20
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
        logutil.log(
            "MAINLOOP",
            f"update sector_ms={sector_ms:.2f} update_sectors_ms={update_sectors_ms:.2f} "
            f"mesh_jobs_ms={mesh_jobs_ms:.2f} physics_ms={physics_ms:.2f} anim_ms={anim_ms:.2f} "
            f"total_ms={total_ms:.2f}",
        )
        logutil.log(
            "MAINLOOP",
            f"entities enabled={enabled_entities} updates={entity_updates_total} substeps={substeps} ms={entity_ms_total:.2f}",
        )
        self.last_update_ms = total_ms


    def _update(self, dt):
        """ Private implementation of the `update()` method. This is where most
        of the motion logic lives, along with gravity and collision detection.

        Parameters
        ----------
        dt : float
            The change in time since the last call.

        """
        motion_vector = self.get_motion_vector()
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
            "apply_camera_rotation": is_moving,
            "player_position": self.player_entity.position.copy(),
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
        self._entity_persist_timer -= dt
        if self._entity_persist_timer <= 0:
            self._persist_entity_states()
            self._entity_persist_timer = self._entity_persist_interval
        return entity_ms, update_count

    def _entity_is_enabled(self, entity):
        if entity is self.snake_entity:
            return self.snake_enabled
        if entity is self.snail_entity:
            return self.snail_enabled
        if entity is self.seagull_entity:
            return self.seagull_enabled
        if entity is self.dog_entity:
            return self.dog_enabled
        return True

    def _persist_entity_states(self):
        for entity in self.entity_objects.values():
            state = entity.serialize_state()
            if state is not None:
                # print('persisting',entity, state)
                world_entity_store.save_entity_state(entity.type, state)

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
        self.sector_label.y = self.label.y - self.label.content_height - 4
        self.sector_debug_label.y = self.sector_label.y - self.sector_label.content_height - 4
        self.entity_label.y = self.sector_debug_label.y - self.sector_debug_label.content_height - 4
        self.keybind_label.y = self.entity_label.y - self.entity_label.content_height - 4
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
        
        t0 = time.perf_counter()
        # Draw world (opaque pass only so water can overlay entities).
        self.model.draw(
            self.position,
            self.get_frustum_circle(),
            frame_start,
            upload_budget,
            defer_uploads=True,
            draw_water=False,
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
        self.draw_label()
        self.draw_reticle()
        self.draw_inventory_item()
        self.draw_focused_block()
        overlay_ms = (time.perf_counter() - t0) * 1000.0
        
        # Use leftover budget to upload meshes at the end of the frame.
        elapsed = time.perf_counter() - frame_start
        if elapsed < frame_budget:  # optional safety guard
            upload_start = time.perf_counter()
            # Decide how much *extra* time you’re willing to spend on uploads now.
            extra_budget = min(upload_budget, frame_budget - (time.perf_counter() - frame_start))
            if extra_budget > 0:
                self.model.process_pending_uploads(upload_start, extra_budget)
        upload_ms = (time.perf_counter() - upload_start) * 1000.0 if elapsed < frame_budget else 0.0
        logutil.log(
            "MAINLOOP",
            f"draw world_ms={world_ms:.2f} entity_ms={entity_draw_ms:.2f} "
            f"water_ms={water_ms:.2f} overlay_ms={overlay_ms:.2f} upload_ms={upload_ms:.2f}",
        )
        self.last_draw_ms = (time.perf_counter()-frame_start)*1000.0
        logutil.log("FRAME", f"end ms={self.last_draw_ms:.2f}")

    def draw_label(self):
        """ Draw the label in the top left of the screen.

        """
        x, y, z = self.position
        rx, ry = self.rotation
        fps = self._current_fps()
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
        self.label.text = 'FPS(%.1f), pos(%.2f, %.2f, %.2f) sector(%d, 0, %d) rot(%.1f, %.1f) void %s mush %s' % (
            fps, x, y, z, sector[0], sector[2], rx, ry, void_text, mush_text
        )
        sector_state = []
        sector_state.append(f"Sector({sector[0]}, 0, {sector[2]})")
        sector_debug = []
        s = self.model.sectors.get(sector)
        if s is None:
            sector_state.append("state=missing")
            sector_debug.append("state=missing")
        else:
            light = 'Y' if s.light is not None else 'N'
            mesh_ready = self.model._mesh_ready(s)
            neighbors_missing = self.model._neighbors_missing(s)
            needs_mesh = (s.vt_data is None and not s.mesh_built)
            needs_light = self.model._needs_light(s)
            if s.mesh_job_pending:
                waiting = "mesh_job"
            elif neighbors_missing:
                waiting = "neighbors"
            elif not s.seam_synced:
                waiting = "seam"
            elif not s.light_combined:
                waiting = "light"
            elif needs_mesh:
                waiting = "mesh"
            elif needs_light:
                waiting = "light_recalc"
            else:
                waiting = "idle"

            def _quad_count(entry):
                if not entry or entry[0] <= 0:
                    return 0
                return int(entry[0] // 4)

            def _float_count(entry):
                if not entry or entry[0] <= 0:
                    return 0
                _, v, t, n, c = entry
                return len(v) + len(t) + len(n) + len(c)

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
                vt_info = f"vt=Y q={solid_quads}/{water_quads} kb={kb:.1f}"
            else:
                solid_quads = int(getattr(s, "vt_solid_quads", 0) or 0)
                water_quads = int(getattr(s, "vt_water_quads", 0) or 0)
                if solid_quads or water_quads:
                    vt_info = f"vt=N q={solid_quads}/{water_quads}"

            uploaded = "Y" if (s.vt or s.vt_water) else "N"
            solid_verts = sum(getattr(vt, "count", 0) for vt in s.vt)
            water_verts = sum(getattr(vt, "count", 0) for vt in s.vt_water)
            upload_solid = f"{s.vt_upload_solid}/{s.vt_solid_quads}"
            upload_water = f"{s.vt_upload_water}/{s.vt_water_quads}"
            solid_tris_expected = int(s.vt_solid_quads * 2)
            water_tris_expected = int(s.vt_water_quads * 2)
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

            sector_state.append(
                f"seam={'Y' if s.seam_synced else 'N'} light={light} "
                f"combined={'Y' if s.light_combined else 'N'} "
                f"mesh_ready={'Y' if mesh_ready else 'N'} "
                f"pending={'Y' if s.mesh_job_pending else 'N'} "
                f"built={'Y' if s.mesh_built else 'N'} "
                f"uploaded={uploaded} vt_lists={len(s.vt)}/{len(s.vt_water)} "
                f"verts={solid_verts}/{water_verts} "
                f"upload={upload_solid}/{upload_water} "
                f"{vt_info} wait={waiting}"
            )
            sector_debug.append(
                f"tris={solid_verts}/{solid_tris_expected} water={water_verts}/{water_tris_expected} "
                f"pending_vt={pending_vt}/{pending_vt_water} use_pending={use_pending} "
                f"prep={upload_prepared} clear_pending={clear_pending} token={token}/{active_token_text} "
                f"gen={s.mesh_gen} dirty={dirty} inflight={inflight}"
            )
        self.sector_label.text = " | ".join(sector_state)
        self.sector_debug_label.text = " | ".join(sector_debug)
        entity_lines = []
        for entity_state in self.entities.values():
            if entity_state['type'] == 'player':
                continue
            pos = entity_state.get('pos')
            if pos is None:
                continue
            px, py, pz = pos
            entity_lines.append(f"{entity_state['type']}({px:.1f},{py:.1f},{pz:.1f})")
        if entity_lines:
            entity_text = "Ent: " + " | ".join(entity_lines)
        else:
            entity_text = "Ent: none"
        
        keybind_text = "Toggle: (N)ake, (B)Snail, (M)Seagull, (V)Dog | (F8)Copy HUD"

        line_spacing = 4
        self.sector_label.y = self.label.y - self.label.content_height - line_spacing
        self.sector_debug_label.y = self.sector_label.y - self.sector_label.content_height - line_spacing
        self.entity_label.text = entity_text
        self.entity_label.y = self.sector_debug_label.y - self.sector_debug_label.content_height - line_spacing
        
        self.keybind_label.text = keybind_text
        self.keybind_label.y = self.entity_label.y - self.entity_label.content_height - line_spacing

        # Light backdrop to keep text readable on bright backgrounds.
        pad_x = 6
        pad_y = 3
        top = self.label.y
        bottom = self.keybind_label.y - self.keybind_label.content_height
        entity_width = max(
            self.label.content_width,
            self.sector_label.content_width,
            self.sector_debug_label.content_width,
            self.entity_label.content_width,
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
        self._label_bg.draw()
        self.label.draw()
        self.sector_label.draw()
        self.sector_debug_label.draw()
        self.entity_label.draw()
        self.keybind_label.draw()

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
    gl.glClearColor(0.5, 0.69, 1.0, 1)
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
    window = Window(width=300, height=200, caption='Pyglet', resizable=True, vsync=True)
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
