import math
from typing import Dict, Any
import numpy as np
from blocks import BLOCK_SOLID
from config import GRAVITY
from entities.tetrapod import build_tetrapod_model, Tetrapod

def build_dog_model() -> Dict[str, Any]:
    return build_tetrapod_model(
        body_size=(0.3, 0.3, 0.6),
        leg_size=(0.1, 0.3, 0.1),
        head_size=(0.25, 0.25, 0.25),
        ear_size=(0.04, 0.1, 0.04),
        snout_size=(0.15, 0.15, 0.4),
        tail_size=(0.04, 0.04, 0.2),
        body_color=(210, 180, 140),
        leg_color=(210, 180, 140),
        head_color=(210, 180, 140),
        ear_color=(200, 170, 130),
        snout_color=(210, 180, 140),
        tail_color=(210, 180, 140),
    )

DOG_MODEL = build_dog_model()

class Dog(Tetrapod):
    SPEED = 1.5
    FOLLOW_RANGE =8.0
    MAX_JUMP_HEIGHT = 2.4
    MAX_JUMP_SPAN = 3.0
    JUMP_LOOKAHEAD = 3
    MAX_DROP = 5.0
    STUCK_TIME = 0.8
    BACKOFF_TIME = 0.4
    BACKOFF_SPEED = 0.8

    def __init__(self, world, entity_id=4, saved_state=None):
        super().__init__(
            world,
            entity_type="dog",
            model_definition=DOG_MODEL,
            entity_id=entity_id,
            saved_state=saved_state,
        )
        self.bounding_box = np.array([0.4, 0.6, 0.7])
        self._stuck_timer = 0.0
        self._jump_boost_timer = 0.0
        self._jump_boost_speed = self.SPEED
        self._backoff_timer = 0.0

    def update(self, dt, context):
        if dt <= 0:
            return
        player_position = context.get("player_position")
        if player_position is None:
            return
        if self._jump_boost_timer > 0.0:
            self._jump_boost_timer = max(0.0, self._jump_boost_timer - dt)
        if self._backoff_timer > 0.0:
            self._backoff_timer = max(0.0, self._backoff_timer - dt)
        prev_pos = self.position.copy()
        moving = self._follow_player(dt, player_position)
        super().update(dt)
        self._update_stuck(dt, prev_pos, moving)

    def _follow_player(self, dt, player_position):
        px, py, pz = player_position
        dx = px - self.position[0]
        dz = pz - self.position[2]
        dist_xz = math.hypot(dx, dz)
        if dist_xz > 1e-6:
            move_dir = np.array([dx / dist_xz, dz / dist_xz], dtype=float)
        else:
            move_dir = np.array([0.0, 0.0], dtype=float)

        if dist_xz > 1e-6:
            self.rotation[0] = math.degrees(math.atan2(-move_dir[0], -move_dir[1]))

        if self._backoff_timer > 0.0 and dist_xz > 1e-6:
            self.velocity[0] = -move_dir[0] * self.SPEED * self.BACKOFF_SPEED
            self.velocity[2] = -move_dir[1] * self.SPEED * self.BACKOFF_SPEED
            self.current_animation = "walk"
            return True

        if dist_xz <= self.FOLLOW_RANGE or self._stuck_timer >= self.STUCK_TIME:
            self.velocity[0] = 0.0
            self.velocity[2] = 0.0
            self.current_animation = "idle"
            return False

        blocked, jump_height, too_close = self._scan_ahead(move_dir)
        if too_close and self.on_ground:
            self._backoff_timer = max(self._backoff_timer, self.BACKOFF_TIME)
            self.velocity[0] = -move_dir[0] * self.SPEED * self.BACKOFF_SPEED
            self.velocity[2] = -move_dir[1] * self.SPEED * self.BACKOFF_SPEED
            self.current_animation = "walk"
            return True
        if blocked:
            self.velocity[0] = 0.0
            self.velocity[2] = 0.0
            self.current_animation = "idle"
            return False

        speed = self.SPEED
        if self._jump_boost_timer > 0.0:
            speed = max(speed, self._jump_boost_speed)
        self.velocity[0] = move_dir[0] * speed
        self.velocity[2] = move_dir[1] * speed
        self.current_animation = "walk"
        if jump_height > 0.0:
            if too_close:
                jump_height = min(self.MAX_JUMP_HEIGHT, jump_height + 0.65)
            self._jump_with_height(jump_height * 1.5)
        return True

    def _jump_with_height(self, height):
        if self.on_ground:
            height = min(height, self.MAX_JUMP_HEIGHT)
            self.velocity[1] = math.sqrt(2.0 * GRAVITY * height)
            self.on_ground = False
            flight_time = 2.0 * math.sqrt(2.0 * height / GRAVITY)
            if flight_time > 1e-6:
                self._jump_boost_speed = max(self.SPEED, self.MAX_JUMP_SPAN / flight_time)
                self._jump_boost_timer = flight_time

    def _scan_ahead(self, move_dir):
        if not hasattr(self.world, "find_surface_y"):
            return True, 0.0, False

        current_y = self.world.find_surface_y(self.position[0], self.position[2])
        if current_y is None:
            current_y = float(self.position[1])
        jump_height = 0.0
        blocked = False
        too_close = False

        for step in range(1, self.JUMP_LOOKAHEAD + 1):
            sample_x = self.position[0] + move_dir[0] * step
            sample_z = self.position[2] + move_dir[1] * step
            surface_y = self.world.find_surface_y(sample_x, sample_z)
            if surface_y is None:
                blocked = True
                break

            delta = surface_y - current_y
            if current_y - surface_y > self.MAX_DROP + 1e-6:
                blocked = True
                break

            if delta > self.MAX_JUMP_HEIGHT + 1e-6:
                blocked = True
                break

            if delta > 0.2:
                if step == 1:
                    too_close = True
                if not self._has_clearance(sample_x, surface_y, sample_z):
                    blocked = True
                    break
                jump_height = max(jump_height, min(delta, self.MAX_JUMP_HEIGHT))

        return blocked, jump_height, too_close

    def _has_clearance(self, x, surface_y, z):
        ix = int(round(x))
        iz = int(round(z))
        base_y = int(math.floor(surface_y))
        height_blocks = int(math.ceil(self.bounding_box[1] + 0.5))
        for dy in range(height_blocks):
            block_id = self._get_block(ix, base_y + dy, iz)
            if block_id and BLOCK_SOLID[block_id]:
                return False
        return True

    def _get_block(self, x, y, z):
        if not hasattr(self.world, "__getitem__"):
            return None
        try:
            return self.world[(int(x), int(y), int(z))]
        except Exception:
            return None

    def _update_stuck(self, dt, prev_pos, moving):
        if not moving:
            self._stuck_timer = max(0.0, self._stuck_timer - dt)
            return
        delta = self.position - prev_pos
        moved = math.hypot(delta[0], delta[2])
        if moved < 0.01:
            self._stuck_timer += dt
        else:
            self._stuck_timer = 0.0

    def to_network_dict(self):
        data = super().to_network_dict()
        data["animation"] = self.current_animation
        return data

    def serialize_state(self):
        return {"pos": tuple(self.position.tolist()), "rot": tuple(self.rotation.tolist())}
