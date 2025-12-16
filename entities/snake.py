"""
snake.py

Procedural model + entity definition for a segmented "snake" built from cubes.

- 60 body segments in a strict daisy-chain hierarchy.
- Alternating black/red body cubes.
- Black head composed of two overlapping cubes (main + snout).
- Red eyes (two small cubes).
- Red tongue (one thin cube).

Model format matches the dictionary structure used by AnimatedEntityRenderer in renderer.py:
    - root_part, root_offset
    - parts: {name: {parent, pivot, position, size, material:{color:(r,g,b)}}}
    - animations: { ... }

Usage (example integration in main.py):
    from snake import SNAKE_MODEL, SnakeEntity

    # register a renderer (client-side)
    self.entity_renderers['snake'] = renderer.AnimatedEntityRenderer(self.block_program, SNAKE_MODEL)

    # create an entity instance (server-side or local test)
    snake = SnakeEntity(world=self.model, position=(0, 120, 0))
    self.entities[1] = snake
"""
from __future__ import annotations

import math
import random
from collections import deque
from typing import Dict, Any, Tuple
import numpy as np

import config
import util
from blocks import BLOCK_SOLID
from entity import BaseEntity


RGB = Tuple[int, int, int]


BLACK: RGB = (0, 0, 0)
RED: RGB = (220, 40, 40)


def build_snake_model(
    head_main_size: Tuple[float, float, float] = (0.55, 0.35, 0.55),
    head_snout_size: Tuple[float, float, float] = (0.50, 0.28, 0.40),
    eye_size: Tuple[float, float, float] = (0.06, 0.06, 0.04),
    tongue_size: Tuple[float, float, float] = (0.10, 0.06, 0.35),
) -> Dict[str, Any]:
    """
    Create a snake model definition compatible with AnimatedEntityRenderer.

    Coordinate convention (consistent with your renderer):
      - Each part:
          pivot: point on parent where this part attaches/rotates (parent-local)
          position: offset from pivot to the center of this part's mesh (part-local)
      - We'll align the snake along -Z (head faces +Z), so the tail extends toward -Z.

    Returns:
        model_definition dict
    """
    head_w, head_h, head_d = head_main_size
    sn_w, sn_h, sn_d = head_snout_size

    parts: Dict[str, Dict[str, Any]] = {}

    # Root: head main block
    parts["head_main"] = {
        "parent": None,
        "pivot": [0.0, 0.0, 0.0],
        "position": [0.0, 0.0, 0.0],
        "size": [head_w, head_h, head_d],
        "material": {"color": BLACK},
    }

    # Overlapping head cube (snout) slightly forward (+Z) and slightly down
    parts["head_snout"] = {
        "parent": "head_main",
        # attach at head center and treat as rigid extension
        "pivot": [0.0, 0.0, 0.0],
        "position": [0.0, -0.03, +0.12],  # overlap forward
        "size": [sn_w, sn_h, sn_d],
        "material": {"color": BLACK},
    }

    # Eyes (small red cubes) attached to head_main, slightly forward and up
    eye_w, eye_h, eye_d = eye_size
    eye_color = RED
    parts["eye_left"] = {
        "parent": "head_main",
        "pivot": [0.0, 0.0, 0.0],
        "position": [-0.18, +0.10, +0.33],
        "size": [eye_w, eye_h, eye_d],
        "material": {"color": eye_color},
    }
    parts["eye_right"] = {
        "parent": "head_main",
        "pivot": [0.0, 0.0, 0.0],
        "position": [+0.18, +0.10, +0.33],
        "size": [eye_w, eye_h, eye_d],
        "material": {"color": eye_color},
    }

    # Tongue (thin red slab) attached to snout, protruding forward (+Z)
    t_w, t_h, t_d = tongue_size
    parts["tongue"] = {
        "parent": "head_snout",
        "pivot": [0.0, 0.0, +sn_d / 2.0],  # front face of snout
        "position": [0.0, -0.08, +t_d / 2.0],
        "size": [t_w, t_h, t_d],
        "material": {"color": RED},
    }

    # Minimal idle animation stub (keeps renderer happy even if you never drive it)
    # You can later populate rotations procedurally (e.g., wave down the chain).
    animations: Dict[str, Any] = {
        "idle": {
            "loop": True,
            "length": 1.0,
            "keyframes": [
                {"time": 0.0, "rotations": {}},
                {"time": 1.0, "rotations": {}},
            ],
        }
    }

    # Place the model slightly above the entity origin so it doesn't clip into the ground.
    # (Your humanoid uses a negative Y offset because its origin is at feet; for the snake
    #  we just lift the whole chain a bit.)
    model = {
        "root_part": "head_main",
        "root_offset": [0.0, +0.05, 0.0],
        "parts": parts,
        "animations": animations,
    }
    return model


# A ready-to-use model instance (client-side renderer can consume this directly)
SNAKE_MODEL: Dict[str, Any] = build_snake_model()


class SnakeEntity(BaseEntity):
    """
    Snake entity with segmented body that follows the player.
    """

    SEGMENT_COUNT = 60
    FOLLOW_DISTANCE = 10.0
    SPEED = 4.0
    SEGMENT_SPACING = 0.36
    HEAD_CLEARANCE = 0.05
    HISTORY_SCALAR = 3.0
    MAX_ROAM_DISTANCE = 20.0

    def __init__(self, world, player_position, entity_id=1, saved_state=None):
        self.world = world
        spawn = self._find_spawn_position(player_position)
        super().__init__(world, position=spawn)
        self.type = "snake"
        self.model_definition = SNAKE_MODEL
        self.id = entity_id
        self.bounding_box = np.array([0.6, 0.4, 0.6], dtype=float)
        self.current_animation = "idle"
        self.rotation = np.array([0.0, 0.0], dtype=float)
        self.segment_positions = np.zeros((self.SEGMENT_COUNT, 3), dtype=float)
        self._grounded = False
        self._last_head_pos = self.position.copy()
        self.history = deque()
        self._distances = deque()
        self._path_length = 0.0
        self._desired_path_length = self.SEGMENT_SPACING * (self.SEGMENT_COUNT - 1)
        self._max_path_length = self._desired_path_length * self.HISTORY_SCALAR
        self._targets = np.arange(self.SEGMENT_COUNT, dtype=float) * self.SEGMENT_SPACING
        self._wander_angle = random.random() * math.pi * 2
        self._wander_speed = 0.6
        if saved_state:
            self._restore_state(saved_state)
        else:
            self._initialize_segments()
        self._settle_height(self.position[1])

    def update(self, dt, context):
        if dt <= 0:
            return
        head = self.position
        player_position = context.get("player_position")
        if player_position is None:
            return
        px, py, pz = player_position
        dx = px - head[0]
        dz = pz - head[2]
        dist_xz = math.hypot(dx, dz)
        moving = False
        move_dir = np.array([0.0, 0.0], dtype=float)
        step = 0.0
        wander = self._update_wander_direction(dt)
        steer_target = np.zeros(2, dtype=float)
        if dist_xz > 1e-6:
            steer_target = np.array([dx / dist_xz, dz / dist_xz], dtype=float)
        if dist_xz > self.MAX_ROAM_DISTANCE:
            if self._path_clear(px, pz):
                move_dir = steer_target
            else:
                move_dir = steer_target * 0.65 + wander * 0.35
            norm = np.linalg.norm(move_dir)
            if norm > 1e-6:
                move_dir /= norm
            step = self.SPEED * dt
        else:
            move_dir = wander
            blend = 0.6
            strength = 0.0
            if dist_xz > 1e-6:
                strength = max(0.0, 1.0 - (dist_xz / self.MAX_ROAM_DISTANCE))
            repulsion = -steer_target * strength
            move_dir = move_dir * blend + repulsion * (1 - blend)
            norm = np.linalg.norm(move_dir)
            if norm < 1e-6:
                fallback = wander
                fallback_norm = np.linalg.norm(fallback)
                if fallback_norm > 1e-6:
                    move_dir = fallback / fallback_norm
                    norm = 1.0
                else:
                    move_dir = np.array([1.0, 0.0], dtype=float)
                    norm = 1.0
            move_dir /= norm
            step = self.SPEED * 0.5 * dt
        if step > 1e-8:
            head[0] += move_dir[0] * step
            head[2] += move_dir[1] * step
            moving = True
        self._settle_height(py)
        if moving and np.linalg.norm(move_dir) > 1e-6:
            self.rotation[0] = math.degrees(math.atan2(move_dir[0], move_dir[1]))
        self.current_animation = "walk" if moving else "idle"
        head_dxz = math.hypot(head[0] - self._last_head_pos[0], head[2] - self._last_head_pos[2])
        if head_dxz > 1e-4:
            self._last_head_pos[:] = head
            self._record_history_position(head.copy())
        self._update_segments()

    def to_network_dict(self):
        d = super().to_network_dict()
        d["type"] = "snake"
        d["animation"] = self.current_animation
        d["pos"] = tuple(self.position.tolist())
        d["rot"] = (self.rotation[0], self.rotation[1])
        d["segment_positions"] = [tuple(pos.tolist()) for pos in self.segment_positions]
        return d

    def serialize_state(self):
        return {
            "head": self.position.tolist(),
            "rotation": self.rotation.tolist(),
            "segments": [list(pos) for pos in self.segment_positions],
        }

    def _find_spawn_position(self, player_position):
        px, py, pz = player_position
        start_y = py + 24.0
        primary = self._sample_ground(px, pz, start_y)
        if primary is not None:
            return primary
        for radius in range(1, 5):
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dz) != radius:
                        continue
                    sample = self._sample_ground(px + dx, pz + dz, start_y)
                    if sample is not None:
                        return sample
        return np.array([px, py, pz], dtype=float)

    def _restore_state(self, state):
        head = state.get("head")
        if head:
            self.position = np.array(head, dtype=float)
        rotation = state.get("rotation")
        if rotation:
            self.rotation = np.array(rotation, dtype=float)
        self._last_head_pos = self.position.copy()
        segments = state.get("segments")
        if segments and len(segments) == self.SEGMENT_COUNT:
            self.segment_positions = np.array(segments, dtype=float)
        else:
            self._initialize_segments()
        path = list(self.segment_positions[::-1])
        path.append(self.position.copy())
        self._initialize_history(path)
        self._update_segments()
        self._grounded = False
        self._settle_height(self.position[1])

    def _initialize_segments(self):
        for i in range(self.SEGMENT_COUNT):
            offset = np.array([0.0, 0.0, -i * self.SEGMENT_SPACING], dtype=float)
            self.segment_positions[i] = self.position + offset
        tail_path = list(self.segment_positions[::-1])
        tail_path.append(self.position.copy())
        self._initialize_history(tail_path)

    def _initialize_history(self, path=None):
        self.history.clear()
        self._distances.clear()
        self._path_length = 0.0
        if path:
            for pos in path:
                self._append_history_point(np.array(pos, dtype=float))
        else:
            self._append_history_point(self.position.copy())

    def _record_history_position(self, position):
        position = np.array(position, dtype=float)
        if not self.history:
            self._append_history_point(position)
            return
        last = self.history[-1]
        delta = position - last
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            return
        steps = max(int(math.ceil(dist / self.SEGMENT_SPACING)), 1)
        for step in range(1, steps):
            t = step / steps
            inter_pos = last + delta * t
            self._append_history_point(inter_pos)
        self._append_history_point(position)
        self._prune_history()

    def _append_history_point(self, point):
        if self.history:
            prev = self.history[-1]
            delta = point - prev
            dist = np.linalg.norm(delta)
            if dist < 1e-6:
                return
            self._distances.append(dist)
            self._path_length += dist
        self.history.append(point)

    def _prune_history(self):
        while (
            self._path_length > self._max_path_length
            and len(self.history) > 2
            and self._distances
            and self._path_length - self._distances[0] >= self._desired_path_length
        ):
            self.history.popleft()
            removed = self._distances.popleft()
            self._path_length -= removed

    def _ordered_history(self):
        if not self.history:
            return np.empty((0, 3), dtype=float)
        return np.array(self.history, dtype=float)[::-1]

    def _sample_ground(self, x, z, reference_y):
        surface = self._find_surface_y(x, z, reference_y)
        if surface is None:
            return None
        return np.array([x, surface, z], dtype=float)

    def _find_surface_y(self, x, z, reference_y):
        max_y = min(int(math.ceil(reference_y + 8.0)), config.SECTOR_HEIGHT - 1)
        min_y = max(int(math.floor(reference_y - 64.0)), -config.SECTOR_HEIGHT)
        for y in range(max_y, min_y - 1, -1):
            if self._is_solid(x, y, z):
                head_y = y + 0.5 + self.HEAD_CLEARANCE
                if self._has_clearance(x, head_y, z):
                    return head_y
        return None

    def _settle_height(self, reference_y):
        target = self._find_surface_y(self.position[0], self.position[2], self.position[1] + 8.0)
        if target is not None:
            self.position[1] = target
            self._grounded = True
            return True
        if not self._grounded:
            fallback = reference_y - 2.0
            if fallback < self.position[1]:
                self.position[1] = fallback
        return False

    def _path_clear(self, target_x, target_z):
        head = self.position
        dx = target_x - head[0]
        dz = target_z - head[2]
        dist = math.hypot(dx, dz)
        if dist < 1e-4:
            return True
        samples = max(6, min(14, int(dist * 2) + 1))
        for i in range(1, samples + 1):
            t = i / samples
            sample_x = head[0] + dx * t
            sample_z = head[2] + dz * t
            if self._find_surface_y(sample_x, sample_z, head[1] + 8.0) is None:
                return False
        return True

    def _has_clearance(self, x, y, z):
        return (not self._is_solid(x, y, z)) and (not self._is_solid(x, y + 1.0, z))

    def _is_solid(self, x, y, z):
        block_id = self._block_id(x, y, z)
        return BLOCK_SOLID[block_id]

    def _block_id(self, x, y, z):
        coords = util.normalize((x, y, z))
        block = self.world[coords]
        return int(block or 0)

    def _update_segments(self):
        ordered = self._ordered_history()
        if ordered.shape[0] < 2:
            self.segment_positions[:] = np.tile(self.position, (self.SEGMENT_COUNT, 1))
            return
        diffs = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
        cumdist = np.concatenate(([0.0], np.cumsum(diffs)))
        total_length = cumdist[-1]
        last_dir = ordered[-1] - ordered[-2]
        last_norm = np.linalg.norm(last_dir)
        if last_norm > 1e-6:
            last_dir /= last_norm
        else:
            last_dir = np.array([0.0, 0.0, -1.0], dtype=float)

        targets = self._targets
        idx = np.searchsorted(cumdist, targets, side="right") - 1
        idx = np.clip(idx, 0, len(cumdist) - 2)

        segment_len = cumdist[idx + 1] - cumdist[idx]
        safe_len = np.where(segment_len > 1e-6, segment_len, 1.0)
        t = (targets - cumdist[idx]) / safe_len

        base = ordered[idx]
        next_pts = ordered[idx + 1]
        positions = base + (next_pts - base) * t[:, None]

        if total_length < 0:
            total_length = 0.0
        beyond = targets > total_length
        if np.any(beyond):
            extras = targets[beyond] - total_length
            positions[beyond] = ordered[-1] + last_dir * extras[:, None]

        self.segment_positions[:] = positions

    def _update_wander_direction(self, dt):
        self._wander_angle += (dt * self._wander_speed) + random.uniform(-0.1, 0.1)
        return np.array([math.cos(self._wander_angle), math.sin(self._wander_angle)], dtype=float)
