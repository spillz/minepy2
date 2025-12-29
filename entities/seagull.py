from __future__ import annotations

import math
import random
from typing import Dict, Any, Tuple

import numpy as np

import config
import util
from blocks import BLOCK_SOLID
from entity import BaseEntity


def build_seagull_model(
    body_size: Tuple[float, float, float] = (0.30, 0.18, 0.60),
    wing_size: Tuple[float, float, float] = (0.75, 0.02, 0.20),
    tail_size: Tuple[float, float, float] = (0.15, 0.02, 0.25),
    beak_size: Tuple[float, float, float] = (0.12, 0.05, 0.25),
 ) -> Dict[str, Any]:
    parts: Dict[str, Dict[str, Any]] = {}
    body_w, body_h, body_d = body_size
    parts["body"] = {
        "parent": None,
        "pivot": [0.0, 0.0, 0.0],
        "position": [0.0, 0.0, 0.0],
        "size": [body_w, body_h, body_d],
        "material": {"color": (240, 240, 240)},
    }
    head_size_val = [0.18, 0.14, 0.18]
    parts["head"] = {
        "parent": "body",
        "pivot": [0.0, 0.0, head_size_val[2] / 2.0],
        "position": [0.0, 0.0, -body_d / 2.0 - head_size_val[2] / 2.0],
        "size": head_size_val,
        "material": {"color": (220, 220, 220)},
    }
    parts["beak"] = {
        "parent": "head",
        "pivot": [0.0, 0.0, beak_size[2] / 2.0],
        "position": [0.0, 0.0, -head_size_val[2] / 2.0 - beak_size[2] / 2.0],
        "size": [beak_size[0], beak_size[1], beak_size[2]],
        "material": {"color": (255, 200, 50)},
    }
    parts["left_wing"] = {
        "parent": "body",
        "pivot": [0.0, 0.0, 0.0],
        "position": [-wing_size[0] / 2.0, 0.0, 0.0],
        "size": [wing_size[0], wing_size[1], wing_size[2]],
        "material": {"color": (230, 230, 230)},
    }
    parts["right_wing"] = {
        "parent": "body",
        "pivot": [0.0, 0.0, 0.0],
        "position": [wing_size[0] / 2.0, 0.0, 0.0],
        "size": [wing_size[0], wing_size[1], wing_size[2]],
        "material": {"color": (230, 230, 230)},
    }
    parts["tail"] = {
        "parent": "body",
        "pivot": [0.0, 0.0, -tail_size[2] / 2.0],
        "position": [0.0, 0.0, body_d / 2.0 + tail_size[2] / 2.0],
        "size": [tail_size[0], tail_size[1], tail_size[2]],
        "material": {"color": (210, 210, 210)},
    }
    model = {
        "root_part": "body",
        "root_offset": [0.0, 0.0, 0.0],
        "parts": parts,
        "animations": {
            "idle": {"loop": True, "length": 1.0, "keyframes": [{"time": 0.0, "rotations": {}}, {"time": 1.0, "rotations": {}}]}
        },
    }
    return model


SEAGULL_MODEL = build_seagull_model()


class SeagullEntity(BaseEntity):
    MAX_RADIUS = 60.0
    MIN_RADIUS = 12.0
    SPEED = 6.0
    VERTICAL_SPEED = 3.0
    MIN_ALTITUDE = 10.0
    ALTITUDE_OFFSET = 12.0
    ALTITUDE_SPAN = 10.0

    def __init__(self, world, player_position, entity_id=3, saved_state=None):
        super().__init__(world, position=player_position)
        self.type = "seagull"
        self.model_definition = SEAGULL_MODEL
        self.id = entity_id
        self.bounding_box = np.array([0.5, 0.4, 0.5], dtype=float)
        self.rotation = np.array([0.0, 0.0], dtype=float)
        self.flying = True
        self._wander_angle = random.random() * math.pi * 2
        self._wander_speed = 0.2
        self._player_spawn = np.array(player_position, dtype=float)
        if saved_state:
            self.position = np.array(saved_state.get("pos", self.position), dtype=float)
            rot = saved_state.get("rot")
            if rot:
                self.rotation = np.array(rot, dtype=float)
        else:
            self.position = self._find_spawn_position(player_position)
        
        ground_y = self.world.find_surface_y(self.position[0], self.position[2])
        if ground_y is None:
            ground_y = self.position[1] - self.ALTITUDE_OFFSET
        self._ground_reference = ground_y
        
        self._flight_circle_sign = random.choice([-1.0, 1.0])
        self._alt_phase = random.random() * math.pi * 2
        self._heading = np.array([1.0, 0.0], dtype=float)

    def update(self, dt, context):
        if dt <= 0:
            return
        player_pos = context.get("player_position")
        if player_pos is None:
            return
        px, _, pz = player_pos
        head = self.position
        dx = px - head[0]
        dz = pz - head[2]
        dist = math.hypot(dx, dz)
        if dist > 0:
            target_dir = np.array([dx / dist, dz / dist], dtype=float)
        else:
            target_dir = np.zeros(2, dtype=float)
        
        wander = self._update_wander_direction(dt) * 0.2
        if dist >= self.MAX_RADIUS:
            move_dir = target_dir * 0.92 + wander * 0.08
        elif dist <= self.MIN_RADIUS:
            perp = np.array([-target_dir[1], target_dir[0]]) * self._flight_circle_sign
            move_dir = perp * 0.6 + wander * 0.1
        else:
            strength = (dist - self.MIN_RADIUS) / (self.MAX_RADIUS - self.MIN_RADIUS)
            perp = np.array([-target_dir[1], target_dir[0]]) * self._flight_circle_sign * 0.3
            move_dir = wander * (0.15 + 0.3 * (1 - strength)) + target_dir * strength * 0.6 + perp * (0.2 + 0.2 * (1 - strength))
        
        norm = np.linalg.norm(move_dir)
        if norm > 1e-6:
            move_dir /= norm
            self.velocity[0] = move_dir[0] * self.SPEED
            self.velocity[2] = move_dir[1] * self.SPEED
            turn_weight = 0.01
            self._heading = self._heading * (1 - turn_weight) + move_dir * turn_weight
            norm_heading = np.linalg.norm(self._heading)
            if norm_heading > 1e-6:
                self._heading /= norm_heading
            self.rotation[0] = math.degrees(math.atan2(-self._heading[0], -self._heading[1]))
        
        ground_y = self.world.find_surface_y(head[0], head[2])
        if ground_y is None:
            fallback_ground = self.world.find_surface_y(px, pz)
            if fallback_ground is not None:
                ground_y = fallback_ground
        
        if ground_y is not None:
            self._ground_reference = ground_y
        
        min_alt = self._ground_reference + self.MIN_ALTITUDE
        max_alt = min_alt + self.ALTITUDE_SPAN
        self._alt_phase += dt
        bias = math.sin(self._alt_phase) * (self.ALTITUDE_SPAN * 0.25)
        target_alt = min_alt + (self.ALTITUDE_SPAN * 0.5) + bias
        target_alt = float(np.clip(target_alt, min_alt, max_alt))
        diff_y = target_alt - head[1]
        
        self.velocity[1] = np.sign(diff_y) * min(abs(diff_y), self.VERTICAL_SPEED)
        
        super().update(dt)
        
        self.rotation[1] = 0.0

    def serialize_state(self):
        return {"pos": tuple(self.position.tolist()), "rot": tuple(self.rotation.tolist())}

    def to_network_dict(self):
        data = super().to_network_dict()
        data["animation"] = "idle"
        return data

    def serialize_state(self):
        return {"pos": tuple(self.position.tolist()), "rot": tuple(self.rotation.tolist())}

    def _find_spawn_position(self, player_position):
        px, py, pz = player_position
        ground = self.world.find_surface_y(px, pz)
        base_y = ground + self.ALTITUDE_OFFSET if ground is not None else py + self.ALTITUDE_OFFSET
        return np.array([px + 2.0, base_y, pz + 2.0], dtype=float)

    def _update_wander_direction(self, dt):
        self._wander_angle += (dt * self._wander_speed) + random.uniform(-0.2, 0.2)
        return np.array([math.cos(self._wander_angle), math.sin(self._wander_angle)], dtype=float)
