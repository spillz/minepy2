from __future__ import annotations

import math
import random
from typing import Dict, Any, Tuple

import numpy as np

from entity import BaseEntity


RGB = Tuple[int, int, int]

FISH_BODY: RGB = (90, 190, 210)
FISH_FIN: RGB = (60, 130, 160)
FISH_EYE: RGB = (240, 240, 240)
FISH_SEGMENT_CONFIGS = [
    ((0.50, 0.22, 0.70), (240, 120, 60)),
    ((0.60, 0.26, 0.80), (60, 200, 220)),
    ((0.70, 0.30, 0.95), (200, 80, 160)),
    ((0.80, 0.32, 1.05), (90, 220, 120)),
    ((0.55, 0.22, 0.75), (230, 220, 70)),
]
FISH_TAIL_LENGTH = 1


def build_fish_model() -> Dict[str, Any]:
    parts: Dict[str, Dict[str, Any]] = {}
    body_size = (0.50, 0.24, 0.75)
    parts["body"] = {
        "parent": None,
        "pivot": [0.0, 0.0, 0.0],
        "position": [0.0, 0.0, 0.0],
        "size": body_size,
        "material": {"color": FISH_BODY},
    }
    tail_size = (0.10, 0.30, 0.40)
    parts["tail"] = {
        "parent": "body",
        "pivot": [0.0, 0.0, body_size[2] / 2.0],
        "position": [0.0, 0.0, tail_size[2] / 2.0 + 0.02],
        "size": tail_size,
        "material": {"color": FISH_FIN},
    }
    for side, sign in (("left", -1.0), ("right", 1.0)):
        parts[f"{side}_eye"] = {
            "parent": "body",
            "pivot": [0.0, 0.0, 0.0],
            "position": [sign * 0.16, 0.06, -0.18],
            "size": (0.06, 0.06, 0.04),
            "material": {"color": FISH_EYE},
        }
        parts[f"{side}_fin"] = {
            "parent": "body",
            "pivot": [sign * body_size[0] / 2.0, 0.0, 0.05],
            "position": [sign * 0.10, -0.02, 0.0],
            "size": (0.18, 0.04, 0.20),
            "material": {"color": FISH_FIN},
        }

    return {
        "root_part": "body",
        "parts": parts,
        "animations": {
            "idle": {"loop": True, "length": 1.0, "keyframes": [{"time": 0.0, "rotations": {}}, {"time": 1.0, "rotations": {}}]}
        },
    }


FISH_MODEL = build_fish_model()


class FishSchoolEntity(BaseEntity):
    SCHOOL_SIZE = 14
    SPEED = 1.2
    VERTICAL_SPEED = 1.1
    WANDER_SPEED = 0.5
    TURN_WEIGHT = 0.06
    SURFACE_CLEARANCE = 1.2
    FLOOR_CLEARANCE = 1.2
    SCHOOL_RADIUS = 2.4

    def __init__(self, world, player_position=None, entity_id=8, saved_state=None):
        initial_position = player_position if player_position is not None else (0, 120, 0)
        if saved_state and "pos" in saved_state:
            initial_position = saved_state["pos"]
        super().__init__(world, position=initial_position, entity_type="fish_school")
        self.id = entity_id
        self.model_definition = FISH_MODEL
        self.bounding_box = np.array([1.0, 1.0, 1.0], dtype=float)
        self.rotation = np.array([0.0, 0.0], dtype=float)
        self.current_animation = "idle"
        self.flying = True
        self._wander_angle = random.random() * math.pi * 2.0
        self._heading = np.array([1.0, 0.0], dtype=float)
        self._alt_phase = random.random() * math.pi * 2.0
        self._time = 0.0
        self._offsets = self._init_offsets()
        self._phases = np.random.uniform(0.0, math.pi * 2.0, size=self.SCHOOL_SIZE)
        self.segment_positions = np.zeros((self.SCHOOL_SIZE, 3), dtype=float)
        if saved_state:
            rot = saved_state.get("rot")
            if rot is not None:
                self.rotation = np.array(rot, dtype=float)
        else:
            self._snap_to_water_band()
        self._update_segments()

    def update(self, dt, context):
        if dt <= 0:
            return
        self._time += dt

        self._wander_angle += (dt * self.WANDER_SPEED) + random.uniform(-0.2, 0.2)
        move_dir = np.array([math.cos(self._wander_angle), math.sin(self._wander_angle)], dtype=float)
        norm = np.linalg.norm(move_dir)
        if norm > 1e-6:
            move_dir /= norm
            self._heading = self._heading * (1.0 - self.TURN_WEIGHT) + move_dir * self.TURN_WEIGHT
            heading_norm = np.linalg.norm(self._heading)
            if heading_norm > 1e-6:
                self._heading /= heading_norm
            self.velocity[0] = self._heading[0] * self.SPEED
            self.velocity[2] = self._heading[1] * self.SPEED
            self.rotation[0] = math.degrees(math.atan2(-self._heading[0], -self._heading[1]))
        else:
            self.velocity[0] = 0.0
            self.velocity[2] = 0.0

        min_y, max_y = self._water_bounds(self.position[0], self.position[2])
        if min_y is not None and max_y is not None and max_y > min_y:
            span = max_y - min_y
            self._alt_phase += dt
            target_y = min_y + span * 0.5 + math.sin(self._alt_phase * 1.2) * (span * 0.25)
            target_y = float(np.clip(target_y, min_y, max_y))
            diff = target_y - float(self.position[1])
            self.velocity[1] = np.sign(diff) * min(abs(diff), self.VERTICAL_SPEED)
        else:
            self.velocity[1] = 0.0

        super().update(dt)
        self.rotation[1] = 0.0
        self._update_segments()

    def to_network_dict(self):
        data = super().to_network_dict()
        data["type"] = "fish_school"
        data["animation"] = "idle"
        data["segment_positions"] = [tuple(pos.tolist()) for pos in self.segment_positions]
        return data

    def serialize_state(self):
        return {"pos": tuple(self.position.tolist()), "rot": tuple(self.rotation.tolist())}

    def _init_offsets(self):
        angles = np.random.uniform(0.0, math.pi * 2.0, size=self.SCHOOL_SIZE)
        radii = np.random.uniform(0.3, self.SCHOOL_RADIUS, size=self.SCHOOL_SIZE)
        heights = np.random.uniform(-0.5, 0.5, size=self.SCHOOL_SIZE)
        offsets = np.zeros((self.SCHOOL_SIZE, 3), dtype=float)
        offsets[:, 0] = np.cos(angles) * radii
        offsets[:, 2] = np.sin(angles) * radii
        offsets[:, 1] = heights
        return offsets

    def _update_segments(self):
        angle = self._time * 0.4
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotated = self._offsets.copy()
        rotated[:, 0] = self._offsets[:, 0] * cos_a - self._offsets[:, 2] * sin_a
        rotated[:, 2] = self._offsets[:, 0] * sin_a + self._offsets[:, 2] * cos_a
        bob = np.sin(self._phases + self._time * 2.5) * 0.15
        rotated[:, 1] = self._offsets[:, 1] + bob
        positions = rotated + self.position
        min_y, max_y = self._water_bounds(self.position[0], self.position[2])
        if min_y is not None and max_y is not None and max_y > min_y:
            positions[:, 1] = np.clip(positions[:, 1], min_y + 0.2, max_y - 0.2)
        self.segment_positions[:] = positions

    def _water_bounds(self, x, z):
        surface_y = None
        solid_y = None
        if hasattr(self.world, "find_surface_y"):
            surface_y = self.world.find_surface_y(x, z)
        if hasattr(self.world, "find_solid_surface_y"):
            solid_y = self.world.find_solid_surface_y(x, z)
        if surface_y is None or solid_y is None:
            return None, None
        min_y = float(solid_y + self.FLOOR_CLEARANCE)
        max_y = float(surface_y - self.SURFACE_CLEARANCE)
        return min_y, max_y

    def _snap_to_water_band(self):
        bounds = self._water_bounds(self.position[0], self.position[2])
        min_y, max_y = bounds
        if min_y is None or max_y is None or max_y <= min_y:
            return
        self.position[1] = float((min_y + max_y) * 0.5)
