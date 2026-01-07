from __future__ import annotations

import math
import random
from typing import Dict, Any, Tuple

import numpy as np

from entity import BaseEntity


RGB = Tuple[int, int, int]

MOSA_DARK: RGB = (28, 60, 80)
MOSA_MID: RGB = (46, 86, 108)
MOSA_LIGHT: RGB = (78, 120, 140)
MOSA_BELLY: RGB = (120, 160, 170)
MOSA_FIN: RGB = (50, 96, 118)
MOSA_EYE: RGB = (230, 220, 170)
MOSA_TEETH: RGB = (235, 235, 230)


def _scale_part(part: Dict[str, Any], scale: float) -> Dict[str, Any]:
    part = dict(part)
    part["pivot"] = [v * scale for v in part["pivot"]]
    part["position"] = [v * scale for v in part["position"]]
    part["size"] = [v * scale for v in part["size"]]
    return part


def _scale_model(model: Dict[str, Any], scale: float) -> Dict[str, Any]:
    model = dict(model)
    parts = model.get("parts", {})
    model["parts"] = {name: _scale_part(part, scale) for name, part in parts.items()}
    if "root_offset" in model:
        model["root_offset"] = [v * scale for v in model["root_offset"]]
    return model


def build_mosasaurus_model() -> Dict[str, Any]:
    parts: Dict[str, Dict[str, Any]] = {}
    gap = 0.06

    body_sizes = [
        (1.20, 0.90, 2.20),
        (1.10, 0.85, 2.00),
        (1.00, 0.80, 1.80),
    ]
    parts["body_1"] = {
        "parent": None,
        "pivot": [0.0, 0.0, 0.0],
        "position": [0.0, 0.0, 0.0],
        "size": body_sizes[0],
        "material": {"color": MOSA_DARK},
    }
    parent = "body_1"
    parent_size = body_sizes[0]
    for idx, size in enumerate(body_sizes[1:], start=2):
        name = f"body_{idx}"
        parts[name] = {
            "parent": parent,
            "pivot": [0.0, 0.0, -parent_size[2] / 2.0],
            "position": [0.0, 0.0, -size[2] / 2.0 - gap],
            "size": size,
            "material": {"color": MOSA_MID if idx % 2 else MOSA_LIGHT},
        }
        parent = name
        parent_size = size

    neck_size = (0.90, 0.70, 1.40)
    parts["neck"] = {
        "parent": parent,
        "pivot": [0.0, 0.0, -parent_size[2] / 2.0],
        "position": [0.0, 0.0, -neck_size[2] / 2.0 - gap],
        "size": neck_size,
        "material": {"color": MOSA_MID},
    }

    head_size = (0.95, 0.65, 1.60)
    parts["head"] = {
        "parent": "neck",
        "pivot": [0.0, 0.02, -neck_size[2] / 2.0],
        "position": [0.0, 0.0, -head_size[2] / 2.0 - gap],
        "size": head_size,
        "material": {"color": MOSA_DARK},
    }
    snout_size = (0.60, 0.40, 1.00)
    parts["snout"] = {
        "parent": "head",
        "pivot": [0.0, 0.0, -head_size[2] / 2.0],
        "position": [0.0, -0.05, -snout_size[2] / 2.0 - gap],
        "size": snout_size,
        "material": {"color": MOSA_MID},
    }
    jaw_size = (0.60, 0.22, 1.00)
    parts["jaw"] = {
        "parent": "head",
        "pivot": [0.0, -head_size[1] / 2.0 + 0.04, -head_size[2] / 2.0 + 0.1],
        "position": [0.0, -jaw_size[1] / 2.0 - gap, -jaw_size[2] / 2.0 - 0.1],
        "size": jaw_size,
        "material": {"color": MOSA_BELLY},
    }

    for side, sign in (("left", -1.0), ("right", 1.0)):
        parts[f"{side}_eye"] = {
            "parent": "head",
            "pivot": [0.0, 0.0, 0.0],
            "position": [sign * 0.24, 0.14, -0.30],
            "size": (0.10, 0.10, 0.08),
            "material": {"color": MOSA_EYE},
        }

    tooth_size = (0.06, 0.18, 0.06)
    for idx in range(6):
        z_offset = -snout_size[2] / 2.0 + 0.18 + idx * 0.12
        for side, sign in (("left", -1.0), ("right", 1.0)):
            parts[f"tooth_upper_{side}_{idx}"] = {
                "parent": "snout",
                "pivot": [0.0, -snout_size[1] / 2.0 + 0.02, z_offset],
                "position": [sign * 0.12, -tooth_size[1] / 2.0, 0.0],
                "size": tooth_size,
                "material": {"color": MOSA_TEETH},
            }
            parts[f"tooth_lower_{side}_{idx}"] = {
                "parent": "jaw",
                "pivot": [0.0, jaw_size[1] / 2.0 - 0.02, z_offset - 0.05],
                "position": [sign * 0.12, tooth_size[1] / 2.0, 0.0],
                "size": tooth_size,
                "material": {"color": MOSA_TEETH},
            }

    fin_size = (1.00, 0.08, 0.60)
    for side, sign in (("left", -1.0), ("right", 1.0)):
        parts[f"{side}_pectoral_fin"] = {
            "parent": "body_2",
            "pivot": [sign * body_sizes[1][0] / 2.0, -0.20, -0.30],
            "position": [sign * fin_size[0] / 2.0, 0.0, 0.0],
            "size": fin_size,
            "material": {"color": MOSA_FIN},
        }

    dorsal_size = (0.18, 0.70, 0.50)
    parts["dorsal_fin"] = {
        "parent": "body_2",
        "pivot": [0.0, body_sizes[1][1] / 2.0, 0.1],
        "position": [0.0, dorsal_size[1] / 2.0, 0.0],
        "size": dorsal_size,
        "material": {"color": MOSA_FIN},
    }

    tail_sizes = [
        (1.00, 0.70, 1.80),
        (0.90, 0.60, 1.60),
        (0.80, 0.50, 1.40),
        (0.70, 0.45, 1.20),
        (0.60, 0.40, 1.00),
        (0.50, 0.35, 0.80),
    ]
    tail_parent = "body_1"
    parent_size = body_sizes[0]
    for idx, size in enumerate(tail_sizes, start=1):
        name = f"tail_{idx}"
        parts[name] = {
            "parent": tail_parent,
            "pivot": [0.0, 0.0, parent_size[2] / 2.0],
            "position": [0.0, 0.0, size[2] / 2.0 + gap],
            "size": size,
            "material": {"color": MOSA_DARK if idx % 2 else MOSA_MID},
        }
        tail_parent = name
        parent_size = size

    tail_fluke_size = (1.60, 0.08, 0.80)
    parts["tail_fluke"] = {
        "parent": tail_parent,
        "pivot": [0.0, 0.0, parent_size[2] / 2.0],
        "position": [0.0, 0.0, tail_fluke_size[2] / 2.0 + gap],
        "size": tail_fluke_size,
        "material": {"color": MOSA_FIN},
    }

    animations = {
        "idle": {
            "loop": True,
            "length": 2.0,
            "keyframes": [
                {
                    "time": 0.0,
                    "rotations": {
                        "tail_1": {"yaw": 6},
                        "tail_2": {"yaw": 10},
                        "tail_3": {"yaw": 14},
                        "tail_4": {"yaw": 10},
                        "tail_5": {"yaw": 8},
                        "tail_6": {"yaw": 6},
                        "tail_fluke": {"yaw": 12},
                    },
                },
                {
                    "time": 1.0,
                    "rotations": {
                        "tail_1": {"yaw": -6},
                        "tail_2": {"yaw": -10},
                        "tail_3": {"yaw": -14},
                        "tail_4": {"yaw": -10},
                        "tail_5": {"yaw": -8},
                        "tail_6": {"yaw": -6},
                        "tail_fluke": {"yaw": -12},
                    },
                },
                {
                    "time": 2.0,
                    "rotations": {
                        "tail_1": {"yaw": 6},
                        "tail_2": {"yaw": 10},
                        "tail_3": {"yaw": 14},
                        "tail_4": {"yaw": 10},
                        "tail_5": {"yaw": 8},
                        "tail_6": {"yaw": 6},
                        "tail_fluke": {"yaw": 12},
                    },
                },
            ],
        }
    }

    return {
        "root_part": "body_1",
        "parts": parts,
        "animations": animations,
    }


MOSASAURUS_MODEL = _scale_model(build_mosasaurus_model(), 1.15)


class MosasaurusEntity(BaseEntity):
    SPEED = 2.2
    VERTICAL_SPEED = 1.8
    TURN_WEIGHT = 0.015
    SURFACE_CLEARANCE = 2.5
    FLOOR_CLEARANCE = 2.5
    WANDER_SPEED = 0.25

    def __init__(self, world, player_position=None, entity_id=7, saved_state=None):
        initial_position = player_position if player_position is not None else (0, 120, 0)
        if saved_state and "pos" in saved_state:
            initial_position = saved_state["pos"]
        super().__init__(world, position=initial_position, entity_type="mosasaurus")
        self.id = entity_id
        self.model_definition = MOSASAURUS_MODEL
        self.bounding_box = np.array([4.0, 3.0, 10.0], dtype=float)
        self.rotation = np.array([0.0, 0.0], dtype=float)
        self.current_animation = "idle"
        self.flying = True
        self._wander_angle = random.random() * math.pi * 2.0
        self._heading = np.array([1.0, 0.0], dtype=float)
        self._alt_phase = random.random() * math.pi * 2.0
        if saved_state:
            rot = saved_state.get("rot")
            if rot is not None:
                self.rotation = np.array(rot, dtype=float)
        else:
            self._snap_to_water_band()

    def update(self, dt, context):
        if dt <= 0:
            return

        self._wander_angle += (dt * self.WANDER_SPEED) + random.uniform(-0.08, 0.08)
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
            target_y = min_y + span * 0.5 + math.sin(self._alt_phase) * (span * 0.2)
            target_y = float(np.clip(target_y, min_y, max_y))
            diff = target_y - float(self.position[1])
            self.velocity[1] = np.sign(diff) * min(abs(diff), self.VERTICAL_SPEED)
        else:
            self.velocity[1] = 0.0

        super().update(dt)
        self.rotation[1] = 0.0

    def to_network_dict(self):
        data = super().to_network_dict()
        data["type"] = "mosasaurus"
        data["animation"] = "idle"
        return data

    def serialize_state(self):
        return {"pos": tuple(self.position.tolist()), "rot": tuple(self.rotation.tolist())}

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
