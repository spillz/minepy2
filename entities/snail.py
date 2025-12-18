from __future__ import annotations

import math
import random
from typing import Dict, Any, Tuple

import numpy as np

from entity import BaseEntity


def build_snail_model(
    foot_size: Tuple[float, float, float] = (0.20, 0.20, 1.30),
    shell_size: Tuple[float, float, float] = (0.25, 0.48, 0.48),
    shell_inner_size: Tuple[float, float, float] = (0.35, 0.36, 0.36),
    stalk_size: Tuple[float, float, float] = (0.04, 0.24, 0.04),
) -> Dict[str, Any]:
    parts: Dict[str, Dict[str, Any]] = {}
    foot_w, foot_h, foot_d = foot_size
    parts["foot"] = {
        "parent": None,
        "pivot": [0.0, 0.0, 0.0],
        "position": [0.0, 0.0, 0.0],
        "size": [foot_w, foot_h, foot_d],
        "material": {"color": (90, 140, 80)},
    }
    parts["shell"] = {
        "parent": "foot",
        "pivot": [0.0, foot_h / 2.0, 0.0],
        "position": [0.0, shell_size[1] / 2.0, 0.0],
        "size": [shell_size[0], shell_size[1], shell_size[2]],
        "material": {"color": (180, 110, 70)},
    }
    parts["shell_inner"] = {
        "parent": "foot",
        "pivot": [0.0, foot_h / 2.0, 0.0],
        "position": [0.0, shell_size[1] / 2.0, 0.0],
        "size": [shell_inner_size[0], shell_inner_size[1], shell_inner_size[2]],
        "material": {"color": (120, 90, 60)},
    }
    for side, offset in (("stalk_left", -0.08), ("stalk_right", 0.08)):
        parts[side] = {
            "parent": "foot",
            "pivot": [0.0, -stalk_size[1] / 2.0, 0.0],
            "position": [offset, foot_h / 2.0 + stalk_size[1] / 2.0, -foot_d / 2.0],
            "size": [stalk_size[0], stalk_size[1], stalk_size[2]],
            "material": {"color": (60, 60, 60)},
        }
    model = {
        "root_part": "foot",
        "root_offset": [0.0, 0.1, 0.0],
        "parts": parts,
        "animations": {
            "idle": {"loop": True, "length": 1.0, "keyframes": [{"time": 0.0, "rotations": {}}, {"time": 1.0, "rotations": {}}]}
        },
    }
    return model


SNAIL_MODEL = build_snail_model()


class SnailEntity(BaseEntity):
    SPEED = 0.14
    FOLLOW_DISTANCE = 5.0
    MAX_ROAM_DISTANCE = 14.0
    HEAD_CLEARANCE = 0.05
    HISTORY_SCALAR = 2.0

    def __init__(self, world, player_position, entity_id=2, saved_state=None):
        super().__init__(world, position=player_position)
        self.type = "snail"
        self.model_definition = SNAIL_MODEL
        self.id = entity_id
        self.bounding_box = np.array([0.6, 0.35, 0.6], dtype=float)
        self.rotation = np.array([0.0, 0.0], dtype=float)
        self.current_animation = "idle"
        self._wander_angle = random.random() * math.pi * 2
        self._wander_speed = 0.4
        self._heading = np.array([0.0, 1.0], dtype=float)
        if saved_state:
            self.position = np.array(saved_state.get("pos", self.position), dtype=float)
            rot = saved_state.get("rot")
            if rot:
                self.rotation = np.array(rot, dtype=float)
        else:
            self.position = player_position
            self.snap_to_ground()

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
        if dist > 1e-6:
            target_dir = np.array([dx / dist, dz / dist], dtype=float)
        else:
            target_dir = np.zeros(2, dtype=float)
        moving = False
        desired_dir = None
        if dist > self.MAX_ROAM_DISTANCE:
            desired_dir = target_dir
        elif dist < self.FOLLOW_DISTANCE:
            desired_dir = -target_dir
        else:
            desired_dir = self._heading

        if desired_dir is not None:
            norm_dir = np.linalg.norm(desired_dir)
            if norm_dir > 1e-6:
                desired_dir = desired_dir / norm_dir
                turn_weight = 0.02
                self._heading = self._heading * (1 - turn_weight) + desired_dir * turn_weight
        noise_angle = random.uniform(-0.01, 0.01)
        heading_angle = math.atan2(self._heading[1], self._heading[0]) + noise_angle
        self._heading = np.array([math.cos(heading_angle), math.sin(heading_angle)], dtype=float)

        move_dir = self._heading
        norm = np.linalg.norm(move_dir)
        if norm > 1e-6:
            move_dir /= norm
            self.velocity[0] = move_dir[0] * self.SPEED
            self.velocity[2] = move_dir[1] * self.SPEED
            moving = True
            self.rotation[0] = math.degrees(math.atan2(-self._heading[0], -self._heading[1]))
        else:
            self.velocity[0] = 0
            self.velocity[2] = 0
        
        super().update(dt)
        
        self.current_animation = "walk" if moving else "idle"

    def serialize_state(self):
        return {"pos": tuple(self.position.tolist()), "rot": tuple(self.rotation.tolist())}
