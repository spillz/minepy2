from __future__ import annotations

import math
import random
from typing import Dict, Any, Tuple

import numpy as np

from entity import BaseEntity


RGB = Tuple[int, int, int]

BONE_MAIN: RGB = (235, 235, 230)
BONE_LIGHT: RGB = (222, 222, 218)
BONE_DARK: RGB = (200, 200, 200)
EYE_GLOW: RGB = (30, 255, 80)


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


def build_dinotrex_model() -> Dict[str, Any]:
    parts: Dict[str, Dict[str, Any]] = {}

    gap = 0.04

    hip_size = (0.55, 0.35, 0.45)
    hip_pivot_y = 1.5
    parts["hip"] = {
        "parent": None,
        "pivot": [0.0, hip_pivot_y, 0.0],
        "position": [0.0, 0.0, 0.0],
        "size": hip_size,
        "material": {"color": BONE_MAIN},
    }

    spine_sizes = [
        (0.52, 0.30, 0.36),
        (0.50, 0.29, 0.36),
        (0.48, 0.28, 0.34),
        (0.46, 0.27, 0.32),
    ]
    spine_parent = "hip"
    parent_size = hip_size
    for idx, size in enumerate(spine_sizes, start=1):
        parts[f"spine_{idx}"] = {
            "parent": spine_parent,
            "pivot": [0.0, parent_size[1] * 0.2, -parent_size[2] / 2.0],
            "position": [0.0, 0.0, -size[2] / 2.0 - gap],
            "size": size,
            "material": {"color": BONE_LIGHT if idx % 2 else BONE_MAIN},
        }
        spine_parent = f"spine_{idx}"
        parent_size = size

    rib_size = (0.42, 0.08, 0.18)
    for idx in range(1, 4):
        spine_name = f"spine_{idx}"
        spine_w = spine_sizes[idx - 1][0]
        rib_offset_x = spine_w / 2.0 + rib_size[0] / 2.0 + gap
        for side, sign in (("left", -1.0), ("right", 1.0)):
            parts[f"ribs_{idx}_{side}"] = {
                "parent": spine_name,
                "pivot": [0.0, 0.0, 0.0],
                "position": [sign * rib_offset_x, -0.05, 0.0],
                "size": rib_size,
                "material": {"color": BONE_DARK},
            }

    tail_sizes = [
        (0.26, 0.24, 0.36),
        (0.24, 0.22, 0.34),
        (0.22, 0.20, 0.30),
        (0.18, 0.18, 0.26),
        (0.14, 0.14, 0.22),
        (0.10, 0.10, 0.18),
    ]
    tail_parent = "hip"
    parent_size = hip_size
    for idx, size in enumerate(tail_sizes, start=1):
        parts[f"tail_{idx}"] = {
            "parent": tail_parent,
            "pivot": [0.0, 0.02, parent_size[2] / 2.0],
            "position": [0.0, 0.0, size[2] / 2.0 + gap],
            "size": size,
            "material": {"color": BONE_MAIN if idx <= 2 else BONE_DARK},
        }
        tail_parent = f"tail_{idx}"
        parent_size = size

    upper_leg_size = (0.234, 0.75, 0.234)
    lower_leg_size = (0.208, 0.675, 0.208)
    foot_size = (0.06, 0.09, 0.18)
    foot_gap = 0.03
    leg_x = hip_size[0] / 2.0 + upper_leg_size[0] / 2.0 + 0.02
    for side, sign in (("left", -1.0), ("right", 1.0)):
        upper_name = f"{side}_upper_leg"
        lower_name = f"{side}_lower_leg"
        parts[upper_name] = {
            "parent": "hip",
            "pivot": [sign * leg_x, -hip_size[1] / 2.0 + 0.02, 0.05],
            "position": [0.0, -upper_leg_size[1] / 2.0 - gap, 0.0],
            "size": upper_leg_size,
            "material": {"color": BONE_LIGHT},
        }
        parts[lower_name] = {
            "parent": upper_name,
            "pivot": [0.0, -upper_leg_size[1] / 2.0, 0.0],
            "position": [0.0, -lower_leg_size[1] / 2.0 - gap, 0.0],
            "size": lower_leg_size,
            "material": {"color": BONE_MAIN},
        }
        toe_offsets = [(-0.06, -0.02), (0.0, -0.03), (0.06, -0.02), (0.0, 0.12)]
        for foot_idx, (toe_x, toe_z) in enumerate(toe_offsets, start=1):
            foot_name = f"{side}_foot_{foot_idx}"
            parts[foot_name] = {
                "parent": lower_name,
                "pivot": [
                    toe_x,
                    -lower_leg_size[1] / 2.0 + foot_size[1] / 2.0,
                    -lower_leg_size[2] / 2.0 + toe_z + 0.04,
                ],
                "position": [0.0, 0.0, -foot_size[2] / 2.0 - foot_gap],
                "size": foot_size,
                "material": {"color": BONE_DARK},
            }

    upper_arm_size = (0.12, 0.28, 0.12)
    lower_arm_size = (0.10, 0.24, 0.10)
    hand_size = (0.04, 0.06, 0.10)
    arm_x = spine_sizes[2][0] / 2.0 + upper_arm_size[0] / 2.0 + 0.02
    for side, sign in (("left", -1.0), ("right", 1.0)):
        upper_name = f"{side}_upper_arm"
        lower_name = f"{side}_lower_arm"
        parts[upper_name] = {
            "parent": "spine_3",
            "pivot": [sign * arm_x, 0.02, -spine_sizes[2][2] / 2.0 + 0.04],
            "position": [0.0, -upper_arm_size[1] / 2.0 - gap, 0.0],
            "size": upper_arm_size,
            "material": {"color": BONE_DARK},
        }
        parts[lower_name] = {
            "parent": upper_name,
            "pivot": [0.0, -upper_arm_size[1] / 2.0, 0.0],
            "position": [0.0, -lower_arm_size[1] / 2.0 - gap, 0.0],
            "size": lower_arm_size,
            "material": {"color": BONE_MAIN},
        }
        hand_offsets = [(-0.03, 0.0), (0.03, 0.0)]
        for hand_idx, (hand_x, hand_z) in enumerate(hand_offsets, start=1):
            hand_name = f"{side}_hand_{hand_idx}"
            parts[hand_name] = {
                "parent": lower_name,
                "pivot": [
                    hand_x,
                    -lower_arm_size[1] / 2.0 + hand_size[1] / 2.0,
                    -lower_arm_size[2] / 2.0 + hand_z,
                ],
                "position": [0.0, 0.0, -hand_size[2] / 2.0 - foot_gap],
                "size": hand_size,
                "material": {"color": BONE_LIGHT},
            }

    skull_top_size = (0.30, 0.20, 0.30)
    skull_side_size = (0.16, 0.20, 0.30)
    snout_size = (0.12, 0.16, 0.32)
    jaw_size = (0.12, 0.08, 0.28)
    parts['skull_top'] = {
            "parent": "spine_4",
            "pivot": [0.0, spine_sizes[3][1] * 0.1, -spine_sizes[3][2] / 2.0],
            "position": [0, 0.2, -skull_top_size[2] / 2.0 - gap],
            "size": skull_top_size,
            "material": {"color": BONE_LIGHT},        
    }
    for side, sign in (("left", -1.0), ("right", 1.0)):
        skull_name = f"skull_{side}"
        snout_name = f"snout_{side}"
        jaw_name = f"jaw_{side}"
        parts[skull_name] = {
            "parent": "skull_top",
            "pivot": [0.0, -skull_top_size[1]/2, 0],
            "position": [sign * (skull_side_size[0] / 2.0 + 0.07), 0.03, 0],
            "size": skull_side_size,
            "material": {"color": BONE_LIGHT},
        }
        parts[snout_name] = {
            "parent": skull_name,
            "pivot": [0.0, -0.02, -skull_side_size[2] / 2.0],
            "position": [0.0, 0.0, -snout_size[2] / 2.0 - gap],
            "size": snout_size,
            "material": {"color": BONE_MAIN},
        }
        parts[f"{side}_snout_tooth_front"] = {
            "parent": snout_name,
            "pivot": [0.0, -snout_size[1] / 2.0, -snout_size[2] / 2.0],
            "position": [sign * 0.02, -0.05, 0.06],
            "size": (0.03, 0.10, 0.03),
            "material": {"color": BONE_LIGHT},
        }
        parts[jaw_name] = {
            "parent": skull_name,
            "pivot": [0.0, -skull_side_size[1] / 2.0 + 0.02, -0.08],
            "position": [0.0, -jaw_size[1] / 2.0 - gap, -jaw_size[2] / 2.0 - 0.10],
            "size": jaw_size,
            "material": {"color": BONE_DARK},
        }
        for tooth_idx in range(4):
            z_offset = -jaw_size[2] / 2.0 + 0.12 + tooth_idx * 0.04
            x_offset = -0.03 if tooth_idx % 2 == 0 else 0.03
            parts[f"{side}_tooth_{tooth_idx + 1}"] = {
                "parent": jaw_name,
                "pivot": [0.0, 0.0, 0.0],
                "position": [x_offset, jaw_size[1] / 2.0 - 0.02, z_offset],
                "size": (0.02, 0.08, 0.02),
                "material": {"color": BONE_LIGHT},
            }
    for side, sign in (("left", -1.0), ("right", 1.0)):
        eye_name = f"{side}_eye"
        parts[eye_name] = {
            "parent": f"skull_{side}",
            "pivot": [0.0, 0.0, 0.0],
            "position": [sign * 0.04, 0.12, -0.16],
            "size": (0.05, 0.05, 0.05),
            "material": {"color": EYE_GLOW},
        }

    animations = {
        "idle": {
            "loop": True,
            "length": 2.0,
            "keyframes": [
                {
                    "time": 0.0,
                    "rotations": {
                        "hip": {"pitch": 45},
                        "spine_1": {"pitch": -10},
                        "spine_2": {"pitch": -10},
                        "spine_3": {"pitch": -10},
                        "spine_4": {"pitch": -10},
                        "skull_top": {"pitch": -25},
                        # "skull_left": {"pitch": -25},
                        # "skull_right": {"pitch": -25},
                        "left_upper_leg": {"pitch": -45},
                        "right_upper_leg": {"pitch": -45},
                        "left_lower_leg": {"pitch": -15},
                        "right_lower_leg": {"pitch": -15},
                        "left_foot_1": {"pitch": -35, "yaw": 16},
                        "left_foot_2": {"pitch": -35, "yaw": 0},
                        "left_foot_3": {"pitch": -35, "yaw": -16},
                        "left_foot_4": {"pitch": 35, "yaw": 180},
                        "right_foot_1": {"pitch": -35, "yaw": 16},
                        "right_foot_2": {"pitch": -35, "yaw": 0},
                        "right_foot_3": {"pitch": -35, "yaw": -16},
                        "right_foot_4": {"pitch": 35, "yaw": 180},
                        "left_hand_1": {"yaw": 16},
                        "left_hand_2": {"yaw": -16},
                        "right_hand_1": {"yaw": 16},
                        "right_hand_2": {"yaw": -16},
                        "jaw_left": {"pitch": -12},
                        "jaw_right": {"pitch": -12},
                        "tail_1": {"yaw": 6, "pitch": -10},
                        "tail_2": {"yaw": 8, "pitch": -12},
                        "tail_3": {"yaw": 10, "pitch": -12},
                        "tail_4": {"yaw": 8, "pitch": -10},
                        "tail_5": {"yaw": 6, "pitch": -10},
                        "tail_6": {"yaw": 4, "pitch": -8},
                    },
                },
                {
                    "time": 1.0,
                    "rotations": {
                        "hip": {"pitch": 45},
                        "spine_1": {"pitch": -10},
                        "spine_2": {"pitch": -10},
                        "spine_3": {"pitch": -10},
                        "spine_4": {"pitch": -10},
                        "skull_top": {"pitch": -25},
                        # "skull_left": {"pitch": -25},
                        # "skull_right": {"pitch": -25},
                        "left_upper_leg": {"pitch": -45},
                        "right_upper_leg": {"pitch": -45},
                        "left_lower_leg": {"pitch": -15},
                        "right_lower_leg": {"pitch": -15},
                        "left_foot_1": {"pitch": -35, "yaw": 16},
                        "left_foot_2": {"pitch": -35, "yaw": 0},
                        "left_foot_3": {"pitch": -35, "yaw": -16},
                        "left_foot_4": {"pitch": 35, "yaw": 180},
                        "right_foot_1": {"pitch": -35, "yaw": 16},
                        "right_foot_2": {"pitch": -35, "yaw": 0},
                        "right_foot_3": {"pitch": -35, "yaw": -16},
                        "right_foot_4": {"pitch": 35, "yaw": 180},
                        "left_hand_1": {"yaw": 16},
                        "left_hand_2": {"yaw": -16},
                        "right_hand_1": {"yaw": 16},
                        "right_hand_2": {"yaw": -16},
                        "jaw_left": {"pitch": -12},
                        "jaw_right": {"pitch": -12},
                        "tail_1": {"yaw": -6, "pitch": -10},
                        "tail_2": {"yaw": -8, "pitch": -12},
                        "tail_3": {"yaw": -10, "pitch": -12},
                        "tail_4": {"yaw": -8, "pitch": -10},
                        "tail_5": {"yaw": -6, "pitch": -10},
                        "tail_6": {"yaw": -4, "pitch": -8},
                    },
                },
                {
                    "time": 2.0,
                    "rotations": {
                        "hip": {"pitch": 45},
                        "spine_1": {"pitch": -10},
                        "spine_2": {"pitch": -10},
                        "spine_3": {"pitch": -10},
                        "spine_4": {"pitch": -10},
                        "skull_top": {"pitch": -25},
                        # "skull_left": {"pitch": -25},
                        # "skull_right": {"pitch": -25},
                        "left_upper_leg": {"pitch": -45},
                        "right_upper_leg": {"pitch": -45},
                        "left_lower_leg": {"pitch": -15},
                        "right_lower_leg": {"pitch": -15},
                        "left_foot_1": {"pitch": -35, "yaw": 16},
                        "left_foot_2": {"pitch": -35, "yaw": 0},
                        "left_foot_3": {"pitch": -35, "yaw": -16},
                        "left_foot_4": {"pitch": 35, "yaw": 180},
                        "right_foot_1": {"pitch": -35, "yaw": 16},
                        "right_foot_2": {"pitch": -35, "yaw": 0},
                        "right_foot_3": {"pitch": -35, "yaw": -16},
                        "right_foot_4": {"pitch": 35, "yaw": 180},
                        "left_hand_1": {"yaw": 16},
                        "left_hand_2": {"yaw": -16},
                        "right_hand_1": {"yaw": 16},
                        "right_hand_2": {"yaw": -16},
                        "jaw_left": {"pitch": -12},
                        "jaw_right": {"pitch": -12},
                        "tail_1": {"yaw": 6, "pitch": -10},
                        "tail_2": {"yaw": 8, "pitch": -12},
                        "tail_3": {"yaw": 10, "pitch": -12},
                        "tail_4": {"yaw": 8, "pitch": -10},
                        "tail_5": {"yaw": 6, "pitch": -10},
                        "tail_6": {"yaw": 4, "pitch": -8},
                    },
                },
            ],
        },
        "collapse": {
            "loop": False,
            "length": 1.0,
            "keyframes": [
                {
                    "time": 0.0,
                    "transforms": {
                        "hip": {"pitch": 0, "roll": 15, "dpy":-7},
                        "spine_1": {"pitch": 0, "roll": 8, "dpy":-0.25},
                        "spine_2": {"pitch": 0, "roll": -6, "dpy":-0.25},
                        "spine_3": {"pitch": 0, "roll": 6, "dpy":-0.25},
                        "spine_4": {"pitch": 0, "roll": -8, "dpy":-0.25},
                        "left_upper_leg": {"pitch": 90, "roll": -40},
                        "right_upper_leg": {"pitch": 90, "roll": 40},
                        "left_lower_leg": {"pitch": 90, "roll": -20},
                        "right_lower_leg": {"pitch": 0, "roll": 20},
                        "left_upper_arm": {"pitch": 90, "roll": -45},
                        "right_upper_arm": {"pitch": 90, "roll": 45},
                        "left_lower_arm": {"pitch": 0, "roll": -20},
                        "right_lower_arm": {"pitch": 0, "roll": 20},
                        "tail_1": {"pitch": 0},
                        "tail_2": {"pitch": 0, "yaw": -10},
                        "tail_3": {"pitch": 0, "yaw": 10},
                        "tail_4": {"pitch": 0, "yaw": -10},
                        "tail_5": {"pitch": 0, "yaw": 8},
                        "tail_6": {"pitch": 0, "yaw": -6},
                        "jaw_left": {"pitch": 0},
                        "jaw_right": {"pitch": 0},
                        "left_eye": {"dpy":-0.25, "dpz":0.5},
                        "right_eye": {"dpy":-0.25, "dpz":0.5},
                    },
                }
            ],
        },
        "walk": {
            "loop": True,
            "length": 4.0,
            "keyframes": [
                {
                    "time": 0.0,
                    "rotations": {
                        "left_upper_leg": {"pitch": 45},
                        "right_upper_leg": {"pitch": 5},
                        "left_lower_leg": {"pitch": -20},
                        "right_lower_leg": {"pitch": -40},
                        "left_foot_1": {"pitch": -35, "yaw": 16},
                        "left_foot_2": {"pitch": -35, "yaw": 0},
                        "left_foot_3": {"pitch": -35, "yaw": -16},
                        "left_foot_4": {"pitch": -35, "yaw": 180},
                        "right_foot_1": {"pitch": 35, "yaw": 16},
                        "right_foot_2": {"pitch": -35, "yaw": 0},
                        "right_foot_3": {"pitch": -35, "yaw": -16},
                        "right_foot_4": {"pitch": 35, "yaw": 180},
                        "left_hand_1": {"yaw": 16},
                        "left_hand_2": {"yaw": -16},
                        "right_hand_1": {"yaw": 16},
                        "right_hand_2": {"yaw": -16},
                        "jaw_left": {"pitch": -12},
                        "jaw_right": {"pitch": -12},
                        "left_upper_arm": {"pitch": -10},
                        "right_upper_arm": {"pitch": 10},
                        "tail_1": {"yaw": 14},
                        "tail_2": {"yaw": 18},
                        "tail_3": {"yaw": 20},
                        "tail_4": {"yaw": 18},
                        "tail_5": {"yaw": 14},
                        "tail_6": {"yaw": 10},
                    },
                },
                {
                    "time": 2.0,
                    "rotations": {
                        "left_upper_leg": {"pitch": 5},
                        "right_upper_leg": {"pitch": 45},
                        "left_lower_leg": {"pitch": -40},
                        "right_lower_leg": {"pitch": -20},
                        "left_foot_1": {"pitch": -35, "yaw": 16},
                        "left_foot_2": {"pitch": -35, "yaw": 0},
                        "left_foot_3": {"pitch": -35, "yaw": -16},
                        "left_foot_4": {"pitch": 35, "yaw": 180},
                        "right_foot_1": {"pitch": -35, "yaw": 16},
                        "right_foot_2": {"pitch": -35, "yaw": 0},
                        "right_foot_3": {"pitch": -35, "yaw": -16},
                        "right_foot_4": {"pitch": 35, "yaw": 180},
                        "left_hand_1": {"yaw": 16},
                        "left_hand_2": {"yaw": -16},
                        "right_hand_1": {"yaw": 16},
                        "right_hand_2": {"yaw": -16},
                        "jaw_left": {"pitch": -12},
                        "jaw_right": {"pitch": -12},
                        "left_upper_arm": {"pitch": 10},
                        "right_upper_arm": {"pitch": -10},
                        "tail_1": {"yaw": -14},
                        "tail_2": {"yaw": -18},
                        "tail_3": {"yaw": -20},
                        "tail_4": {"yaw": -18},
                        "tail_5": {"yaw": -14},
                        "tail_6": {"yaw": -10},
                    },
                },
                {
                    "time": 3.99,
                    "rotations": {
                        "left_upper_leg": {"pitch": 45},
                        "right_upper_leg": {"pitch": 5},
                        "left_lower_leg": {"pitch": -20},
                        "right_lower_leg": {"pitch": -40},
                        "left_foot_1": {"pitch": -35, "yaw": 16},
                        "left_foot_2": {"pitch": -35, "yaw": 0},
                        "left_foot_3": {"pitch": -35, "yaw": -16},
                        "left_foot_4": {"pitch": -35, "yaw": 180},
                        "right_foot_1": {"pitch": -35, "yaw": 16},
                        "right_foot_2": {"pitch": -35, "yaw": 0},
                        "right_foot_3": {"pitch": -35, "yaw": -16},
                        "right_foot_4": {"pitch": -35, "yaw": 180},
                        "left_hand_1": {"yaw": 16},
                        "left_hand_2": {"yaw": -16},
                        "right_hand_1": {"yaw": 16},
                        "right_hand_2": {"yaw": -16},
                        "jaw_left": {"pitch": -12},
                        "jaw_right": {"pitch": -12},
                        "left_upper_arm": {"pitch": -10},
                        "right_upper_arm": {"pitch": 10},
                        "tail_1": {"yaw": 14},
                        "tail_2": {"yaw": 18},
                        "tail_3": {"yaw": 20},
                        "tail_4": {"yaw": 18},
                        "tail_5": {"yaw": 14},
                        "tail_6": {"yaw": 10},
                    },
                },
            ],
        },
        "step": {
            "loop": False,
            "length": 0.25,
            "keyframes": [
                {
                    "time": 0.0,
                    "rotations": {
                        "left_upper_leg": {"pitch": 45},
                        "right_upper_leg": {"pitch": 5},
                        "left_lower_leg": {"pitch": -20},
                        "right_lower_leg": {"pitch": -40},
                        "left_foot_1": {"pitch": -35, "yaw": 16},
                        "left_foot_2": {"pitch": -35, "yaw": 0},
                        "left_foot_3": {"pitch": -35, "yaw": -16},
                        "left_foot_4": {"pitch": -35, "yaw": 180},
                        "right_foot_1": {"pitch": 35, "yaw": 16},
                        "right_foot_2": {"pitch": -35, "yaw": 0},
                        "right_foot_3": {"pitch": -35, "yaw": -16},
                        "right_foot_4": {"pitch": 35, "yaw": 180},
                        "left_hand_1": {"yaw": 16},
                        "left_hand_2": {"yaw": -16},
                        "right_hand_1": {"yaw": 16},
                        "right_hand_2": {"yaw": -16},
                        "jaw_left": {"pitch": -12},
                        "jaw_right": {"pitch": -12},
                        "left_upper_arm": {"pitch": -10},
                        "right_upper_arm": {"pitch": 10},
                        "tail_1": {"yaw": 14},
                        "tail_2": {"yaw": 18},
                        "tail_3": {"yaw": 20},
                        "tail_4": {"yaw": 18},
                        "tail_5": {"yaw": 14},
                        "tail_6": {"yaw": 10},
                    },
                }
            ],
        },
    }

    return {
        "root_part": "hip",
        "parts": parts,
        "animations": animations,
        "animation_blend": 1.0,
    }


DINOTREX_MODEL = _scale_model(build_dinotrex_model(), 4.0)


class DinoTrexEntity(BaseEntity):
    SPEED = 1.2
    TURN_JITTER = 0.6
    TURN_INTERVAL = (1.4, 3.2)
    IDLE_DURATION = (5.0, 10.0)
    IDLE_COOLDOWN = (8.0, 16.0)
    WAKE_DISTANCE = 2.0
    TURN_LERP_SPEED = (1.2, 2.2)
    STEP_DURATION = 0.28
    STEP_HEIGHT = 1.05
    ANIM_BLEND = 0.15

    def __init__(self, world, entity_id=6, saved_state=None):
        initial_position = (0, 160, 0)
        if saved_state and "pos" in saved_state:
            initial_position = saved_state["pos"]
        super().__init__(world, position=initial_position, entity_type="dinotrex")
        self.id = entity_id
        self.model_definition = DINOTREX_MODEL
        self.bounding_box = np.array([2.4, 3.2, 3.0], dtype=float)
        self.rotation = np.array([0.0, 0.0], dtype=float)
        self.current_animation = "collapse"
        self._wander_angle = random.random() * math.pi * 2.0
        self._wander_timer = random.uniform(*self.TURN_INTERVAL)
        self._turn_target_angle = None
        self._turn_lerp_speed = random.uniform(*self.TURN_LERP_SPEED)
        self._idle_timer = 0.0
        self._idle_cooldown = random.uniform(*self.IDLE_COOLDOWN)
        self._collapsed = True
        self._step_timer = 0.0
        self._step_start = None
        self._step_target = None
        if saved_state:
            rot = saved_state.get("rot")
            if rot is not None:
                self.rotation = np.array(rot, dtype=float)
        else:
            self.snap_to_ground()

    def update(self, dt, context):
        if dt <= 0:
            return
        if self._step_timer > 0.0:
            self._advance_step(dt)
            return
        player_pos = context.get("player_position") if context else None
        if self._collapsed:
            self.velocity[:] = 0.0
            self.current_animation = "collapse"
            if player_pos is not None:
                delta = np.array(player_pos, dtype=float) - self.position
                if float(np.dot(delta, delta)) <= self.WAKE_DISTANCE * self.WAKE_DISTANCE:
                    self._collapsed = False
                    self._idle_timer = random.uniform(*self.IDLE_DURATION)
                    self._idle_cooldown = random.uniform(*self.IDLE_COOLDOWN)
            super().update(dt)
            return

        if self._idle_timer > 0.0:
            self._idle_timer = max(0.0, self._idle_timer - dt)
            self.velocity[:] = 0.0
            self.current_animation = "idle"
            super().update(dt)
            return
        if self._idle_cooldown > 0.0:
            self._idle_cooldown = max(0.0, self._idle_cooldown - dt)
        else:
            self._idle_timer = random.uniform(*self.IDLE_DURATION)
            self._idle_cooldown = random.uniform(*self.IDLE_COOLDOWN)
            self.velocity[:] = 0.0
            self.current_animation = "idle"
            super().update(dt)
            return

        self._wander_timer -= dt
        if self._wander_timer <= 0.0:
            self._wander_timer = random.uniform(*self.TURN_INTERVAL)
            self._wander_angle += random.uniform(-self.TURN_JITTER, self.TURN_JITTER)

        if self._turn_target_angle is not None:
            self._wander_angle = self._lerp_angle(
                self._wander_angle, self._turn_target_angle, dt * self._turn_lerp_speed
            )
            if abs(self._angle_delta(self._wander_angle, self._turn_target_angle)) < 1e-3:
                self._turn_target_angle = None

        move_dir = np.array([math.cos(self._wander_angle), math.sin(self._wander_angle)], dtype=float)
        norm = np.linalg.norm(move_dir)
        moving = False
        prev_pos = self.position.copy()
        if norm > 1e-6:
            move_dir /= norm
            self.velocity[0] = move_dir[0] * self.SPEED
            self.velocity[2] = move_dir[1] * self.SPEED
            moving = True
            self.rotation[0] = math.degrees(math.atan2(-move_dir[0], -move_dir[1]))
        else:
            self.velocity[0] = 0.0
            self.velocity[2] = 0.0

        super().update(dt)
        if moving:
            delta = self.position - prev_pos
            if (delta[0] * delta[0] + delta[2] * delta[2]) < 1e-4:
                if not self._try_step():
                    self._turn_target_angle = self._pick_clear_direction()
                    self._turn_lerp_speed = random.uniform(*self.TURN_LERP_SPEED)
                    self._wander_timer = min(self._wander_timer, 1.2)
        self.current_animation = "walk" if moving else "idle"


    def to_network_dict(self):
        data = super().to_network_dict()
        data["type"] = "dinotrex"
        data["animation"] = self.current_animation
        data["anim_blend"] = self.ANIM_BLEND
        return data

    def serialize_state(self):
        return {"pos": tuple(self.position.tolist()), "rot": tuple(self.rotation.tolist())}

    def _try_step(self):
        if not hasattr(self.world, "find_solid_surface_y"):
            return False
        base_y = self.world.find_solid_surface_y(self.position[0], self.position[2])
        if base_y is None:
            return False
        forward = np.array([math.cos(self._wander_angle), math.sin(self._wander_angle)], dtype=float)
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            return False
        forward /= forward_norm
        probe = float(self.bounding_box[2] * 0.5 + 0.2)
        target_x = float(self.position[0] + forward[0] * probe)
        target_z = float(self.position[2] + forward[1] * probe)
        target_y = self.world.find_solid_surface_y(target_x, target_z)
        if target_y is None:
            return False
        rise = target_y - base_y
        if rise < 0.6 or rise > 1.2:
            return False
        self._step_timer = self.STEP_DURATION
        self._step_start = self.position.copy()
        self._step_target = np.array([target_x, target_y, target_z], dtype=float)
        self.velocity[:] = 0.0
        self.current_animation = "step"
        return True

    def _advance_step(self, dt):
        if self._step_start is None or self._step_target is None:
            self._step_timer = 0.0
            return
        elapsed = max(0.0, self.STEP_DURATION - self._step_timer)
        self._step_timer = max(0.0, self._step_timer - dt)
        t = min(1.0, elapsed / self.STEP_DURATION) if self.STEP_DURATION > 0 else 1.0
        lift_phase = 0.4
        if t <= lift_phase:
            up_t = t / lift_phase if lift_phase > 0 else 1.0
            target = self._step_start.copy()
            target[1] = self._step_start[1] + self.STEP_HEIGHT * up_t
        else:
            move_t = (t - lift_phase) / (1.0 - lift_phase)
            target = self._step_start + (self._step_target - self._step_start) * move_t
        self.position = target
        self.velocity[:] = 0.0
        self.on_ground = True
        self.current_animation = "step"
        if self._step_timer <= 0.0:
            self.position = self._step_target.copy()
            self._step_start = None
            self._step_target = None
            self.current_animation = "walk"

    def _pick_clear_direction(self):
        base = self._wander_angle
        offsets = [0.0, math.pi / 3, -math.pi / 3, math.pi / 2, -math.pi / 2, math.pi * 0.75, -math.pi * 0.75]
        forward_dist = 1.2
        current_y = None
        if hasattr(self.world, "find_surface_y"):
            current_y = self.world.find_surface_y(self.position[0], self.position[2])
        if current_y is None:
            current_y = float(self.position[1])

        candidates = []
        for offset in offsets:
            angle = base + offset + random.uniform(-0.2, 0.2)
            dx = math.cos(angle) * forward_dist
            dz = math.sin(angle) * forward_dist
            test_x = float(self.position[0] + dx)
            test_z = float(self.position[2] + dz)
            if hasattr(self.world, "find_surface_y"):
                test_y = self.world.find_surface_y(test_x, test_z)
                if test_y is None:
                    continue
                if abs(test_y - current_y) > 1.6:
                    continue
            candidates.append(angle)
        if candidates:
            return random.choice(candidates)
        return base + random.uniform(-math.pi, math.pi)

    def _angle_delta(self, a, b):
        return (b - a + math.pi) % (2 * math.pi) - math.pi

    def _lerp_angle(self, a, b, t):
        t = max(0.0, min(1.0, float(t)))
        return a + self._angle_delta(a, b) * t
