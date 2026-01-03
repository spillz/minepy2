from __future__ import annotations
import math
from typing import Dict, Any, Tuple

import numpy as np

from entity import BaseEntity
from config import GRAVITY, TERMINAL_VELOCITY, JUMP_SPEED




def build_tetrapod_model(
    body_size: Tuple[float, float, float] = (0.4, 0.4, 0.8),
    leg_size: Tuple[float, float, float] = (0.15, 0.5, 0.15),
    head_size: Tuple[float, float, float] = (0.3, 0.3, 0.3),
    snout_size: Tuple[float, float, float] = (0.2, 0.2, 0.6),
    ear_size: Tuple[float, float, float] = (0.05, 0.1, 0.05),
    eye_size: Tuple[float, float, float] = (0.05, 0.05, 0.05),
    nose_size: Tuple[float, float, float] = (0.05, 0.05, 0.05),
    tail_size: Tuple[float, float, float] = (0.05, 0.05, 0.3),
    body_color: Tuple[int, int, int] = (160, 160, 160),
    leg_color: Tuple[int, int, int] = (140, 140, 140),
    head_color: Tuple[int, int, int] = (150, 150, 150),
    ear_color: Tuple[int, int, int] = (130, 130, 130),
    eye_color: Tuple[int, int, int] = (0, 0, 0),
    snout_color: Tuple[int, int, int] = (150, 150, 150),
    nose_color: Tuple[int, int, int] = (0, 0, 0),
    tail_color: Tuple[int, int, int] = (140, 140, 140),
) -> Dict[str, Any]:
    parts: Dict[str, Dict[str, Any]] = {}

    # Body
    parts["body"] = {
        "parent": None,
        "pivot": [0.0, leg_size[1], 0.0],
        "position": [0.0, 0.0, 0.0],
        "size": body_size,
        "material": {"color": body_color},
    }

    # Legs
    leg_positions = {
        "front_left_leg": [-body_size[0] / 2 + leg_size[0] /2 , -body_size[1]/2, -body_size[2] / 2 + leg_size[2] /2],
        "front_right_leg": [body_size[0] / 2 - leg_size[0] /2 , -body_size[1]/2, -body_size[2] / 2 + leg_size[2] /2],
        "back_left_leg": [-body_size[0] / 2 + leg_size[0] /2 ,  -body_size[1]/2, body_size[2] / 2 - leg_size[2] /2],
        "back_right_leg": [body_size[0] / 2 - leg_size[0] /2 ,  -body_size[1]/2, body_size[2] / 2 - leg_size[2] /2],
    }
    for name, pos in leg_positions.items():
        parts[name] = {
            "parent": "body",
            "pivot": pos,
            "position": [0, -leg_size[1] / 2, 0],
            "size": leg_size,
            "material": {"color": leg_color},
        }
    
    # Head
    parts["head"] = {
        "parent": "body",
        "pivot": [0, body_size[1] / 2, -body_size[2] / 2],
        "position": [0, head_size[1] / 2, 0],
        "size": head_size,
        "material": {"color": head_color},
    }

    # Ears
    parts["left_ear"] = {
        "parent": "head",
        "pivot": [-head_size[0]/2, head_size[1]/2, 0],
        "position": [0, ear_size[1]/2, 0],
        "size": ear_size,
        "material": {"color": ear_color},
    }
    parts["right_ear"] = {
        "parent": "head",
        "pivot": [head_size[0]/2, head_size[1]/2, 0],
        "position": [0, ear_size[1]/2, 0],
        "size": ear_size,
        "material": {"color": ear_color},
    }

    # Eyes
    parts["left_eye"] = {
        "parent": "head",
        "pivot": [-head_size[0]/4, 0.15*head_size[1], -head_size[2]/2],
        "position": [0, 0, 0],
        "size": eye_size,
        "material": {"color": eye_color},
    }
    parts["right_eye"] = {
        "parent": "head",
        "pivot": [head_size[0]/4, 0.15*head_size[1], -head_size[2]/2],
        "position": [0, 0, 0],
        "size": eye_size,
        "material": {"color": eye_color},
    }

    # snout
    parts["snout"] = {
        "parent": "head",
        "pivot": [0, -head_size[1]/2, +head_size[2]/2],
        "position": [0, snout_size[1]/2, -snout_size[2]/2],
        "size": snout_size,
        "material": {"color": snout_color},
    }

    # Nose
    parts["nose"] = {
        "parent": "snout",
        "pivot": [0, snout_size[1]/2, -snout_size[2]/2],
        "position": [0, -0.4*nose_size[1], 0],
        "size": nose_size,
        "material": {"color": nose_color},
    }

    # Tail
    parts["tail"] = {
        "parent": "body",
        "pivot": [0, 0, body_size[2] / 2],
        "position": [0, 0, tail_size[2] / 2],
        "size": tail_size,
        "material": {"color": tail_color},
    }

    model = {
        "root_part": "body",
        "parts": parts,
        "animations": {
            "idle": {
                "loop": True,
                "length": 2.0,
                "keyframes": [
                    {"time": 0.0, "rotations": {"tail": {"pitch": 10, "yaw": 10}}},
                    {"time": 1.0, "rotations": {"tail": {"pitch": 10, "yaw": -10}}},
                    {"time": 2.0, "rotations": {"tail": {"pitch": 10, "yaw": 10}}},
                ],
            },
            "walk": {
                "loop": True,
                "length": 1.0,
                "keyframes": [
                    {
                        "time": 0.0,
                        "rotations": {
                            "front_left_leg": {"pitch": 45},
                            "back_right_leg": {"pitch": 45},
                            "front_right_leg": {"pitch": -45},
                            "back_left_leg": {"pitch": -45},
                        },
                    },
                    {
                        "time": 0.5,
                        "rotations": {
                            "front_left_leg": {"pitch": -45},
                            "back_right_leg": {"pitch": -45},
                            "front_right_leg": {"pitch": 45},
                            "back_left_leg": {"pitch": 45},
                        },
                    },
                    {
                        "time": 1.0,
                        "rotations": {
                            "front_left_leg": {"pitch": 45},
                            "back_right_leg": {"pitch": 45},
                            "front_right_leg": {"pitch": -45},
                            "back_left_leg": {"pitch": -45},
                        },
                    },
                ],
            },
        },
    }
    
    return model

class Tetrapod(BaseEntity):
    def __init__(
        self,
        world,
        entity_type="tetrapod",
        model_definition=None,
        jump_height=1,
        jump_span=1,
        entity_id=None,
        saved_state=None,
    ):
        initial_position = (0, 160, 0)
        if saved_state and "pos" in saved_state:
            initial_position = saved_state["pos"]

        super().__init__(world, position=initial_position, entity_type=entity_type)

        if entity_id is not None:
            self.id = entity_id

        if saved_state and "rot" in saved_state:
            self.rotation = np.array(saved_state["rot"], dtype=float)

        self.model_definition = model_definition
        self.jump_height = jump_height
        self.jump_span = jump_span
        self.bounding_box = np.array([0.25, 0.5, 0.25])
        # bp = model_definition['parts'][model_definition['root_part']]
        # self.bounding_box = np.array([bp['size'][0], bp['size'][1], bp['size'][2]])
        self.current_animation = "idle"

    def jump(self):
        if self.on_ground:
            self.velocity[1] = JUMP_SPEED
            self.on_ground = False
