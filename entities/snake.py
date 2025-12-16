
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

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np

from entity import BaseEntity


RGB = Tuple[int, int, int]


BLACK: RGB = (0, 0, 0)
RED: RGB = (220, 40, 40)


def build_snake_model(
    num_segments: int = 60,
    segment_size: Tuple[float, float, float] = (0.40, 0.30, 0.35),
    head_main_size: Tuple[float, float, float] = (0.55, 0.35, 0.55),
    head_snout_size: Tuple[float, float, float] = (0.50, 0.28, 0.40),
    eye_size: Tuple[float, float, float] = (0.10, 0.10, 0.10),
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
    seg_w, seg_h, seg_d = segment_size
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
    parts["eye_left"] = {
        "parent": "head_main",
        "pivot": [0.0, 0.0, 0.0],
        "position": [-0.16, +0.10, +0.18],
        "size": [eye_w, eye_h, eye_d],
        "material": {"color": RED},
    }
    parts["eye_right"] = {
        "parent": "head_main",
        "pivot": [0.0, 0.0, 0.0],
        "position": [+0.16, +0.10, +0.18],
        "size": [eye_w, eye_h, eye_d],
        "material": {"color": RED},
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

    # Body segments: seg_0..seg_{n-1}, daisy-chained off the head.
    # Attach each segment to the back face of its parent, with no gaps:
    #   parent pivot at -parent_depth/2, child center at -child_depth/2 from that pivot.
    #
    # Parent direction: -Z is "back".
    first_parent = "head_main"
    first_parent_depth = head_d

    for i in range(num_segments):
        name = f"seg_{i:02d}"
        parent = first_parent if i == 0 else f"seg_{i-1:02d}"
        parent_depth = first_parent_depth if i == 0 else seg_d

        color: RGB = BLACK if (i % 2 == 0) else RED

        parts[name] = {
            "parent": parent,
            "pivot": [0.0, 0.0, -parent_depth / 2.0],     # back face of parent
            "position": [0.0, 0.0, -seg_d / 2.0],         # center of this segment
            "size": [seg_w, seg_h, seg_d],
            "material": {"color": color},
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
        "root_offset": [0.0, +0.20, 0.0],
        "parts": parts,
        "animations": animations,
    }
    return model


# A ready-to-use model instance (client-side renderer can consume this directly)
SNAKE_MODEL: Dict[str, Any] = build_snake_model(num_segments=60)


class SnakeEntity(BaseEntity):
    """
    Simple entity wrapper for the snake model.

    Notes:
      - Your current main.py uses `entity.type` to select an AnimatedEntityRenderer.
      - We set `self.type = 'snake'` so you can register a renderer under that key.
    """
    def __init__(self, world, position=(0, 100, 0)):
        super().__init__(world, position=position)
        self.type = "snake"
        self.model_definition = SNAKE_MODEL

        # A tighter bounding box than the humanoid, roughly matching the head.
        self.bounding_box = np.array([0.6, 0.4, 0.6], dtype=float)

        # If your client uses this field to select animation:
        self.current_animation = "idle"

    def to_network_dict(self):
        d = super().to_network_dict()
        d["type"] = "snake"
        d["animation"] = "idle"
        return d
