import math
import random
from typing import Dict, Any
import numpy as np
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

    def __init__(self, world, entity_id=4, saved_state=None):
        super().__init__(
            world,
            entity_type="dog",
            model_definition=DOG_MODEL,
            entity_id=entity_id,
            saved_state=saved_state,
        )
        self.bounding_box = np.array([0.4, 0.6, 0.7])
        self._wander_angle = random.random() * math.pi * 2
        self._heading = np.array([0.0, 1.0], dtype=float)

    def update(self, dt, context):
        self._wander(dt)
        super().update(dt)

    def _wander(self, dt):
        # Snail's wandering logic
        self._wander_angle += dt * (0.4 + random.uniform(-0.2, 0.2))
        self._heading = np.array([math.cos(self._wander_angle), math.sin(self._wander_angle)], dtype=float)
        
        move_dir = self._heading
        norm = np.linalg.norm(move_dir)
        moving = False
        if norm > 1e-6:
            move_dir /= norm
            self.velocity[0] = move_dir[0] * self.SPEED
            self.velocity[2] = move_dir[1] * self.SPEED
            moving = True
            self.rotation[0] = math.degrees(math.atan2(-self._heading[0], -self._heading[1]))
        else:
            self.velocity[0] = 0
            self.velocity[2] = 0

        if moving:
            self.current_animation = "walk"
        else:
            self.current_animation = "idle"

    def to_network_dict(self):
        data = super().to_network_dict()
        data["animation"] = self.current_animation
        return data

    def serialize_state(self):
        return {"pos": tuple(self.position.tolist()), "rot": tuple(self.rotation.tolist())}
