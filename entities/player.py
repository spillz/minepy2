import numpy as np
from entity import BaseEntity
from config import FLYING_SPEED, WALKING_SPEED, GRAVITY, TERMINAL_VELOCITY, PLAYER_HEIGHT

# This file contains a dictionary-based definition for an animated,
# multi-part entity model. It's designed to be human-readable and
# easy to create and edit directly in code.

# Later, this could be loaded from a JSON file.

HUMANOID_MODEL = {
    # 'parts' defines the "bones" and their shapes.
    # Each part is a block, positioned relative to its parent.
    # 'pivot' is the point ON THE PARENT where this part attaches and rotates.
    # 'position' is the offset FROM THE PIVOT to the center of this part's mesh.
    'root_part': 'torso',
    'root_offset': [0.0, -0.9, 0.0],
    'parts': {
        'torso': {
            'parent': None,
            'pivot': [0, 0.8, 0],   # The model's origin is at its feet. The torso pivot is 0.8 units up.
            'position': [0, 0.0, 0], 
            'size': [0.6, 0.8, 0.3],
            'material': {'color': (58, 110, 165)}
        },
        'head': {
            'parent': 'torso',
            'pivot': [0, 0.4, 0],   # Pivot is at the top-center of the torso (the "neck").
            'position': [0, 0.2, 0], # Head mesh is shifted up from the neck pivot.
            'size': [0.4, 0.4, 0.4],
            'material': {'color': (224, 172, 125)}
        },
        'hair': {
            'parent': 'torso',
            'pivot': [0, 0.4, 0],   # Pivot is at the top-center of the torso (the "neck").
            'position': [0, 0.3, 0.1], # Head mesh is shifted up from the neck pivot.
            'size': [0.5, 0.5, 0.5],
            'material': {'color': (148, 121, 95)}
        },
        'left_arm': {
            'parent': 'torso',
            'pivot': [-0.4, 0.3, 0], # Left shoulder pivot is on the side of the torso.
            'position': [0, -0.4, 0],# Arm mesh is shifted down from the shoulder pivot.
            'size': [0.2, 0.9, 0.2],
            'material': {'color': (224, 172, 125)}
        },
        'right_arm': {
            'parent': 'torso',
            'pivot': [0.4, 0.3, 0],
            'position': [0, -0.4, 0],
            'size': [0.2, 0.9, 0.2],
            'material': {'color': (224, 172, 125)}
        },
        'left_leg': {
            'parent': 'torso',
            'pivot': [-0.15, -0.4, 0], # Left hip pivot.
            'position': [0, -0.45, 0], # Leg mesh is shifted down from the hip pivot.
            'size': [0.3, 0.9, 0.3],
            'material': {'color': (40, 50, 100)}
        },
        'right_leg': {
            'parent': 'torso',
            'pivot': [0.15, -0.4, 0],
            'position': [0, -0.45, 0],
            'size': [0.3, 0.9, 0.3],
            'material': {'color': (40, 50, 100)}
        },
    },

    # 'animations' defines movements by rotating parts over time.
    'animations': {
        'idle': {
            'loop': True,
            'length': 2.0, # seconds
            'keyframes': [
                {
                    'time': 0.0,
                    'rotations': { # Default pose, (pitch, yaw, roll) in degrees
                        'left_arm': {'pitch': 5},
                        'right_arm': {'pitch': -5},
                    }
                },
                {
                    'time': 1.0,
                    'rotations': {
                        'left_arm': {'pitch': 5, 'roll': 3},
                        'right_arm': {'pitch': -5, 'roll': -3},
                    }
                },
                 {
                    'time': 2.0,
                    'rotations': { 
                        'left_arm': {'pitch': 5},
                        'right_arm': {'pitch': -5},
                    }
                },
            ]
        },
        'walk': {
            'loop': True,
            'length': 1.0,
            'keyframes': [
                {
                    'time': 0.0,
                    'rotations': {
                        'left_arm': {'pitch': 45},
                        'right_arm': {'pitch': -45},
                        'left_leg': {'pitch': -45},
                        'right_leg': {'pitch': 45},
                    }
                },
                {
                    'time': 0.5,
                    'rotations': {
                        'left_arm': {'pitch': -45},
                        'right_arm': {'pitch': 45},
                        'left_leg': {'pitch': 45},
                        'right_leg': {'pitch': -45},
                    }
                },
                {
                    'time': 1.0,
                    'rotations': {
                        'left_arm': {'pitch': 45},
                        'right_arm': {'pitch': -45},
                        'left_leg': {'pitch': -45},
                        'right_leg': {'pitch': 45},
                    }
                }
            ]
        }
    }
}




class Player(BaseEntity):
    """
    Represents the player entity in the game.
    Inherits from BaseEntity and adds player-specific properties and logic.
    """
    def __init__(self, world, model_definition, position=(0, 160, 0)):
        """
        Initializes a new Player entity.

        Args:
            world: An object representing the game world for collision detection.
            model_definition (dict): A dictionary describing the entity's model and animations.
            position (tuple): The initial (x, y, z) coordinates of the player.
        """
        super().__init__(world, position=position, entity_type='player')
        
        self.model_definition = model_definition
        self.name = "Player"  # Default name

        # Player-specific physics might differ slightly
        self.bounding_box = np.array([0.8, PLAYER_HEIGHT, 0.8])

    def serialize_state(self):
        return {
            "pos": tuple(self.position.tolist()),
            "rot": tuple(self.rotation.tolist()),
            "flying": self.flying,
        }

    def get_camera_position(self):
        """
        Calculates the position of the camera, which is located in the head of the model.
        This enables a first-person perspective where the player can see their own body.

        Returns:
            np.ndarray: The 3D world coordinates for the camera.
        """
        # The player's self.position is at their feet.
        # Find the torso's pivot relative to the feet.
        torso_pivot = self.model_definition['parts']['torso']['pivot']
        
        # Find the head's pivot relative to the torso.
        head_pivot = self.model_definition['parts']['head']['pivot']
        
        # Calculate the absolute position of the head's pivot point.
        # Camera is at: player_pos + torso_pivot_offset + head_pivot_offset
        camera_pos = self.position + np.array(torso_pivot) + np.array(head_pivot)
        
        # Add a slight forward offset to prevent clipping into the head model
        sight_vector = self.get_sight_vector()
        camera_pos += sight_vector * 0.2
        
        return camera_pos

    def update(self, dt, context):
        """
        Updates the player's state for the current frame.

        Args:
            dt (float): The time delta since the last frame.
            context (dict): Shared context from the main loop (motion_vector, flying, rotation, world_model).
        """
        motion_vector = context.get("motion_vector", (0.0, 0.0, 0.0))
        flying = context.get("flying", False)
        camera_rotation = context.get("camera_rotation")
        apply_rotation = context.get("apply_camera_rotation", False)

        if camera_rotation is not None:
            if apply_rotation:
                self.rotation[0] = float(camera_rotation[0])
            self.rotation[1] = float(camera_rotation[1])

        self.flying = bool(flying)
        speed = FLYING_SPEED if flying else WALKING_SPEED
        
        dx, dy, dz = motion_vector
        
        self.velocity[0] = dx * speed
        self.velocity[2] = dz * speed
        if flying:
            self.velocity[1] = dy * speed
        
        super().update(dt)

    def to_network_dict(self):
        """
        Extends the base network dictionary with player-specific state.
        """
        state = super().to_network_dict()
        state.update({
            'name': self.name,
            # Add other player-specific states here, e.g., animation state
        })
        return state
