import numpy as np
from entity import BaseEntity

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

def compute_bind_pose_min_y(model):
    parts = model["parts"]
    root = model["root_part"]

    # Build children map
    children = {name: [] for name in parts}
    for name, p in parts.items():
        parent = p.get("parent")
        if parent is not None:
            children[parent].append(name)

    min_y = float("inf")

    def dfs(part_name, parent_joint_y):
        nonlocal min_y
        p = parts[part_name]

        pivot_y = float(p["pivot"][1])
        pos_y   = float(p["position"][1])
        size_y  = float(p["size"][1])

        # Joint position in model space (bind pose; ignoring rotations)
        joint_y = parent_joint_y + pivot_y

        # Mesh center in model space
        center_y = joint_y + pos_y

        # Bottom of this box mesh
        bottom_y = center_y - size_y * 0.5
        min_y = min(min_y, bottom_y)

        for ch in children.get(part_name, []):
            dfs(ch, joint_y)

    # parent_joint_y for root is 0.0 by convention
    dfs(root, 0.0)
    return min_y

if False: #THIS MATH IS BROKEN
    min_y = compute_bind_pose_min_y(HUMANOID_MODEL)
    root_offset_y = -min_y
    print("bind-pose min_y:", min_y, "=> root_offset_y:", root_offset_y)
    HUMANOID_MODEL["root_offset"] = [0.0, root_offset_y, 0.0]


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
        super().__init__(world, entity_type='player', position=position)
        
        self.model_definition = model_definition
        self.name = "Player"  # Default name

        # Player-specific physics might differ slightly
        self.bounding_box = np.array([0.8, 1.8, 0.8]) # width, height, depth

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

    def update(self, dt, motion_vector):
        """
        Updates the player's state for the current frame.

        Args:
            dt (float): The time delta since the last frame.
            motion_vector (tuple): The (dx, dy, dz) motion vector from user input.
        """
        # Player-specific movement logic
        speed = 15.0 if self.flying else 5.0 # Example speeds
        self.velocity[0] = motion_vector[0] * speed
        self.velocity[2] = motion_vector[2] * speed
        
        # If jumping and on the ground
        if motion_vector[1] > 0 and self.on_ground:
            self.velocity[1] = 8.0 # Jump speed

        # Let the base class handle gravity and collision
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
