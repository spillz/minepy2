import numpy as np
from entity import BaseEntity

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
