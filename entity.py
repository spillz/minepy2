import numpy as np

class BaseEntity:
    """
    The base class for all non-block objects in the world.
    This class is managed by the server.
    """
    def __init__(self, world, position=(0, 100, 0)):
        # Every entity needs these properties
        self.id = None  # Will be assigned by the server
        self.type = 'base_entity'
        self.world = world  # A reference to the world instance to query blocks

        # Physics and Position
        self.position = np.array(position, dtype=float)
        self.velocity = np.array([0, 0, 0], dtype=float)
        self.rotation = np.array([0, 0], dtype=float)  # (yaw, pitch)

        # A simple box for collision detection
        self.bounding_box = np.array([0.8, 1.8, 0.8]) # width, height, depth

        # State
        self.on_ground = False

    def update(self, dt):
        """
        This method is called by the server on every game tick.
        The base implementation handles universal physics like gravity and collisions.
        
        NOTE: This assumes the `world` object has a `collide` method that can
        handle entity collisions, similar to the one in `main.py`. This will
        likely require some refactoring to move collision logic to a shared location.
        """
        # 1. Apply gravity
        if not self.on_ground:
            self.velocity[1] -= 20.0 * dt # Gravity constant

        # 2. Apply velocity to position
        self.position += self.velocity * dt

        # 3. Handle collisions with the world
        # The collide method should return the new position and a boolean for on_ground
        if hasattr(self.world, 'collide') and callable(self.world.collide):
            new_pos, vertical_collision = self.world.collide(self.position, self.bounding_box)
            self.position = np.array(new_pos, dtype=float)
            self.on_ground = vertical_collision
        
        # Stop vertical velocity if on ground
        if self.on_ground:
            self.velocity[1] = 0

    def to_network_dict(self):
        """
        Creates a simple dictionary of this entity's state to be sent to clients.
        """
        return {
            'id': self.id,
            'type': self.type,
            'pos': list(self.position),
            'rot': list(self.rotation),
            # Also include animation state for the client renderer
            'animation': 'walk' if np.linalg.norm(self.velocity) > 0.1 else 'idle'
        }

    def from_network_dict(self, data):
        """
        Updates the entity's state from a dictionary received from the network.
        Used on the client-side to keep a local copy of the entity state.
        """
        self.id = data.get('id', self.id)
        self.type = data.get('type', self.type)
        self.position = np.array(data.get('pos', self.position))
        self.rotation = np.array(data.get('rot', self.rotation))
        # Client can use this to drive the right animation
        self.current_animation = data.get('animation', 'idle')
