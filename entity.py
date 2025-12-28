import numpy as np
import logutil

class BaseEntity:
    """
    The base class for all non-block objects in the world.
    This class is managed by the server.
    """
    def __init__(self, world, position=(0, 100, 0), entity_type='base_entity'):
        # Every entity needs these properties
        self.id = None  # Will be assigned by the server
        self.type = entity_type
        self.world = world  # A reference to the world instance to query blocks

        # Physics and Position
        self.position = np.array(position, dtype=float)
        self.velocity = np.array([0, 0, 0], dtype=float)
        self.rotation = np.array([0, 0], dtype=float)  # (yaw, pitch)

        # A simple box for collision detection
        self.bounding_box = np.array([0.8, 1.8, 0.8]) # width, height, depth

        # State
        self.on_ground = False
        self.flying = False

        # Client can use this to drive the right animation
        self.current_animation = 'idle'


    def snap_to_ground(self):
        """
        Moves the entity to be on top of the ground.
        """
        # Set a high initial position to ensure we are above any terrain
        self.position = np.array([self.position[0], 160, self.position[2]], dtype=float)
        # Use the world's collide method to find the ground position (returns bottom of bbox)
        grounded_pos, _ = self.world.collide(tuple(self.position), self.bounding_box)
        self.position = np.array(grounded_pos, dtype=float)
        # Adjust y to be the integer block coordinate, as collide returns top surface of block
        # self.position[1] -= 0.5
        self.on_ground = True
        if hasattr(self, 'velocity'):
            self.velocity[1] = 0

    def update(self, dt):
        """
        This method is called by the server on every game tick.
        The base implementation handles universal physics like gravity and collisions.
        
        NOTE: This assumes the `world` object has a `collide` method that can
        handle entity collisions, similar to the one in `main.py`. This will
        likely require some refactoring to move collision logic to a shared location.
        """
        # 1. Apply gravity
        if not self.on_ground and not self.flying:
            self.velocity[1] -= 20.0 * dt # Gravity constant

        # 2. Apply velocity to position
        prev_pos = self.position.copy()
        self.position += self.velocity * dt

        # 3. Handle collisions with the world
        # The collide method should return the new position and a boolean for on_ground
        if hasattr(self.world, 'collide') and callable(self.world.collide):
            new_pos, vertical_collision = self.world.collide(
                self.position,
                self.bounding_box,
                velocity=self.velocity,
                prev_position=prev_pos,
            )
            def dist(a,b):
                return ((np.array(b)-np.array(a))**2).sum()**0.5
            if self.__class__.__name__ == 'Player':
                if(dist(self.position, new_pos)>0.01):
                    logutil.log(
                        "ENTITY",
                        f"collide pos={self.position} new_pos={new_pos} vertical={vertical_collision}",
                        level="DEBUG",
                    )
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
