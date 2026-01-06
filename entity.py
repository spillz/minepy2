import numpy as np
import logutil
from blocks import BLOCK_COLLIDES

ENTITY_TYPE_IDS = {
    "base_entity": 0,
    "player": 1,
    "snake": 2,
    "snail": 3,
    "seagull": 4,
    "dog": 5,
    "dinotrex": 6,
}
ENTITY_TYPE_NAMES = {value: key for key, value in ENTITY_TYPE_IDS.items()}
ENTITY_ANIM_IDS = {
    "idle": 0,
    "walk": 1,
}
ENTITY_ANIM_NAMES = {value: key for key, value in ENTITY_ANIM_IDS.items()}

ENTITY_FLAG_ON_GROUND = 1
_BASE_ENTITY_KEYS = {
    "id",
    "type",
    "pos",
    "rot",
    "vel",
    "on_ground",
    "animation",
}


def pack_entity_batch(entity_states):
    ids = []
    type_ids = []
    anim_ids = []
    pos = []
    rot = []
    vel = []
    flags = []
    extras = {}
    types = []
    anims = []
    for state in entity_states:
        if not isinstance(state, dict):
            continue
        entity_id = state.get("id")
        if entity_id is None:
            continue
        ids.append(int(entity_id))
        entity_type = state.get("type", "base_entity")
        type_id = ENTITY_TYPE_IDS.get(entity_type, 0)
        type_ids.append(type_id)
        types.append(entity_type)
        anim_name = state.get("animation", "idle")
        anim_id = ENTITY_ANIM_IDS.get(anim_name, 0)
        anim_ids.append(anim_id)
        anims.append(anim_name)
        pos.append(state.get("pos", (0.0, 0.0, 0.0)))
        rot.append(state.get("rot", (0.0, 0.0)))
        vel.append(state.get("vel", (0.0, 0.0, 0.0)))
        flags.append(ENTITY_FLAG_ON_GROUND if state.get("on_ground") else 0)
        extra = {k: v for k, v in state.items() if k not in _BASE_ENTITY_KEYS}
        if extra:
            extras[int(entity_id)] = extra
    payload = {
        "ids": np.asarray(ids, dtype=np.int32),
        "type_id": np.asarray(type_ids, dtype=np.uint8),
        "anim_id": np.asarray(anim_ids, dtype=np.uint8),
        "pos": np.asarray(pos, dtype=np.float32),
        "rot": np.asarray(rot, dtype=np.float32),
        "vel": np.asarray(vel, dtype=np.float32),
        "flags": np.asarray(flags, dtype=np.uint8),
    }
    if extras:
        payload["extras"] = extras
    if any(tid == 0 for tid in type_ids):
        payload["types"] = types
    if any(aid == 0 for aid in anim_ids):
        payload["anims"] = anims
    return payload


def unpack_entity_batch(payload):
    if payload is None:
        return []
    ids = np.asarray(payload.get("ids", []), dtype=np.int32)
    type_ids = np.asarray(payload.get("type_id", []), dtype=np.uint8)
    anim_ids = np.asarray(payload.get("anim_id", []), dtype=np.uint8)
    pos = np.asarray(payload.get("pos", []), dtype=np.float32)
    rot = np.asarray(payload.get("rot", []), dtype=np.float32)
    vel = np.asarray(payload.get("vel", []), dtype=np.float32)
    flags = np.asarray(payload.get("flags", []), dtype=np.uint8)
    extras = payload.get("extras", {}) or {}
    types = payload.get("types")
    anims = payload.get("anims")

    count = int(len(ids))
    states = []
    for i in range(count):
        entity_id = int(ids[i])
        type_id = int(type_ids[i]) if i < len(type_ids) else 0
        anim_id = int(anim_ids[i]) if i < len(anim_ids) else 0
        entity_type = ENTITY_TYPE_NAMES.get(type_id)
        if entity_type is None and types and i < len(types):
            entity_type = types[i]
        if entity_type is None:
            entity_type = "base_entity"
        anim_name = ENTITY_ANIM_NAMES.get(anim_id)
        if anim_name is None and anims and i < len(anims):
            anim_name = anims[i]
        if anim_name is None:
            anim_name = "idle"
        state = {
            "id": entity_id,
            "type": entity_type,
            "pos": pos[i] if i < len(pos) else np.zeros(3, dtype=np.float32),
            "rot": rot[i] if i < len(rot) else np.zeros(2, dtype=np.float32),
            "vel": vel[i] if i < len(vel) else np.zeros(3, dtype=np.float32),
            "on_ground": bool(flags[i] & ENTITY_FLAG_ON_GROUND) if i < len(flags) else False,
            "animation": anim_name,
        }
        extra = extras.get(entity_id)
        if extra:
            state.update(extra)
        states.append(state)
    return states

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
        x = float(self.position[0])
        z = float(self.position[2])
        if hasattr(self.world, "get_vertical_column"):
            column = self.world.get_vertical_column(x, z)
            if column is not None and column.size > 0:
                collides = BLOCK_COLLIDES[column]
                if collides.any():
                    top = int(np.nonzero(collides)[0][-1])
                    self.position = np.array([x, float(top + 1), z], dtype=float)
                    self.on_ground = True
                    if hasattr(self, "velocity"):
                        self.velocity[1] = 0
                    return
        if hasattr(self.world, "collide") and callable(self.world.collide):
            grounded_pos, _ = self.world.collide(tuple(self.position), self.bounding_box)
            self.position = np.array(grounded_pos, dtype=float)
            self.on_ground = True
            if hasattr(self, "velocity"):
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
        on_ladder = getattr(self, "on_ladder", False)
        if not self.on_ground and not self.flying and not on_ladder:
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
            'vel': list(self.velocity),
            'on_ground': bool(self.on_ground),
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
        self.velocity = np.array(data.get('vel', self.velocity))
        self.on_ground = bool(data.get('on_ground', self.on_ground))
        # Client can use this to drive the right animation
        self.current_animation = data.get('animation', 'idle')
