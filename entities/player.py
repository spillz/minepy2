import math
import time
import numpy as np
import util
from entity import BaseEntity
from config import FLYING_SPEED, WALKING_SPEED, GRAVITY, TERMINAL_VELOCITY, PLAYER_HEIGHT, JUMP_SPEED
from blocks import (
    LADDER_IDS,
    LADDER_ORIENT,
    BLOCK_SOLID,
    ORIENT_SOUTH,
    ORIENT_WEST,
    ORIENT_NORTH,
    ORIENT_EAST,
)

# This file contains a dictionary-based definition for an animated,
# multi-part entity model. It's designed to be human-readable and
# easy to create and edit directly in code.

# Later, this could be loaded from a JSON file.

BASE_PLAYER_HEIGHT = 1.8
PLAYER_SCALE = float(PLAYER_HEIGHT) / BASE_PLAYER_HEIGHT

def _scale_part(part, scale):
    part = dict(part)
    part['pivot'] = [v * scale for v in part['pivot']]
    part['position'] = [v * scale for v in part['position']]
    part['size'] = [v * scale for v in part['size']]
    return part

def _scale_model(model, scale):
    model = dict(model)
    parts = model.get('parts', {})
    model['parts'] = {name: _scale_part(part, scale) for name, part in parts.items()}
    return model

head_size = np.array((0.3, 0.3, 0.3))
eye_color = (0, 0, 0)

HUMANOID_MODEL = {
    # 'parts' defines the "bones" and their shapes.
    # Each part is a block, positioned relative to its parent.
    # 'pivot' is the point ON THE PARENT where this part attaches and rotates.
    # 'position' is the offset FROM THE PIVOT to the center of this part's mesh.
    'root_part': 'torso',
    'parts': {
        'torso': {
            'parent': None,
            'pivot': [0, 0.8, 0],   # The model's origin is at its feet. The torso pivot matches the leg height.
            'position': [0, 0.0, 0],
            'size': [0.5, 0.7, 0.25],
            'material': {'color': (58, 110, 165)}
        },
        'head': {
            'parent': 'torso',
            'pivot': [0, 0.5, 0],   # Pivot tuned so camera sits around 1.7 units above feet.
            'position': [0, 0.0, 0], # Seat the head on the torso top.
            'size': [0.3, 0.3, 0.3],
            'material': {'color': (224, 172, 125)}
        },
        "left_eye": {
            "parent": "head",
            "pivot": [-head_size[0]/4, 0.15*head_size[1], -head_size[2]/2],
            "position": [0, 0, 0],
            "size": 0.2*head_size,
            "material": {"color": eye_color},
        },
        "right_eye": {
            "parent": "head",
            "pivot": [head_size[0]/4, 0.15*head_size[1], -head_size[2]/2],
            "position": [0, 0, 0],
            "size": 0.2*head_size,
            "material": {"color": eye_color},
        },
        "mouth": {
            "parent": "head",
            "pivot": [0, -0.15*head_size[1], -head_size[2]/2],
            "position": [0, 0, 0],
            "size": head_size*(0.3,0.1,0.1),
            "material": {"color": eye_color},
        },
        'hair': {
            'parent': 'torso',
            'pivot': [0, 0.5, 0],   # Pivot matches the head pivot for eye-level camera.
            'position': [0, 0.2*head_size[1], 0.2*(1.25*head_size[2])], # Hair sits on top of the head.
            'size': head_size*1.25,
            'material': {'color': (148, 121, 95)}
        },
        'left_arm': {
            'parent': 'torso',
            'pivot': [-0.3, 0.3, 0], # Left shoulder pivot is on the side of the torso.
            'position': [0, -0.3, 0],# Arm mesh is shifted down from the shoulder pivot.
            'size': [0.18, 0.7, 0.18],
            'material': {'color': (224, 172, 125)}
        },
        'right_arm': {
            'parent': 'torso',
            'pivot': [0.3, 0.3, 0],
            'position': [0, -0.3, 0],
            'size': [0.18, 0.7, 0.18],
            'material': {'color': (224, 172, 125)}
        },
        'left_leg': {
            'parent': 'torso',
            'pivot': [-0.12, -0.35, 0], # Left hip pivot.
            'position': [0, -0.4, 0], # Leg mesh is shifted down from the hip pivot.
            'size': [0.22, 0.8, 0.22],
            'material': {'color': (40, 50, 100)}
        },
        'right_leg': {
            'parent': 'torso',
            'pivot': [0.12, -0.35, 0],
            'position': [0, -0.4, 0],
            'size': [0.22, 0.8, 0.22],
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

if abs(PLAYER_SCALE - 1.0) > 1e-6:
    HUMANOID_MODEL = _scale_model(HUMANOID_MODEL, PLAYER_SCALE)




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

        # Player-specific physics might differ slightly.
        # Collision box: 0.5x0.5 footprint centered on player, PLAYER_HEIGHT blocks tall from feet.
        self.bounding_box = np.array([0.5, float(PLAYER_HEIGHT), 0.5])
        self.on_ladder = False
        self.ladder_climb_speed = 3.0
        self.ladder_strafe_speed = WALKING_SPEED * 0.6
        self.ladder_snap_distance = 0.15
        self.ladder_snap_speed = 6.0
        self.ladder_align_duration = 0.2
        self._ladder_align_elapsed = 0.0
        self._ladder_align_start = 0.0
        self._ladder_align_target = 0.0
        self._ladder_last_orient = None
        self.ladder_pitch_reverse_deg = -60.0
        self.ladder_jump_push = WALKING_SPEED * 0.6
        self._jump_was_down = False
        self.ladder_mount_duration = 0.25
        self.ladder_collider_pad = 0.12
        self.ladder_align_enabled = True
        self.ladder_remount_cooldown = 0.5
        self.ladder_dismount_push = WALKING_SPEED * 0.6
        self.ladder_dismount_yaw_duration = 0.2
        self.ladder_dismount_nudge = 0.18
        self._ladder_mount_timer = 0.0
        self._ladder_mount_orient = None
        self._ladder_mount_start_pos = None
        self._ladder_mount_target_pos = None
        self.ladder_exit_grace = 0.1
        self._ladder_exit_timer = 0.0
        self._ladder_contact_cache = None
        self._ladder_remount_timer = 0.0
        self._ladder_dismount_yaw_timer = 0.0
        self._ladder_dismount_yaw_start = 0.0
        self._ladder_dismount_yaw_target = 0.0
        self._ladder_requires_clear = False
        self._ladder_dismount_nudge_timer = 0.0
        self._ladder_dismount_nudge_vec = np.zeros(3, dtype=float)
        self._ladder_mount_from = None
        self.ladder_mount_pitch_bottom = 70.0
        self.ladder_mount_pitch_top = -70.0
        self.ladder_mount_pitch_neutral = 0.0
        self.camera_yaw_follow = False
        self.camera_yaw_target = 0.0
        self.camera_yaw_duration = self.ladder_align_duration
        self.camera_pitch_follow = False
        self.camera_pitch_target = 0.0
        self.camera_pitch_duration = self.ladder_align_duration

    def serialize_state(self):
        return {
            "pos": tuple(self.position.tolist()),
            "rot": tuple(self.rotation.tolist()),
            "vel": tuple(self.velocity.tolist()),
            "flying": self.flying,
            "on_ground": self.on_ground,
        }

    def get_camera_position(self):
        """
        Calculates the position of the camera, which is located in the head of the model.
        This enables a first-person perspective where the player can see their own body.

        Returns:
            np.ndarray: The 3D world coordinates for the camera.
        """
        # The player's self.position is at their feet.
        # Renderer anchors the root mesh at the base, so include that offset here too.
        root_part = self.model_definition.get('root_part', 'torso')
        root_size = self.model_definition.get('parts', {}).get(root_part, {}).get('size', [0.0, 0.0, 0.0])
        base_offset = np.array([0.0, root_size[1] / 2.0, 0.0], dtype=float)

        # Find the torso's pivot relative to the feet.
        torso_pivot = self.model_definition['parts']['torso']['pivot']
        
        # Find the head's pivot relative to the torso.
        head_pivot = self.model_definition['parts']['head']['pivot']
        
        # Calculate the absolute position of the head's pivot point.
        # Camera is at: player_pos + torso_pivot_offset + head_pivot_offset
        camera_pos = self.position + base_offset + np.array(torso_pivot) + np.array(head_pivot)
        
        # Keep the camera at the head pivot to avoid peeking past the collider.
        return camera_pos

    def update(self, dt, context):
        """
        Updates the player's state for the current frame.

        Args:
            dt (float): The time delta since the last frame.
            context (dict): Shared context from the main loop (motion_vector, flying, rotation, world_model).
        """
        motion_vector = context.get("motion_vector", (0.0, 0.0, 0.0))
        strafe = context.get("strafe", (0, 0))
        jump_down = bool(context.get("jump", False))
        flying = context.get("flying", False)
        camera_rotation = context.get("camera_rotation")
        apply_rotation = context.get("apply_camera_rotation", False)
        camera_mode = context.get("camera_mode", "first_person")

        if camera_rotation is not None:
            if apply_rotation:
                self.rotation[0] = float(camera_rotation[0])
            self.rotation[1] = float(camera_rotation[1])

        self.flying = bool(flying)
        speed = FLYING_SPEED if flying else WALKING_SPEED
        
        dx, dy, dz = motion_vector
        
        self.velocity[0] = dx * speed
        self.velocity[2] = dz * speed
        ladder_contact = None if self.flying else self._ladder_contact()
        if ladder_contact is not None:
            _, _, _, block_pos, _ = ladder_contact
            bx, by, bz = block_pos
            player_above_ladder = (self.position[1] > by + 0.6) or (self.velocity[1] < 0)
            ladder_continues = self.world[util.normalize((bx, by - 1, bz))]
            if ladder_continues not in LADDER_IDS and player_above_ladder:
                ladder_contact = None
        forward = -float(strafe[0])
        right_input = float(strafe[1])
        wants_ladder = (abs(forward) > 1e-3 or self.on_ladder) and self._ladder_remount_timer <= 1e-6
        if (
            not self.on_ladder
            and ladder_contact
            and self._ladder_mount_timer <= 1e-6
            and self._ladder_remount_timer <= 1e-6
            and not self._ladder_requires_clear
        ):
            self._start_ladder_mount(ladder_contact)

        if self._ladder_mount_timer > 1e-6:
            self._update_ladder_mount(dt)
            self.on_ladder = True
            self.velocity[0] = 0.0
            self.velocity[1] = 0.0
            self.velocity[2] = 0.0
        else:
            if ladder_contact:
                self._ladder_contact_cache = ladder_contact
                self._ladder_exit_timer = self.ladder_exit_grace
                self.on_ladder = wants_ladder
            else:
                self._ladder_requires_clear = False
                self._ladder_exit_timer = max(0.0, self._ladder_exit_timer - dt)
                if self._ladder_exit_timer <= 1e-6:
                    self.on_ladder = False
        if self.on_ladder and self._ladder_mount_timer <= 1e-6:
            if ladder_contact is None:
                ladder_contact = self._ladder_contact_cache
            if ladder_contact is None:
                self.on_ladder = False
                self._ladder_align_elapsed = 0.0
                self._ladder_last_orient = None
            else:
                orient, plane_axis, plane_pos, block_pos, _ = ladder_contact
                right_vec = self._yaw_right_vector(self.rotation[0])
                self.velocity[0] = right_vec[0] * right_input * self.ladder_strafe_speed
                self.velocity[2] = right_vec[2] * right_input * self.ladder_strafe_speed
                climb_dir = -1.0 if self.rotation[1] <= self.ladder_pitch_reverse_deg else 1.0
                if forward > 1e-3:
                    self.velocity[1] = self.ladder_climb_speed * climb_dir
                elif forward < -1e-3:
                    self.velocity[1] = -self.ladder_climb_speed * climb_dir
                else:
                    self.velocity[1] = 0.0
                snap_offset = plane_pos - self.position[plane_axis]
                if abs(snap_offset) > 1e-4:
                    self.velocity[plane_axis] += snap_offset * self.ladder_snap_speed
                self._align_to_ladder(orient, dt)
                dismount_reason = self._ladder_should_dismount(block_pos)
                if dismount_reason is not None:
                    self._ladder_dismount(
                        orient=orient,
                        forward_input=forward,
                        pitch=self.rotation[1],
                        reason=dismount_reason,
                    )
                if jump_down and not self._jump_was_down:
                    jump_normal = self._ladder_normal(orient)
                    self.velocity[0] += jump_normal[0] * self.ladder_jump_push
                    self.velocity[2] += jump_normal[2] * self.ladder_jump_push
                    self.velocity[1] = max(self.velocity[1], JUMP_SPEED * 0.5)
                    self._ladder_dismount(
                        orient=orient,
                        forward_input=forward,
                        pitch=self.rotation[1],
                        reason="jump",
                        remount_cooldown=self.ladder_remount_cooldown,
                    )
        else:
            self._ladder_align_elapsed = 0.0
            self._ladder_last_orient = None
        if flying:
            self.velocity[1] = dy * speed
        if self._ladder_dismount_nudge_timer > 0.0:
            self._ladder_dismount_nudge_timer = max(0.0, self._ladder_dismount_nudge_timer - dt)
            if self.ladder_dismount_yaw_duration > 1e-6:
                self.velocity[0] += self._ladder_dismount_nudge_vec[0] / self.ladder_dismount_yaw_duration
                self.velocity[2] += self._ladder_dismount_nudge_vec[2] / self.ladder_dismount_yaw_duration
        if self._ladder_dismount_yaw_timer > 0.0:
            self._update_ladder_dismount_yaw(dt)
        if self.ladder_align_enabled and (self._ladder_mount_timer > 1e-6 or self._ladder_dismount_yaw_timer > 1e-6):
            target_yaw = None
            if self._ladder_dismount_yaw_timer > 1e-6:
                target_yaw = self._ladder_dismount_yaw_target
            elif self._ladder_mount_timer > 1e-6 and self._ladder_mount_orient is not None:
                target_yaw = self._ladder_target_yaw(self._ladder_mount_orient)
            elif self._ladder_last_orient is not None:
                target_yaw = self._ladder_target_yaw(self._ladder_last_orient)
            else:
                target_yaw = float(self.rotation[0])
            self.camera_yaw_follow = True
            self.camera_yaw_target = -float(target_yaw)
            self.camera_yaw_duration = (
                self.ladder_dismount_yaw_duration
                if self._ladder_dismount_yaw_timer > 1e-6
                else self.ladder_align_duration
            )
            self.camera_pitch_follow = True
            if self._ladder_dismount_yaw_timer > 1e-6:
                pitch_target = 0.0
            elif camera_mode == "third_person":
                pitch_target = 0.0
            elif self._ladder_mount_timer > 1e-6:
                if self._ladder_mount_from == "bottom":
                    pitch_target = self.ladder_mount_pitch_bottom
                elif self._ladder_mount_from == "top":
                    pitch_target = self.ladder_mount_pitch_top
                else:
                    pitch_target = self.ladder_mount_pitch_neutral
            else:
                pitch_target = 0.0
            self.camera_pitch_target = pitch_target
            self.camera_pitch_duration = self.camera_yaw_duration
        else:
            self.camera_yaw_follow = False
            self.camera_pitch_follow = False
        self._jump_was_down = jump_down
        if self._ladder_remount_timer > 0.0:
            self._ladder_remount_timer = max(0.0, self._ladder_remount_timer - dt)
        
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

    def _ladder_contact(self):
        if not hasattr(self.world, '__getitem__'):
            return None
        width, height, depth = self.bounding_box
        min_x = self.position[0] - width / 2
        max_x = self.position[0] + width / 2
        min_y = self.position[1]
        max_y = self.position[1] + height
        min_z = self.position[2] - depth / 2
        max_z = self.position[2] + depth / 2
        eps = 1e-6
        y_pad = 0.25
        plane_pad = self.ladder_snap_distance
        collider_pad = self.ladder_collider_pad
        face_eps = 1e-3

        def block_range(min_v, max_v, centered):
            if centered:
                lo = int(math.floor(min_v - 0.5 + eps))
                hi = int(math.floor(max_v + 0.5 - eps))
            else:
                lo = int(math.floor(min_v + eps))
                hi = int(math.floor(max_v - eps))
            return range(lo, hi + 1)

        def dist_to_plane(plane, min_v, max_v):
            if min_v <= plane <= max_v:
                return 0.0
            return min(abs(plane - min_v), abs(plane - max_v))

        best = None
        best_d2 = None
        px, py, pz = self.position
        for bx in block_range(min_x, max_x, centered=True):
            for by in block_range(min_y - y_pad, max_y + y_pad, centered=False):
                for bz in block_range(min_z, max_z, centered=True):
                    block_id = self.world[util.normalize((bx, by, bz))]
                    if block_id not in LADDER_IDS:
                        continue
                    orient = LADDER_ORIENT.get(block_id)
                    if orient is None:
                        continue
                    block_min_x = bx - 0.5 - collider_pad
                    block_max_x = bx + 0.5 + collider_pad
                    block_min_y = by - collider_pad
                    block_max_y = by + 1.0 + collider_pad
                    block_min_z = bz - 0.5 - collider_pad
                    block_max_z = bz + 0.5 + collider_pad
                    if max_x < block_min_x or min_x > block_max_x:
                        continue
                    if max_z < block_min_z or min_z > block_max_z:
                        continue
                    if max_y < block_min_y or min_y > block_max_y:
                        continue
                    if orient in (ORIENT_SOUTH, ORIENT_NORTH):
                        plane_face = (bz + 0.5) if orient == ORIENT_SOUTH else (bz - 0.5)
                        if dist_to_plane(plane_face, min_z, max_z) > plane_pad + collider_pad:
                            continue
                        normal = 1.0 if orient == ORIENT_SOUTH else -1.0
                        plane_pos = plane_face - normal * (depth / 2 + face_eps)
                        plane_axis = 2
                    else:
                        plane_face = (bx + 0.5) if orient == ORIENT_EAST else (bx - 0.5)
                        if dist_to_plane(plane_face, min_x, max_x) > plane_pad + collider_pad:
                            continue
                        normal = 1.0 if orient == ORIENT_EAST else -1.0
                        plane_pos = plane_face - normal * (width / 2 + face_eps)
                        plane_axis = 0
                    d2 = (bx + 0.0 - px) ** 2 + (by + 0.0 - py) ** 2 + (bz + 0.0 - pz) ** 2
                    if best is None or d2 < best_d2:
                        best = (orient, plane_axis, plane_pos, (bx, by, bz), d2)
                        best_d2 = d2
        return best

    def _yaw_right_vector(self, yaw_deg):
        rad = math.radians(yaw_deg)
        return np.array([math.cos(rad), 0.0, math.sin(rad)], dtype=float)

    def _yaw_forward_vector(self, yaw_deg):
        rad = math.radians(yaw_deg - 90.0)
        return np.array([math.cos(rad), 0.0, math.sin(rad)], dtype=float)

    def _start_ladder_mount(self, ladder_contact):
        orient, plane_axis, plane_pos, block_pos, _ = ladder_contact
        bx, by, bz = block_pos
        ladder_above = self.world[util.normalize((bx, by + 1, bz))]
        ladder_below = self.world[util.normalize((bx, by - 1, bz))]
        has_neighbor = ladder_above in LADDER_IDS or ladder_below in LADDER_IDS
        if not (has_neighbor and self.position[1] >= by + 0.9):
            return
        target = np.array(self.position, dtype=float)
        target[1] = float(by)
        target[plane_axis] = plane_pos
        lateral_axis = 0 if plane_axis == 2 else 2
        target[lateral_axis] = float(bx) if lateral_axis == 0 else float(bz)
        self._ladder_mount_timer = self.ladder_mount_duration
        self._ladder_mount_orient = orient
        self._ladder_mount_start_pos = self.position.copy()
        self._ladder_mount_target_pos = target
        if self.position[1] <= by + 0.1:
            self._ladder_mount_from = "bottom"
        elif self.position[1] >= by + 0.9:
            self._ladder_mount_from = "top"
        else:
            self._ladder_mount_from = None

    def _update_ladder_mount(self, dt):
        if self._ladder_mount_timer <= 0.0:
            return
        self._ladder_mount_timer = max(0.0, self._ladder_mount_timer - dt)
        if self._ladder_mount_start_pos is None or self._ladder_mount_target_pos is None:
            self._ladder_mount_timer = 0.0
            return
        if self.ladder_mount_duration > 1e-6:
            t = 1.0 - (self._ladder_mount_timer / self.ladder_mount_duration)
        else:
            t = 1.0
        self.position = self._ladder_mount_start_pos * (1.0 - t) + self._ladder_mount_target_pos * t
        if self._ladder_mount_orient is not None and self.ladder_align_enabled:
            target = self._ladder_target_yaw(self._ladder_mount_orient)
            self.rotation[0] = self._yaw_lerp(self.rotation[0], target, t)

    def _ladder_should_dismount(self, block_pos):
        bx, by, bz = block_pos
        if self.velocity[1] < -1e-3:
            below_id = self.world[util.normalize((bx, by - 1, bz))]
            if below_id and BLOCK_SOLID[below_id] and self.position[1] <= by + 0.1:
                return "down"
        if self.velocity[1] > 1e-3:
            above_id = self.world[util.normalize((bx, by + 1, bz))]
            if above_id not in LADDER_IDS and self.position[1] >= by + 0.98:
                return "up"
        return None

    def _ladder_dismount(self, orient=None, forward_input=0.0, pitch=0.0, reason=None, remount_cooldown=None):
        if orient is not None:
            normal = self._ladder_normal(orient)
            if reason == "up":
                push_dir = -normal
            else:
                push_dir = normal
            self.velocity[0] += push_dir[0] * self.ladder_dismount_push
            self.velocity[2] += push_dir[2] * self.ladder_dismount_push
            if reason in ("down", "jump"):
                self._ladder_dismount_nudge_timer = self.ladder_dismount_yaw_duration
                self._ladder_dismount_nudge_vec = normal * self.ladder_dismount_nudge
            if self.ladder_align_enabled:
                target = None
                if reason == "up":
                    target = self._ladder_target_yaw(orient)
                elif reason == "down":
                    if forward_input > 1e-3:
                        target = self._ladder_target_yaw(orient) + 180.0
                    elif forward_input < -1e-3:
                        target = self._ladder_target_yaw(orient)
                elif forward_input < -1e-3 and pitch >= self.ladder_pitch_reverse_deg:
                    target = self._ladder_target_yaw(orient)
                elif forward_input > 1e-3 and pitch <= self.ladder_pitch_reverse_deg:
                    target = self._ladder_target_yaw(orient) + 180.0
                if target is not None:
                    self._start_ladder_dismount_yaw(target)
        self.on_ladder = False
        self._ladder_mount_timer = 0.0
        self._ladder_mount_orient = None
        self._ladder_mount_start_pos = None
        self._ladder_mount_target_pos = None
        self._ladder_exit_timer = 0.0
        self._ladder_contact_cache = None
        self._ladder_requires_clear = True
        self._ladder_mount_from = None
        if remount_cooldown is None:
            remount_cooldown = self.ladder_remount_cooldown
        self._ladder_remount_timer = max(self._ladder_remount_timer, remount_cooldown)

    def _ladder_normal(self, orient):
        if orient == ORIENT_SOUTH:
            return np.array([0.0, 0.0, -1.0], dtype=float)
        if orient == ORIENT_NORTH:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        if orient == ORIENT_EAST:
            return np.array([-1.0, 0.0, 0.0], dtype=float)
        return np.array([1.0, 0.0, 0.0], dtype=float)

    def _start_ladder_dismount_yaw(self, target):
        self._ladder_dismount_yaw_timer = self.ladder_dismount_yaw_duration
        self._ladder_dismount_yaw_start = float(self.rotation[0])
        self._ladder_dismount_yaw_target = float(target)

    def _update_ladder_dismount_yaw(self, dt):
        self._ladder_dismount_yaw_timer = max(0.0, self._ladder_dismount_yaw_timer - dt)
        if self.ladder_dismount_yaw_duration > 1e-6:
            t = 1.0 - (self._ladder_dismount_yaw_timer / self.ladder_dismount_yaw_duration)
        else:
            t = 1.0
        self.rotation[0] = self._yaw_lerp(
            self._ladder_dismount_yaw_start,
            self._ladder_dismount_yaw_target,
            t,
        )

    def _align_to_ladder(self, orient, dt):
        target = self._ladder_target_yaw(orient)
        if self._ladder_last_orient != orient or self._ladder_align_elapsed <= 1e-6:
            self._ladder_align_start = float(self.rotation[0])
            self._ladder_align_target = target
            self._ladder_align_elapsed = 0.0
            self._ladder_last_orient = orient
        if self._ladder_align_elapsed < self.ladder_align_duration:
            self._ladder_align_elapsed += dt
            t = min(1.0, self._ladder_align_elapsed / self.ladder_align_duration)
            self.rotation[0] = self._yaw_lerp(self._ladder_align_start, target, t)

    def _ladder_target_yaw(self, orient):
        if orient == ORIENT_SOUTH:
            return 180.0
        if orient == ORIENT_NORTH:
            return 0.0
        if orient == ORIENT_EAST:
            return 270.0
        return 90.0

    def _yaw_lerp(self, start, end, t):
        delta = ((end - start + 180.0) % 360.0) - 180.0
        return start + delta * t
