import math
import numpy as np
import pyglet
import pyglet.gl as gl
from pyglet.math import Mat4, Vec3
from entities.snake import SnakeEntity

_cube_mesh_cache = {}

def get_cube_vertices(size):
    size_key = tuple(size)
    if size_key in _cube_mesh_cache:
        return _cube_mesh_cache[size_key]

    w, h, d = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0

    vertices = np.array([
        # top
        -w, h, -d,   -w, h, d,    w, h, d,    w, h, -d,
        # bottom
        -w, -h, -d,   w, -h, -d,   w, -h, d,   -w, -h, d,
        # left
        -w, -h, -d,  -w, -h, d,   -w, h, d,   -w, h, -d,
        # right
        w, -h, d,    w, -h, -d,   w, h, -d,   w, h, d,
        # front
        -w, -h, d,    w, -h, d,    w, h, d,   -w, h, d,
        # back
        w, -h, -d,   -w, -h, -d,  -w, h, -d,   w, h, -d,
    ], dtype=np.float32)

    normals = np.array([
        0,1,0, 0,1,0, 0,1,0, 0,1,0,
        0,-1,0, 0,-1,0, 0,-1,0, 0,-1,0,
        -1,0,0, -1,0,0, -1,0,0, -1,0,0,
        1,0,0, 1,0,0, 1,0,0, 1,0,0,
        0,0,1, 0,0,1, 0,0,1, 0,0,1,
        0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1,
    ], dtype=np.float32) # Corrected dtype to float32

    indices = np.array([
         0, 1, 2,  0, 2, 3,    # top
         4, 5, 6,  4, 6, 7,    # bottom
         8, 9,10,  8,10,11,    # left
        12,13,14, 12,14,15,    # right
        16,17,18, 16,18,19,    # front
        20,21,22, 20,22,23     # back
    ], dtype=np.uint32)

    _cube_mesh_cache[size_key] = (vertices, normals, indices)
    return vertices, normals, indices


class AnimatedEntityRenderer:
    def __init__(self, program, model_definition):
        self.model = model_definition
        self.program = program
        self.animation_time = 0.0
        self.current_animation_name = 'idle'
        self._fallback_animation_name = 'idle'
        self._entity_anim_time = {}
        self._entity_anim_name = {}
        self._entity_last_seen = {}
        self._entity_blend = {}
        self._frame_counter = 0
        self._last_dt = 0.0
        self._active_blend = None
        self._active_entity_id = None

        self.batch = pyglet.graphics.Batch()
        self.group = pyglet.graphics.Group()

        self.part_meshes = {}
        for part_name, part_data in self.model['parts'].items():
            size = part_data['size']

            # base_rgb is in 0..255 space (consistent with your world pipeline / shader)
            base_rgb = np.array(
                part_data['material'].get('color', (255, 255, 255)),
                dtype='f4'
            )

            vertices, normals, indices = get_cube_vertices(size)

            # Expand indexed cube to explicit triangles
            v3 = vertices.reshape(-1, 3)          # (24, 3)
            n3 = normals.reshape(-1, 3)           # (24, 3)
            tri_verts = v3[indices]               # (36, 3)
            tri_norms = n3[indices]               # (36, 3)
            count = tri_verts.shape[0]

            # Dummy tex coords so the shader attribute exists
            tri_tex = np.zeros((count, 2), dtype='f4')
            tri_light = np.ones((count, 2), dtype='f4')

            # *** CHANGE: build RGBA (count, 4), not RGB (count, 3) ***
            base_rgba = np.array([base_rgb[0], base_rgb[1], base_rgb[2], 1.0], dtype='f4')
            tri_col = np.broadcast_to(base_rgba, (count, 4)).astype('f4')
            # --------------------------------------------------------

            self.part_meshes[part_name] = self.program.vertex_list(
                count,
                gl.GL_TRIANGLES,
                batch=self.batch,
                group=self.group,
                position=('f', tri_verts.ravel().astype('f4')),
                tex_coords=('f', tri_tex.ravel().astype('f4')),
                normal=('f', tri_norms.ravel().astype('f4')),
                color=('f', tri_col.ravel().astype('f4')),   # now 4 floats/vertex
                light=('f', tri_light.ravel().astype('f4')),
            )

    def set_animation(self, anim_name):
        self._fallback_animation_name = anim_name

    def update(self, dt):
        self._last_dt = float(dt)
        self._frame_counter += 1
        if self._entity_last_seen:
            cutoff = self._frame_counter - 300
            stale = [eid for eid, seen in self._entity_last_seen.items() if seen < cutoff]
            for eid in stale:
                self._entity_last_seen.pop(eid, None)
                self._entity_anim_name.pop(eid, None)
                self._entity_anim_time.pop(eid, None)
                self._entity_blend.pop(eid, None)

    def get_interpolated_rotation(self, part_name):
        anim = self.model['animations'].get(self.current_animation_name)
        return self._get_interpolated_rotation_for(part_name, anim, self.animation_time)

    def _get_interpolated_rotation_for(self, part_name, anim, anim_time):
        default_rot = {'pitch': 0, 'yaw': 0, 'roll': 0}
        if not anim or 'keyframes' not in anim:
            return self._get_parent_rotation_for(part_name, anim, anim_time)
        length = float(anim.get('length', 1.0) or 1.0)
        loop = bool(anim.get('loop', False))
        eps = 1e-6

        keyframes = [kf for kf in anim['keyframes'] if 'rotations' in kf and part_name in kf['rotations']]
        if loop and length > 0:
            keyframes = [kf for kf in keyframes if kf.get('time', 0.0) < (length - eps)]
        if not keyframes:
            return self._get_parent_rotation_for(part_name, anim, anim_time)

        keyframes = sorted(keyframes, key=lambda kf: kf.get('time', 0.0))
        anim_time = float(anim_time)
        if loop and length > 0:
            anim_time %= length

        if anim_time < keyframes[0]['time']:
            if loop:
                prev_kf = keyframes[-1]
                next_kf = keyframes[0]
                time_diff = (length - prev_kf['time']) + next_kf['time']
                t = 0.0 if time_diff <= 0 else (anim_time + (length - prev_kf['time'])) / time_diff
            else:
                return keyframes[0]['rotations'][part_name]
        else:
            prev_kf = keyframes[0]
            next_kf = None
            for kf in keyframes:
                if kf['time'] > anim_time:
                    next_kf = kf
                    break
                prev_kf = kf
            if next_kf is None:
                if loop:
                    next_kf = keyframes[0]
                    time_diff = (length - prev_kf['time']) + next_kf['time']
                    t = 0.0 if time_diff <= 0 else (anim_time - prev_kf['time']) / time_diff
                else:
                    return prev_kf['rotations'][part_name]
            else:
                time_diff = next_kf['time'] - prev_kf['time']
                t = 0.0 if time_diff <= 0 else (anim_time - prev_kf['time']) / time_diff

        prev_rot = prev_kf['rotations'][part_name]
        next_rot = next_kf['rotations'][part_name]
        interpolated_rot = {}
        for angle in ['pitch', 'yaw', 'roll']:
            start_angle = prev_rot.get(angle, 0)
            end_angle = next_rot.get(angle, 0)
            interpolated_rot[angle] = start_angle + t * (end_angle - start_angle)
        return interpolated_rot

    def get_interpolated_transform(self, part_name):
        anim = self.model['animations'].get(self.current_animation_name)
        return self._get_interpolated_transform_for(part_name, anim, self.animation_time)

    def _get_interpolated_transform_for(self, part_name, anim, anim_time):
        default_transform = {
            'pitch': 0, 'yaw': 0, 'roll': 0,
            'dx': 0, 'dy': 0, 'dz': 0,
            'dwx': 0, 'dwy': 0, 'dwz': 0,
            'sx': None, 'sy': None, 'sz': None,
            'dpx': 0, 'dpy': 0, 'dpz': 0,
        }
        if not anim or 'keyframes' not in anim:
            rot = self._get_parent_rotation_for(part_name, anim, anim_time)
            transform = dict(default_transform)
            transform.update(rot)
            return transform
        length = float(anim.get('length', 1.0) or 1.0)
        loop = bool(anim.get('loop', False))
        eps = 1e-6

        keyframes = [kf for kf in anim['keyframes'] if 'transforms' in kf and part_name in kf['transforms']]
        if loop and length > 0:
            keyframes = [kf for kf in keyframes if kf.get('time', 0.0) < (length - eps)]
        if not keyframes:
            rot = self._get_interpolated_rotation_for(part_name, anim, anim_time)
            transform = dict(default_transform)
            transform.update(rot)
            return transform

        keyframes = sorted(keyframes, key=lambda kf: kf.get('time', 0.0))
        anim_time = float(anim_time)
        if loop and length > 0:
            anim_time %= length

        if anim_time < keyframes[0]['time']:
            if loop:
                prev_kf = keyframes[-1]
                next_kf = keyframes[0]
                time_diff = (length - prev_kf['time']) + next_kf['time']
                t = 0.0 if time_diff <= 0 else (anim_time + (length - prev_kf['time'])) / time_diff
            else:
                transform = dict(default_transform)
                transform.update(keyframes[0]['transforms'][part_name])
                return transform
        else:
            prev_kf = keyframes[0]
            next_kf = None
            for kf in keyframes:
                if kf['time'] > anim_time:
                    next_kf = kf
                    break
                prev_kf = kf
            if next_kf is None:
                if loop:
                    next_kf = keyframes[0]
                    time_diff = (length - prev_kf['time']) + next_kf['time']
                    t = 0.0 if time_diff <= 0 else (anim_time - prev_kf['time']) / time_diff
                else:
                    transform = dict(default_transform)
                    transform.update(prev_kf['transforms'][part_name])
                    return transform
            else:
                time_diff = next_kf['time'] - prev_kf['time']
                t = 0.0 if time_diff <= 0 else (anim_time - prev_kf['time']) / time_diff

        prev_tr = prev_kf['transforms'][part_name]
        next_tr = next_kf['transforms'][part_name]
        interpolated = dict(default_transform)
        keys = set(prev_tr.keys()) | set(next_tr.keys())
        for key in keys:
            start_val = prev_tr.get(key, default_transform.get(key, 0))
            end_val = next_tr.get(key, default_transform.get(key, 0))
            if key in ('sx', 'sy', 'sz'):
                if start_val is None:
                    start_val = 1.0
                if end_val is None:
                    end_val = 1.0
            interpolated[key] = start_val + t * (end_val - start_val)
        return interpolated

    def _blend_transforms(self, a, b, t):
        default_transform = {
            'pitch': 0, 'yaw': 0, 'roll': 0,
            'dx': 0, 'dy': 0, 'dz': 0,
            'dwx': 0, 'dwy': 0, 'dwz': 0,
            'sx': None, 'sy': None, 'sz': None,
            'dpx': 0, 'dpy': 0, 'dpz': 0,
        }
        out = dict(default_transform)
        keys = set(a.keys()) | set(b.keys()) | set(default_transform.keys())
        for key in keys:
            start_val = a.get(key, default_transform.get(key, 0))
            end_val = b.get(key, default_transform.get(key, 0))
            if key in ('sx', 'sy', 'sz'):
                if start_val is None:
                    start_val = 1.0
                if end_val is None:
                    end_val = 1.0
            out[key] = start_val + t * (end_val - start_val)
        return out

    def get_parent_rotation(self, part_name, anim):
        """Recursively find rotation from parent part."""
        return self._get_parent_rotation_for(part_name, anim, self.animation_time)

    def _get_parent_rotation_for(self, part_name, anim, anim_time):
        part_data = self.model['parts'].get(part_name, {})
        parent_name = part_data.get('parent')
        if parent_name:
            return self._get_interpolated_rotation_for(parent_name, anim, anim_time)
        return {'pitch': 0, 'yaw': 0, 'roll': 0}

    def draw(self, entity_state):
        pos = entity_state['pos']
        rot = entity_state['rot']
        anim_name = entity_state.get('animation', self._fallback_animation_name)
        entity_id = entity_state.get('id')
        if entity_id is None:
            entity_id = id(entity_state)
        prev_anim = self._entity_anim_name.get(entity_id)
        prev_time = self._entity_anim_time.get(entity_id, 0.0)
        if prev_anim != anim_name:
            blend_duration = entity_state.get('anim_blend')
            if blend_duration is None:
                blend_duration = self.model.get('animation_blend', 0.0)
            if blend_duration and prev_anim is not None:
                self._entity_blend[entity_id] = {
                    'prev_anim': prev_anim,
                    'prev_time': prev_time,
                    'elapsed': 0.0,
                    'duration': float(blend_duration),
                }
            else:
                self._entity_blend.pop(entity_id, None)
            self._entity_anim_name[entity_id] = anim_name
            self._entity_anim_time[entity_id] = 0.0
        self._entity_last_seen[entity_id] = self._frame_counter
        anim_time = self._entity_anim_time.get(entity_id, 0.0)
        if self._last_dt:
            anim_time += self._last_dt
            self._entity_anim_time[entity_id] = anim_time
            blend = self._entity_blend.get(entity_id)
            if blend is not None:
                blend['elapsed'] = float(blend.get('elapsed', 0.0)) + self._last_dt
                if blend['elapsed'] >= blend['duration']:
                    self._entity_blend.pop(entity_id, None)
        self.current_animation_name = anim_name
        self.animation_time = anim_time
        self._active_entity_id = entity_id
        self._active_blend = self._entity_blend.get(entity_id)

        root_part_name = self.model.get('root_part', 'body')
        root_part_size = None
        if root_part_name in self.model['parts']:
            root_part_size = self.model['parts'][root_part_name]['size']

        root_offset = self.model.get('root_offset', (0.0, 0.0, 0.0))
        base_offset_y = root_part_size[1] / 2.0 if root_part_size is not None else 0.0
        base_offset = Vec3(0.0, base_offset_y, 0.0)
        pos_vec = Vec3(pos[0], pos[1], pos[2])
        model_matrix = Mat4.from_translation(pos_vec + Vec3(*root_offset) + base_offset)
        model_matrix = model_matrix.rotate(math.radians(rot[0]), Vec3(0, 1, 0))

        # Find the root part and start the recursive drawing
        if root_part_name in self.model['parts']:
            self._draw_part(root_part_name, model_matrix)
        self._active_blend = None
        self._active_entity_id = None

    def _draw_part(self, part_name, parent_matrix):
        part_data = self.model['parts'][part_name]

        # Get transformations for this part
        pivot = part_data.get('pivot', [0, 0, 0])
        if self._active_blend:
            blend = self._active_blend
            duration = blend.get('duration', 0.0) or 0.0
            t = 1.0 if duration <= 0 else min(max(blend.get('elapsed', 0.0) / duration, 0.0), 1.0)
            prev_anim = self.model['animations'].get(blend.get('prev_anim'))
            prev_time = blend.get('prev_time', 0.0)
            prev_transform = self._get_interpolated_transform_for(part_name, prev_anim, prev_time)
            next_anim = self.model['animations'].get(self.current_animation_name)
            next_transform = self._get_interpolated_transform_for(part_name, next_anim, self.animation_time)
            transform = self._blend_transforms(prev_transform, next_transform, t)
        else:
            transform = self.get_interpolated_transform(part_name)
        rotation = {
            'pitch': transform.get('pitch', 0),
            'yaw': transform.get('yaw', 0),
            'roll': transform.get('roll', 0),
        }
        position = part_data.get('position', [0, 0, 0])
        dx = transform.get('dx', 0)
        dy = transform.get('dy', 0)
        dz = transform.get('dz', 0)
        sx = transform.get('sx')
        sy = transform.get('sy')
        sz = transform.get('sz')
        scale = Vec3(
            sx if sx is not None else 1.0 + transform.get('dwx', 0),
            sy if sy is not None else 1.0 + transform.get('dwy', 0),
            sz if sz is not None else 1.0 + transform.get('dwz', 0),
        )

        # Create transformation matrices using pyglet.math methods
        pivot_vec = Vec3(
            pivot[0] + transform.get('dpx', 0),
            pivot[1] + transform.get('dpy', 0),
            pivot[2] + transform.get('dpz', 0),
        )
        pivot_mat = Mat4.from_translation(pivot_vec)

        pos_vec = Vec3(position[0] + dx, position[1] + dy, position[2] + dz)
        pos_mat = Mat4.from_translation(pos_vec)
        scale_mat = Mat4.from_scale(scale)

        # Apply rotations
        rot_mat = Mat4()
        rot_mat = rot_mat.rotate(math.radians(rotation.get('roll', 0)), Vec3(0, 0, 1))
        rot_mat = rot_mat.rotate(math.radians(rotation.get('pitch', 0)), Vec3(1, 0, 0))
        rot_mat = rot_mat.rotate(math.radians(rotation.get('yaw', 0)), Vec3(0, 1, 0))

        # Final transformation matrix to draw this part's mesh
        draw_matrix = parent_matrix @ pivot_mat @ rot_mat @ pos_mat @ scale_mat 
        
        # Set the model matrix uniform and draw the mesh
        self.program['u_model'] = draw_matrix
        mesh = self.part_meshes.get(part_name)
        if mesh:
            mesh.draw(gl.GL_TRIANGLES)

        # Recursively draw children, passing the correct parent matrix
        for child_name, child_data in self.model['parts'].items():
            if child_data.get('parent') == part_name:
                self._draw_part(child_name, draw_matrix)


class SnakeRenderer:
    def __init__(self, program, model_definition, segment_configs=None, tail_length=12, segment_capacity=None):
        self.program = program
        self.head_renderer = AnimatedEntityRenderer(program, model_definition)
        self.tail_length = tail_length
        configs = segment_configs or [
            ((0.40, 0.28, 0.36), (0, 0, 0)),
            ((0.34, 0.24, 0.33), (220, 40, 40)),
            ((0.28, 0.20, 0.30), (180, 40, 30)),
        ]
        self.segment_capacity = SnakeEntity.SEGMENT_COUNT if segment_capacity is None else int(segment_capacity)
        self._segment_sizes, self._segment_colors = self._build_segment_styles(configs)
        self._buffer = self._prepare_segment_buffer(self._segment_sizes, self._segment_colors)
        self._mesh = self.program.vertex_list(
            self._buffer["vertex_count"],
            gl.GL_TRIANGLES,
            position=('f', self._buffer["positions"].ravel()),
            tex_coords=('f', self._buffer["tex"].ravel()),
            normal=('f', self._buffer["normals"].ravel()),
            color=('f', self._buffer["colors"].ravel()),
            light=('f', self._buffer["light"].ravel()),
        )

    def _build_mesh(self, size):
        verts, normals, indices = get_cube_vertices(size)
        tri_verts = verts.reshape(-1, 3)[indices]
        tri_norms = normals.reshape(-1, 3)[indices]
        tri_tex = np.zeros((tri_verts.shape[0], 2), dtype='f4')
        return tri_verts, tri_norms, tri_tex

    def _build_segment_styles(self, configs):
        variant_count = len(configs)
        sizes = np.zeros((self.segment_capacity, 3), dtype='f4')
        colors = np.zeros((self.segment_capacity, 3), dtype='f4')
        head_len = max(0, self.segment_capacity - self.tail_length)
        for idx in range(self.segment_capacity):
            if variant_count <= 1:
                variant_idx = 0
            elif idx < head_len:
                variant_idx = idx % (variant_count - 1)
            else:
                variant_idx = variant_count - 1
            size, color = configs[variant_idx]
            sizes[idx] = size
            colors[idx] = color
        return sizes, colors

    def _prepare_segment_buffer(self, sizes, colors):
        mesh_cache = {}
        base_positions = []
        normals = []
        tex = []
        vertex_colors = []
        for idx in range(self.segment_capacity):
            size = tuple(sizes[idx].tolist())
            color = colors[idx]
            if size not in mesh_cache:
                mesh_cache[size] = self._build_mesh(size)
            tri_verts, tri_norms, tri_tex = mesh_cache[size]
            verts = tri_verts.copy()
            verts[:, 1] += size[1] / 2.0
            base_positions.append(verts)
            normals.append(tri_norms)
            tex.append(tri_tex)
            base_rgba = np.array([color[0], color[1], color[2], 1.0], dtype='f4')
            tri_col = np.broadcast_to(base_rgba, (tri_verts.shape[0], 4))
            vertex_colors.append(tri_col)

        base_positions = np.vstack(base_positions).astype('f4')
        normals = np.vstack(normals).astype('f4')
        tex = np.vstack(tex).astype('f4')
        colors = np.vstack(vertex_colors).astype('f4')
        light = np.ones((base_positions.shape[0], 2), dtype='f4')
        tri_count = base_positions.shape[0] // self.segment_capacity
        segment_index = np.repeat(np.arange(self.segment_capacity), tri_count).astype(np.int32)
        positions = base_positions.copy()
        target_trans = np.full((self.segment_capacity, 3), 1e6, dtype='f4')
        return {
            "tri_count": tri_count,
            "vertex_count": base_positions.shape[0],
            "base_positions": base_positions,
            "positions": positions,
            "normals": normals,
            "tex": tex,
            "colors": colors,
            "light": light,
            "segment_index": segment_index,
            "target_trans": target_trans,
        }

    def _update_mesh(self, translations):
        buffer = self._buffer
        target_trans = buffer["target_trans"]
        target_trans[:] = 1e6
        if translations is not None and len(translations):
            usable = min(len(translations), self.segment_capacity)
            target_trans[:usable] = translations[:usable]
            if usable < self.segment_capacity:
                if usable > 0:
                    target_trans[usable:] = target_trans[usable - 1]
                else:
                    target_trans[usable:] = np.array([1e6, 1e6, 1e6], dtype='f4')

        buffer["positions"][:] = buffer["base_positions"] + target_trans[buffer["segment_index"]]
        self._mesh.position[:] = buffer["positions"].ravel()

    def draw(self, entity_state):
        positions = entity_state.get("segment_positions") or []
        if not positions:
            head = entity_state.get("pos")
            if head is None:
                return
            positions = [head]
        head_pos = positions[0]
        head_state = {
            "pos": tuple(head_pos) if isinstance(head_pos, (tuple, list, np.ndarray)) else entity_state.get("pos"),
            "rot": entity_state.get("rot", (0.0, 0.0)),
        }
        self.head_renderer.draw(head_state)

        body_positions = np.array(positions[1:], dtype='f4')
        self.program["u_model"] = Mat4()
        self._update_mesh(body_positions)
        self._mesh.draw(gl.GL_TRIANGLES)

    def update(self, dt):
        pass
