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
            )

    def set_animation(self, anim_name):
        if anim_name != self.current_animation_name:
            self.current_animation_name = anim_name
            self.animation_time = 0.0

    def update(self, dt):
        anim = self.model['animations'].get(self.current_animation_name)
        if anim:
            self.animation_time += dt
            duration = anim.get('length', 1.0)
            if anim.get('loop', False) and duration > 0:
                self.animation_time %= duration

    def get_interpolated_rotation(self, part_name):
        anim = self.model['animations'].get(self.current_animation_name)
        default_rot = {'pitch': 0, 'yaw': 0, 'roll': 0}
        if not anim or 'keyframes' not in anim:
            return self.get_parent_rotation(part_name, anim)

        keyframes = [kf for kf in anim['keyframes'] if 'rotations' in kf and part_name in kf['rotations']]
        
        if not keyframes:
            return self.get_parent_rotation(part_name, anim)

        if keyframes[0]['time'] > self.animation_time:
            # If before the first keyframe, use its rotation but don't interpolate
            return keyframes[0]['rotations'][part_name]
        
        prev_kf = keyframes[0]
        next_kf = None
        for kf in keyframes:
            if kf['time'] > self.animation_time:
                next_kf = kf
                break
            prev_kf = kf
        
        if next_kf is None:
             return prev_kf['rotations'][part_name]

        prev_rot = prev_kf['rotations'][part_name]
        next_rot = next_kf['rotations'][part_name]
        
        time_diff = next_kf['time'] - prev_kf['time']
        if time_diff <= 0:
            t = 0
        else:
            t = (self.animation_time - prev_kf['time']) / time_diff
        
        interpolated_rot = {}
        for angle in ['pitch', 'yaw', 'roll']:
            start_angle = prev_rot.get(angle, 0)
            end_angle = next_rot.get(angle, 0)
            interpolated_rot[angle] = start_angle + t * (end_angle - start_angle)

        return interpolated_rot

    def get_parent_rotation(self, part_name, anim):
        """Recursively find rotation from parent part."""
        part_data = self.model['parts'].get(part_name, {})
        parent_name = part_data.get('parent')
        if parent_name:
            return self.get_interpolated_rotation(parent_name)
        return {'pitch': 0, 'yaw': 0, 'roll': 0}

    def draw(self, entity_state):
        pos = entity_state['pos']
        rot = entity_state['rot']

        # Find the root part and its size to position its base flush with entity_state['pos'].y
        root_part_name = self.model.get('root_part', 'body')
        initial_y_offset = -0.5
        if root_part_name in self.model['parts']:
            root_part_size = self.model['parts'][root_part_name]['size']
            initial_y_offset += root_part_size[1] / 2.0

        model_matrix = Mat4.from_translation(entity_state['pos'] + Vec3(0.0, initial_y_offset, 0.0))
        model_matrix = model_matrix.rotate(math.radians(rot[0]), Vec3(0, 1, 0))

        # Find the root part and start the recursive drawing
        root_part_name = self.model.get('root_part', 'body')
        if root_part_name in self.model['parts']:
            self._draw_part(root_part_name, model_matrix)

    def _draw_part(self, part_name, parent_matrix):
        part_data = self.model['parts'][part_name]

        # Get transformations for this part
        pivot = part_data.get('pivot', [0, 0, 0])
        rotation = self.get_interpolated_rotation(part_name)
        position = part_data.get('position', [0, 0, 0])

        # Create transformation matrices using pyglet.math methods
        pivot_vec = Vec3(*pivot)
        pivot_mat = Mat4.from_translation(pivot_vec)

        pos_vec = Vec3(*position)
        pos_mat = Mat4.from_translation(pos_vec)

        # Apply rotations
        rot_mat = Mat4()
        rot_mat = rot_mat.rotate(math.radians(rotation.get('roll', 0)), Vec3(0, 0, 1))
        rot_mat = rot_mat.rotate(math.radians(rotation.get('pitch', 0)), Vec3(1, 0, 0))
        rot_mat = rot_mat.rotate(math.radians(rotation.get('yaw', 0)), Vec3(0, 1, 0))

        # Final transformation matrix to draw this part's mesh
        draw_matrix = parent_matrix @ pivot_mat @rot_mat @ pos_mat 
        
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
    def __init__(self, program, model_definition, segment_configs=None, tail_length=12):
        self.program = program
        self.head_renderer = AnimatedEntityRenderer(program, model_definition)
        self.tail_length = tail_length
        configs = segment_configs or [
            ((0.40, 0.28, 0.36), (0, 0, 0)),
            ((0.34, 0.24, 0.33), (220, 40, 40)),
            ((0.28, 0.20, 0.30), (180, 40, 30)),
        ]
        self.segment_capacity = SnakeEntity.SEGMENT_COUNT
        self._variant_buffers = []
        self._active_meshes = []
        for size, color in configs:
            buffer = self._prepare_variant_buffer(size, color)
            self._variant_buffers.append(buffer)
            mesh = self.program.vertex_list(
                buffer["vertex_count"],
                gl.GL_TRIANGLES,
                position=('f', buffer["base_positions"].ravel()),
                tex_coords=('f', buffer["tex"].ravel()),
                normal=('f', buffer["normals"].ravel()),
                color=('f', buffer["colors"].ravel()),
            )
            self._active_meshes.append(mesh)

    def _build_mesh(self, size, color):
        verts, normals, indices = get_cube_vertices(size)
        tri_verts = verts.reshape(-1, 3)[indices]
        tri_norms = normals.reshape(-1, 3)[indices]
        tri_tex = np.zeros((tri_verts.shape[0], 2), dtype='f4')
        base_rgba = np.array([color[0], color[1], color[2], 1.0], dtype='f4')
        tri_col = np.broadcast_to(base_rgba, (tri_verts.shape[0], 4))
        return tri_verts, tri_norms, tri_tex, tri_col

    def _prepare_variant_buffer(self, size, color):
        tri_verts, tri_norms, tri_tex, tri_col = self._build_mesh(size, color)
        tri_count = tri_verts.shape[0]
        repeats = self.segment_capacity
        vertex_count = tri_count * repeats
        base_positions = np.tile(tri_verts, (repeats, 1)).astype('f4')
        normals = np.tile(tri_norms, (repeats, 1)).astype('f4')
        tex = np.tile(tri_tex, (repeats, 1)).astype('f4')
        colors = np.tile(tri_col, (repeats, 1)).astype('f4')
        return {
            "tri_count": tri_count,
            "vertex_count": vertex_count,
            "base_positions": base_positions,
            "normals": normals,
            "tex": tex,
            "colors": colors,
            "size": size, # Add size here
        }

    def _update_variant_mesh(self, idx, translations):
        mesh = self._active_meshes[idx]
        buffer = self._variant_buffers[idx]
        tri_count = buffer["tri_count"]
        vertex_count = buffer["vertex_count"]
        target_trans = np.full((self.segment_capacity, 3), 1e6, dtype='f4')

        if translations is not None and len(translations):
            usable = min(len(translations), self.segment_capacity)
            target_trans[:usable] = translations[:usable]
            if usable < self.segment_capacity:
                if usable > 0:
                    target_trans[usable:] = target_trans[usable - 1]
                else:
                    target_trans[usable:] = np.array([1e6, 1e6, 1e6], dtype='f4')

        positions = buffer["base_positions"].copy()
        initial_segment_y_offset = self._variant_buffers[idx]["size"][1] / 2.0 # Get segment height from buffer data

        offsets = np.repeat(target_trans, tri_count, axis=0)
        offsets[:, 1] += initial_segment_y_offset # Add the vertical offset to the y-component

        positions += offsets[:vertex_count]
        mesh.position[:] = positions.ravel()

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
        variant_count = len(self._variant_buffers)
        if variant_count == 0:
            return
        partitions = []
        if body_positions.size > 0:
            total = len(body_positions)
            tail_start = max(0, total - self.tail_length)
            if variant_count == 1:
                partitions = [body_positions]
            else:
                variant_indices = np.full(total, variant_count - 1, dtype=int)
                if tail_start > 0:
                    pattern_len = variant_count - 1
                    variant_indices[:tail_start] = np.arange(tail_start) % pattern_len
                partitions = [
                    body_positions[variant_indices == idx]
                    for idx in range(variant_count)
                ]
        else:
            partitions = [[] for _ in range(variant_count)]

        self.program["u_model"] = Mat4()
        for idx, translations in enumerate(partitions):
            if idx >= len(self._active_meshes):
                continue
            self._update_variant_mesh(idx, translations if isinstance(translations, np.ndarray) else np.array(translations, dtype='f4'))
            mesh = self._active_meshes[idx]
            if mesh:
                mesh.draw(gl.GL_TRIANGLES)

    def update(self, dt):
        pass
