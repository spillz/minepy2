import math
import numpy as np
import pyglet
import pyglet.gl as gl
from pyglet.math import Mat4, Vec3

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
        
        # Start with the entity's base transformation matrix (position and yaw)
        root_offset = Vec3(*self.model.get('root_offset', (0.0, 0.0, 0.0)))
        model_matrix = Mat4.from_translation(entity_state['pos'] + root_offset)
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


        # Transformation for the part's coordinate system (to be inherited by children)
        child_parent_matrix = parent_matrix @ pivot_mat @ rot_mat
        
        # Final transformation matrix to draw this part's mesh
        draw_matrix = child_parent_matrix @ pos_mat
        
        # Set the model matrix uniform and draw the mesh
        self.program['u_model'] = draw_matrix
        mesh = self.part_meshes.get(part_name)
        if mesh:
            mesh.draw(gl.GL_TRIANGLES)

        # Recursively draw children, passing the correct parent matrix
        for child_name, child_data in self.model['parts'].items():
            if child_data.get('parent') == part_name:
                self._draw_part(child_name, child_parent_matrix)