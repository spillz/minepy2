from pyglet.graphics.shader import Shader, ShaderProgram


VERTEX_SOURCE = """
#version 330 core

uniform mat4 u_projection;
uniform mat4 u_view;
uniform mat4 u_model;
uniform vec3 u_camera_pos;

in vec3 position;
in vec2 tex_coords;
in vec3 normal;
in vec4 color;
in vec2 light;

out vec2 v_tex_coords;
out vec3 v_normal;
out vec4 v_color;
out vec2 v_light;
out vec3 v_view_position;

void main() {
    // Transform vertex to world space using the model matrix
    vec4 world_position = u_model * vec4(position, 1.0);

    // Work in camera-relative space to reduce floating point error at large coords.
    vec3 rel = world_position.xyz - u_camera_pos;
    vec4 view_position = u_view * vec4(rel, 1.0);
    gl_Position = u_projection * view_position;

    v_tex_coords = tex_coords;

    // Transform normal to view space
    // Assumes u_model has no non-uniform scaling, so inverse transpose is not needed.
    v_normal = mat3(u_view) * mat3(u_model) * normal;
    
    v_color = color / 255.0;
    v_light = light;
    v_view_position = view_position.xyz;
}
"""


FRAGMENT_SOURCE = """
#version 330 core

uniform sampler2D u_texture;
uniform vec3 u_light_dir;
uniform float u_ambient_light;
uniform float u_sky_intensity;
uniform vec3 u_fog_color;
uniform float u_fog_start;
uniform float u_fog_end;
uniform bool u_water_pass;
uniform float u_water_alpha;
uniform bool u_use_vertex_color;
uniform bool u_use_texture;

in vec2 v_tex_coords;
in vec3 v_normal;
in vec4 v_color;
in vec2 v_light;
in vec3 v_view_position;

out vec4 out_color;

void main() {
    vec3 n = normalize(v_normal);
    float light = max(dot(n, normalize(u_light_dir)), 0.0);
    vec3 color = u_use_vertex_color ? v_color.rgb : vec3(1.0);
    vec4 tex_color = u_use_texture ? texture(u_texture, v_tex_coords) : vec4(1.0);
    float alpha = tex_color.a * (u_water_pass ? u_water_alpha : 1.0);
    if (alpha < 0.05) {
        discard;
    }
    float torch_light = v_light.x;
    float sky_light = v_light.y * u_sky_intensity;
    float voxel_light = clamp(u_ambient_light + (torch_light + sky_light) * (1.0 - u_ambient_light), 0.0, 1.0);
    vec3 base_color = tex_color.rgb * color;
    vec3 lit_color = base_color * (0.3 + 0.7 * light) * voxel_light;
    vec3 emissive = tex_color.rgb * v_color.a;
    vec3 final_color = lit_color + emissive;
    float distance = length(v_view_position);
    float fog_factor = clamp(
        (u_fog_end - distance) / (u_fog_end - u_fog_start),
        0.0,
        1.0
    );
    vec3 fogged_color = mix(u_fog_color, final_color, fog_factor);
    out_color = vec4(fogged_color, alpha);
}
"""


def create_block_shader():
    """Create the shader program used for world rendering."""
    vertex_shader = Shader(VERTEX_SOURCE, "vertex")
    fragment_shader = Shader(FRAGMENT_SOURCE, "fragment")
    program = ShaderProgram(vertex_shader, fragment_shader)
    program['u_use_texture'] = True
    program['u_use_vertex_color'] = True
    program['u_ambient_light'] = 0.0
    program['u_sky_intensity'] = 1.0
    return program
