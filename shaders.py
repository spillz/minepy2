from pyglet.graphics.shader import Shader, ShaderProgram


VERTEX_SOURCE = """
#version 330 core

uniform mat4 u_projection;
uniform mat4 u_view;
uniform vec3 u_camera_pos;

in vec3 position;
in vec2 tex_coords;
in vec3 normal;
in vec3 color;

out vec2 v_tex_coords;
out vec3 v_normal;
out vec3 v_color;
out vec3 v_view_position;

void main() {
    // Work in camera-relative space to reduce floating point error at large coords.
    vec3 rel = position - u_camera_pos;
    vec4 view_position = u_view * vec4(rel, 1.0);
    gl_Position = u_projection * view_position;
    v_tex_coords = tex_coords;
    v_normal = mat3(u_view) * normal;
    v_color = color / 255.0;
    v_view_position = view_position.xyz;
}
"""


FRAGMENT_SOURCE = """
#version 330 core

uniform sampler2D u_texture;
uniform vec3 u_light_dir;
uniform vec3 u_fog_color;
uniform float u_fog_start;
uniform float u_fog_end;
uniform bool u_water_pass;
uniform float u_water_alpha;

in vec2 v_tex_coords;
in vec3 v_normal;
in vec3 v_color;
in vec3 v_view_position;

out vec4 out_color;

void main() {
    vec3 n = normalize(v_normal);
    float light = max(dot(n, normalize(u_light_dir)), 0.0);
    vec4 tex_color = texture(u_texture, v_tex_coords);
    float alpha = tex_color.a * (u_water_pass ? u_water_alpha : 1.0);
    if (alpha < 0.05) {
        discard;
    }
    vec3 base_color = tex_color.rgb * v_color;
    vec3 lit_color = base_color * (0.3 + 0.7 * light);
    float distance = length(v_view_position);
    float fog_factor = clamp((u_fog_end - distance) / (u_fog_end - u_fog_start), 0.0, 1.0);
    vec3 fogged_color = mix(u_fog_color, lit_color, fog_factor);
    out_color = vec4(fogged_color, alpha);
}
"""


def create_block_shader():
    """Create the shader program used for world rendering."""
    vertex_shader = Shader(VERTEX_SOURCE, "vertex")
    fragment_shader = Shader(FRAGMENT_SOURCE, "fragment")
    return ShaderProgram(vertex_shader, fragment_shader)
