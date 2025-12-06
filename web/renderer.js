/**
 * WebGL2 renderer for voxel geometry with atlas textures.
 * @module renderer
 */
import { BLOCK_TEXTURES, ATLAS_TILES } from './blocks.js';
import { FAR, FOV, NEAR, TEXTURE_ATLAS } from './config.js';
import { multiply, perspective, view } from './math.js';

const VERTEX_SRC = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_position;
layout(location=1) in vec3 a_normal;
layout(location=2) in vec2 a_uv;
layout(location=3) in float a_face;
layout(location=4) in vec3 a_translation;
layout(location=5) in float a_blockId;
layout(location=6) in float a_light;
uniform mat4 u_pv;
uniform vec2 u_blockUV[${(BLOCK_TEXTURES.length / 2).toFixed(0)}];
out vec2 v_uv;
out vec3 v_normal;
out float v_light;
void main(){
  int bid = int(a_blockId);
  int faceIndex = int(a_face);
  int uvIndex = bid * 6 + faceIndex;
  vec2 tileOrigin = u_blockUV[uvIndex];
  vec2 tileUV = tileOrigin + a_uv / float(${ATLAS_TILES});
  v_uv = tileUV;
  v_normal = a_normal;
  v_light = a_light;
  vec3 pos = a_position + a_translation;
  gl_Position = u_pv * vec4(pos,1.0);
}`;

const FRAGMENT_SRC = `#version 300 es
precision highp float;
in vec2 v_uv;
in vec3 v_normal;
in float v_light;
uniform sampler2D u_atlas;
uniform vec3 u_light;
out vec4 fragColor;
void main(){
  vec3 color = texture(u_atlas, v_uv).rgb;
  float sun = clamp(dot(normalize(v_normal), normalize(u_light)), 0.25, 1.0);
  float lighting = mix(0.15, sun, v_light);
  fragColor = vec4(color * lighting, 1.0);
}`;

/**
 * Handles WebGL2 rendering of voxel geometry.
 */
export class Renderer {
  /**
   * @param {HTMLCanvasElement} canvas - Target canvas for WebGL2.
   */
  constructor(canvas) {
    const gl = canvas.getContext('webgl2');
    if (!gl) {
      throw new Error('WebGL2 not supported');
    }
    this.gl = gl;
    this.program = this.createProgram();
    this.cube = this.createCubeGeometry();
    this.instanceBuffer = gl.createBuffer();
    this.blockIdBuffer = gl.createBuffer();
    this.lightBuffer = gl.createBuffer();
    this.instanceCount = 0;
    this.projection = perspective(FOV, canvas.clientWidth / canvas.clientHeight, NEAR, FAR);
    this.atlas = null;
    this.uploadUVs();
  }

  /**
   * Load the texture atlas image into GL.
   * @returns {Promise<void>}
   */
  async loadAtlas() {
    const gl = this.gl;
    const img = await loadImage(TEXTURE_ATLAS);
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    this.atlas = tex;
  }

  /**
   * Resize viewport and projection.
   * @param {number} width - Canvas width.
   * @param {number} height - Canvas height.
   */
  resize(width, height) {
    this.gl.viewport(0, 0, width, height);
    this.projection = perspective(FOV, width / height, NEAR, FAR);
  }

  /**
   * Upload instance data and render the scene.
   * @param {Float32Array} translations - Per-instance positions.
   * @param {Float32Array} blocks - Per-instance block ids.
   * @param {Float32Array} lights - Per-instance light multipliers.
   * @param {number} count - Number of instances.
   * @param {Float32Array} cameraPos - Camera position.
   * @param {number} yaw - Camera yaw in radians.
   * @param {number} pitch - Camera pitch in radians.
   */
  render(translations, blocks, lights, count, cameraPos, yaw, pitch) {
    if (!this.atlas) return;
    const gl = this.gl;
    this.instanceCount = count;
    gl.clearColor(0.7, 0.85, 1.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);

    gl.useProgram(this.program);
    const viewMat = view(cameraPos, yaw, pitch);
    const pv = multiply(this.projection, viewMat);

    gl.uniformMatrix4fv(gl.getUniformLocation(this.program, 'u_pv'), false, pv);
    gl.uniform3fv(gl.getUniformLocation(this.program, 'u_light'), new Float32Array([0.35, 1, 0.65]));

    this.uploadInstances(translations, blocks, lights);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.atlas);
    gl.uniform1i(gl.getUniformLocation(this.program, 'u_atlas'), 0);

    gl.bindVertexArray(this.cube.vao);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, this.cube.count, this.instanceCount);
  }

  /**
   * Upload instance buffers.
   * @param {Float32Array} translations - Positions.
   * @param {Float32Array} blocks - Block IDs.
   * @param {Float32Array} lights - Light values per instance.
   */
  uploadInstances(translations, blocks, lights) {
    const gl = this.gl;
    gl.bindVertexArray(this.cube.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, translations, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 3, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(4, 1);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.blockIdBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, blocks, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(5);
    gl.vertexAttribPointer(5, 1, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(5, 1);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.lightBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, lights, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(6);
    gl.vertexAttribPointer(6, 1, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(6, 1);
  }

  /**
   * Compile shaders and create program.
   * @returns {WebGLProgram} Compiled program.
   */
  createProgram() {
    const gl = this.gl;
    const vs = this.compileShader(gl.VERTEX_SHADER, VERTEX_SRC);
    const fs = this.compileShader(gl.FRAGMENT_SHADER, FRAGMENT_SRC);
    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program));
    }
    return program;
  }

  /**
   * Compile a shader object.
   * @param {number} type - GL shader type.
   * @param {string} source - GLSL code.
   * @returns {WebGLShader} Compiled shader.
   */
  compileShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader));
    }
    return shader;
  }

  /**
   * Upload block UV array into the shader uniform.
   */
  uploadUVs() {
    const gl = this.gl;
    gl.useProgram(this.program);
    gl.uniform2fv(gl.getUniformLocation(this.program, 'u_blockUV'), BLOCK_TEXTURES);
  }

  /**
   * Create cube VAO with per-face uv and face id attributes.
   * @returns {{vao: WebGLVertexArrayObject, count: number}} Geometry handle.
   */
  createCubeGeometry() {
    const gl = this.gl;
    const positions = [];
    const normals = [];
    const uvs = [];
    const faces = [];
    const facesDef = [
      { n: [0, 1, 0], p: [[-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]] },
      { n: [0, -1, 0], p: [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5]] },
      { n: [-1, 0, 0], p: [[-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5]] },
      { n: [1, 0, 0], p: [[0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5]] },
      { n: [0, 0, 1], p: [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]] },
      { n: [0, 0, -1], p: [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]] },
    ];
    const baseUV = [0, 0, 1, 0, 1, 1, 0, 1];
    for (let faceIndex = 0; faceIndex < facesDef.length; faceIndex++) {
      const face = facesDef[faceIndex];
      const p = face.p;
      // two triangles per face
      const order = [0, 1, 2, 0, 2, 3];
      for (let i = 0; i < order.length; i++) {
        const v = p[order[i]];
        positions.push(v[0], v[1], v[2]);
        normals.push(face.n[0], face.n[1], face.n[2]);
        const uvIndex = order[i];
        uvs.push(baseUV[uvIndex * 2], baseUV[uvIndex * 2 + 1]);
        faces.push(faceIndex);
      }
    }

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    const normalBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

    const uvBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(uvs), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 0, 0);

    const faceBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, faceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(faces), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 1, gl.FLOAT, false, 0, 0);

    return { vao, count: positions.length / 3 };
  }
}

/**
 * Load an image from a URL.
 * @param {string} url
 * @returns {Promise<HTMLImageElement>}
 */
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}
