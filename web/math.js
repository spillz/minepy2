/**
 * Minimal linear algebra utilities for the renderer.
 * @module math
 */

/**
 * Create a perspective projection matrix.
 * @param {number} fov - Field of view in radians.
 * @param {number} aspect - Aspect ratio.
 * @param {number} near - Near clipping plane.
 * @param {number} far - Far clipping plane.
 * @returns {Float32Array} 4x4 matrix.
 */
export function perspective(fov, aspect, near, far) {
  const f = 1.0 / Math.tan(fov / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, (2 * far * near) * nf, 0,
  ]);
}

/**
 * Create a view matrix from position and yaw/pitch orientation.
 * @param {Float32Array} position - Camera position [x, y, z].
 * @param {number} yaw - Horizontal rotation in radians.
 * @param {number} pitch - Vertical rotation in radians.
 * @returns {Float32Array} 4x4 matrix.
 */
export function view(position, yaw, pitch) {
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);

  const fx = cp * cy;
  const fy = sp;
  const fz = cp * sy;

  const rx = -fz;
  const ry = 0;
  const rz = fx;
  const rl = Math.max(Math.hypot(rx, ry, rz), 1e-6);
  const rnx = rx / rl;
  const rny = ry / rl;
  const rnz = rz / rl;

  const ux = rny * fz - rnz * fy;
  const uy = rnz * fx - rnx * fz;
  const uz = rnx * fy - rny * fx;

  const tx = -(rnx * position[0] + rny * position[1] + rnz * position[2]);
  const ty = -(ux * position[0] + uy * position[1] + uz * position[2]);
  const tz = -(-fx * position[0] - fy * position[1] - fz * position[2]);

  return new Float32Array([
    rnx, ux, -fx, 0,
    rny, uy, -fy, 0,
    rnz, uz, -fz, 0,
    tx, ty, tz, 1,
  ]);
}

/**
 * Multiply two 4x4 matrices.
 * @param {Float32Array} a - Matrix A.
 * @param {Float32Array} b - Matrix B.
 * @returns {Float32Array} Product matrix.
 */
export function multiply(a, b) {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      out[i * 4 + j] =
        a[i * 4 + 0] * b[0 * 4 + j] +
        a[i * 4 + 1] * b[1 * 4 + j] +
        a[i * 4 + 2] * b[2 * 4 + j] +
        a[i * 4 + 3] * b[3 * 4 + j];
    }
  }
  return out;
}
