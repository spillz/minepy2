/**
 * Utility helpers used across the web client.
 * @module util
 */

/**
 * Create a deterministic pseudo random generator.
 * @param {number} seed - Seed integer.
 * @returns {() => number} Generator returning values in [0, 1).
 */
export function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let m = Math.imul(t ^ (t >>> 15), 1 | t);
    m ^= m + Math.imul(m ^ (m >>> 7), 61 | m);
    return ((m ^ (m >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Generate a unique key string for the given chunk coordinate.
 * @param {number} cx - Chunk x coordinate.
 * @param {number} cz - Chunk z coordinate.
 * @returns {string} Key for the chunk map.
 */
export function chunkKey(cx, cz) {
  return `${cx},${cz}`;
}

/**
 * Clamp a value to the provided range.
 * @param {number} value - Value to clamp.
 * @param {number} min - Minimum value.
 * @param {number} max - Maximum value.
 * @returns {number} Clamped value.
 */
export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

/**
 * Raycast through the world using incremental stepping.
 * @param {import('./world.js').World} world
 * @param {Float32Array} origin
 * @param {Float32Array} direction
 * @param {number} maxDistance
 * @returns {{hit:boolean, position:[number,number,number], normal:[number,number,number], block:number}}
 */
export function raycast(world, origin, direction, maxDistance) {
  const step = 0.1;
  const pos = new Float32Array(origin);
  const dir = direction;
  const normal = [0, 0, 0];
  for (let traveled = 0; traveled <= maxDistance; traveled += step) {
    const bx = Math.floor(pos[0]);
    const by = Math.floor(pos[1]);
    const bz = Math.floor(pos[2]);
    const block = world.getBlock(bx, by, bz);
    if (block) {
      return { hit: true, position: [bx, by, bz], normal, block };
    }
    pos[0] += dir[0] * step;
    pos[1] += dir[1] * step;
    pos[2] += dir[2] * step;
    normal[0] = -Math.sign(dir[0]);
    normal[1] = -Math.sign(dir[1]);
    normal[2] = -Math.sign(dir[2]);
  }
  return { hit: false, position: [0, 0, 0], normal: [0, 0, 0], block: 0 };
}
