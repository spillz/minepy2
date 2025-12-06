/**
 * Chunk container and mesh conversion helpers.
 * @module chunk
 */
import { CHUNK_HEIGHT, CHUNK_SIZE } from './config.js';
import { BLOCK_SOLID } from './blocks.js';

/**
 * Represents a single chunk of voxel data.
 */
export class Chunk {
  /**
   * @param {number} cx - Chunk x coordinate.
   * @param {number} cz - Chunk z coordinate.
   * @param {Uint8Array} blocks - Block IDs laid out in x,z,y order.
   */
  constructor(cx, cz, blocks) {
    this.cx = cx;
    this.cz = cz;
    this.blocks = blocks;
    this.instanceCount = 0;
    this.instances = null;
    this.blockIds = null;
    this.lightLevels = null;
    this.computeLighting();
    this.buildInstanceData();
  }

  /**
   * Lookup a block within the chunk.
   * @param {number} x
   * @param {number} y
   * @param {number} z
   * @returns {number}
   */
  getBlock(x, y, z) {
    if (x < 0 || x >= CHUNK_SIZE || z < 0 || z >= CHUNK_SIZE || y < 0 || y >= CHUNK_HEIGHT) {
      return 0;
    }
    return this.blocks[(y * CHUNK_SIZE + z) * CHUNK_SIZE + x];
  }

  /**
   * Set a block and rebuild lighting and instance data.
   * @param {number} x
   * @param {number} y
   * @param {number} z
   * @param {number} id
   */
  setBlock(x, y, z, id) {
    if (x < 0 || x >= CHUNK_SIZE || z < 0 || z >= CHUNK_SIZE || y < 0 || y >= CHUNK_HEIGHT) {
      return;
    }
    this.blocks[(y * CHUNK_SIZE + z) * CHUNK_SIZE + x] = id;
    this.computeLighting();
    this.buildInstanceData();
  }

  /**
   * Recompute skylight/ambient values per block.
   */
  computeLighting() {
    const size = CHUNK_SIZE;
    const height = CHUNK_HEIGHT;
    const light = new Float32Array(this.blocks.length);
    // Skylight pass: march downward from top of column.
    for (let z = 0; z < size; z++) {
      for (let x = 0; x < size; x++) {
        let brightness = 1.0;
        for (let y = height - 1; y >= 0; y--) {
          const idx = (y * size + z) * size + x;
          const solid = BLOCK_SOLID[this.blocks[idx]];
          if (solid) {
            brightness = Math.max(0.1, brightness - 0.18);
          } else {
            brightness = Math.max(brightness, 0.05);
          }
          light[idx] = brightness;
        }
      }
    }
    // Simple diffusion: pull light from neighbors.
    const neighborOffsets = [
      -1,
      1,
      -size,
      size,
      -size * size,
      size * size,
    ];
    for (let iter = 0; iter < 2; iter++) {
      for (let y = 0; y < height; y++) {
        for (let z = 0; z < size; z++) {
          for (let x = 0; x < size; x++) {
            const idx = (y * size + z) * size + x;
            let maxNeighbor = light[idx];
            for (let n = 0; n < neighborOffsets.length; n++) {
              const offset = neighborOffsets[n];
              const nx = x + (n === 0 ? -1 : n === 1 ? 1 : 0);
              const nz = z + (n === 2 ? -1 : n === 3 ? 1 : 0);
              const ny = y + (n === 4 ? -1 : n === 5 ? 1 : 0);
              if (nx < 0 || nx >= size || nz < 0 || nz >= size || ny < 0 || ny >= height) continue;
              const neighborIdx = idx + offset;
              const candidate = light[neighborIdx] - 0.05;
              if (candidate > maxNeighbor) {
                maxNeighbor = candidate;
              }
            }
            light[idx] = Math.max(0.05, maxNeighbor);
          }
        }
      }
    }
    this.lightLevels = light;
  }

  /**
   * Compute whether a block is hidden by neighbors.
   * @param {number} x
   * @param {number} y
   * @param {number} z
   * @returns {boolean}
   */
  occluded(x, y, z) {
    const size = CHUNK_SIZE;
    const height = CHUNK_HEIGHT;
    const idx = (y * size + z) * size + x;
    const self = this.blocks[idx];
    if (self === 0) return true;
    // If on boundary, never occluded to avoid holes at chunk edges.
    if (x === 0 || x === size - 1 || z === 0 || z === size - 1 || y === 0 || y === height - 1) {
      return false;
    }
    const neighbors = [
      this.blocks[idx - 1],
      this.blocks[idx + 1],
      this.blocks[idx - size],
      this.blocks[idx + size],
      this.blocks[idx - size * size],
      this.blocks[idx + size * size],
    ];
    for (let i = 0; i < neighbors.length; i++) {
      if (!BLOCK_SOLID[neighbors[i]]) {
        return false;
      }
    }
    return true;
  }

  /**
   * Create compact instance arrays for rendering.
   * The block storage layout matches python's chunk convention.
   */
  buildInstanceData() {
    const size = CHUNK_SIZE;
    const height = CHUNK_HEIGHT;
    const translations = new Float32Array(this.blocks.length * 3);
    const blockIds = new Float32Array(this.blocks.length);
    const lights = new Float32Array(this.blocks.length);
    let index = 0;

    for (let y = 0; y < height; y++) {
      for (let z = 0; z < size; z++) {
        for (let x = 0; x < size; x++) {
          const block = this.blocks[(y * size + z) * size + x];
          if (block === 0) {
            continue;
          }
          if (this.occluded(x, y, z)) {
            continue;
          }
          const bx = this.cx * size + x;
          const by = y;
          const bz = this.cz * size + z;
          translations[index * 3 + 0] = bx;
          translations[index * 3 + 1] = by;
          translations[index * 3 + 2] = bz;
          blockIds[index] = block;
          lights[index] = this.lightLevels[(y * size + z) * size + x];
          index++;
        }
      }
    }
    this.instanceCount = index;
    this.instances = translations.subarray(0, index * 3);
    this.blockIds = blockIds.subarray(0, index);
    this.instanceLights = lights.subarray(0, index);
  }
}
