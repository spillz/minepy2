/**
 * Chunk streaming management for the browser client.
 * @module world
 */
import { CHUNK_SIZE, VIEW_DISTANCE } from './config.js';
import { Chunk } from './chunk.js';
import { chunkKey } from './util.js';

/**
 * World orchestrates chunk streaming and background generation.
 */
export class World {
  constructor() {
    this.chunks = new Map();
    this.pending = new Set();
    this.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    this.worker.onmessage = (event) => {
      const { cx, cz, buffer } = event.data;
      const blocks = new Uint8Array(buffer);
      const chunk = new Chunk(cx, cz, blocks);
      this.chunks.set(chunkKey(cx, cz), chunk);
      this.pending.delete(chunkKey(cx, cz));
    };
  }

  /**
   * Request chunks around the given world position.
   * @param {Float32Array} position - Player position.
   */
  update(position) {
    const cx = Math.floor(position[0] / CHUNK_SIZE);
    const cz = Math.floor(position[2] / CHUNK_SIZE);
    for (let dz = -VIEW_DISTANCE; dz <= VIEW_DISTANCE; dz++) {
      for (let dx = -VIEW_DISTANCE; dx <= VIEW_DISTANCE; dx++) {
        const nx = cx + dx;
        const nz = cz + dz;
        const key = chunkKey(nx, nz);
        if (this.chunks.has(key) || this.pending.has(key)) {
          continue;
        }
        this.pending.add(key);
        this.worker.postMessage({ type: 'generate', cx: nx, cz: nz });
      }
    }
  }

  /**
   * Retrieve a chunk if available.
   * @param {number} cx
   * @param {number} cz
   * @returns {Chunk | null}
   */
  getChunk(cx, cz) {
    return this.chunks.get(chunkKey(cx, cz)) ?? null;
  }

  /**
   * Sample a block at world coordinates.
   * @param {number} x
   * @param {number} y
   * @param {number} z
   * @returns {number}
   */
  getBlock(x, y, z) {
    const cx = Math.floor(x / CHUNK_SIZE);
    const cz = Math.floor(z / CHUNK_SIZE);
    const chunk = this.getChunk(cx, cz);
    if (!chunk) return 0;
    const lx = x - cx * CHUNK_SIZE;
    const lz = z - cz * CHUNK_SIZE;
    return chunk.getBlock(lx, y, lz);
  }

  /**
   * Set a block at world coordinates.
   * @param {number} x
   * @param {number} y
   * @param {number} z
   * @param {number} id
   */
  setBlock(x, y, z, id) {
    const cx = Math.floor(x / CHUNK_SIZE);
    const cz = Math.floor(z / CHUNK_SIZE);
    const chunk = this.getChunk(cx, cz);
    if (!chunk) return;
    const lx = x - cx * CHUNK_SIZE;
    const lz = z - cz * CHUNK_SIZE;
    chunk.setBlock(lx, y, lz, id);
  }

  /**
   * Collect instance data for all loaded chunks.
   * @returns {{ translations: Float32Array, blocks: Float32Array, lights: Float32Array, count: number }}
   */
  gatherInstances() {
    let total = 0;
    for (const chunk of this.chunks.values()) {
      total += chunk.instanceCount;
    }
    const translations = new Float32Array(total * 3);
    const blocks = new Float32Array(total);
    const lights = new Float32Array(total);
    let offset = 0;
    for (const chunk of this.chunks.values()) {
      translations.set(chunk.instances, offset * 3);
      blocks.set(chunk.blockIds, offset);
      lights.set(chunk.instanceLights, offset);
      offset += chunk.instanceCount;
    }
    return { translations, blocks, lights, count: total };
  }
}
