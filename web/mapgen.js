/**
 * Procedural terrain generation mirroring the Python mapgen module.
 * @module mapgen
 */
import { CHUNK_HEIGHT, CHUNK_SIZE } from './config.js';
import { BLOCK_ID } from './blocks.js';
import { SimplexNoise } from './noise.js';

/**
 * Wrapper to sample 2D simplex noise with tiling-friendly coordinates.
 */
class SectorNoise2D {
  /**
   * @param {number} seed
   * @param {number} step
   * @param {number} stepOffset
   * @param {number} scale
   * @param {number} offset
   */
  constructor(seed, step, stepOffset, scale, offset) {
    this.noise = new SimplexNoise(seed);
    this.step = step;
    this.stepOffset = stepOffset;
    this.scale = scale;
    this.offset = offset;
  }

  /**
   * @param {[number, number]} position - sector x/z
   * @returns {Float32Array} (CHUNK_SIZE+2)^2 heights
   */
  sample(position) {
    const out = new Float32Array((CHUNK_SIZE + 2) * (CHUNK_SIZE + 2));
    let idx = 0;
    const originX = position[0] + this.stepOffset;
    const originZ = position[1] + this.stepOffset;
    for (let z = -1; z <= CHUNK_SIZE; z++) {
      for (let x = -1; x <= CHUNK_SIZE; x++) {
        const nx = (originX + x) / this.step;
        const nz = (originZ + z) / this.step;
        out[idx++] = this.noise.noise2D(nx, nz) * this.scale + this.offset;
      }
    }
    return out;
  }
}

/**
 * Light 3D simplex helper used for caves and exploration details.
 */
class SectorNoise3D {
  /**
   * @param {number} seed
   * @param {number} step
   * @param {number} scale
   * @param {number} offset
   */
  constructor(seed, step, scale, offset) {
    this.noise = new SimplexNoise(seed);
    this.step = step;
    this.scale = scale;
    this.offset = offset;
  }

  /**
   * @param {[number, number]} position - sector x/z
   * @returns {Float32Array} (CHUNK_SIZE+2)^2*CHUNK_HEIGHT values
   */
  sample(position) {
    const sx = position[0];
    const sz = position[1];
    const out = new Float32Array((CHUNK_SIZE + 2) * (CHUNK_SIZE + 2) * CHUNK_HEIGHT);
    let idx = 0;
    for (let y = 0; y < CHUNK_HEIGHT; y++) {
      for (let z = -1; z <= CHUNK_SIZE; z++) {
        for (let x = -1; x <= CHUNK_SIZE; x++) {
          const nx = (sx + x) / this.step;
          const ny = y / this.step;
          const nz = (sz + z) / this.step;
          out[idx++] = this.noise.noise3D(nx, ny, nz) * this.scale + this.offset;
        }
      }
    }
    return out;
  }
}

/**
 * Main biome generator.
 */
export class BiomeGenerator {
  /**
   * @param {number} seed
   */
  constructor(seed = Date.now()) {
    this.seed = seed;
    // Broad elevation controls.
    this.noiseHill = new SectorNoise2D(seed + 12, 40.0, 30, 5, 5);
    this.noiseHillB = new SectorNoise2D(seed + 16, 40.0, 900, 5, 5);
    this.noiseContinental = new SectorNoise2D(seed + 14, 1500.0, 531, 40.0, 80);
    this.noiseGain = new SectorNoise2D(seed + 18, 3000.0, 8123, 5, 5);
    this.ridgeNoise = new SectorNoise2D(seed + 102, 220.0, 900, 12.0, 0.0);
    this.terrainNoise = new SectorNoise2D(seed + 108, 700.0, 1337, 1.0, 0.0);
    this.detailNoise = new SectorNoise2D(seed + 109, 90.0, 1650, 6.0, 0.0);
    this.jaggedNoise = new SectorNoise2D(seed + 111, 60.0, 987, 10.0, 0.0);
    this.fineTerrain = new SectorNoise2D(seed + 110, 180.0, 733, 1.0, 0.0);
    this.moisture = new SectorNoise2D(seed + 105, 520.0, 700, 1.0, 0.0);
    this.temperature = new SectorNoise2D(seed + 106, 680.0, 123, 1.0, 0.0);
    this.decor = new SectorNoise2D(seed + 150, 140.0, 3300, 1.0, 0.0);
    // Underground and exploratory features.
    this.caveHeight = new SectorNoise2D(seed + 120, 120.0, 3100, 1.0, 0.0);
    this.caveDensity = new SectorNoise2D(seed + 121, 60.0, 4100, 1.0, 0.0);
    this.caveBand = 0.2;
    this.caveOutletDepth = 3;
    this.caveDetail3d = new SectorNoise3D(seed + 122, 55.0, 0.8, 0.0);
    this.oreHeight = new SectorNoise2D(seed + 140, 160.0, 5200, 1.0, 0.0);
    this.oreDensity = new SectorNoise2D(seed + 141, 90.0, 6200, 1.0, 0.0);
    this.structureMask = new SectorNoise2D(seed + 114, 950.0, 5100, 1.0, 0.0);
    this.structureFine = new SectorNoise2D(seed + 115, 210.0, 4111, 1.0, 0.0);
  }

  /**
   * Compute ground height field for a sector.
   * @param {[number, number]} position
   * @returns {Float32Array}
   */
  elevation(position) {
    const a = this.noiseHill.sample(position);
    const b = this.noiseHillB.sample(position);
    const continental = this.noiseContinental.sample(position);
    const gain = this.noiseGain.sample(position);
    const ridge = this.ridgeNoise.sample(position);
    const terrain = this.terrainNoise.sample(position);
    const detail = this.detailNoise.sample(position);
    const fine = this.fineTerrain.sample(position);
    const jagged = this.jaggedNoise.sample(position);
    const out = new Float32Array((CHUNK_SIZE + 2) * (CHUNK_SIZE + 2));
    for (let i = 0; i < out.length; i++) {
      const roll = a[i] * gain[i] + continental[i];
      const roll2 = b[i] + continental[i];
      const rough = ridge[i] * 0.6 + detail[i] * 0.35;
      const landType = terrain[i] * 0.7 + fine[i] * 0.3;
      const jag = Math.abs(jagged[i]);
      let h;
      if (landType < -0.35) {
        // plains
        h = roll * 0.35 + rough * 0.4 + jag * 0.15;
      } else if (landType > 0.35) {
        // plateaus and cliffs
        h = roll2 * 0.65 + rough * 1.45 + jag * 1.35;
      } else {
        // hills
        h = (roll + roll2) * 0.5 + rough * 0.9 + jag * 0.55;
      }
      out[i] = h;
    }
    return out;
  }

  /**
   * Generate a chunk of blocks.
   * @param {number} cx
   * @param {number} cz
   * @returns {Uint8Array}
   */
  generate(cx, cz) {
    const blocks = new Uint8Array(CHUNK_SIZE * CHUNK_SIZE * CHUNK_HEIGHT);
    const field = this.elevation([cx * CHUNK_SIZE, cz * CHUNK_SIZE]);
    const moist = this.moisture.sample([cx * CHUNK_SIZE, cz * CHUNK_SIZE]);
    const temp = this.temperature.sample([cx * CHUNK_SIZE, cz * CHUNK_SIZE]);
    const decor = this.decor.sample([cx * CHUNK_SIZE, cz * CHUNK_SIZE]);
    const caveHeight = this.caveHeight.sample([cx * CHUNK_SIZE, cz * CHUNK_SIZE]);
    const caveDensity = this.caveDensity.sample([cx * CHUNK_SIZE, cz * CHUNK_SIZE]);
    const oreHeight = this.oreHeight.sample([cx * CHUNK_SIZE, cz * CHUNK_SIZE]);
    const oreDensity = this.oreDensity.sample([cx * CHUNK_SIZE, cz * CHUNK_SIZE]);
    const caveDetail = this.caveDetail3d.sample([cx * CHUNK_SIZE, cz * CHUNK_SIZE]);
    const structures = this.structureMask.sample([cx * CHUNK_SIZE, cz * CHUNK_SIZE]);
    const structureFine = this.structureFine.sample([cx * CHUNK_SIZE, cz * CHUNK_SIZE]);
    const sandId = BLOCK_ID['Sand'];
    const grassId = BLOCK_ID['Grass'];
    const stoneId = BLOCK_ID['Stone'];
    const roseId = BLOCK_ID['Rose'];
    const coalId = BLOCK_ID['Coal Ore'];
    const ironId = BLOCK_ID['Iron Ore'];
    const goldId = BLOCK_ID['Gold Ore'];
    const diamondId = BLOCK_ID['Diamond Ore'];
    const redstoneId = BLOCK_ID['Redstone Ore'];
    const emeraldId = BLOCK_ID['Emerald Ore'];

    for (let z = 0; z < CHUNK_SIZE; z++) {
      for (let x = 0; x < CHUNK_SIZE; x++) {
        const idx2 = (z + 1) * (CHUNK_SIZE + 2) + (x + 1);
        const h = Math.max(1, Math.min(CHUNK_HEIGHT - 2, Math.floor(field[idx2])));
        const moistVal = moist[idx2];
        const tempVal = temp[idx2];
        const decorVal = decor[idx2];
        const isBeach = moistVal > 0.15 && h < 20;
        const surfaceBlock = isBeach ? sandId : grassId;
        const caveH = caveHeight[idx2];
        const caveD = caveDensity[idx2];
        const oreH = oreHeight[idx2];
        const oreD = oreDensity[idx2];
        for (let y = 0; y <= h; y++) {
          const index = (y * CHUNK_SIZE + z) * CHUNK_SIZE + x;
          const caveIndex = (y * (CHUNK_SIZE + 2) + (z + 1)) * (CHUNK_SIZE + 2) + (x + 1);
          const caveSample = caveDetail[caveIndex];
          const caveBand = Math.abs(caveSample - caveH) - this.caveBand;
          const carve = caveBand < 0 && caveD > -0.25 && y > 4 ? 1 : 0;
          if (carve) {
            continue;
          }
          if (y === h) {
            blocks[index] = surfaceBlock;
          } else if (y > h - 3) {
            blocks[index] = surfaceBlock === grassId ? grassId : sandId;
          } else {
            blocks[index] = stoneId;
            if (y < h - 6 && decorVal > 0.2 && decorVal < 0.28 && (y & 3) === 0) {
              blocks[index] = ironId;
            } else if (y < h - 8 && decorVal < -0.45 && (y & 7) === 0) {
              blocks[index] = coalId;
            } else if (y < 48 && oreD > 0.45 && oreH > 0.1 && (y % 4 === 0)) {
              blocks[index] = goldId;
            } else if (y < 32 && oreD > 0.6 && oreH < -0.2) {
              blocks[index] = redstoneId;
            } else if (y < 20 && oreD > 0.65 && oreH < -0.35) {
              blocks[index] = diamondId;
            } else if (y < 72 && oreD < -0.55 && oreH > 0.5 && (x & 1) === 0 && (z & 1) === 0) {
              blocks[index] = emeraldId;
            }
          }
        }
        if (surfaceBlock === grassId && decorVal > 0.45 && decorVal < 0.6 && h + 5 < CHUNK_HEIGHT) {
          this.placeTree(blocks, x, h + 1, z);
        }
        if (surfaceBlock === grassId && tempVal > 0.35 && decorVal > 0.7 && decorVal < 0.74) {
          const idx = ((h + 1) * CHUNK_SIZE + z) * CHUNK_SIZE + x;
          blocks[idx] = roseId;
        }
        if (surfaceBlock === grassId && structures[idx2] > 0.55 && structureFine[idx2] > 0.1 && h + 6 < CHUNK_HEIGHT) {
          this.placeCabin(blocks, x, h + 1, z);
        }
      }
    }
    return blocks;
  }

  /**
   * Paint a minimal tree structure.
   * @param {Uint8Array} blocks
   * @param {number} x
   * @param {number} y
   * @param {number} z
   */
  placeTree(blocks, x, y, z) {
    const woodId = BLOCK_ID['Wood'];
    const leavesId = BLOCK_ID['Leaves'];
    const height = 4;
    for (let i = 0; i < height; i++) {
      const idx = ((y + i) * CHUNK_SIZE + z) * CHUNK_SIZE + x;
      if (y + i < CHUNK_HEIGHT) {
        blocks[idx] = woodId;
      }
    }
    for (let dy = 2; dy <= 4; dy++) {
      for (let dz = -2; dz <= 2; dz++) {
        for (let dx = -2; dx <= 2; dx++) {
          const dist = Math.abs(dx) + Math.abs(dy - 3) + Math.abs(dz);
          if (dist > 4) continue;
          const tx = x + dx;
          const tz = z + dz;
          const ty = y + dy;
          if (tx < 0 || tx >= CHUNK_SIZE || tz < 0 || tz >= CHUNK_SIZE || ty < 0 || ty >= CHUNK_HEIGHT) continue;
          const idx = (ty * CHUNK_SIZE + tz) * CHUNK_SIZE + tx;
          if (blocks[idx] === 0) {
            blocks[idx] = leavesId;
          }
        }
      }
    }
  }

  /**
   * Place a tiny cabin for exploration landmarks.
   * @param {Uint8Array} blocks
   * @param {number} x
   * @param {number} y
   * @param {number} z
   */
  placeCabin(blocks, x, y, z) {
    const plank = BLOCK_ID['Plank'];
    const wood = BLOCK_ID['Wood'];
    const leaves = BLOCK_ID['Leaves'];
    const jack = BLOCK_ID["Jack O'Lantern"];
    const w = 4;
    const h = 3;
    for (let dy = 0; dy <= h; dy++) {
      for (let dz = -w; dz <= w; dz++) {
        for (let dx = -w; dx <= w; dx++) {
          const worldX = x + dx;
          const worldY = y + dy;
          const worldZ = z + dz;
          if (worldX < 0 || worldX >= CHUNK_SIZE || worldZ < 0 || worldZ >= CHUNK_SIZE || worldY < 1 || worldY >= CHUNK_HEIGHT) continue;
          const idx = (worldY * CHUNK_SIZE + worldZ) * CHUNK_SIZE + worldX;
          const edge = Math.abs(dx) === w || Math.abs(dz) === w;
          const roof = dy === h;
          if (roof) {
            blocks[idx] = plank;
          } else if (dy === 0) {
            blocks[idx] = plank;
          } else if (edge) {
            blocks[idx] = wood;
          } else if (dy === 1) {
            blocks[idx] = plank;
          }
        }
      }
    }
    // lantern and a simple leaf patch on top for variety
    const lanternIdx = ((y + 1) * CHUNK_SIZE + z) * CHUNK_SIZE + x;
    blocks[lanternIdx] = jack;
    for (let dz = -1; dz <= 1; dz++) {
      for (let dx = -1; dx <= 1; dx++) {
        const wx = x + dx;
        const wz = z + dz;
        const wy = y + h + 1;
        if (wx < 0 || wx >= CHUNK_SIZE || wz < 0 || wz >= CHUNK_SIZE || wy >= CHUNK_HEIGHT) continue;
        const idx = (wy * CHUNK_SIZE + wz) * CHUNK_SIZE + wx;
        blocks[idx] = leaves;
      }
    }
  }
}

/**
 * Convenience singleton to avoid re-initializing the generator per worker call.
 */
export const globalGenerator = new BiomeGenerator();
