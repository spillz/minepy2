/**
 * Block definitions mirroring the Python edition.
 * @module blocks
 */

/**
 * Shape identifiers used by the renderer.
 * @enum {number}
 */
export const BlockShape = {
  /** Standard cube. */
  Cube: 0,
};

/**
 * Create a block descriptor.
 * @param {string} name
 * @param {[number, number][]} coords - Texture tile coordinates (top, bottom, side in atlas units).
 * @param {boolean} solid
 * @param {BlockShape} shape
 * @returns {{id:number,name:string,coords:[number,number][],solid:boolean,shape:BlockShape}}
 */
function block(name, coords, solid = true, shape = BlockShape.Cube) {
  return { id: 0, name, coords, solid, shape };
}

const BLOCK_DESCRIPTORS = [
  block('Grass', [[0, 15], [2, 15], [3, 15]]),
  block('Leaves', [[4, 7], [4, 7], [4, 7]], false),
  block('Sand', [[2, 14], [2, 14], [2, 14]]),
  block('Brick', [[7, 15], [7, 15], [7, 15]]),
  block('Stone', [[1, 15], [1, 15], [1, 15]]),
  block('Cobblestone', [[0, 14], [0, 14], [0, 14]]),
  block('Iron Block', [[6, 14], [6, 14], [6, 14]]),
  block('Wood', [[5, 14], [5, 14], [4, 14]]),
  block('Plank', [[4, 15], [4, 15], [4, 15]]),
  block('Crafting Table', [[11, 13], [4, 15], [11, 12]]),
  block('Pumpkin', [[6, 9], [6, 8], [7, 8]]),
  block("Jack O'Lantern", [[6, 9], [6, 8], [8, 8]]),
  block('Rose', [[12, 15], [12, 15], [12, 15]], false),
  block('Gobbledeblock', [[9, 14], [13, 1], [13, 8]]),
  block('Iron Ore', [[1, 13], [1, 13], [1, 13]]),
  block('Gold Ore', [[0, 13], [0, 13], [0, 13]]),
  block('Coal Ore', [[2, 13], [2, 13], [2, 13]]),
  block('Diamond Ore', [[2, 12], [2, 12], [2, 12]]),
  block('Redstone Ore', [[3, 12], [3, 12], [3, 12]]),
  block('Emerald Ore', [[11, 5], [11, 5], [11, 5]]),
  block('Bookshelf', [[4, 15], [4, 15], [3, 13]]),
  block('TNT', [[9, 15], [10, 15], [8, 15]]),
  block('Cake', [[9, 8], [12, 8], [10, 8]], false),
];

// Assign stable IDs.
for (let i = 0; i < BLOCK_DESCRIPTORS.length; i++) {
  BLOCK_DESCRIPTORS[i].id = i + 1;
}

/**
 * Number of tiles along one axis of the atlas.
 */
export const ATLAS_TILES = 16;

/**
 * Map block name to ID.
 * @type {Record<string, number>}
 */
export const BLOCK_ID = Object.fromEntries(BLOCK_DESCRIPTORS.map((b) => [b.name, b.id]));

/**
 * Ordered block descriptors.
 */
export const BLOCKS = BLOCK_DESCRIPTORS;

/**
 * Solid flags indexed by block id (0 is air).
 */
export const BLOCK_SOLID = (() => {
  const flags = new Uint8Array(BLOCK_DESCRIPTORS.length + 1);
  for (const b of BLOCK_DESCRIPTORS) {
    flags[b.id] = b.solid ? 1 : 0;
  }
  return flags;
})();

/**
 * Face texture lookup: [blockId][faceIndex] => vec2 tile origin.
 * Face order: top, bottom, left, right, front, back (matching util.py faces).
 */
export const BLOCK_TEXTURES = (() => {
  const faces = new Float32Array((BLOCK_DESCRIPTORS.length + 1) * 6 * 2);
  const tileSize = 1 / ATLAS_TILES;
  for (const b of BLOCK_DESCRIPTORS) {
    const [top, bottom, side] = b.coords;
    const coords = [top, bottom, side, side, side, side];
    for (let f = 0; f < 6; f++) {
      const idx = ((b.id * 6 + f) * 2);
      faces[idx + 0] = coords[f][0] * tileSize;
      faces[idx + 1] = coords[f][1] * tileSize;
    }
  }
  return faces;
})();

/**
 * Palette colors for minimap/lighting fallback (RGB normalized).
 */
export const BLOCK_COLORS = (() => {
  const colors = new Float32Array((BLOCK_DESCRIPTORS.length + 1) * 3);
  const defaults = [
    [0.76, 0.69, 0.5],
    [0.18, 0.35, 0.18],
    [0.86, 0.79, 0.64],
    [0.64, 0.54, 0.49],
    [0.65, 0.65, 0.65],
    [0.5, 0.5, 0.5],
    [0.83, 0.83, 0.83],
    [0.51, 0.32, 0.19],
    [0.76, 0.62, 0.42],
    [0.67, 0.55, 0.39],
    [0.93, 0.56, 0.13],
    [1.0, 0.79, 0.33],
    [0.9, 0.14, 0.28],
    [0.8, 0.58, 0.16],
    [0.53, 0.5, 0.48],
    [0.87, 0.72, 0.3],
    [0.23, 0.23, 0.23],
    [0.25, 0.94, 0.93],
    [0.74, 0.12, 0.12],
    [0.09, 0.68, 0.37],
    [0.66, 0.52, 0.36],
    [0.88, 0.15, 0.15],
    [1.0, 1.0, 1.0],
  ];
  for (const b of BLOCK_DESCRIPTORS) {
    const c = defaults[b.id - 1] || [0.7, 0.7, 0.7];
    colors[b.id * 3 + 0] = c[0];
    colors[b.id * 3 + 1] = c[1];
    colors[b.id * 3 + 2] = c[2];
  }
  return colors;
})();

export const AIR_BLOCK = 0;
