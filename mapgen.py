#std/external libs
import time
import numpy

#local libs
from config import SECTOR_SIZE, SECTOR_HEIGHT, LOADED_SECTORS
from blocks import BLOCK_VERTICES, BLOCK_COLORS, BLOCK_NORMALS, BLOCK_TEXTURES, BLOCK_ID, BLOCK_SOLID, TEXTURE_PATH
import noise
import config

STONE = BLOCK_ID['Stone']
SAND = BLOCK_ID['Sand']
GRASS = BLOCK_ID['Grass']
WOOD = BLOCK_ID['Wood']
LEAVES = BLOCK_ID['Leaves']
PLANK = BLOCK_ID['Plank']
BRICK = BLOCK_ID['Brick']
COBBLE = BLOCK_ID['Cobblestone']
IRON_ORE = BLOCK_ID['Iron Ore']
GOLD_ORE = BLOCK_ID['Gold Ore']
COAL_ORE = BLOCK_ID['Coal Ore']
DIAMOND_ORE = BLOCK_ID['Diamond Ore']
REDSTONE_ORE = BLOCK_ID['Redstone Ore']
EMERALD_ORE = BLOCK_ID['Emerald Ore']
PUMPKIN = BLOCK_ID['Pumpkin']
JACK = BLOCK_ID["Jack O'Lantern"]
TNT = BLOCK_ID['TNT']
CAKE = BLOCK_ID['Cake']
ROSE = BLOCK_ID['Rose']
WATER = BLOCK_ID['Water']

WATER_LEVEL = 70
GLOBAL_WATER_LEVEL = WATER_LEVEL


class SectorNoise2D(object):
    def __init__(self, seed, step = SECTOR_SIZE, step_offset = 0, scale = 1, offset = 0):
        self.noise = noise.SimplexNoise(seed = seed)
        self.seed = seed
        self.step = step
        self.scale = scale
        self.offset = offset
        Z = numpy.mgrid[-1:SECTOR_SIZE+1,-1:SECTOR_SIZE+1].T #overgenerate by one block in each direction
        shape = Z.shape
        self.Z = Z.reshape((shape[0]*shape[1],2))+step_offset

    def __call__(self, position):
        Z = self.Z + numpy.array([position[0],position[2]])
        N=self.noise.noise(Z/self.step)*self.scale + self.offset
        return N.reshape((SECTOR_SIZE+2,SECTOR_SIZE+2))


class SectorNoise3D(object):
    """3D Simplex noise helper with overgeneration to avoid seams."""
    def __init__(self, seed, step=SECTOR_SIZE, step_offset=0, scale=1.0, offset=0.0):
        self.noise = noise.SimplexNoise(seed=seed)
        self.step = step
        self.scale = scale
        self.offset = offset
        self.step_offset = step_offset
        Z = numpy.mgrid[-1:SECTOR_SIZE+1, 0:SECTOR_HEIGHT, -1:SECTOR_SIZE+1].T
        shape = Z.shape
        self.Z = Z.reshape((shape[0]*shape[1]*shape[2], 3))

    def __call__(self, position):
        # position is (sector x,z); y stays absolute
        offset = numpy.array([position[0], 0, position[2]])
        Z = self.Z + offset + self.step_offset
        coords = numpy.mod(Z / self.step, 64.0).astype(numpy.float32)  # keep coordinates small for noise
        N = self.noise.noise(coords) * self.scale + self.offset
        return N.reshape((SECTOR_SIZE+2, SECTOR_HEIGHT, SECTOR_SIZE+2))

def initialize_map_generator(seed = None):
    global noise1, noise2, noise3, noise4
    HILL_STEP = 40.0
    HILL_SCALE = 5
    HILL_OFFSET = 5
    CONTINENTAL_STEP = 1500.0
    CONTINENTAL_SCALE = 40.0
    CONTINENTAL_OFFSET = 80
    GAIN_STEP = 3000.0
    GAIN_SCALE = 5
    GAIN_OFFSET = 5
    if seed == None:
        seed = int(time.time())
    noise1 = SectorNoise2D(seed = seed+12, step = HILL_STEP, step_offset = 30,  
        scale = HILL_SCALE, offset = HILL_OFFSET)
    noise2 = SectorNoise2D(seed = seed+16, step = HILL_STEP, step_offset = 900,
        scale = HILL_SCALE, offset = HILL_OFFSET)
    noise3 = SectorNoise2D(seed = seed+14, step = CONTINENTAL_STEP, step_offset = 531,
        scale = CONTINENTAL_SCALE, offset = CONTINENTAL_OFFSET)
    noise4 = SectorNoise2D(seed = seed+18, step = GAIN_STEP, step_offset = 8123,
        scale = GAIN_SCALE, offset = GAIN_OFFSET)

def generate_sector(position, sector, world):
    """ Initialize the sector by procedurally generating terrain using
    simplex noise.

    """
    N1=noise1(position)
    N2=noise2(position)
    N3=noise3(position)
    N4=noise4(position)

    N1 = N1*N4+N3
    N2 = N2+N3

    b = numpy.zeros((SECTOR_HEIGHT,SECTOR_SIZE + 2,SECTOR_SIZE + 2),dtype='u2')
    for y in range(SECTOR_HEIGHT):
        b[y] = ((y<N1-3)*STONE + (((y>=N1-3) & (y<N1))*GRASS))
        thresh = ((y>N3)*(y<N2)*(y>10))>0
        b[y] = b[y]*(1 - thresh) + SAND * thresh
    return b.swapaxes(0,1).swapaxes(0,2)


# -------- Experimental biome-based generator (keeps the legacy generator intact) --------

class BiomeGenerator:
    """Composable biome generator that mixes multiple noise fields for elevation and variety."""

    def __init__(self, seed=None):
        if seed is None:
            seed = int(time.time())
        self.seed = seed
        self.fast = getattr(config, 'BIOME_FAST_MODE', False)
        # Broad elevation and details.
        self.continents = SectorNoise2D(seed=seed + 101, step=1600.0, step_offset=240,
            scale=55.0, offset=70.0)
        self.hills = SectorNoise2D(seed=seed + 102, step=220.0, step_offset=900,
            scale=12.0, offset=0.0)
        self.ridges = SectorNoise2D(seed=seed + 103, step=380.0, step_offset=1250,
            scale=22.0, offset=0.0)
        self.canyons = SectorNoise2D(seed=seed + 104, step=260.0, step_offset=450,
            scale=1.0, offset=0.0)
        self.terrain_type = SectorNoise2D(seed=seed + 108, step=700.0, step_offset=1337,
            scale=1.0, offset=0.0)
        self.terrain_type_fine = SectorNoise2D(seed=seed + 110, step=180.0, step_offset=733,
            scale=1.0, offset=0.0)
        self.hill_detail = SectorNoise2D(seed=seed + 109, step=90.0, step_offset=1650,
            scale=6.0, offset=0.0)
        self.jagged = SectorNoise2D(seed=seed + 113, step=60.0, step_offset=987,
            scale=10.0, offset=0.0)
        # Climate controls.
        self.moisture = SectorNoise2D(seed=seed + 105, step=520.0, step_offset=700,
            scale=1.0, offset=0.0)
        self.temperature = SectorNoise2D(seed=seed + 106, step=680.0, step_offset=123,
            scale=1.0, offset=0.0)
        # Structure placement.
        self.structure_mask = SectorNoise2D(seed=seed + 107, step=1100.0, step_offset=3000,
            scale=1.0, offset=0.0)
        self.structure_fine = SectorNoise2D(seed=seed + 111, step=210.0, step_offset=4111,
            scale=1.0, offset=0.0)
        self.tree_jitter = SectorNoise2D(seed=seed + 112, step=75.0, step_offset=2511,
            scale=1.0, offset=0.0)
        self.building_cluster = SectorNoise2D(seed=seed + 114, step=950.0, step_offset=5100,
            scale=1.0, offset=0.0)
        self.decor_noise = SectorNoise2D(seed=seed + 150, step=140.0, step_offset=3300,
            scale=1.0, offset=0.0)
        self.decor_detail = SectorNoise2D(seed=seed + 151, step=55.0, step_offset=4300,
            scale=1.0, offset=0.0)
        # Underground detail
        if not self.fast:
            # Caves derived from the difference of two 2D noise fields (faster than full 3D).
            self.cave_height_noise = SectorNoise2D(seed=seed + 120, step=120.0, step_offset=3100, scale=1.0, offset=0.0)
            self.cave_density_noise = SectorNoise2D(seed=seed + 121, step=60.0, step_offset=4100, scale=1.0, offset=0.0)
            self.cave_height_fine = SectorNoise2D(seed=seed + 122, step=45.0, step_offset=5100, scale=0.6, offset=0.0)
            self.cave_band = 0.18  # controls cave thickness
            self.cave_surface_outlet_depth = 3  # how close to surface caves can open
            # Ore detail: height preference and density from 2D noise fields (saves 3D passes).
            self.ore_height_noise = SectorNoise2D(seed=seed + 140, step=160.0, step_offset=5200, scale=1.0, offset=0.0)
            self.ore_density_noise = SectorNoise2D(seed=seed + 141, step=90.0, step_offset=6200, scale=1.0, offset=0.0)
            # Ore settings: band controls clump thickness, max_y controls depth distribution.
            self.ore_settings = [
                {'id': COAL_ORE, 'band': 0.26, 'max_y': 128},
                {'id': IRON_ORE, 'band': 0.21, 'max_y': 96},
                {'id': GOLD_ORE, 'band': 0.17, 'max_y': 60},
                {'id': REDSTONE_ORE, 'band': 0.15, 'max_y': 50},
                {'id': DIAMOND_ORE, 'band': 0.12, 'max_y': 45},
                {'id': EMERALD_ORE, 'band': 0.10, 'max_y': 80},
            ]
        # Prebuilt tree templates to avoid heavy per-block loops.
        self.tree_templates = self._build_tree_templates()

    def _height_field(self, position):
        base = self.continents(position)
        hill = self.hills(position) + self.hill_detail(position) * 0.5
        ridge = self.ridges(position)
        canyon_noise = self.canyons(position)
        terrain = 0.7 * self.terrain_type(position) + 0.3 * self.terrain_type_fine(position)
        jagged = numpy.abs(self.jagged(position))

        flat_mask = terrain < -0.35
        plateau_mask = terrain > 0.35
        hill_mask = (~flat_mask) & (~plateau_mask)

        height = base.copy()
        # Gentle rolling plains.
        height[flat_mask] += hill[flat_mask] * 0.25
        # Hills with ridge variation.
        height[hill_mask] += hill[hill_mask] * 1.2
        height[hill_mask] += numpy.clip(ridge[hill_mask], 0, None) * 1.4
        height[hill_mask] -= numpy.clip(-ridge[hill_mask], 0, None) * 1.0
        # Plateaus with sheer faces and step-like terraces.
        plateau_variation = ridge[plateau_mask] * 0.6 + hill[plateau_mask] * 0.2
        terraced = numpy.round(plateau_variation / 8.0) * 8.0
        height[plateau_mask] += terraced + 18.0
        height[plateau_mask] += numpy.clip(ridge[plateau_mask], 0, None) * 2.0
        # Jagged peaks for mountains/highlands.
        height[hill_mask] += jagged[hill_mask] * 0.8
        height[plateau_mask] += jagged[plateau_mask] * 0.6

        # Flatten very low-gradient areas into level patches to avoid endless gentle slopes.
        gx, gz = numpy.gradient(height)
        grad = numpy.hypot(gx, gz)
        flat_zone = (grad < 0.45) & (~plateau_mask)
        avg = (height + numpy.roll(height, 1, 0) + numpy.roll(height, -1, 0) +
               numpy.roll(height, 1, 1) + numpy.roll(height, -1, 1)) / 5.0
        flattened = numpy.round(avg / 2.0) * 2.0
        height = numpy.where(flat_zone, flattened, height)

        # Canyon carving slices through everything.
        height -= numpy.clip(-canyon_noise - 0.25, 0, None) * 18.0
        height = numpy.clip(height, 4, SECTOR_HEIGHT - 6)
        return height, canyon_noise

    def _biome_masks(self, moisture, temperature, elevation, canyon_noise):
        moist01 = 0.5 + 0.5 * numpy.tanh(moisture)
        temp01 = 0.5 + 0.5 * numpy.tanh(temperature)
        elev01 = numpy.clip((elevation - 48.0) / 80.0, 0.0, 1.0)

        # Base biome indices: 0 plains, 1 forest, 2 desert, 3 highland, 4 canyon.
        biome = numpy.zeros(moist01.shape, dtype=numpy.int8)
        biome[(moist01 > 0.48) & (elev01 < 0.7)] = 1   # forest
        biome[(moist01 < 0.22) & (temp01 > 0.6)] = 2   # desert (rarer)
        biome[(elev01 > 0.75)] = 3                     # highlands (smaller area)
        biome[canyon_noise < -0.35] = 4                # canyon/mesa
        return biome

    def _build_tree_templates(self):
        templates = []
        def build(height, radius):
            trunk = [(0, dy, 0) for dy in range(1, height + 1)]
            canopy = []
            top = height
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    for dy in range(-1, 2):
                        if abs(dx) + abs(dz) + abs(dy) > radius + 1:
                            continue
                        canopy.append((dx, top + dy, dz))
            return trunk, canopy
        templates.append(build(4, 2))
        templates.append(build(5, 2))
        templates.append(build(6, 2))
        return templates

    def _place_tree(self, blocks, x, z, ground_y, structure_value):
        # Pick template based on noise to get variety.
        idx = int(abs(structure_value) * len(self.tree_templates)) % len(self.tree_templates)
        trunk, canopy = self.tree_templates[idx]
        # Keep trees spaced out: skip if near edge or too tall for ceiling.
        max_height = max(t[1] for t in trunk + canopy) + ground_y
        if max_height >= SECTOR_HEIGHT or x < 2 or z < 2 or x >= SECTOR_SIZE - 1 or z >= SECTOR_SIZE - 1:
            return
        # Draw trunk and canopy with bounds checks.
        for dx, dy, dz in trunk:
            cx, cy, cz = x + dx, ground_y + dy, z + dz
            if 0 <= cy < SECTOR_HEIGHT:
                blocks[cy, cx, cz] = WOOD
        for dx, dy, dz in canopy:
            cx, cy, cz = x + dx, ground_y + dy, z + dz
            if 0 <= cy < SECTOR_HEIGHT:
                blocks[cy, cx, cz] = LEAVES

    def _place_cabin(self, blocks, x, z, ground_y):
        # Small 4x4 hut with plank walls, cobble base, and brick chimney.
        width = 4
        height = 4
        if x + width + 1 >= SECTOR_SIZE + 1 or z + width + 1 >= SECTOR_SIZE + 1:
            return
        if ground_y + height + 2 >= SECTOR_HEIGHT:
            return
        floor_y = ground_y + 1
        roof_y = ground_y + height
        for dx in range(width):
            for dz in range(width):
                wx, wz = x + dx, z + dz
                blocks[floor_y, wx, wz] = PLANK
                # walls
                if dx in (0, width - 1) or dz in (0, width - 1):
                    for dy in range(1, height):
                        blocks[floor_y + dy, wx, wz] = PLANK
        # roof and trim
        for dx in range(-1, width + 1):
            for dz in range(-1, width + 1):
                wx, wz = x + dx, z + dz
                blocks[roof_y, wx, wz] = WOOD if (dx in (-1, width) or dz in (-1, width)) else PLANK
        # foundation
        for dx in range(width):
            for dz in range(width):
                blocks[ground_y, x + dx, z + dz] = COBBLE
        # chimney
        chimney_x, chimney_z = x + width // 2, z + width - 1
        for dy in range(2):
            blocks[roof_y + dy, chimney_x, chimney_z] = BRICK

    def _clear_area(self, blocks, elevation, x, z, width, depth, target_height):
        """Flatten terrain under a footprint. If blocks is provided, also set ground/grass."""
        x0, x1 = x, x + width
        z0, z1 = z, z + depth
        for xi in range(x0, x1):
            for zi in range(z0, z1):
                h = int(target_height)
                h = numpy.clip(h, 1, SECTOR_HEIGHT - 3)
                if blocks is not None:
                    col = blocks[:, xi, zi]
                    col[:h - 1] = STONE
                    col[h - 1] = STONE
                    col[h] = GRASS
                elevation[xi, zi] = h

    def _place_rect_building(self, blocks, x, z, ground_y, width, depth, height, wall_block, roof_block, pitched=True, doorway_dir='z+', windows=True):
        """Generic rectangular building with door openings and optional windows."""
        base_y = ground_y + 1
        roof_y = base_y + height
        # Floor and foundation
        for dx in range(width):
            for dz in range(depth):
                wx, wz = x + dx, z + dz
                blocks[ground_y, wx, wz] = COBBLE
                blocks[base_y, wx, wz] = PLANK
        # Walls with doorway and windows
        door_x = x + width // 2
        door_z = z + depth - 1 if doorway_dir == 'z+' else (z if doorway_dir == 'z-' else z + depth // 2)
        for dy in range(1, height):
            wy = base_y + dy
            for dx in range(width):
                for dz in range(depth):
                    wx, wz = x + dx, z + dz
                    at_edge = dx == 0 or dx == width - 1 or dz == 0 or dz == depth - 1
                    if not at_edge:
                        continue
                    # Door opening 2 blocks tall
                    if (wx == door_x and ((doorway_dir == 'z+' and wz == z + depth - 1) or (doorway_dir == 'z-' and wz == z)) and dy <= 2):
                        continue
                    # Windows in middle height
                    mid_band = (dy == max(2, height // 2)) and windows
                    if mid_band and ((dx % (width - 1) == 0 and dz % 2 == 0) or (dz % (depth - 1) == 0 and dx % 2 == 0)):
                        continue
                    blocks[wy, wx, wz] = wall_block
        # Roof
        if pitched:
            left = 0
            right = width - 1
            step = 0
            while left <= right:
                y = roof_y + step
                for dx in range(left, right + 1):
                    for dz in range(depth):
                        wx, wz = x + dx, z + dz
                        blocks[y, wx, wz] = roof_block
                left += 1
                right -= 1
                step += 1
        else:
            for dx in range(width):
                for dz in range(depth):
                    blocks[roof_y, x + dx, z + dz] = roof_block

    def _place_building_by_type(self, blocks, x, z, ground_y, btype, orientation_noise):
        """Dispatch to specific building layouts with variation."""
        orient = 'z+' if orientation_noise > 0 else 'z-'
        if btype == 'cabin':
            self._place_rect_building(blocks, x, z, ground_y, width=4, depth=4, height=3, wall_block=PLANK, roof_block=WOOD, pitched=True, doorway_dir=orient)
        elif btype == 'house':
            self._place_rect_building(blocks, x, z, ground_y, width=6, depth=5, height=4, wall_block=PLANK, roof_block=WOOD, pitched=True, doorway_dir=orient)
        elif btype == 'mansion':
            self._place_rect_building(blocks, x, z, ground_y, width=10, depth=8, height=6, wall_block=BRICK, roof_block=WOOD, pitched=True, doorway_dir=orient)
        elif btype == 'castle':
            w, d, h = 12, 12, 6
            self._place_rect_building(blocks, x, z, ground_y, width=w, depth=d, height=h, wall_block=BRICK, roof_block=BRICK, pitched=False, doorway_dir=orient, windows=False)
            # courtyard hollow
            for dx in range(2, w - 2):
                for dz in range(2, d - 2):
                    wx, wz = x + dx, z + dz
                    for dy in range(1, h):
                        blocks[ground_y + 1 + dy, wx, wz] = 0
            # battlements
            top = ground_y + 1 + h
            for dx in range(w):
                for dz in range(d):
                    if dx in (0, w - 1) or dz in (0, d - 1):
                        if (dx + dz) % 2 == 0:
                            blocks[top, x + dx, z + dz] = BRICK
        elif btype == 'church':
            self._place_rect_building(blocks, x, z, ground_y, width=7, depth=11, height=5, wall_block=PLANK, roof_block=WOOD, pitched=True, doorway_dir=orient)
            # tower on one end
            tower_x = x + 2
            tower_z = z + (0 if orient == 'z-' else 7)
            self._place_rect_building(blocks, tower_x, tower_z, ground_y, width=3, depth=3, height=7, wall_block=BRICK, roof_block=WOOD, pitched=False, doorway_dir=orient, windows=True)

    def _place_boulder(self, blocks, x, z, ground_y, radius=2):
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx * dx + dz * dz > radius * radius + 1:
                    continue
                wx, wz = x + dx, z + dz
                if 1 <= wx < SECTOR_SIZE + 1 and 1 <= wz < SECTOR_SIZE + 1:
                    h = ground_y + max(0, 1 - abs(dx) - abs(dz))
                    if h < SECTOR_HEIGHT:
                        blocks[h, wx, wz] = COBBLE if (dx + dz) % 2 else STONE

    def _reserve_area(self, taken_mask, x, z, width, depth, padding=1):
        x0 = max(0, x - padding)
        z0 = max(0, z - padding)
        x1 = min(SECTOR_SIZE + 2, x + width + padding)
        z1 = min(SECTOR_SIZE + 2, z + depth + padding)
        if taken_mask[x0:x1, z0:z1].any():
            return False
        taken_mask[x0:x1, z0:z1] = True
        return True

    def _draw_path(self, blocks, p0, p1):
        """Simple L-shaped cobble path connecting two building centers."""
        (x0, y0, z0) = p0
        (x1, y1, z1) = p1

        def lay_line(xs, zs, ys):
            for xx, zz, yy in zip(xs, zs, ys):
                h = int(yy)
                h = numpy.clip(h, 1, SECTOR_HEIGHT - 2)
                blocks[h, xx, zz] = COBBLE
                blocks[h - 1, xx, zz] = STONE

        if x0 != x1:
            xs = range(min(x0, x1), max(x0, x1) + 1)
            zs = [z0] * len(xs)
            ys = [y0] * len(xs)
            lay_line(xs, zs, ys)
        if z0 != z1:
            zs = range(min(z0, z1), max(z0, z1) + 1)
            xs = [x1] * len(zs)
            ys = [y1] * len(zs)
            lay_line(xs, zs, ys)

    def _carve_caves(self, position, blocks, elevation, cliff_mask):
        """Carve caves whose roofs sit between bedrock+20 and the terrain surface."""
        if not cliff_mask.any():
            return

        # Height selection: absolute roof height drawn from noise; discard columns where the target roof would poke above ground.
        height_pref = self.cave_height_noise(position)             # (X,Z)
        height_fine = self.cave_height_fine(position)              # (X,Z)
        height01 = numpy.clip(0.5 + 0.5 * height_pref + 0.25 * height_fine, 0.0, 1.0)
        min_roof = 20.0
        max_roof = SECTOR_HEIGHT - 6.0
        ground_cap = numpy.clip(elevation, min_roof, max_roof)
        # Map roof height into [min_roof, ground height]; if ground is below min_roof we skip that column.
        target_roof = min_roof + height01 * (ground_cap - min_roof)

        roof_limit = ground_cap
        has_space = target_roof <= roof_limit  # skip slices where the noise would rise above ground
        if not has_space.any():
            return
        # roof = numpy.where(has_space, numpy.minimum(target_roof, roof_limit), 0.0)

        roof_offset = 18.0 * height01  # ~6..16 blocks below surface
        roof = numpy.clip(ground_cap - roof_offset, min_roof, max_roof)


        density2d = 0.5 + 0.5 * self.cave_density_noise(position)  # 0..1 (X,Z)
        # Smooth density laterally to encourage wider, connected patches.
        density2d = (density2d +
                     numpy.roll(density2d, 1, 0) + numpy.roll(density2d, -1, 0) +
                     numpy.roll(density2d, 1, 1) + numpy.roll(density2d, -1, 1)) / 5.0
        depth = 4.0 + 12.0 * density2d                             # 4..16-ish
        floor = numpy.maximum(min_roof - 1.0, roof - depth)

        y_grid = numpy.arange(SECTOR_HEIGHT, dtype=float)[:, None, None]  # (H,1,1)
        elev_grid = elevation[None, :, :]                                 # (1,X,Z)
        cliff3d = cliff_mask[None, :, :]                                  # (1,X,Z)
        land3d = (elevation > WATER_LEVEL)[None, :, :]                    # avoid flooding seafloor caves
        area_mask = cliff3d | (density2d[None, :, :] > 0.6)               # allow some non-cliff caves where dense

        vertical_mask = (y_grid >= floor[None, :, :]) & (y_grid <= roof[None, :, :])
        below_surface = y_grid < (elev_grid - 2.0)
        density_gate = density2d > 0.35
        column_gate = has_space[None, :, :] & land3d

        carve = vertical_mask & below_surface & density_gate[None, :, :] & column_gate & area_mask

        # Occasional surface outlets only when the roof is comfortably below the surface.
        outlet_band = (y_grid >= (elev_grid - self.cave_surface_outlet_depth)) & (y_grid <= (elev_grid - 1.0))
        deep_roof = (elevation - roof) > 2.5
        outlet_gate = density2d > 0.7
        carve |= outlet_band & outlet_gate[None, :, :] & column_gate & cliff3d & deep_roof[None, :, :]

        # Small lateral dilation to connect nearby passages while respecting height masks.
        for _ in range(2):
            neighbor = numpy.zeros_like(carve, dtype=bool)
            neighbor[:, 1:, :] |= carve[:, :-1, :]
            neighbor[:, :-1, :] |= carve[:, 1:, :]
            neighbor[:, :, 1:] |= carve[:, :, :-1]
            neighbor[:, :, :-1] |= carve[:, :, 1:]
            spread = neighbor & vertical_mask & below_surface & density_gate[None, :, :] & column_gate & area_mask
            carve |= spread

        # print("carve shape", carve.shape)  # expect (H, X, Z)
        # print("carve voxels", int(carve.sum()))
        # print("carve columns", int(carve.any(axis=0).sum()))  # number of XZ columns that got any cave voxels
        # print("cliff columns", int(cliff_mask.sum()))
        # print("land columns", int((elevation > WATER_LEVEL).sum()))
        # print("has_space columns", int(has_space.sum()))
        # print("density>gate columns", int((density2d > 0.52).sum()))
        # carved_cols = carve.any(axis=0)
        # roof_c = roof[carved_cols]
        # floor_c = floor[carved_cols]
        # if carved_cols.any():
        #     print(
        #         "CARVED roof min/max/mean",
        #         float(roof_c.min()), float(roof_c.max()), float(roof_c.mean()),
        #         "floor min/max/mean",
        #         float(floor_c.min()), float(floor_c.max()), float(floor_c.mean()),
        #     )
        # total_voxels = carve.size
        # print("carve fraction", carve.sum() / total_voxels)
        blocks[carve] = 0

    def _place_ores(self, position, blocks, elevation, cliff_mask):
        """Sprinkle ore clumps in stone within cliffy areas using band-pass 3D noise."""
        if not cliff_mask.any():
            return
        y_grid = numpy.arange(SECTOR_HEIGHT)[:, None, None]  # (H,1,1)
        stone_mask = blocks == STONE                         # (H,X,Z)
        cliff3d = cliff_mask[None, :, :]                     # (1,X,Z)
        height_pref = self.ore_height_noise(position)        # (X,Z)
        density2d = 0.5 + 0.5 * self.ore_density_noise(position)  # (X,Z) in 0..1
        # ground heights to bias ore vertically
        ground = numpy.clip(elevation, 2, SECTOR_HEIGHT - 3)
        for setting in self.ore_settings:
            roof_offset = 3 + 6 * (0.5 + 0.5 * height_pref)  # 3..9 below surface
            roof = ground - roof_offset
            roof = numpy.clip(roof, 2, setting['max_y'] - 1)
            above_ground = roof >= ground - 1

            depth = 0.5 + 2.5 * density2d  # up to ~3
            shallow = depth < 0.8
            floor = numpy.clip(roof - depth, 1, SECTOR_HEIGHT - 2)

            vertical_mask = (y_grid <= roof[None, :, :]) & (y_grid >= floor[None, :, :])
            depth_mask = y_grid < setting['max_y']
            density_gate = density2d > 0.85
            density3d = density_gate[None, :, :]

            mask = vertical_mask & depth_mask & stone_mask & cliff3d & density3d & (~above_ground) & (~shallow[None, :, :])
            blocks[mask] = setting['id']

    def generate(self, position):
        elevation, canyon_noise = self._height_field(position)
        moisture = self.moisture(position)
        temperature = self.temperature(position)
        moist01 = 0.5 + 0.5 * numpy.tanh(moisture)
        temp01 = 0.5 + 0.5 * numpy.tanh(temperature)
        structure = self.structure_mask(position)
        structure_fine = self.structure_fine(position)
        tree_jitter = self.tree_jitter(position)
        building_cluster = self.building_cluster(position)
        if self.fast:
            # Reuse an existing layer for building clustering to save a perlin call.
            building_cluster = structure
        decor_noise = self.decor_noise(position)
        decor_detail = self.decor_detail(position)

        biome = self._biome_masks(moisture, temperature, elevation, canyon_noise)
        soil_depth = numpy.full_like(elevation, 3, dtype=numpy.int8)
        surface_block = numpy.full_like(elevation, GRASS, dtype=numpy.int16)
        filler_block = numpy.full_like(elevation, STONE, dtype=numpy.int16)

        desert_mask = biome == 2
        forest_mask = biome == 1
        highland_mask = biome == 3
        canyon_mask = biome == 4
        water_biome = elevation <= WATER_LEVEL
        biome[water_biome] = 5

        surface_block[desert_mask] = SAND
        filler_block[desert_mask] = SAND
        soil_depth[desert_mask] = 4

        surface_block[canyon_mask] = STONE
        filler_block[canyon_mask] = STONE
        soil_depth[canyon_mask] = 2

        surface_block[highland_mask] = STONE
        soil_depth[highland_mask] = 2

        # Identify cliffy regions (steep gradients).
        gx, gz = numpy.gradient(elevation)
        grad = numpy.hypot(gx, gz)
        cliff_mask = grad > 0.85

        # Flat global water level to avoid seams.
        water_level_i = numpy.full_like(elevation, WATER_LEVEL, dtype=int)

        # Plan buildings on mostly flat plains; gate density with multiple noises.
        building_spots = []
        building_mask = numpy.zeros_like(biome, dtype=bool)
        max_buildings = 2 if self.fast else 4
        step = 12 if self.fast else 8
        for x in range(2, SECTOR_SIZE - 6, step):
            for z in range(2, SECTOR_SIZE - 6, step):
                if len(building_spots) >= max_buildings:
                    break
                if elevation[x, z] < WATER_LEVEL:
                    continue
                if biome[x, z] != 0:
                    continue
                if building_cluster[x, z] < 0.55 or structure[x, z] < 0.65 or structure_fine[x, z] < 0.1:
                    continue
                # Pick type based on noise bands.
                val = structure[x, z] + 0.5 * structure_fine[x, z]
                if val > 1.25:
                    btype, size = 'castle', (12, 12)
                elif val > 1.05:
                    btype, size = 'mansion', (10, 8)
                elif val > 0.9:
                    btype, size = 'house', (6, 5)
                elif val > 0.8:
                    btype, size = 'church', (7, 11)
                else:
                    btype, size = 'cabin', (4, 4)
                w, d = size
                # Ensure space and flatten.
                target_h = int(round(numpy.mean(elevation[max(x-1,0):min(x+w+1, SECTOR_SIZE+2), max(z-1,0):min(z+d+1, SECTOR_SIZE+2)])))
                if target_h + 10 >= SECTOR_HEIGHT:
                    continue
                if not self._reserve_area(building_mask, x, z, w, d, padding=2):
                    continue
                self._clear_area(blocks=None, elevation=elevation, x=x, z=z, width=w, depth=d, target_height=target_h)
                building_spots.append((x, z, target_h, btype, size, building_cluster[x, z]))

        sx = SECTOR_SIZE + 2
        sz = SECTOR_SIZE + 2
        y_grid = numpy.arange(SECTOR_HEIGHT)[:, None, None]
        ground_h = numpy.clip(elevation, 2, SECTOR_HEIGHT - 3).astype(int)
        depth = soil_depth.astype(int)
        stone_cap = numpy.maximum(1, ground_h - depth)
        # Fill all columns below the global water level; building placement already avoids these.
        water_mask = (ground_h <= water_level_i)
        if water_mask.any():
            sandy = (~highland_mask) & (~canyon_mask)
            surface_block = numpy.where(water_mask & sandy, SAND, surface_block)
            filler_block = numpy.where(water_mask & sandy, SAND, filler_block)
            soil_depth = numpy.where(water_mask & sandy, numpy.maximum(soil_depth, 4), soil_depth)
            depth = soil_depth.astype(int)
            stone_cap = numpy.maximum(1, ground_h - depth)

        blocks = numpy.zeros((SECTOR_HEIGHT, sx, sz), dtype='u2')
        # vectorized bulk fill
        blocks[y_grid < stone_cap[None, :, :]] = STONE
        mid_mask = (y_grid >= stone_cap[None, :, :]) & (y_grid < ground_h[None, :, :])
        filler_broadcast = numpy.broadcast_to(filler_block, (SECTOR_HEIGHT, sx, sz))
        blocks[mid_mask] = filler_broadcast[mid_mask]
        xs, zs = numpy.meshgrid(numpy.arange(sx), numpy.arange(sz), indexing='ij')
        blocks[ground_h, xs, zs] = surface_block
        # Underwater columns: sandier bottoms and water fill up to water line.
        if water_mask.any():
            wl = numpy.clip(water_level_i, 1, SECTOR_HEIGHT - 1)
            water_fill = (y_grid > ground_h[None, :, :]) & (y_grid <= wl[None, :, :]) & water_mask[None, :, :]
            blocks[water_fill] = WATER

        # Per-column decorations that need decisions.
        for x in range(sx):
            for z in range(sz):
                ground = ground_h[x, z]
                if water_mask[x, z]:
                    continue
                # Forest canopy: sparse, jittered placement for walkability.
                if forest_mask[x, z] and not building_mask[x, z]:
                    allow_tree = (structure[x, z] > 0.35 if self.fast else structure[x, z] > 0.32) and tree_jitter[x, z] > 0.05
                    stride = 6 if self.fast else 5
                    coarse_gate = ((x + z + int(tree_jitter[x, z] * 7)) % stride == 0)
                    if allow_tree and coarse_gate:
                        self._place_tree(blocks, x, z, ground, structure[x, z] + tree_jitter[x, z])
                # Sparse shrubs in canyons
                if canyon_mask[x, z] and structure[x, z] > 0.45 and ground + 1 < SECTOR_HEIGHT:
                    blocks[ground + 1, x, z] = LEAVES
                # Decorations (skip buildings and cliffs near steep drops)
                if building_mask[x, z]:
                    continue
                dn = decor_noise[x, z]
                dd = decor_detail[x, z]
                dense_hit = (dn > 0.7) and (dd > 0.6)
                rare_hit = (dn < -0.7) and (dd < -0.6)
                if dense_hit and not cliff_mask[x, z]:
                    rad = 1 + int(abs(dn + dd) * 1.2)
                    self._place_boulder(blocks, x, z, ground, radius=min(2, rad))
                elif dense_hit and biome[x, z] in (0, 1) and ground + 1 < SECTOR_HEIGHT:
                    blocks[ground + 1, x, z] = ROSE
                elif rare_hit and biome[x, z] == 0 and ground + 1 < SECTOR_HEIGHT and (x + z) % 15 == 0:
                    blocks[ground + 1, x, z] = PUMPKIN
                elif rare_hit and biome[x, z] == 0 and ground + 1 < SECTOR_HEIGHT and (x + 2 * z) % 19 == 0:
                    blocks[ground + 1, x, z] = JACK
                elif dense_hit and biome[x, z] == 0 and ground + 1 < SECTOR_HEIGHT and (x + 3 * z) % 21 == 0:
                    blocks[ground + 1, x, z] = TNT if not self.fast else COBBLE
                elif dense_hit and biome[x, z] == 0 and ground + 1 < SECTOR_HEIGHT and (2 * x + z) % 23 == 0:
                    blocks[ground + 1, x, z] = CAKE

        # Ores and caves after terrain fill, limited to cliff regions.
        if not self.fast:
            self._place_ores(position, blocks, elevation, cliff_mask)
            self._carve_caves(position, blocks, elevation, cliff_mask)

        # Place planned buildings and connect them with paths.
        centers = []
        for x, z, gh, btype, size, orient_val in building_spots:
            w, d = size
            self._clear_area(blocks, elevation, x, z, w, d, gh)
            self._place_building_by_type(blocks, x, z, gh, btype, orient_val)
            centers.append((x + w // 2, gh + 1, z + d // 2))
        for i in range(1, len(centers)):
            self._draw_path(blocks, centers[i - 1], centers[i])

        # Final water seal: ensure any air pockets below water level are filled.
        y_grid = numpy.arange(SECTOR_HEIGHT)[:, None, None]
        water_cols = (ground_h <= water_level_i)
        if water_cols.any():
            wl = numpy.clip(water_level_i, 1, SECTOR_HEIGHT - 1)
            below_surface = (y_grid <= wl[None, :, :]) & water_cols[None, :, :]
            air_pockets = below_surface & (blocks == 0)
            blocks[air_pockets] = WATER
            # Let water flow sideways into shallow cave openings at/below sea level.
            water_mask = blocks == WATER
            air_mask = (blocks == 0) & (y_grid <= wl[None, :, :])
            for _ in range(4):  # limited iterations to avoid flooding far inland
                neighbor = numpy.zeros_like(water_mask, dtype=bool)
                neighbor[:-1, :, :] |= water_mask[1:, :, :]
                neighbor[1:, :, :] |= water_mask[:-1, :, :]
                neighbor[:, :-1, :] |= water_mask[:, 1:, :]
                neighbor[:, 1:, :] |= water_mask[:, :-1, :]
                neighbor[:, :, :-1] |= water_mask[:, :, 1:]
                neighbor[:, :, 1:] |= water_mask[:, :, :-1]
                spread = air_mask & neighbor
                if not spread.any():
                    break
                blocks[spread] = WATER
                water_mask |= spread
                air_mask &= ~spread

        return blocks.swapaxes(0,1).swapaxes(0,2)


biome_generator = None

def initialize_biome_map_generator(seed=None):
    global biome_generator
    biome_generator = BiomeGenerator(seed=seed)


def generate_biome_sector(position, sector, world):
    """Experimental biome-based terrain generator with varied landscapes and structures."""
    global biome_generator
    if biome_generator is None:
        initialize_biome_map_generator()
    return biome_generator.generate(position)
