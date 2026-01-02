#std/external libs
import time
import math
import numpy

#local libs
from config import SECTOR_SIZE, SECTOR_HEIGHT, LOADED_SECTORS
from blocks import BLOCK_VERTICES, BLOCK_COLORS, BLOCK_NORMALS, BLOCK_TEXTURES, BLOCK_ID, BLOCK_SOLID, TEXTURE_PATH
import noise
import config
import logutil

STONE = BLOCK_ID['Stone']
SAND = BLOCK_ID['Sand']
GRASS = BLOCK_ID['Grass']
DIRT = BLOCK_ID['Dirt']
WOOD = BLOCK_ID['Wood']
LEAVES = BLOCK_ID['Leaves']
PLANK = BLOCK_ID['Plank']
BRICK = BLOCK_ID['Brick']
COBBLE = BLOCK_ID['Cobblestone']
BETTERSTONE = BLOCK_ID['BetterStone']
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
MUSHROOM = BLOCK_ID['Mushroom']

LADDER_SOUTH = BLOCK_ID['Ladder South']
LADDER_WEST = BLOCK_ID['Ladder West']
LADDER_NORTH = BLOCK_ID['Ladder North']
LADDER_EAST = BLOCK_ID['Ladder East']

WATER_LEVEL = 70
GLOBAL_WATER_LEVEL = WATER_LEVEL
# Debug tracker for nearest placed mushroom.
_debug_mushroom_hint = {'best': None, 'best_dist2': None}


def _record_mushroom_hint(world_pos, sector_pos, local_pos):
    """Track and print the closest mushroom position found so far."""
    global _debug_mushroom_hint
    x, _, z = world_pos
    dist2 = x * x + z * z
    best = _debug_mushroom_hint.get('best')
    best_d2 = _debug_mushroom_hint.get('best_dist2')
    if best is None or dist2 < best_d2:
        _debug_mushroom_hint['best'] = world_pos
        _debug_mushroom_hint['best_dist2'] = dist2
        horiz = math.sqrt(dist2)
        logutil.log(
            "MAPGEN",
            f"mushroom placed world {world_pos} sector {sector_pos} local {local_pos} horiz~{horiz:.1f}",
            level="DEBUG",
        )


class SectorNoise2D(object):
    def __init__(self, seed, step = SECTOR_SIZE, step_offset = 0, scale = 1, offset = 0):
        self.noise = noise.SimplexNoise(seed = seed)
        self.seed = seed
        self.step = step
        self.scale = scale
        self.offset = offset
        xs, zs = numpy.meshgrid(
            numpy.arange(SECTOR_SIZE, dtype=numpy.float32),
            numpy.arange(SECTOR_SIZE, dtype=numpy.float32),
            indexing='ij',
        )
        self.Z = numpy.stack([xs, zs], axis=-1).reshape((-1, 2)) + step_offset

    def __call__(self, position):
        Z = self.Z + numpy.array([position[0],position[2]])
        N=self.noise.noise(Z/self.step)*self.scale + self.offset
        return N.reshape((SECTOR_SIZE, SECTOR_SIZE))


class SectorNoise3D(object):
    """3D Simplex noise helper with overgeneration to avoid seams."""
    def __init__(self, seed, step=SECTOR_SIZE, step_offset=0, scale=1.0, offset=0.0):
        self.noise = noise.SimplexNoise(seed=seed)
        self.step = step
        self.scale = scale
        self.offset = offset
        self.step_offset = step_offset
        Z = numpy.mgrid[0:SECTOR_SIZE, 0:SECTOR_HEIGHT, 0:SECTOR_SIZE].T
        shape = Z.shape
        self.Z = Z.reshape((shape[0]*shape[1]*shape[2], 3))

    def __call__(self, position):
        # position is (sector x,z); y stays absolute
        offset = numpy.array([position[0], 0, position[2]])
        Z = self.Z + offset + self.step_offset
        coords = numpy.mod(Z / self.step, 64.0).astype(numpy.float32)  # keep coordinates small for noise
        N = self.noise.noise(coords) * self.scale + self.offset
        return N.reshape((SECTOR_SIZE, SECTOR_HEIGHT, SECTOR_SIZE))

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

    b = numpy.zeros((SECTOR_HEIGHT, SECTOR_SIZE, SECTOR_SIZE), dtype='u2')
    for y in range(SECTOR_HEIGHT):
        b[y] = ((y<N1-3)*STONE + (((y>=N1-3) & (y<N1))*GRASS))
        thresh = ((y>N3)*(y<N2)*(y>10))>0
        b[y] = b[y]*(1 - thresh) + SAND * thresh
    return b.swapaxes(0,1)

# -------- Experimental road generation helpers -----------

import numpy as np
import math

ROAD_BLOCK = COBBLE
HIWAY_BLOCK = BETTERSTONE
TRAIL_BLOCK = DIRT
TRAIL_SUPPORT_BLOCK = PLANK




# -------- Experimental biome-based generator (keeps the legacy generator intact) --------

class BiomeGenerator:
    """Composable biome generator that mixes multiple noise fields for elevation and variety."""

    def __init__(self, seed=None):
        if seed is None:
            seed = int(time.time())
        self.seed = seed
        self.fast = getattr(config, 'BIOME_FAST_MODE', False)
        self.spawn_point = tuple(getattr(config, 'SPAWN_POINT', (0.0, 0.0)))
        self.spawn_bias_radius = float(getattr(config, 'SPAWN_BIAS_RADIUS', 40.0))
        self.spawn_bias_boost = float(getattr(config, 'SPAWN_BIAS_BOOST', 10.0))
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
        # Macro feature controls.
        self.enable_rivers = getattr(config, 'ENABLE_RIVER_NETWORKS', False)
        self.enable_roads = getattr(config, 'ENABLE_ROAD_NETWORKS', True)
        self.river_spacing = float(getattr(config, 'RIVER_NETWORK_SPACING', 320))
        self.road_spacing = float(getattr(config, 'ROAD_NETWORK_SPACING', 220))
        self.road_grade_limit = float(getattr(config, 'ROAD_MAX_GRADE', 1.25))
        # Macro path helpers use standalone simplex noise for global lookups.
        self.river_height_field = noise.SimplexNoise(seed=seed + 200)
        self.river_vec_u = noise.SimplexNoise(seed=seed + 201)
        self.river_vec_v = noise.SimplexNoise(seed=seed + 202)
        self.river_width_noise = noise.SimplexNoise(seed=seed + 203)
        self.road_vec_u = noise.SimplexNoise(seed=seed + 204)
        self.road_vec_v = noise.SimplexNoise(seed=seed + 205)
        self.road_height_noise = noise.SimplexNoise(seed=seed + 206)
        self._macro_path_cache = {'river': {}, 'road': {}}
        self.river_step = 12.0
        self.road_step = 7.0
        self.river_max_steps = 1400
        self.road_max_steps = 1000
        # Underground detail
        if not self.fast:
            # Caves derived from the difference of two 2D noise fields (faster than full 3D).
            self.cave_height_noise = SectorNoise2D(seed=seed + 120, step=120.0, step_offset=3100, scale=1.0, offset=0.0)
            self.cave_density_noise = SectorNoise2D(seed=seed + 121, step=60.0, step_offset=4100, scale=1.0, offset=0.0)
            self.cave_height_fine = SectorNoise2D(seed=seed + 122, step=45.0, step_offset=5100, scale=0.6, offset=0.0)
            self.cave_region_noise = SectorNoise2D(seed=seed + 123, step=520.0, step_offset=7100, scale=1.0, offset=0.0)
            self.cave_region_detail = SectorNoise2D(seed=seed + 124, step=180.0, step_offset=8100, scale=0.7, offset=0.0)
            self.cave_level_noise = SectorNoise2D(seed=seed + 125, step=160.0, step_offset=9100, scale=1.0, offset=0.0)
            self.cave_depth_noise = SectorNoise2D(seed=seed + 126, step=90.0, step_offset=10100, scale=1.0, offset=0.0)
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
        # Jagged peaks for mountains/highlands.
        height[hill_mask] += jagged[hill_mask] * 0.8
        height_no_plateau = height.copy()

        # Road height: extend the flat biome formula everywhere to avoid
        # hill/flat boundary cliffs in the road base.
        road_height = base + hill * 0.25

        # Plateaus with sheer faces and step-like terraces.
        plateau_variation = ridge[plateau_mask] * 0.6 + hill[plateau_mask] * 0.2
        terraced = numpy.round(plateau_variation / 8.0) * 8.0
        height[plateau_mask] += terraced + 18.0
        height[plateau_mask] += numpy.clip(ridge[plateau_mask], 0, None) * 2.0
        height[plateau_mask] += jagged[plateau_mask] * 0.6

        # Flatten very low-gradient areas into level patches to avoid endless gentle slopes.
        def _flatten_low_grad(height_in):
            gx, gz = numpy.gradient(height_in)
            grad = numpy.hypot(gx, gz)
            flat_zone = (grad < 0.45) & (~plateau_mask)
            # Avoid wraparound smoothing that can introduce seam artifacts.
            pad = numpy.pad(height_in, 1, mode="edge")
            avg = (
                pad[1:-1, 1:-1]
                + pad[:-2, 1:-1]
                + pad[2:, 1:-1]
                + pad[1:-1, :-2]
                + pad[1:-1, 2:]
            ) / 5.0
            flattened = numpy.round(avg / 2.0) * 2.0
            return numpy.where(flat_zone, flattened, height_in)

        height = _flatten_low_grad(height)
        height_no_plateau = _flatten_low_grad(height_no_plateau)
        road_height = _flatten_low_grad(road_height)
        road_height = numpy.minimum(road_height, height)

        # Canyon carving slices through terrain (roads use pre-canyon height for continuity).
        carve = numpy.clip(-canyon_noise - 0.25, 0, None) * 18.0
        height -= carve
        height = numpy.clip(height, 4, SECTOR_HEIGHT - 6)
        road_height = numpy.clip(road_height, 4, SECTOR_HEIGHT - 6)
        return height, canyon_noise, road_height

    def _apply_spawn_bias(self, elevation, position):
        """Raise terrain near the spawn point so new maps don't start over water."""
        if self.spawn_bias_radius <= 0.0 or self.spawn_bias_boost <= 0.0:
            return elevation
        xs, zs = self._world_axes(position)
        dx = xs[:, None] - self.spawn_point[0]
        dz = zs[None, :] - self.spawn_point[1]
        dist = numpy.hypot(dx, dz)
        factor = numpy.clip(1.0 - dist / self.spawn_bias_radius, 0.0, 1.0)
        min_height = float(WATER_LEVEL) + self.spawn_bias_boost * factor
        return numpy.maximum(elevation, min_height)

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

    # ----- Macro feature helpers -----

    def _sector_origin(self, position):
        if len(position) >= 3:
            return int(position[0]), int(position[2])
        if len(position) == 2:
            return int(position[0]), int(position[1])
        return int(position[0]), 0

    def _world_axes(self, position):
        sx = SECTOR_SIZE
        sz = SECTOR_SIZE
        ox, oz = self._sector_origin(position)
        xs = numpy.arange(sx, dtype=float) + ox
        zs = numpy.arange(sz, dtype=float) + oz
        return xs, zs

    def _macro_random(self, gx, gz, salt):
        # Splitmix64-style integer hash for deterministic floats in [0,1).
        h = (int(gx) * 0x632BE59BD9B4E019) ^ (int(gz) * 0x9E3779B97F4A7C15) ^ (salt * 0x94D049BB133111EB) ^ int(self.seed)
        h &= (1 << 64) - 1
        h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9 & ((1 << 64) - 1)
        h = (h ^ (h >> 27)) * 0x94d049bb133111eb & ((1 << 64) - 1)
        h ^= (h >> 31)
        return (h & ((1 << 53) - 1)) / float(1 << 53)

    def _trim_path_to_bounds(self, path, bounds, margin):
        xmin, xmax, zmin, zmax = bounds
        xmin -= margin
        xmax += margin
        zmin -= margin
        zmax += margin
        keep = [idx for idx, (x, z) in enumerate(path)
                if xmin <= x <= xmax and zmin <= z <= zmax]
        if not keep:
            return None
        start = max(0, keep[0] - 1)
        end = min(len(path), keep[-1] + 2)
        return path[start:end]

    def _ensure_macro_path(self, kind, gx, gz, start):
        cache = self._macro_path_cache[kind]
        key = (int(gx), int(gz))
        if key in cache:
            return cache[key]
        path = self._build_macro_path(kind, start)
        cache[key] = path
        return path

    def _build_macro_path(self, kind, start):
        start_vec = numpy.array(start, dtype=float)
        forward = self._integrate_macro_path(kind, start_vec.copy(), direction=1)
        backward = self._integrate_macro_path(kind, start_vec.copy(), direction=-1)
        backward.reverse()
        return backward[:-1] + forward

    def _integrate_macro_path(self, kind, pos, direction):
        pts = [tuple(pos)]
        step = self.river_step if kind == 'river' else self.road_step
        max_steps = self.river_max_steps if kind == 'river' else self.road_max_steps
        for _ in range(max_steps):
            vec = self._river_vector(pos[0], pos[1]) if kind == 'river' else self._road_vector(pos[0], pos[1])
            norm = math.hypot(vec[0], vec[1])
            if norm < 1e-5:
                break
            pos = pos + (vec / norm) * (step * direction)
            pts.append((float(pos[0]), float(pos[1])))
            if abs(pos[0]) > 8_000_000 or abs(pos[1]) > 8_000_000:
                break
        return pts

    def _river_vector(self, x, z):
        scale = 1.0 / 1800.0
        delta = 35.0
        center = numpy.array([[x * scale, z * scale]], dtype=float)
        h0 = self.river_height_field.noise(center)[0]
        hx = self.river_height_field.noise(numpy.array([[(x + delta) * scale, z * scale]], dtype=float))[0]
        hz = self.river_height_field.noise(numpy.array([[x * scale, (z + delta) * scale]], dtype=float))[0]
        grad = numpy.array([h0 - hx, h0 - hz], dtype=float)
        jitter = numpy.array([
            self.river_vec_u.noise(numpy.array([[x / 620.0, z / 620.0]], dtype=float))[0],
            self.river_vec_v.noise(numpy.array([[x / 620.0, z / 620.0]], dtype=float))[0],
        ])
        vec = grad * 0.8 + jitter * 0.4
        norm = math.hypot(vec[0], vec[1])
        if norm < 1e-4:
            return numpy.array([0.0, -1.0])
        return vec / norm

    def _road_vector(self, x, z):
        freq = 1.0 / 320.0
        jitter = 1.0 / 1500.0
        u = self.road_vec_u.noise(numpy.array([[x * freq, z * freq]], dtype=float))[0]
        v = self.road_vec_v.noise(numpy.array([[x * freq, z * freq]], dtype=float))[0]
        warp_x = self.river_vec_u.noise(numpy.array([[x * jitter, z * jitter]], dtype=float))[0] * 0.2
        warp_z = self.river_vec_v.noise(numpy.array([[x * jitter, z * jitter]], dtype=float))[0] * 0.2
        vec = numpy.array([u + warp_x, v + warp_z], dtype=float)
        norm = math.hypot(vec[0], vec[1])
        if norm < 1e-4:
            return numpy.array([1.0, 0.0])
        return vec / norm

    def _macro_surface_height(self, wx, wz):
        coarse = self.river_height_field.noise(numpy.array([[wx / 2600.0, wz / 2600.0]], dtype=float))[0] * 55.0
        detail = self.road_height_noise.noise(numpy.array([[wx / 650.0, wz / 650.0]], dtype=float))[0] * 12.0
        return numpy.clip(68.0 + coarse + detail, 6.0, SECTOR_HEIGHT - 6.0)

    def _sample_local_height(self, wx, wz, elevation, position):
        xs, zs = self._world_axes(position)
        base_x = xs[0]
        base_z = zs[0]
        lx = int(round(wx - base_x))
        lz = int(round(wz - base_z))
        sx = elevation.shape[0]
        sz = elevation.shape[1]
        if 0 <= lx < sx and 0 <= lz < sz:
            return float(elevation[lx, lz])
        return self._macro_surface_height(wx, wz)

    def _segment_local_grid(self, xs, zs, ax, az, bx, bz, width):
        seg_dx = bx - ax
        seg_dz = bz - az
        seg_len2 = seg_dx * seg_dx + seg_dz * seg_dz
        if seg_len2 < 1e-6:
            return None
        sx = len(xs)
        sz = len(zs)
        ix0 = max(0, int(math.floor(min(ax, bx) - width - xs[0])))
        ix1 = min(sx - 1, int(math.ceil(max(ax, bx) + width - xs[0])))
        iz0 = max(0, int(math.floor(min(az, bz) - width - zs[0])))
        iz1 = min(sz - 1, int(math.ceil(max(az, bz) + width - zs[0])))
        if ix1 < ix0 or iz1 < iz0:
            return None
        sub_xs = xs[ix0:ix1 + 1]
        sub_zs = zs[iz0:iz1 + 1]
        grid_x, grid_z = numpy.meshgrid(sub_xs, sub_zs, indexing='ij')
        dx = grid_x - ax
        dz = grid_z - az
        t = ((dx * seg_dx) + (dz * seg_dz)) / seg_len2
        t = numpy.clip(t, 0.0, 1.0)
        closest_x = ax + seg_dx * t
        closest_z = az + seg_dz * t
        dist = numpy.sqrt((grid_x - closest_x) ** 2 + (grid_z - closest_z) ** 2)
        influence = numpy.clip(1.0 - dist / max(width, 1.0), 0.0, 1.0)
        apply = influence > 0.0
        if not apply.any():
            return None
        return ix0, ix1, iz0, iz1, influence, t, apply

    def _gather_macro_paths(self, kind, position, bounds):
        spacing = self.river_spacing if kind == 'river' else self.road_spacing
        threshold = 0.82 if kind == 'river' else 0.58
        margin = 96 if kind == 'river' else 64
        xmin, xmax, zmin, zmax = bounds
        gx_min = int(math.floor((xmin - margin) / spacing)) - 1
        gx_max = int(math.ceil((xmax + margin) / spacing)) + 1
        gz_min = int(math.floor((zmin - margin) / spacing)) - 1
        gz_max = int(math.ceil((zmax + margin) / spacing)) + 1
        paths = []
        for gx in range(gx_min, gx_max + 1):
            for gz in range(gz_min, gz_max + 1):
                if self._macro_random(gx, gz, salt=13 if kind == 'river' else 29) < threshold:
                    continue
                start_x = gx * spacing + spacing * 0.5
                start_z = gz * spacing + spacing * 0.5
                path = self._ensure_macro_path(kind, gx, gz, (start_x, start_z))
                trimmed = self._trim_path_to_bounds(path, bounds, margin)
                if trimmed:
                    paths.append(trimmed)
        return paths


    def _transport_plans_vectorized(self, position, elevation, road_height=None):
        """
        Returns:
        rural_plan: dict or None
        trail_plan: dict or None
        """
        rural = self._compute_rural_road_plan_vec(position, elevation, road_height=road_height)
        trail = self._compute_trail_plan_vec(position, elevation)

        # DEBUG: plan presence + density
        sx, sz = elevation.shape
        if rural is None:
            rural_n = 0
        else:
            rural_n = int(np.count_nonzero(rural["mask"]))
        if trail is None:
            trail_n = 0
        else:
            trail_n = int(np.count_nonzero(trail["mask"]))

        logutil.log(
            "MAPGEN_TRANSPORT",
            f"transport plans sector={position} rural={rural is not None} rural_cells={rural_n}/{sx*sz} "
            f"trail={trail is not None} trail_cells={trail_n}/{sx*sz}",
            level="DEBUG",
        )

        return rural, trail

    # ---------------------------------------------------------------------
    # 1) Vectorized Voronoi query (two nearest distances + the two nearest site positions)
    # ---------------------------------------------------------------------
    def _hash2_u32(self, x: np.ndarray, z: np.ndarray, seed: int) -> np.ndarray:
        # x,z int32 arrays -> uint32 hash
        h = (x.astype(np.uint32) * np.uint32(0x8da6b343)) ^ (z.astype(np.uint32) * np.uint32(0xd8163841)) ^ np.uint32(seed)
        h ^= (h >> np.uint32(16))
        h *= np.uint32(0x7feb352d)
        h ^= (h >> np.uint32(15))
        h *= np.uint32(0x846ca68b)
        h ^= (h >> np.uint32(16))
        return h

    def _hash_to_unit_float(self, h: np.ndarray) -> np.ndarray:
        # uint32 -> float64 in [0,1)
        return (h.astype(np.float64) / 4294967296.0)

    def _road_voronoi_two_sites_vec(self, wx_grid, wz_grid, spacing):
        sx, sz = wx_grid.shape

        cx = np.floor(wx_grid / spacing).astype(np.int32)
        cz = np.floor(wz_grid / spacing).astype(np.int32)

        off = np.array([-1, 0, 1], dtype=np.int32)
        ox, oz = np.meshgrid(off, off, indexing="ij")  # (3,3)

        gx = cx[None, None, :, :] + ox[:, :, None, None]  # (3,3,sx,sz)
        gz = cz[None, None, :, :] + oz[:, :, None, None]

        gx_flat = gx.reshape(-1).astype(np.int32)
        gz_flat = gz.reshape(-1).astype(np.int32)

        base_x = gx_flat.astype(np.float64) * spacing + 0.5 * spacing
        base_z = gz_flat.astype(np.float64) * spacing + 0.5 * spacing

        jitter_scale = 0.4 * spacing
        h1 = self._hash2_u32(gx_flat, gz_flat, seed=12345)
        h2 = self._hash2_u32(gx_flat, gz_flat, seed=67890)

        u = self._hash_to_unit_float(h1) - 0.5
        v = self._hash_to_unit_float(h2) - 0.5

        sx_flat = base_x + u * jitter_scale
        sz_flat = base_z + v * jitter_scale

        sx_33 = sx_flat.reshape(3, 3, sx, sz)
        sz_33 = sz_flat.reshape(3, 3, sx, sz)
        sx9 = sx_33.reshape(9, sx, sz)
        sz9 = sz_33.reshape(9, sx, sz)

        # ALSO reshape gx/gz to match the 9-candidate layout
        gx9 = gx.reshape(3, 3, sx, sz).reshape(9, sx, sz)
        gz9 = gz.reshape(3, 3, sx, sz).reshape(9, sx, sz)

        dx = wx_grid[None, :, :] - sx9
        dz = wz_grid[None, :, :] - sz9
        dist2 = dx * dx + dz * dz

        idx = np.argsort(dist2, axis=0)
        i0 = idx[0]
        i1 = idx[1]

        d0 = np.take_along_axis(dist2, i0[None, :, :], axis=0)[0]
        d1 = np.take_along_axis(dist2, i1[None, :, :], axis=0)[0]

        s1x = np.take_along_axis(sx9, i0[None, :, :], axis=0)[0]
        s1z = np.take_along_axis(sz9, i0[None, :, :], axis=0)[0]
        s2x = np.take_along_axis(sx9, i1[None, :, :], axis=0)[0]
        s2z = np.take_along_axis(sz9, i1[None, :, :], axis=0)[0]

        # NEW: pick the nearest site's integer cell coords (stable globally)
        g1x = np.take_along_axis(gx9, i0[None, :, :], axis=0)[0]
        g1z = np.take_along_axis(gz9, i0[None, :, :], axis=0)[0]

        # NEW: stable site id (uint32) to compare across neighbors/chunks
        site_id0 = self._hash2_u32(g1x.astype(np.int32), g1z.astype(np.int32), seed=424242)

        return np.sqrt(d0), np.sqrt(d1), s1x, s1z, s2x, s2z, site_id0

    # ---------------------------------------------------------------------
    # 2) Rural road planning (vectorized)
    #    - sparse network
    #    - low grade (local slope check)
    #    - prefers stable elevation along tangent (direction-quantized smoothing)
    #    - shallow bridging and tunneling classification
    # ---------------------------------------------------------------------

    def _compute_rural_road_plan_vec(self, position, elevation, road_height=None):
        """
        Returns dict or None:
        {
            "mask": bool[sx,sz],
            "y": int16[sx,sz]   target road deck height,
            "clearance": int,
            "pillar_cap": int,
        }
        """
        xs, zs = self._world_axes(position)
        sx = len(xs)
        sz = len(zs)

        # World coord grids (sx,sz)
        wx, wz = np.meshgrid(np.array(xs, dtype=np.float64),
                            np.array(zs, dtype=np.float64), indexing="ij")

        # Tunables (start values; iterate)
        spacing = float(getattr(self, "rural_road_spacing", max(700.0, float(self.road_spacing) * 3.0)))
        edge_width = float(getattr(self, "rural_road_edge_width", 3.25))
        max_grade = float(getattr(self, "rural_road_max_grade", 0.15))     # blocks per block
        span = int(getattr(self, "rural_road_smooth_span", 5))             # samples along tangent
        clearance = int(getattr(self, "rural_road_clearance", 4))
        pillar_cap = int(getattr(self, "rural_road_pillar_cap", 32))
        max_bridge_depth = int(getattr(self, "rural_road_max_bridge_depth", 12))
        d1, d2, s1x, s1z, s2x, s2z, site_id0 = self._road_voronoi_two_sites_vec(wx, wz, spacing)
        right_diff = np.zeros_like(site_id0, dtype=bool)
        up_diff = np.zeros_like(site_id0, dtype=bool)
        right_diff[:-1, :] = (site_id0[:-1, :] != site_id0[1:, :])
        up_diff[:, :-1] = (site_id0[:, :-1] != site_id0[:, 1:])
        mask = (right_diff | up_diff) & (d1 < spacing * 0.8)

        if not mask.any():
            return None

        elev = elevation.astype(np.float32)
        gx, gz = np.gradient(elev)
        cliff_grad = float(getattr(config, "RURAL_ROAD_CLIFF_GRAD", 0.85))
        cliff_mask = np.hypot(gx, gz) > cliff_grad
        ground = elev
        shallow_water = ground <= WATER_LEVEL

        if road_height is None:
            road_base = elev
        else:
            road_base = road_height.astype(np.float32)
        base_target = road_base.copy()

        # Tangent direction from the two sites:
        # boundary normal ~ (s2-s1), tangent = perp(normal).
        vx = (s2x - s1x).astype(np.float32)
        vz = (s2z - s1z).astype(np.float32)
        norm = np.sqrt(vx * vx + vz * vz) + 1e-6
        nx = vx / norm
        nz = vz / norm
        tx = -nz  # perp
        tz = nx

        # Quantize tangent to 8-connected integer step (dx,dz) in {-1,0,1}.
        # This makes sampling and smoothing fully array-index based.
        ang = np.arctan2(tz, tx)  # [-pi,pi]
        # 8 bins: E,NE,N,NW,W,SW,S,SE
        bin8 = np.round((ang / (np.pi / 4.0))).astype(np.int32) % 8
        dir_dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)[bin8]
        dir_dz = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)[bin8]

        # Sample elevation along +/- tangent for smoothing (median)
        xi, zi = np.meshgrid(np.arange(sx, dtype=np.int32),
                            np.arange(sz, dtype=np.int32), indexing="ij")
        xp = np.clip(xi + dir_dx, 0, sx - 1)
        zp = np.clip(zi + dir_dz, 0, sz - 1)
        xm = np.clip(xi - dir_dx, 0, sx - 1)
        zm = np.clip(zi - dir_dz, 0, sz - 1)

        samples = [base_target]
        for k in range(1, span + 1):
            x1 = np.clip(xi + dir_dx * k, 0, sx - 1)
            z1 = np.clip(zi + dir_dz * k, 0, sz - 1)
            x2 = np.clip(xi - dir_dx * k, 0, sx - 1)
            z2 = np.clip(zi - dir_dz * k, 0, sz - 1)
            samples.append(base_target[x1, z1])
            samples.append(base_target[x2, z2])

        smoothed = np.median(np.stack(samples, axis=0), axis=0).astype(np.float32)

        # Do not go below water; allow shallow bridging by lifting to WATER_LEVEL+1
        shallow_water = ground <= WATER_LEVEL
        depth = (WATER_LEVEL - ground).astype(np.float32)
        ok_bridge = shallow_water & (depth <= float(max_bridge_depth))
        mask &= (~shallow_water) | ok_bridge
        # Fill small gaps along the road direction to avoid short dead ends.
        land_ok = (~shallow_water) | ok_bridge
        mask_line = mask.copy()
        for _ in range(2):
            mask_line = mask_line | mask_line[xp, zp] | mask_line[xm, zm]
            mask_line &= land_ok
        mask = mask_line

        # Target height prefers smoothed, but never under water
        target = smoothed.copy()
        target = np.where(shallow_water, float(WATER_LEVEL + 1), target)
        target = np.where(target <= WATER_LEVEL, float(WATER_LEVEL + 1), target)
        target = np.clip(target, 2.0, float(SECTOR_HEIGHT - 5)).astype(np.float32)

        grade_iters = int(getattr(config, "RURAL_ROAD_GRADE_RELAX_ITERS", 3))
        if grade_iters > 0:
            for _ in range(grade_iters):
                t_p = target[xp, zp]
                t_m = target[xm, zm]
                lower = np.maximum(t_p - max_grade, t_m - max_grade)
                upper = np.minimum(t_p + max_grade, t_m + max_grade)
                target = np.where(mask, np.minimum(np.maximum(target, lower), upper), target)

        # Interpolate along the sector crossing using the two furthest road cells.
        base_mask = mask
        if base_mask.any():
            coords = np.column_stack(np.nonzero(base_mask))
            if coords.shape[0] >= 2:
                dmax = -1.0
                a = coords[0]
                b = coords[1]
                for i in range(coords.shape[0]):
                    dx = coords[:, 0] - coords[i, 0]
                    dz = coords[:, 1] - coords[i, 1]
                    dist2 = dx * dx + dz * dz
                    j = int(np.argmax(dist2))
                    if dist2[j] > dmax:
                        dmax = float(dist2[j])
                        a = coords[i]
                        b = coords[j]
                vx = float(b[0] - a[0])
                vz = float(b[1] - a[1])
                denom = vx * vx + vz * vz
                if denom > 0.0:
                    t = ((xi - a[0]) * vx + (zi - a[1]) * vz) / denom
                    t = np.clip(t, 0.0, 1.0)
                    y0 = target[a[0], a[1]]
                    y1 = target[b[0], b[1]]
                    interp = y0 + t * (y1 - y0)
                    target = np.where(base_mask, interp, target)

        # Centerline should not float above terrain; shoulders may build up.
        if base_mask.any():
            target = np.where(base_mask & (ground > WATER_LEVEL), np.minimum(target, road_base), target)

        # Enforce 1-block steps along the road direction for smoother descents.
        step_iters = int(getattr(config, "RURAL_ROAD_STEP_RELAX_ITERS", 3))
        max_step = float(getattr(config, "RURAL_ROAD_MAX_STEP", 1.0))
        if step_iters > 0:
            for _ in range(step_iters):
                t_p = target[xp, zp]
                t_m = target[xm, zm]
                lower = np.maximum(t_p - max_step, t_m - max_step)
                upper = np.minimum(t_p + max_step, t_m + max_step)
                target = np.where(mask, np.minimum(np.maximum(target, lower), upper), target)
        target = np.where(shallow_water, np.minimum(target, float(WATER_LEVEL + 1)), target)

        # If terrain is far above target (big cut), rural road gives up (trail will handle)
        cliff_mode = getattr(config, "RURAL_ROAD_CLIFF_MODE", "tunnel")
        if cliff_mode == "dead_end":
            mask &= ~cliff_mask

        if not mask.any():
            return None

        y = np.rint(np.clip(target, 2.0, float(SECTOR_HEIGHT - 5))).astype(np.int16)
        ground_i16 = np.clip(ground, 0, SECTOR_HEIGHT - 1).astype(np.int16)

        return {
            "mask": mask,
            "y": y,
            "ground": ground_i16,
            "clearance": clearance,
            "pillar_cap": pillar_cap,
            "origin": self._sector_origin(position),
        }

    # ---------------------------------------------------------------------
    # 4) Fully vectorized road stamping: decks, headroom carve, capped pillars
    # ---------------------------------------------------------------------

    def _apply_rural_roads_vec(self, blocks, plan):
        if plan is None:
            return
        def _binary_dilate_8(mask):
            # 3x3 width: center + NSEW + diagonals
            up = np.zeros_like(mask);   up[:, 1:] = mask[:, :-1]
            dn = np.zeros_like(mask);   dn[:, :-1] = mask[:, 1:]
            lt = np.zeros_like(mask);   lt[1:, :] = mask[:-1, :]
            rt = np.zeros_like(mask);   rt[:-1, :] = mask[1:, :]
            ul = np.zeros_like(mask);   ul[1:, 1:] = mask[:-1, :-1]
            ur = np.zeros_like(mask);   ur[:-1, 1:] = mask[1:, :-1]
            dl = np.zeros_like(mask);   dl[1:, :-1] = mask[:-1, 1:]
            dr = np.zeros_like(mask);   dr[:-1, :-1] = mask[1:, 1:]
            return mask | up | dn | lt | rt | ul | ur | dl | dr
        def _shift_bool(mask, dx, dz):
            out = np.zeros_like(mask, dtype=bool)
            sx_, sz_ = mask.shape
            xs0 = slice(max(0, dx),  min(sx_, sx_ + dx))
            zs0 = slice(max(0, dz),  min(sz_, sz_ + dz))
            xs1 = slice(max(0, -dx), min(sx_, sx_ - dx))
            zs1 = slice(max(0, -dz), min(sz_, sz_ - dz))
            out[xs1, zs1] = mask[xs0, zs0]
            return out
        def _shift_i32(a, dx, dz, fill):
            out = np.full(a.shape, fill, dtype=np.int32)
            sx_, sz_ = a.shape
            xs0 = slice(max(0, dx),  min(sx_, sx_ + dx))
            zs0 = slice(max(0, dz),  min(sz_, sz_ + dz))
            xs1 = slice(max(0, -dx), min(sx_, sx_ - dx))
            zs1 = slice(max(0, -dz), min(sz_, sz_ - dz))
            out[xs1, zs1] = a[xs0, zs0].astype(np.int32)
            return out

        base_mask = plan["mask"]
        mask = _binary_dilate_8(base_mask)
        n_deck = int(np.count_nonzero(mask))
        if n_deck == 0:
            logutil.log("MAPGEN", "rural stamp skipped (mask empty)", level="DEBUG")
            return

        y_center = plan["y"].astype(np.int32)
        y = y_center.copy()
        if mask.any():
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                neighbor = _shift_bool(base_mask, dx, dz)
                neighbor_y = _shift_i32(y_center, dx, dz, fill=-1)
                assign = (~base_mask) & neighbor
                y = np.where(assign, neighbor_y, y)
            y = np.where(mask & (y < 0), y_center, y)

        tunnel_clearance = 4
        pillar_cap = int(plan["pillar_cap"])
        ground = plan.get("ground")
        if ground is not None:
            ground = np.array(ground, dtype=np.int32)
        origin = plan.get("origin", None)

        sx, sz = mask.shape
        H = blocks.shape[0]  # SECTOR_HEIGHT

        yy = np.arange(H, dtype=np.int32)[:, None, None]
        y0 = y[None, :, :]
        # --- Carve only where terrain blocks the road; keep a 4-block tunnel clearance.
        solid = blocks != 0
        carve = mask[None, :, :] & solid & (yy >= y0) & (yy <= (y0 + tunnel_clearance))
        if carve.any():
            road_protect = (blocks == COBBLE) | (blocks == PLANK) | (blocks == JACK)
            carve = carve & (~road_protect)
            blocks[carve] = 0

        # --- Deck placement (advanced indexing)
        water_under = None
        if ground is not None:
            water_under = ground <= WATER_LEVEL
        xs, zs = np.nonzero(mask)
        ys = y[mask]
        blocks[ys, xs, zs] = COBBLE
        if water_under is not None:
            bridge_mask = mask & water_under
            if bridge_mask.any():
                bx, bz = np.nonzero(bridge_mask)
                by = y[bridge_mask]
                blocks[by, bx, bz] = PLANK

        # --- Capped pillar supports down through air/water (bridging shallow gaps)
        # This will not overwrite solid terrain; it only fills air/water in [y-pillar_cap, y).
        # It is vectorized and “self-stops” on solid because we do not overwrite solids.
        support_mask = base_mask
        if origin is not None:
            ox, oz = origin
            wx = np.arange(sx, dtype=np.int64) + int(ox)
            wz = np.arange(sz, dtype=np.int64) + int(oz)
            wxg, wzg = np.meshgrid(wx, wz, indexing="ij")
            support_mask = base_mask & (((wxg + wzg) & 15) == 0)
        solid = blocks != 0
        solid_below = solid & (yy < y0)
        rev = solid_below[::-1, :, :]
        has_solid = rev.any(axis=0)
        first_rev = rev.argmax(axis=0).astype(np.int32)
        solid_y = (H - 1 - first_rev).astype(np.int32)
        solid_y = np.where(has_solid, solid_y, -1).astype(np.int32)
        fill = support_mask[None, :, :] & (yy < y0) & (yy > solid_y[None, :, :])
        air_or_water = (blocks == 0) | (blocks == WATER)
        blocks[fill & air_or_water] = COBBLE

        # --- Tunnel lighting: jack o lantern on the ceiling ~every 32 blocks.
        if origin is not None and ground is not None:
            ox, oz = origin
            wx = np.arange(sx, dtype=np.int64) + int(ox)
            wz = np.arange(sz, dtype=np.int64) + int(oz)
            wxg, wzg = np.meshgrid(wx, wz, indexing="ij")
            tunnel_mask = base_mask & ((ground - y_center) >= 4)
            if tunnel_mask.any():
                hashv = (wxg * 73856093 + wzg * 19349663) & 31
                pick = (hashv == 0)
                ceil_y = y_center + tunnel_clearance + 1
                valid = (ceil_y < H)
                place = tunnel_mask & pick & valid
                if place.any():
                    xs, zs = np.nonzero(place)
                    ys = ceil_y[place]
                    blocks[ys, xs, zs] = JACK

        n_clear = int(np.count_nonzero(carve))
        n_fill = int(np.count_nonzero(fill & air_or_water))
        logutil.log(
            "MAPGEN",
            f"rural stamp deck_cols={n_deck} clear_voxels={n_clear} pillar_voxels={n_fill} "
            f"clearance={tunnel_clearance} pillar_cap={pillar_cap}",
            level="DEBUG",
        )

    def _compute_trail_plan_vec(self, position, elevation):
        import numpy as np

        def _binary_dilate_4(m: np.ndarray) -> np.ndarray:
            up = np.zeros_like(m);   up[:, 1:]  = m[:, :-1]
            dn = np.zeros_like(m);   dn[:, :-1] = m[:, 1:]
            lt = np.zeros_like(m);   lt[1:, :]  = m[:-1, :]
            rt = np.zeros_like(m);   rt[:-1, :] = m[1:, :]
            return m | up | dn | lt | rt

        def _shift_bool(m: np.ndarray, dx: int, dz: int) -> np.ndarray:
            out = np.zeros_like(m, dtype=bool)
            sx_, sz_ = m.shape
            xs0 = slice(max(0, dx),  min(sx_, sx_ + dx))
            zs0 = slice(max(0, dz),  min(sz_, sz_ + dz))
            xs1 = slice(max(0, -dx), min(sx_, sx_ - dx))
            zs1 = slice(max(0, -dz), min(sz_, sz_ - dz))
            out[xs1, zs1] = m[xs0, zs0]
            return out

        def _bridge_diagonals(m: np.ndarray) -> np.ndarray:
            # Add orthogonal bridge tiles only when diagonals touch.
            ne = m & _shift_bool(m, 1, 1)
            nw = m & _shift_bool(m, -1, 1)
            se = m & _shift_bool(m, 1, -1)
            sw = m & _shift_bool(m, -1, -1)

            bridge = (
                _shift_bool(ne, -1, 0) | _shift_bool(ne, 0, -1) |
                _shift_bool(nw, 1, 0) | _shift_bool(nw, 0, -1) |
                _shift_bool(se, -1, 0) | _shift_bool(se, 0, 1) |
                _shift_bool(sw, 1, 0) | _shift_bool(sw, 0, 1)
            )
            return m | bridge

        def _shift_i32(a: np.ndarray, dx: int, dz: int, fill: int) -> np.ndarray:
            out = np.full(a.shape, fill, dtype=np.int32)
            sx_, sz_ = a.shape
            xs0 = slice(max(0, dx),  min(sx_, sx_ + dx))
            zs0 = slice(max(0, dz),  min(sz_, sz_ + dz))
            xs1 = slice(max(0, -dx), min(sx_, sx_ - dx))
            zs1 = slice(max(0, -dz), min(sz_, sz_ - dz))
            out[xs1, zs1] = a[xs0, zs0].astype(np.int32)
            return out

        xs, zs = self._world_axes(position)
        sx = len(xs); sz = len(zs)

        wx, wz = np.meshgrid(
            np.asarray(xs, dtype=np.float64),
            np.asarray(zs, dtype=np.float64),
            indexing="ij",
        )

        spacing     = float(getattr(self, "trail_spacing", max(180.0, float(self.road_spacing) * 0.9)))
        clearance   = int(getattr(self, "trail_clearance", 2))
        pillar_cap  = int(getattr(self, "trail_pillar_cap", 4))
        relax_iters = int(getattr(self, "trail_relax_iters", 4))
        step_limit  = int(getattr(self, "trail_step_limit", 2))
        step_limit  = max(1, step_limit)

        # width control for trails (in blocks): 1 => 1-wide, 2 => a bit thicker, etc.
        width = int(getattr(self, "trail_width_blocks", 1))
        width = max(1, min(4, width))

        xs_ext = np.arange(sx + 1, dtype=np.float64) + float(xs[0])
        zs_ext = np.arange(sz + 1, dtype=np.float64) + float(zs[0])
        wx_ext, wz_ext = np.meshgrid(xs_ext, zs_ext, indexing="ij")

        _, _, _, _, _, _, site_id_ext = self._road_voronoi_two_sites_vec(wx_ext, wz_ext, spacing)
        site_id0 = site_id_ext[:-1, :-1]

        # Edge detection: mark cells where nearest-site id differs from neighbor.
        # Include the neighbor outside the sector to avoid seam gaps.
        right_diff = (site_id0 != site_id_ext[1:, :-1])
        up_diff = (site_id0 != site_id_ext[:-1, 1:])
        mask = right_diff | up_diff

        # Bridge diagonal runs to keep continuity without widening axis-aligned paths.
        mask = _bridge_diagonals(mask)

        # Optional extra thickness (kept separate from diagonal bridging).
        for _ in range(max(0, width - 1)):
            mask = _binary_dilate_4(mask)

        if not mask.any():
            return None

        allow_water = bool(getattr(config, "TRAIL_ALLOW_WATER", False))

        # Elevation base
        y = elevation.astype(np.int16).copy()
        if not allow_water:
            mask &= (y > WATER_LEVEL)
        y = np.where(y <= WATER_LEVEL, WATER_LEVEL + 1, y).astype(np.int16)
        y = np.clip(y, 2, SECTOR_HEIGHT - 4).astype(np.int16)

        # Stair-step relaxation within mask
        INF = 32767
        for _ in range(relax_iters):
            y_i32 = y.astype(np.int32)

            yN = _shift_i32(y_i32, 0,  1, fill=INF)
            yS = _shift_i32(y_i32, 0, -1, fill=INF)
            yE = _shift_i32(y_i32, 1,  0, fill=INF)
            yW = _shift_i32(y_i32, -1, 0, fill=INF)

            mN = _shift_bool(mask, 0,  1)
            mS = _shift_bool(mask, 0, -1)
            mE = _shift_bool(mask, 1,  0)
            mW = _shift_bool(mask, -1, 0)

            upper = np.full((sx, sz), INF, dtype=np.int32)
            lower = np.full((sx, sz), -INF, dtype=np.int32)

            for yn, mn in ((yN, mN), (yS, mS), (yE, mE), (yW, mW)):
                upper = np.minimum(upper, np.where(mn, yn + step_limit, INF))
                lower = np.maximum(lower, np.where(mn, yn - step_limit, -INF))

            y_new = y_i32
            y_new = np.where(mask, np.maximum(y_new, lower), y_new)
            y_new = np.where(mask, np.minimum(y_new, upper), y_new)
            y = np.clip(y_new, 2, SECTOR_HEIGHT - 4).astype(np.int16)

        return {
            "mask": mask,
            "y": y,
            "clearance": clearance,
            "pillar_cap": pillar_cap,
        }

    def _apply_trails_vec(self, blocks, plan):
        if plan is None:
            return

        mask = plan["mask"]
        if not np.any(mask):
            return

        H = blocks.shape[0]

        solid = (blocks != 0) & (blocks != WATER)
        any_solid = solid.any(axis=0)
        top = np.where(any_solid, H - 1 - np.argmax(solid[::-1], axis=0), 0).astype(np.int32)
        trail_mask = mask & any_solid & (top > WATER_LEVEL)
        if not trail_mask.any():
            return
        y = top

        # Deck (path) placement
        xs, zs = np.nonzero(trail_mask)
        ys = y[trail_mask]
        blocks[ys, xs, zs] = TRAIL_BLOCK

        yy = np.arange(H, dtype=np.int32)[:, None, None]
        air_only = (blocks == 0)

        ladder_min_height = int(getattr(self, "trail_ladder_min_height", 2))
        ladder_min_height = max(2, ladder_min_height)

        y_i32 = y.astype(np.int32)
        neg_inf = -32768
        m = trail_mask

        yE = np.full_like(y_i32, neg_inf); yE[:-1, :] = y_i32[1:, :]
        yW = np.full_like(y_i32, neg_inf); yW[1:, :] = y_i32[:-1, :]
        yS = np.full_like(y_i32, neg_inf); yS[:, :-1] = y_i32[:, 1:]
        yN = np.full_like(y_i32, neg_inf); yN[:, 1:] = y_i32[:, :-1]

        mE = np.zeros_like(m); mE[:-1, :] = m[1:, :]
        mW = np.zeros_like(m); mW[1:, :] = m[:-1, :]
        mS = np.zeros_like(m); mS[:, :-1] = m[:, 1:]
        mN = np.zeros_like(m); mN[:, 1:] = m[:, :-1]

        dE = yE - y_i32
        dW = yW - y_i32
        dS = yS - y_i32
        dN = yN - y_i32

        ladder_dir = np.zeros_like(y_i32, dtype=np.int8)
        ladder_delta = np.zeros_like(y_i32, dtype=np.int16)

        cand = m & mE & (dE >= ladder_min_height)
        better = cand & (dE > ladder_delta)
        ladder_delta = np.where(better, dE, ladder_delta)
        ladder_dir = np.where(better, 1, ladder_dir)

        cand = m & mW & (dW >= ladder_min_height)
        better = cand & (dW > ladder_delta)
        ladder_delta = np.where(better, dW, ladder_delta)
        ladder_dir = np.where(better, 2, ladder_dir)

        cand = m & mS & (dS >= ladder_min_height)
        better = cand & (dS > ladder_delta)
        ladder_delta = np.where(better, dS, ladder_delta)
        ladder_dir = np.where(better, 3, ladder_dir)

        cand = m & mN & (dN >= ladder_min_height)
        better = cand & (dN > ladder_delta)
        ladder_delta = np.where(better, dN, ladder_delta)
        ladder_dir = np.where(better, 4, ladder_dir)

        if np.any(ladder_dir):
            start = (y_i32 + 1)[None, :, :]
            end = (y_i32 + ladder_delta)[None, :, :]

            ladder_cells = (ladder_dir == 1)
            if ladder_cells.any():
                ladder_mask = ladder_cells[None, :, :] & (yy >= start) & (yy <= end)
                blocks[ladder_mask & air_only] = LADDER_EAST

            ladder_cells = (ladder_dir == 2)
            if ladder_cells.any():
                ladder_mask = ladder_cells[None, :, :] & (yy >= start) & (yy <= end)
                blocks[ladder_mask & air_only] = LADDER_WEST

            ladder_cells = (ladder_dir == 3)
            if ladder_cells.any():
                ladder_mask = ladder_cells[None, :, :] & (yy >= start) & (yy <= end)
                blocks[ladder_mask & air_only] = LADDER_SOUTH

            ladder_cells = (ladder_dir == 4)
            if ladder_cells.any():
                ladder_mask = ladder_cells[None, :, :] & (yy >= start) & (yy <= end)
                blocks[ladder_mask & air_only] = LADDER_NORTH

    def _compute_river_plan(self, position, elevation):
        """
        Plan rivers as canyons / underground channels that always keep the
        water surface at the global WATER_LEVEL.

        We still use the existing macro paths for horizontal layout, but
        vertical placement is simple:
          - water_top = WATER_LEVEL
          - river floor = WATER_LEVEL - depth(x, z)

        We do NOT lower 'elevation' here; carving happens later in
        _apply_river_columns, so rivers can be underground in uplifted
        regions and surface canyons elsewhere.
        """
        xs, zs = self._world_axes(position)
        bounds = (xs[0], xs[-1], zs[0], zs[-1])
        paths = self._gather_macro_paths('river', position, bounds)
        if not paths:
            return None

        sx = len(xs)
        sz = len(zs)
        mask = numpy.zeros((sx, sz), dtype=bool)
        depth = numpy.zeros((sx, sz), dtype=float)

        # Accumulate desired depth along macro paths.
        for path in paths:
            for (ax, az), (bx, bz) in zip(path[:-1], path[1:]):
                seg_mid_x = 0.5 * (ax + bx)
                seg_mid_z = 0.5 * (az + bz)
                base = 0.5 + 0.5 * self.river_width_noise.noise(
                    numpy.array([[seg_mid_x / 540.0, seg_mid_z / 540.0]], dtype=float)
                )[0]
                width = 5.0 + base * 15.0
                carve_depth = 5.0 + base * 12.0

                res = self._segment_local_grid(xs, zs, ax, az, bx, bz, width)
                if not res:
                    continue
                ix0, ix1, iz0, iz1, influence, _, apply = res

                sub_depth = depth[ix0:ix1 + 1, iz0:iz1 + 1]
                sub_mask = mask[ix0:ix1 + 1, iz0:iz1 + 1]

                sub_mask |= apply
                sub_depth[:] = numpy.maximum(sub_depth, carve_depth * influence)

                depth[ix0:ix1 + 1, iz0:iz1 + 1] = sub_depth
                mask[ix0:ix1 + 1, iz0:iz1 + 1] = sub_mask

        if not mask.any():
            return None

        # Fixed water surface at global sea level.
        water_top = float(WATER_LEVEL)
        carve_depth = numpy.clip(depth, 0.0, 24.0)

        # River floor is below the water surface by 'carve_depth'.
        floor = numpy.clip(
            water_top - numpy.maximum(carve_depth, 2.0),
            2.0,
            SECTOR_HEIGHT - 8.0,
        )
        surface = numpy.clip(
            water_top,
            floor + 1.0,
            SECTOR_HEIGHT - 5.0,
        )

        # Note: we deliberately do NOT modify 'elevation' here.
        # Rivers will be carved directly into the block volume in
        # _apply_river_columns, producing canyons or underground channels.

        return {
            'mask': mask,
            'depth': carve_depth,
            'floor': floor,
            'surface': surface,
        }

    def _sample_path_heights(self, path, elevation, position):
        """
        Sample road heights along a macro path using the global macro surface,
        not the locally modified elevation (which may include cliff boosts).
        This keeps roads continuous across sectors. We still respect a max grade.
        """
        samples = []
        last_h = None
        sample_spacing = 6.0  # world units between samples

        for (ax, az), (bx, bz) in zip(path[:-1], path[1:]):
            seg_len = max(1.0, math.hypot(bx - ax, bz - az))
            steps = max(1, int(seg_len / sample_spacing))
            for step in range(steps):
                t = step / steps
                px = ax + (bx - ax) * t
                pz = az + (bz - az) * t

                # Use macro surface height so roads ignore local cliff uplift.
                target = self._macro_surface_height(px, pz)

                if last_h is None:
                    clamped = target
                else:
                    delta = numpy.clip(
                        target - last_h,
                        -self.road_grade_limit,
                        self.road_grade_limit,
                    )
                    clamped = last_h + delta

                samples.append((px, pz, clamped))
                last_h = clamped

        if not samples:
            # Degenerate path: single sample at start.
            px, pz = path[0]
            h = self._macro_surface_height(px, pz)
            samples.append((px, pz, h))
            return samples

        # Also include the final node, gently blended toward its macro height.
        end_x, end_z = path[-1]
        end_h = self._macro_surface_height(end_x, end_z)
        if last_h is not None:
            delta = numpy.clip(
                end_h - last_h,
                -self.road_grade_limit,
                self.road_grade_limit,
            )
            end_h = last_h + delta
        samples.append((end_x, end_z, end_h))

        return samples

    def _road_cell_seed(self, gx, gz):
        """
        Deterministic jittered site position for Voronoi-style roads.
        gx, gz are integer grid cell coordinates in 'road space'.

        Uses ROAD_NETWORK_SPACING as the base cell size, and jitters the
        site inside each cell using simplex noise so it is globally consistent.
        """
        spacing = self.road_spacing  # from config: ROAD_NETWORK_SPACING
        base_x = gx * spacing + 0.5 * spacing
        base_z = gz * spacing + 0.5 * spacing

        # Jitter sites using the existing road_vec_u / road_vec_v noise fields.
        jitter_scale = 0.4 * spacing  # stay inside the cell
        jx = self.road_vec_u.noise(
            numpy.array([[gx * 0.37, gz * 0.41]], dtype=float)
        )[0] * jitter_scale
        jz = self.road_vec_v.noise(
            numpy.array([[gx * 0.29, gz * 0.53]], dtype=float)
        )[0] * jitter_scale

        return base_x + jx, base_z + jz

    def _road_voronoi_distances(self, wx, wz):
        """
        Distances to the two nearest Voronoi sites in the jittered road grid
        near world position (wx, wz).

        Returns (d1, d2) with d1 <= d2.
        """
        spacing = self.road_spacing
        cx = int(math.floor(wx / spacing))
        cz = int(math.floor(wz / spacing))

        best1 = 1e30  # nearest squared distance
        best2 = 1e30  # second-nearest squared distance

        # 3x3 neighborhood of cells is enough to find nearest two sites.
        for gx in range(cx - 1, cx + 2):
            for gz in range(cz - 1, cz + 2):
                sx, sz = self._road_cell_seed(gx, gz)
                dx = wx - sx
                dz = wz - sz
                d2 = dx * dx + dz * dz
                if d2 < best1:
                    best2 = best1
                    best1 = d2
                elif d2 < best2:
                    best2 = d2

        d1 = math.sqrt(best1)
        d2 = math.sqrt(best2)
        return d1, d2


    def _compute_road_plan(self, position, elevation, ground_before):
        """
        Plan roads as Voronoi borders of a global jittered site grid.

        Roads are where the two nearest sites are at almost equal distance
        (|d1 - d2| small). This yields an infinite, continuous network of
        curves in XZ. Road Y is taken directly from the landscape elevation,
        with a small lift above water.

        We do NOT modify 'elevation' here; roads are stamped later into blocks.
        """
        xs, zs = self._world_axes(position)
        sx = len(xs)
        sz = len(zs)

        mask = numpy.zeros((sx, sz), dtype=bool)
        height_map = elevation.astype(float).copy()
        original_ground = numpy.zeros((sx, sz), dtype=float)

        spacing = self.road_spacing
        edge_width = 3.5  # Voronoi border thickness in world units

        for ix, wx in enumerate(xs):
            for iz, wz in enumerate(zs):
                d1, d2 = self._road_voronoi_distances(wx, wz)
                diff = abs(d1 - d2)

                # Voronoi border: d1 ≈ d2 and reasonably close to a site.
                if diff < edge_width and d1 < spacing * 0.8:
                    mask[ix, iz] = True
                    ground_y = float(elevation[ix, iz])
                    original_ground[ix, iz] = ground_y

                    # Road at landscape height, but never under water.
                    h = ground_y
                    if h <= WATER_LEVEL:
                        h = WATER_LEVEL + 1.0
                    h = float(numpy.clip(h, 2.0, SECTOR_HEIGHT - 2.0))
                    height_map[ix, iz] = h

        if not mask.any():
            return None

        return {
            'mask': mask,
            'height': height_map,
            'original_ground': original_ground,
        }

    def _apply_river_columns(self, blocks, plan, ground_h=None, surface_block=None):
        if not plan:
            return
        sx = plan['mask'].shape[0]
        sz = plan['mask'].shape[1]
        for x in range(1, sx - 1):
            for z in range(1, sz - 1):
                if not plan['mask'][x, z]:
                    continue
                floor = int(plan['floor'][x, z])
                surface = int(plan['surface'][x, z])
                floor = numpy.clip(floor, 2, SECTOR_HEIGHT - 8)
                surface = numpy.clip(surface, floor + 1, SECTOR_HEIGHT - 5)
                col = blocks[:, x, z]
                col[:floor - 1] = STONE
                col[floor - 1] = STONE
                bed_block = SAND if plan['depth'][x, z] > 4 else STONE
                col[floor] = bed_block
                water_top = max(surface, floor + 1)
                col[floor + 1:water_top + 1] = WATER
                if ground_h is not None:
                    level_top = int(ground_h[x, z])
                    bank_start = water_top + 1
                    if level_top >= bank_start:
                        bank_end = min(level_top + 1, SECTOR_HEIGHT)
                        # Replace the old grassy ledge with stone so the carved bank has a clean face.
                        col[bank_start:bank_end] = STONE
                        if surface_block is not None and 0 <= level_top < SECTOR_HEIGHT:
                            col[level_top] = surface_block[x, z]

    def _apply_roads(self, blocks, plan, position):
        """
        Stamp roads into the block volume.

        - XZ positions come from the Voronoi plan.
        - Y is just the planned landscape height (clamped to world bounds
          and above WATER_LEVEL when the plan was made).
        - At each road column:
            * Place a COBBLE deck at target_y.
            * Drop a vertical COBBLE pillar downward through air/water
              until we hit solid ground -> simple bridges / cliff ascents.
            * Clear a few blocks of air above the deck for walkability.
        """
        if not plan:
            return

        xs, zs = self._world_axes(position)
        sx = plan['mask'].shape[0]
        sz = plan['mask'].shape[1]

        for x in range(sx):
            for z in range(sz):
                if not plan['mask'][x, z]:
                    continue

                target_y = int(plan['height'][x, z])
                target_y = int(numpy.clip(target_y, 2, SECTOR_HEIGHT - 3))

                col = blocks[:, x, z]

                # Cobble deck at the planned height.
                col[target_y] = COBBLE

                # Drop a cobble pillar downward through air/water until we hit
                # something solid. This gives simple bridges/columns up cliffs.
                for py in range(target_y - 1, 0, -1):
                    b = col[py]
                    # Treat air/water as non-support; replace with cobble.
                    if b == 0 or b == WATER:
                        col[py] = COBBLE
                    else:
                        # Hit solid ground: stop here.
                        break

                # Clear a small headroom above the deck so roads are usable
                # even when inside hills/cliffs.
                head_top = min(SECTOR_HEIGHT - 1, target_y + 3)
                col[target_y + 1:head_top + 1] = 0

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
            if 0 <= cx < SECTOR_SIZE and 0 <= cz < SECTOR_SIZE and 0 <= cy < SECTOR_HEIGHT:
                blocks[cy, cx, cz] = WOOD
        for dx, dy, dz in canopy:
            cx, cy, cz = x + dx, ground_y + dy, z + dz
            if 0 <= cx < SECTOR_SIZE and 0 <= cz < SECTOR_SIZE and 0 <= cy < SECTOR_HEIGHT:
                blocks[cy, cx, cz] = LEAVES

    def _place_cabin(self, blocks, x, z, ground_y):
        # Small 4x4 hut with plank walls, cobble base, and brick chimney.
        width = 4
        height = 4
        if x < 1 or z < 1 or x + width >= SECTOR_SIZE or z + width >= SECTOR_SIZE:
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
                if 0 <= wx < SECTOR_SIZE and 0 <= wz < SECTOR_SIZE:
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
        if x < 1 or z < 1 or x + width >= SECTOR_SIZE or z + depth >= SECTOR_SIZE:
            return
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
                if y >= SECTOR_HEIGHT:
                    break
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
                    if roof_y < SECTOR_HEIGHT:
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
                        if 0 <= wx < SECTOR_SIZE and 0 <= wz < SECTOR_SIZE and ground_y + 1 + dy < SECTOR_HEIGHT:
                            blocks[ground_y + 1 + dy, wx, wz] = 0
            # battlements
            top = ground_y + 1 + h
            if top < SECTOR_HEIGHT:
                for dx in range(w):
                    for dz in range(d):
                        if dx in (0, w - 1) or dz in (0, d - 1):
                            if (dx + dz) % 2 == 0:
                                wx, wz = x + dx, z + dz
                                if 0 <= wx < SECTOR_SIZE and 0 <= wz < SECTOR_SIZE:
                                    blocks[top, wx, wz] = BRICK
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
                if 0 <= wx < SECTOR_SIZE and 0 <= wz < SECTOR_SIZE:
                    h = ground_y + max(0, 1 - abs(dx) - abs(dz))
                    if h < SECTOR_HEIGHT:
                        blocks[h, wx, wz] = COBBLE if (dx + dz) % 2 else STONE

    def _reserve_area(self, taken_mask, x, z, width, depth, padding=1):
        x0 = max(0, x - padding)
        z0 = max(0, z - padding)
        x1 = min(SECTOR_SIZE, x + width + padding)
        z1 = min(SECTOR_SIZE, z + depth + padding)
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
        region_base = self.cave_region_noise(position)
        region_detail = self.cave_region_detail(position)
        region01 = numpy.clip(0.5 + 0.5 * region_base + 0.25 * region_detail, 0.0, 1.0)
        region_threshold = float(getattr(config, "CAVE_REGION_THRESHOLD", 0.62))
        region_mask = region01 > region_threshold
        if not cliff_mask.any() and not region_mask.any():
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
        # Avoid wraparound smoothing that can introduce seam artifacts.
        pad = numpy.pad(density2d, 1, mode="edge")
        density2d = (
            pad[1:-1, 1:-1]
            + pad[:-2, 1:-1]
            + pad[2:, 1:-1]
            + pad[1:-1, :-2]
            + pad[1:-1, 2:]
        ) / 5.0
        level_noise = 0.5 + 0.5 * self.cave_level_noise(position)
        depth_noise = 0.5 + 0.5 * self.cave_depth_noise(position)
        upper_sep = 5.0 + 6.0 * level_noise
        deep_sep = 10.0 + 12.0 * depth_noise
        roof_mid = roof
        roof_upper = numpy.clip(roof_mid + upper_sep, min_roof, max_roof)
        roof_deep = numpy.clip(roof_mid - deep_sep, min_roof, max_roof)
        depth_mid = 4.0 + 12.0 * density2d                          # 4..16-ish
        depth_upper = 3.0 + 8.0 * density2d
        depth_deep = 6.0 + 18.0 * density2d
        floor_mid = numpy.maximum(min_roof - 1.0, roof_mid - depth_mid)
        floor_upper = numpy.maximum(min_roof - 1.0, roof_upper - depth_upper)
        floor_deep = numpy.maximum(min_roof - 1.0, roof_deep - depth_deep)

        y_grid = numpy.arange(SECTOR_HEIGHT, dtype=float)[:, None, None]  # (H,1,1)
        elev_grid = elevation[None, :, :]                                 # (1,X,Z)
        cliff3d = cliff_mask[None, :, :]                                  # (1,X,Z)
        land3d = (elevation > WATER_LEVEL)[None, :, :]                    # avoid flooding seafloor caves
        cliff_gate = float(getattr(config, "CAVE_CLIFF_DENSITY_GATE", 0.72))
        region3d = region_mask[None, :, :]
        area_mask = region3d | (cliff3d & (density2d[None, :, :] > cliff_gate))

        vertical_mid = (y_grid >= floor_mid[None, :, :]) & (y_grid <= roof_mid[None, :, :])
        vertical_upper = (y_grid >= floor_upper[None, :, :]) & (y_grid <= roof_upper[None, :, :])
        vertical_deep = (y_grid >= floor_deep[None, :, :]) & (y_grid <= roof_deep[None, :, :])
        vertical_any = vertical_mid | vertical_upper | vertical_deep
        below_surface = y_grid < (elev_grid - 2.0)
        density_mid = density2d > 0.35
        density_upper = density2d > 0.5
        density_deep = density2d > 0.3
        column_gate = has_space[None, :, :] & land3d

        carve = (
            (vertical_mid & density_mid[None, :, :])
            | (vertical_upper & density_upper[None, :, :])
            | (vertical_deep & density_deep[None, :, :])
        )
        carve &= below_surface & column_gate & area_mask

        # Occasional surface outlets only when the roof is comfortably below the surface.
        outlet_band = (y_grid >= (elev_grid - self.cave_surface_outlet_depth)) & (y_grid <= (elev_grid - 1.0))
        deep_roof = (elevation - roof_mid) > 2.5
        outlet_gate = density2d > 0.7
        carve |= outlet_band & outlet_gate[None, :, :] & column_gate & cliff3d & deep_roof[None, :, :]

        # Small lateral dilation to connect nearby passages while respecting height masks.
        for _ in range(int(getattr(config, "CAVE_DILATE_ITERS", 3))):
            neighbor = numpy.zeros_like(carve, dtype=bool)
            neighbor[:, 1:, :] |= carve[:, :-1, :]
            neighbor[:, :-1, :] |= carve[:, 1:, :]
            neighbor[:, :, 1:] |= carve[:, :, :-1]
            neighbor[:, :, :-1] |= carve[:, :, 1:]
            spread = neighbor & vertical_any & below_surface & column_gate & area_mask
            carve |= spread

        # Thin connectors between levels in cave regions.
        xs, zs = self._world_axes(position)
        wxg, wzg = numpy.meshgrid(xs, zs, indexing="ij")
        connector_spacing = int(getattr(config, "CAVE_CONNECTOR_SPACING", 16))
        connector_spacing = max(4, connector_spacing)
        hashv = (wxg.astype(numpy.int64) * 73856093 + wzg.astype(numpy.int64) * 19349663 + int(self.seed)) % connector_spacing
        connector_mask = region_mask & (density2d > 0.55) & (hashv == 0)
        connector3d = connector_mask[None, :, :] & column_gate & land3d & below_surface
        conn_upper = connector3d & (y_grid >= roof_mid[None, :, :]) & (y_grid <= floor_upper[None, :, :])
        conn_deep = connector3d & (y_grid >= roof_deep[None, :, :]) & (y_grid <= floor_mid[None, :, :])
        carve |= conn_upper | conn_deep

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
        if carve.any():
            road_protect = (blocks == COBBLE) | (blocks == PLANK) | (blocks == JACK)
            carve = carve & (~road_protect)
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

    def _place_cave_mushrooms(self, position, blocks, ground_h, water_mask, decor_noise, decor_detail):
        """Sprinkle glowing mushrooms on dark cave floors at low density."""
        if self.fast:
            return
        sx = SECTOR_SIZE
        sz = SECTOR_SIZE
        # Ensure deterministic, non-negative seed for RNG
        px = int(position[0])
        pz = int(position[2] if len(position) > 2 else (position[1] if len(position) > 1 else 0))
        mix = numpy.uint64(0x9E3779B97F4A7C15)
        mix ^= numpy.uint64(numpy.int64(self.seed))
        mix ^= numpy.uint64(numpy.int64(px * 928371))
        mix ^= numpy.uint64(numpy.int64(pz * 97241))
        rng_seed = int(mix)
        rng = numpy.random.default_rng(rng_seed)
        stone_like = {STONE, COBBLE, COAL_ORE, IRON_ORE, GOLD_ORE, DIAMOND_ORE, REDSTONE_ORE, EMERALD_ORE}
        for x in range(1, sx - 1):
            for z in range(1, sz - 1):
                if water_mask[x, z]:
                    continue
                max_floor = min(ground_h[x, z] - 2, SECTOR_HEIGHT - 2)
                if max_floor <= 2:
                    continue
                column = blocks[:, x, z]
                skylight = True
                sky_reaches = numpy.zeros(SECTOR_HEIGHT, dtype=bool)
                for y in range(SECTOR_HEIGHT - 1, -1, -1):
                    if column[y] != 0:
                        skylight = False
                    else:
                        sky_reaches[y] = skylight
                # scan upward from the floor for a single candidate per column
                for y in range(2, max_floor):
                    if column[y] != 0:
                        continue
                    if sky_reaches[y]:
                        continue  # skip skylit pockets
                    floor_id = column[y - 1]
                    if floor_id not in stone_like:
                        continue
                    # need a ceiling nearby to feel cave-like darkness
                    ceiling_band = column[y + 1:min(SECTOR_HEIGHT, y + 20)]
                    if not (ceiling_band != 0).any():
                        continue
                    # Require headroom and at least one exposed horizontal face so it isn't buried in a wall.
                    if y + 1 >= SECTOR_HEIGHT or column[y + 1] != 0:
                        continue
                    neighbors = [
                        blocks[y, x + 1, z],
                        blocks[y, x - 1, z],
                        blocks[y, x, z + 1],
                        blocks[y, x, z - 1],
                    ]
                    if all(n != 0 for n in neighbors):
                        continue
                    base_chance = 0.002
                    density_boost = max(0.0, decor_detail[x, z]) * 0.01
                    stripe_gate = 0.6 + 0.4 * (decor_noise[x, z] > 0.15)
                    chance = (base_chance + density_boost) * stripe_gate
                    if rng.random() < chance:
                        if (blocks[y, x-1:x+2, z-1:z+2] == MUSHROOM).any():
                            continue
                        column[y] = MUSHROOM
                        world_pos = (int(position[0] + x), int(y), int(position[2] + z))
                        local_pos = (int(x), int(y), int(z))
                        _record_mushroom_hint(world_pos, position, local_pos)
                        break

    def generate(self, position):
        elevation, canyon_noise, road_height = self._height_field(position)
        if False:
            elevation = self._apply_spawn_bias(elevation, position)
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

        river_plan = None
        if self.enable_rivers:
            river_plan = self._compute_river_plan(position, elevation)
            if river_plan:
                biome[river_plan['mask']] = 5

        rural_plan = None
        trail_plan = None
        if self.enable_roads:
            rural_plan, trail_plan = self._transport_plans_vectorized(position, elevation, road_height=road_height)

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
                target_h = int(round(numpy.mean(elevation[max(x-1,0):min(x+w+1, SECTOR_SIZE), max(z-1,0):min(z+d+1, SECTOR_SIZE)])))
                if target_h + 10 >= SECTOR_HEIGHT:
                    continue
                if not self._reserve_area(building_mask, x, z, w, d, padding=2):
                    continue
                self._clear_area(blocks=None, elevation=elevation, x=x, z=z, width=w, depth=d, target_height=target_h)
                building_spots.append((x, z, target_h, btype, size, building_cluster[x, z]))

        sx = SECTOR_SIZE
        sz = SECTOR_SIZE
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

        if river_plan:
            # Carve river beds and fill with water along the planned river mask.
            self._apply_river_columns(blocks, river_plan)
            # Treat river tiles as water for later decoration / cave logic (even if above sea level).
            water_mask = water_mask | river_plan['mask']

        # Ores and caves after terrain fill, before roads/trails.
        if not self.fast:
            self._place_ores(position, blocks, elevation, cliff_mask)
            self._carve_caves(position, blocks, elevation, cliff_mask)
            self._place_cave_mushrooms(position, blocks, ground_h, water_mask, decor_noise, decor_detail)

        road_mask = None
        trail_mask = None
        if trail_plan:
            trail_mask = trail_plan["mask"]
            # Lay dirt trails
            self._apply_trails_vec(blocks, trail_plan)
        if rural_plan:
            # Lay cobblestone roads after trails so roads overwrite crossings.
            road_mask = rural_plan["mask"]
            if road_mask is not None:
                up = numpy.zeros_like(road_mask);   up[:, 1:] = road_mask[:, :-1]
                dn = numpy.zeros_like(road_mask);   dn[:, :-1] = road_mask[:, 1:]
                lt = numpy.zeros_like(road_mask);   lt[1:, :] = road_mask[:-1, :]
                rt = numpy.zeros_like(road_mask);   rt[:-1, :] = road_mask[1:, :]
                ul = numpy.zeros_like(road_mask);   ul[1:, 1:] = road_mask[:-1, :-1]
                ur = numpy.zeros_like(road_mask);   ur[:-1, 1:] = road_mask[1:, :-1]
                dl = numpy.zeros_like(road_mask);   dl[1:, :-1] = road_mask[:-1, 1:]
                dr = numpy.zeros_like(road_mask);   dr[:-1, :-1] = road_mask[1:, 1:]
                road_mask = road_mask | up | dn | lt | rt | ul | ur | dl | dr
            self._apply_rural_roads_vec(blocks, rural_plan)

        # Per-column decorations that need decisions.
        for x in range(sx):
            for z in range(sz):
                if road_mask is not None and road_mask[x, z]:
                    continue
                if trail_mask is not None and trail_mask[x, z]:
                    continue
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

        # Place planned buildings and connect them with paths.
        centers = []
        for x, z, gh, btype, size, orient_val in building_spots:
            w, d = size
            self._clear_area(blocks, elevation, x, z, w, d, gh)
            self._place_building_by_type(blocks, x, z, gh, btype, orient_val)
            centers.append((x + w // 2, gh + 1, z + d // 2))
        for i in range(1, len(centers)):
            self._draw_path(blocks, centers[i - 1], centers[i])

        if False:
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

        return blocks.swapaxes(0,1)


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

if __name__ == "__main__":
    import numpy as np

    # --- Instantiate generator in the same way your module intends ---
    gen = initialize_biome_map_generator(seed=12345)
    if gen is None:
        # If initializer stores a global instead of returning
        try:
            gen = biome_generator
        except NameError as e:
            raise RuntimeError("initialize_biome_map_generator() returned None and biome_generator not found") from e

    # --- Helpers ---
    def compute_mask_for_sector(ox: int, oz: int):
        """
        Builds a simple elevation plane and runs _compute_trail_plan_vec, returning the mask.
        Uses a 3-length 'position' tuple so _sector_origin reads x/z correctly.
        """
        sx = SECTOR_SIZE
        sz = SECTOR_SIZE

        # Flat-ish elevation well above WATER_LEVEL, within SECTOR_HEIGHT range
        # (the relaxer/clipping won't matter for the mask checks, but avoid water logic surprises)
        elev = np.full((sx, sz), max(WATER_LEVEL + 5, 10), dtype=np.int16)

        # IMPORTANT: position must be length>=3 so _sector_origin uses [0] and [2]
        pos = (ox, 0, oz)

        plan = gen._compute_trail_plan_vec(pos, elev)
        if plan is None:
            return np.zeros((sx, sz), dtype=bool)
        return plan["mask"].astype(bool)

    def seam_indices_core_x():
        """
        For a sector with core axes xs = [ox .. ox+15]:
          core world-x = [ox .. ox+15] => local ix = [0 .. 15]
        The seam between sector O (ox) and X (ox+16) is between world-x=ox+15 and world-x=ox+16.

        So compare:
          O core right edge: world-x = ox+15 => ix = 15
          X core left edge:  world-x = ox+16 => ix = 0
        """
        ix_O = SECTOR_SIZE - 1
        ix_X = 0
        inward_O = SECTOR_SIZE - 2   # one cell inward from O seam edge
        inward_X = 1    # one cell inward from X seam edge
        return ix_O, ix_X, inward_O, inward_X

    def seam_indices_core_z():
        """
        For z axis with core zs = [oz .. oz+15]:
          core world-z = [oz .. oz+15] => local iz = [0 .. 15]
        Seam between sector O (oz) and Z (oz+16) is between world-z=oz+15 and world-z=oz+16.

        Compare:
          O core top edge:  world-z = oz+15 => iz = 15
          Z core bottom:    world-z = oz+16 => iz = 0
        """
        iz_O = SECTOR_SIZE - 1
        iz_Z = 0
        inward_O = SECTOR_SIZE - 2
        inward_Z = 1
        return iz_O, iz_Z, inward_O, inward_Z

    def classify_dir(mask: np.ndarray, x: int, z: int):
        """
        Classify local direction at a mask cell by checking neighbors.
        Returns one of: 'x', 'z', 'both', 'iso', or 'off' (if mask is False).
        """
        if not mask[x, z]:
            return "off"
        sx, sz = mask.shape

        x_conn = False
        z_conn = False

        if x > 0 and mask[x - 1, z]:
            x_conn = True
        if x + 1 < sx and mask[x + 1, z]:
            x_conn = True

        if z > 0 and mask[x, z - 1]:
            z_conn = True
        if z + 1 < sz and mask[x, z + 1]:
            z_conn = True

        if x_conn and z_conn:
            return "both"
        if x_conn:
            return "x"
        if z_conn:
            return "z"
        return "iso"

    def compare_seam_core_x(maskO: np.ndarray, maskX: np.ndarray, label="O vs +X"):
        ixO, ixX, inO, inX = seam_indices_core_x()
        sz = maskO.shape[1]

        # (1) Direct seam continuation: do we have trail on both sides at same z?
        seamO = maskO[ixO, :]
        seamX = maskX[ixX, :]
        mism = seamO != seamX
        n_mism = int(np.count_nonzero(mism))
        total = seamO.size

        # (2) Direction consistency: compare local direction one-step inward on each side
        # Only evaluate where BOTH sides have seam trail (continuation exists).
        dir_mism = 0
        dir_total = 0
        examples = []

        for z in range(sz):
            if seamO[z] and seamX[z]:
                dir_total += 1
                dO = classify_dir(maskO, inO, z)  # inward cell in O
                dX = classify_dir(maskX, inX, z)  # inward cell in X
                # For a cross-seam segment, you'd expect both sides to be 'x' or 'both'
                # (i.e., have x-connectivity). If one side is 'z' only, it is "misoriented".
                okO = (dO in ("x", "both"))
                okX = (dX in ("x", "both"))
                if (okO != okX) or (not okO) or (not okX):
                    dir_mism += 1
                    if len(examples) < 5:
                        examples.append((z, dO, dX))

        print(f"\n[{label}] core seam x-check")
        print(f"  seam mask mismatches: {n_mism} / {total}")
        print(f"  seam direction issues (inward cells): {dir_mism} / {max(dir_total,1)}")
        if examples:
            print("  examples (z, dir_in_O, dir_in_X):")
            for ex in examples:
                print("   ", ex)

    def compare_seam_core_z(maskO: np.ndarray, maskZ: np.ndarray, label="O vs +Z"):
        izO, izZ, inO, inZ = seam_indices_core_z()
        sx = maskO.shape[0]

        seamO = maskO[:, izO]
        seamZ = maskZ[:, izZ]
        mism = seamO != seamZ
        n_mism = int(np.count_nonzero(mism))
        total = seamO.size

        dir_mism = 0
        dir_total = 0
        examples = []

        for x in range(sx):
            if seamO[x] and seamZ[x]:
                dir_total += 1
                dO = classify_dir(maskO, x, inO)  # inward cell in O
                dZ = classify_dir(maskZ, x, inZ)  # inward cell in Z
                # For a cross-seam segment, you'd expect both sides to be 'z' or 'both'
                okO = (dO in ("z", "both"))
                okZ = (dZ in ("z", "both"))
                if (okO != okZ) or (not okO) or (not okZ):
                    dir_mism += 1
                    if len(examples) < 5:
                        examples.append((x, dO, dZ))

        print(f"\n[{label}] core seam z-check")
        print(f"  seam mask mismatches: {n_mism} / {total}")
        print(f"  seam direction issues (inward cells): {dir_mism} / {max(dir_total,1)}")
        if examples:
            print("  examples (x, dir_in_O, dir_in_Z):")
            for ex in examples:
                print("   ", ex)

    # --- Run the 3-sector test ---
    print("Computing masks for 3 sectors: O=(0,0), X=(+16,0), Z=(0,+16) ...")
    maskO = compute_mask_for_sector(0, 0)
    maskX = compute_mask_for_sector(SECTOR_SIZE, 0)
    maskZ = compute_mask_for_sector(0, SECTOR_SIZE)

    compare_seam_core_x(maskO, maskX, label="Origin vs +X sector")
    compare_seam_core_z(maskO, maskZ, label="Origin vs +Z sector")

    print("\nDone.")
