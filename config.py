import math

DIST = 16*12

TICKS_PER_SEC = 60
TARGET_FPS = 60
LOW_FPS = 30
PHYSICS_SUBSTEPS_MAX = 2
UPDATE_SLOW_LOG_MS = 10.0

# UPLOAD_TRIANGLE_CHUNK = 5000
# UPLOAD_TRIANGLE_BUDGET = 10000

# Size of sectors used to ease block loading.
SECTOR_SIZE = 16 #width and depth (x and z)
SECTOR_HEIGHT = 256 #height of world (y)
LOADED_SECTORS = DIST//SECTOR_SIZE + 1 #number of sections in (x,z) directions to load sectors for
LOAD_RADIUS = LOADED_SECTORS + 1
KEEP_RADIUS = LOAD_RADIUS*2
READY_SECTOR_RADIUS = 1  # 1 waits for 3x3; set 0 to only require current sector
READY_SECTOR_FRACTION = 1.0  # fallback when radius is None

WALKING_SPEED = 5
FLYING_SPEED = 15

WATER_COLOR = [40,90,128]
WATER_ALPHA  = 0.95
UNDERWATER_COLOR =  [70,128,168, 228]#

GRAVITY = 20.0
MAX_JUMP_HEIGHT = 1.0 # About the height of a block.
# To derive the formula for calculating jump speed, first solve
#    v_t = v_0 + a * t
# for the time at which you achieve maximum height, where a is the acceleration
# due to gravity and v_t = 0. This gives:
#    t = - v_0 / a
# Use t and the desired MAX_JUMP_HEIGHT to solve for v_0 (jump speed) in
#    s = s_0 + v_0 * t + (a * t^2) / 2
JUMP_SPEED = math.sqrt(2 * GRAVITY * MAX_JUMP_HEIGHT)
TERMINAL_VELOCITY = 50

PLAYER_HEIGHT = 2

SERVER_IP = None
SERVER_PORT = 20226

LOADER_IP = 'localhost'
LOADER_PORT = 20230
# Send mesh (vt_data) from the loader process instead of building in world_proxy.
LOADER_SEND_MESH = False
# Send per-block light field from the loader process even when mesh is client-built.
LOADER_SEND_LIGHT = True

# Debug: skip world loading and render a single block in front of the player.
DEBUG_SINGLE_BLOCK = False

# Mesh build worker counts (edit queue is higher priority).
MESH_EDIT_WORKERS = 1

# Single long-lived mesh worker instead of a thread pool.
MESH_SINGLE_WORKER = True

# Upload throttling: max triangles per sector upload chunk (None for full).
UPLOAD_TRIANGLE_CHUNK = 4000

# Lighting policy: defer lighting until neighbors are present; reuse loader light if available.
DEFER_LIGHTING_UNTIL_NEIGHBORS = True
DEFER_LIGHTING_REQUIRE_DIAGONALS = True
REUSE_LIGHTING_WHEN_AVAILABLE = True

# Patch meshes are a temporary visual hack; disable when sync rebuilds are fast enough.
USE_PATCH_MESH = False

# Mesh readiness: allow meshing without waiting for neighbors for contiguous build-out.
MESH_READY_REQUIRE_NEIGHBORS = False
MESH_READY_REQUIRE_DIAGONALS = False

# When True, wait for neighbors before meshing so lighting can be combined once.
MESH_WAIT_FOR_NEIGHBORS = True
MESH_WAIT_REQUIRE_DIAGONALS = False

# Strict terrain pipeline: seam sync -> light combine -> mesh.
STRICT_TERRAIN_PIPELINE = True

# Enable ANSI colors in logs.
LOG_COLOR = True

# Log main-loop timings and frame boundaries.
LOG_MAIN_LOOP = True

# Log queue/inflight state (loader + mesh).
LOG_QUEUE_STATE = True

# Log missing sectors around the player (3x3).
LOG_MISSING_SECTORS = True
LOG_MISSING_SECTORS_EVERY_N_FRAMES = 30

# Log strict terrain pipeline stage and candidate counts.
LOG_STRICT_PIPELINE = True
LOG_STRICT_PIPELINE_EVERY_N_FRAMES = 30

# Show placeholder blocks while waiting for strict pipeline stages.
STRICT_SHOW_PLACEHOLDER = True

# Log edge mismatch diagnostics for a nearby unmeshed sector.
LOG_SEAM_DIAGNOSTICS = True
LOG_SEAM_DIAGNOSTICS_EVERY_N_FRAMES = 60

# When True, only emit strict/seam logs (and warnings/errors).
LOG_STRICT_ONLY = False
LOG_STRICT_TRACE = True
LOG_STRICT_TRACE_EVERY_N_FRAMES = 1

# Logging for mesh activity.
MESH_LOG = False

# HUD probe/debug info (void distance, mushroom lookup).
HUD_PROBE_ENABLED = False
HUD_PROBE_EVERY_N_FRAMES = 15

# Loader inflight requests (sector loads).
LOADER_MAX_INFLIGHT = 1

# When True, never send any loader request while another is in flight.
LOADER_STRICT_INFLIGHT = True

# When True, pause loader sends while mesh work is pending on loaded sectors.
LOADER_BLOCK_ON_MESH_BACKLOG = True

# Recompute load candidates at most every N ms while staying in same sector.
LOAD_CANDIDATE_REFRESH_MS = 250

# Recompute mesh candidates at most every N ms while staying in same sector.
MESH_CANDIDATE_REFRESH_MS = 250

# Small penalty for sectors outside the frustum (keeps near-to-far contiguous).
FRUSTUM_PRIORITY_PENALTY = 0.1

# Scale for camera-facing priority; set 0 to remove directional bias.
VIEW_PRIORITY_SCALE = 0.0

# Sector streaming / seam rebuild behavior
# How many deferred seam rebuilds to process per tick (None for unlimited).
MAX_SEAM_REBUILDS_PER_TICK = None

# Lighting settings
LIGHT_DECAY = 0.1  # per-step attenuation for flood-fill lighting
AMBIENT_LIGHT = 0.1  # minimum light level (0-1 range)
# Ambient occlusion settings (darken inner edges/corners of exposed faces).
AO_ENABLED = True
AO_STRENGTH = 0.6
AO_MIN = 0.35
AO_GAMMA = 1.0
AO_FORCE_DEBUG_PATTERN = False
AO_DEBUG = False
AO_DEBUG_SHADER = False
AO_DEBUG_SHADER_FORCE = False

# Terrain generation
USE_EXPERIMENTAL_BIOME_GEN = True
# Performance toggle for biome generator: skip caves/ores and limit structures/trees.
BIOME_FAST_MODE = False
# Macro features: large-scale river and road networks.
ENABLE_RIVER_NETWORKS = False
ENABLE_ROAD_NETWORKS = True
RIVER_NETWORK_SPACING = 320
ROAD_NETWORK_SPACING = 220
ROAD_MAX_GRADE = 1.25
