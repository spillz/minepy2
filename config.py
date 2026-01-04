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
MAP_GEN_SECTOR_SIZE = 3*SECTOR_SIZE
MAP_GEN_PAD_SIZE = 1  # extra XZ boundary blocks for mapgen; set 0 to disable
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

PLAYER_HEIGHT = 1.8

# Entity spawning
ENTITY_MAX = 6
ENTITY_SPAWN_CHANCE = 0.01
ENTITY_SPAWN_RADIUS = 3

SERVER_IP = None
SERVER_PORT = 20226

LOADER_IP = 'localhost'
LOADER_PORT = 20230
# Send mesh (vt_data) from the loader process instead of building in world_proxy.
LOADER_SEND_MESH = False

# Debug: skip world loading and render a single block in front of the player.
DEBUG_SINGLE_BLOCK = False

# Mesh build worker counts (edit queue is higher priority).
MESH_EDIT_WORKERS = 1

# Single long-lived mesh worker instead of a thread pool.
MESH_SINGLE_WORKER = True

# Upload throttling: max triangles per sector upload chunk (None for full).
UPLOAD_TRIANGLE_CHUNK = 4000
# Minimum per-frame upload budget (milliseconds).
UPLOAD_MIN_BUDGET_MS = 1.0

# Patch meshes are a temporary visual hack; disable when sync rebuilds are fast enough.
USE_PATCH_MESH = False

# Enable ANSI colors in logs.
LOG_COLOR = True
# Log file path (None or "" to disable file output).
LOG_FILE_PATH = "log.txt"
# Log file path for loader process (None or "" to use LOG_FILE_PATH).
LOG_LOADER_FILE_PATH = None
# When True, append to existing log; when False, truncate on first log (main process).
LOG_FILE_APPEND = True

# Log main-loop timings and frame boundaries.
LOG_MAIN_LOOP = True
# Log detailed map generation timings.
LOG_MAPGEN_TIMINGS = True
# Log per-stage cave carving timings.
LOG_MAPGEN_CAVE_TIMINGS = True

# Enable per-frame HUD stats collection (can be disabled to reduce overhead).
HUD_STATS_ENABLED = False

# Log queue/inflight state (loader + mesh).
LOG_QUEUE_STATE = True

# Log missing sectors around the player (3x3).
LOG_MISSING_SECTORS = True
LOG_MISSING_SECTORS_EVERY_N_FRAMES = 30


# Debug: log load candidate ordering and loader request/response flow.
LOG_LOAD_CANDIDATES = False
LOG_LOAD_CANDIDATES_EVERY_N_FRAMES = 30
LOG_LOADER_FLOW = False

# Logging for mesh activity.
MESH_LOG = False

# HUD probe/debug info (void distance, mushroom lookup).
HUD_PROBE_ENABLED = False
HUD_PROBE_EVERY_N_FRAMES = 15

# HUD profiling (break down draw_label timings).
HUD_PROFILE = True
HUD_PROFILE_LOG_S = 1.0
HUD_PROFILE_SPIKE_MS = 20.0
HUD_DETAIL_REFRESH_S = 1.0

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
AMBIENT_LIGHT = 0.1  # minimum light level (0-1 range)
MAX_LIGHT = 15  # integer light level range (0..MAX_LIGHT)
SKY_INTENSITY = 1.0  # global sky light multiplier (0..1)
SKY_SIDEFILL_ENABLED = True  # indirect sky light from sides/seams
TORCH_FILL_ENABLED = True # torch light propagation to nearby blocks
LIGHT_PROPAGATION_BFS = True  # use numpy frontier BFS instead of dense relaxation
# Send only boundary light values to neighbors to reduce propagation cost.
LIGHT_OUTGOING_BOUNDARY_ONLY = False

# Day/night cycle (visual only; no skylight recompute).
DAY_NIGHT_CYCLE_ENABLED = True
DAY_LENGTH_SECONDS = 1200.0
DAY_START_PHASE = 0.0  # 0.0 = midday, 0.5 = midnight
DAY_LIGHT_CURVE = 1.0  # >1.0 darkens nights, <1.0 brightens nights
SUN_LIGHT_DIR = (0.35, 1.0, 0.65)  # midday light direction (x, y, z)
DAY_AMBIENT_LIGHT = 0.18
NIGHT_AMBIENT_LIGHT = 0.06
DAY_SKY_INTENSITY = 1.0
NIGHT_SKY_INTENSITY = 0.25
DAY_FOG_COLOR = (0.5, 0.69, 1.0)
NIGHT_FOG_COLOR = (0.02, 0.03, 0.06)
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
# Skip expensive cave carving; keeps terrain but no underground caverns.
MAPGEN_CAVES_ENABLED = True
# Sparse ore placement toggle.
MAPGEN_ORES_ENABLED = True
# Base spacing (in blocks) for ore columns; higher = fewer ores.
MAPGEN_ORE_SPACING = 10
# Cave tuning: lower max roof height and lower density for speed.
CAVE_MAX_ROOF = 50
CAVE_REGION_THRESHOLD = 0.30
CAVE_DILATE_ITERS = 2
CAVE_CONNECTOR_SPACING = 20
CAVE_BREACH_SPACING = 48
# Macro features: large-scale river and road networks.
ENABLE_RIVER_NETWORKS = False
ENABLE_ROAD_NETWORKS = True
RIVER_NETWORK_SPACING = 320
ROAD_NETWORK_SPACING = 220
ROAD_MAX_GRADE = 1.25
RURAL_ROAD_ALLOW_WATER = True
RURAL_ROAD_WATER_PILLAR_CAP = 16
TRAIL_ALLOW_WATER = False
RURAL_ROAD_HEIGHT_PRIORITY = "terrain"     # "continuity" or "terrain"
RURAL_ROAD_HEIGHT_BLEND = 0.8              # 0=macro only, 1=local only
RURAL_ROAD_CLIFF_MODE = "tunnel"           # "tunnel" or "dead_end"
RURAL_ROAD_MAX_CUT_CLIFF = 18              # max cut when tunneling through cliffs
RURAL_ROAD_CLIFF_GRAD = 0.85               # gradient threshold for cliff masking
