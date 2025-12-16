import math

DIST = 16*12

TICKS_PER_SEC = 60
TARGET_FPS = 60

# UPLOAD_TRIANGLE_CHUNK = 5000
# UPLOAD_TRIANGLE_BUDGET = 10000

# Size of sectors used to ease block loading.
SECTOR_SIZE = 16 #width and depth (x and z)
SECTOR_HEIGHT = 256 #height of world (y)
LOADED_SECTORS = DIST//SECTOR_SIZE + 1 #number of sections in (x,z) directions to load sectors for
LOAD_RADIUS = LOADED_SECTORS + 1
KEEP_RADIUS = LOAD_RADIUS*2

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

# Debug: skip world loading and render a single block in front of the player.
DEBUG_SINGLE_BLOCK = False

# Sector streaming / seam rebuild behavior
# How many deferred seam rebuilds to process per tick (None for unlimited).
MAX_SEAM_REBUILDS_PER_TICK = 1

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
ENABLE_ROAD_NETWORKS = False
RIVER_NETWORK_SPACING = 320
ROAD_NETWORK_SPACING = 220
ROAD_MAX_GRADE = 1.25
