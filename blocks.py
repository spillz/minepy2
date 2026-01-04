import numpy
from config import WATER_COLOR
from util import (
    tex_coords,
    FACES,
    cb_v,
    de_v,
    cb_v_half,
    cb_v_cake,
    wall_plank_south,
    wall_plank_west,
    wall_plank_north,
    wall_plank_east,
    window_pane_south,
    window_pane_west,
    window_pane_north,
    window_pane_east,
    door_south,
    door_west,
    door_north,
    door_east,
    torch_south,
    torch_west,
    torch_north,
    torch_east,
    ladder_south,
    ladder_west,
    ladder_north,
    ladder_east,
    STAIR_FACE_DIRS,
    stair_south,
    stair_west,
    stair_north,
    stair_east,
    stair_south_ud,
    stair_west_ud,
    stair_north_ud,
    stair_east_ud,
)

TEXTURE_PATH = 'texture_fv.png'

gr = [50,150,70]
white = numpy.tile(numpy.array([255,255,255]),6*4).reshape(6,3*4)
green = numpy.tile(numpy.array(gr),6*4).reshape(6,3*4)
grass_top = numpy.array([77,244,44]*4+[255,255,255]*5*4).reshape(6,3*4)
water_blue = numpy.tile(numpy.array(WATER_COLOR),6*4).reshape(6,3*4)


class Block(object):
    #String name of the block
    name = None
    #tuple of texture coordinates
    coords = None
    #Solid blocks are collision sites
    solid = True
    colors = white
    texture_fn = tex_coords
    vertices = cb_v
    # Occlusion flags: solid/opaque blocks should occlude neighbors; transparent cutouts may selectively occlude same-type.
    occludes = True
    occludes_same = False
    # Which face to show in the HUD picker; defaults to top (0).
    picker_face = 0
    # Minimum on-face glow (0-1) applied after world lighting; useful for emissive look without bright emission.
    glow = 0.0
    # Inventory visibility for block selector.
    show_in_inventory = True
    # Collision behavior: None (no collision), "full" (1x1x1), or "mesh" (use vertices AABB).
    collision = None
    # Render all faces regardless of neighbor occlusion (useful for thin meshes).
    render_all_faces = False

class Decoration(object):
    vertices = de_v
    solid = False
    render_all_faces = True

class DirtWithGrass(Block):
    name = 'Grass'
    coords = ((0, 15), (2, 15), (3, 15))
    colors = grass_top

class Dirt(Block):
    name = 'Dirt'
    coords = ((2,15), )

class Leaves(Block):
    name = 'Leaves'
    coords = ((4, 7), )
    colors = green
    solid = False
    occludes = False
    occludes_same = True

class Sand(Block):
    name = 'Sand'
    coords = ((2, 14), (2, 14), (2, 14))

class Brick(Block):
    name = 'Brick'
    coords = ((7, 15), (7, 15), (7, 15)) #3 brick

class Stone(Block):
    name = 'Stone'
    coords = ((1,15), )

class CobbleStone(Block):
    name = 'Cobblestone'
    coords = ((0, 14), (0, 14), (0, 14))  #4 stone

class BetterStone(Block):
    name = 'Betterstone'
    coords = ((3, 14), )  #4 stone

class IronBlock(Block):
    name = 'Iron Block'
    coords = ((6, 14),)

class Wood(Block):
    name = 'Wood'
    coords = ((5, 14), (5, 14), (4, 14))  #5 wood
    picker_face = 2

class Plank(Block):
    name = 'Plank'
    coords = ((4, 15), (4, 15), (4, 15))  #6 plank
    vertices = cb_v

class PlankSouth(Plank):
    name = 'Plank South'
    vertices = cb_v
    show_in_inventory = False

class PlankWest(Plank):
    name = 'Plank West'
    vertices = cb_v
    show_in_inventory = False

class PlankNorth(Plank):
    name = 'Plank North'
    vertices = cb_v
    show_in_inventory = False

class PlankEast(Plank):
    name = 'Plank East'
    vertices = cb_v
    show_in_inventory = False

class CraftingTable(Block):
    name = 'Crafting Table'
    coords = ((11, 13), (4,15), (11, 12), (11, 12), (12, 12))

class Pumpkin(Block):
    name = 'Pumpkin'
    coords = ((6, 9), (6, 8), (7, 8), (6,8))

class JackOLantern(Block):
    name = 'Jack O\'Lantern'
    coords = ((6, 9), (6, 8), (8, 8), (6,8))
    picker_face = 2  # show the carved face instead of the top

class Rose(Decoration, Block):
    name = 'Rose'
    coords = ((12,15), (12,15), (12,15), (12,15))
    solid = False
    occludes = False

class Mushroom(Decoration, Block):
    name = 'Mushroom'
    coords = ((4, 1), (4, 1), (4, 1), (4, 1))
    solid = False
    occludes = False
    glow = 0.75

class GobbleDeBlock(Block):
    name = 'Gobbledeblock'
    coords = ((9,14), (13,1), (13,8), (8,9), (15,5), (0,12))

class IronOre(Block):
    name = 'Iron Ore'
    coords = ((1,13),)

class GoldOre(Block):
    name = 'Gold Ore'
    coords = ((0,13),)

class CoalOre(Block):
    name = 'Coal Ore'
    coords = ((2,13),)

class DiamondOre(Block):
    name = 'Diamond Ore'
    coords = ((2,12),)

class RedstoneOre(Block):
    name = 'Redstone Ore'
    coords = ((3,12),)

class EmeraldOre(Block):
    name = 'Emerald Ore'
    coords = ((11,5),)

class Bookshelf(Block):
    name = 'Bookshelf'
    coords = ((4,15), (4,15), (3,13))
    picker_face = 2

class TNT(Block):
    name = 'TNT'
    coords = ((9,15), (10,15), (8,15))
    picker_face = 2

class Cake(Block):
    name = 'Cake'
    coords = ((9,8), (12,8), (10,8))
    vertices = cb_v_cake
    solid = False
    occludes = False

class Water(Block):
    name = 'Water'
    coords = ((1, 1), (1, 1), (1, 1))
    # coords = ((3, 11), (3, 11), (3, 11))
    colors = water_blue
    solid = False
    occludes = False
    collision = None

class Door(Block):
    name = 'Door'
    coords = ((4, 15), (4, 15), (4, 15), (4, 15), (1, 10), (1, 10))
    picker_face = 4
    vertices = door_south
    occludes = False
    solid = True
    collision = 'mesh'
    render_all_faces = True

class DoorLower(Block):
    name = 'Door Lower'
    coords = ((4, 15), (4, 15), (4, 15), (4, 15), (1, 9), (1, 9))
    vertices = door_south
    occludes = False
    solid = True
    show_in_inventory = False
    collision = 'mesh'
    render_all_faces = True

class DoorUpper(Block):
    name = 'Door Upper'
    coords = ((4, 15), (4, 15), (4, 15), (4, 15), (1, 10), (1, 10))
    vertices = door_south
    occludes = False
    solid = True
    show_in_inventory = False
    collision = 'mesh'
    render_all_faces = True

class DoorLowerSouth(DoorLower):
    name = 'Door Lower South'
    vertices = door_south
    show_in_inventory = False

class DoorLowerWest(DoorLower):
    name = 'Door Lower West'
    vertices = door_west
    coords = ((4, 15), (4, 15), (1, 9), (1, 9), (4, 15), (4, 15))
    show_in_inventory = False

class DoorLowerNorth(DoorLower):
    name = 'Door Lower North'
    vertices = door_north
    show_in_inventory = False

class DoorLowerEast(DoorLower):
    name = 'Door Lower East'
    vertices = door_east
    coords = ((4, 15), (4, 15), (1, 9), (1, 9), (4, 15), (4, 15))
    show_in_inventory = False

class DoorUpperSouth(DoorUpper):
    name = 'Door Upper South'
    vertices = door_south
    show_in_inventory = False

class DoorUpperWest(DoorUpper):
    name = 'Door Upper West'
    vertices = door_west
    coords = ((4, 15), (4, 15), (1, 10), (1, 10), (4, 15), (4, 15))
    show_in_inventory = False

class DoorUpperNorth(DoorUpper):
    name = 'Door Upper North'
    vertices = door_north
    show_in_inventory = False

class DoorUpperEast(DoorUpper):
    name = 'Door Upper East'
    vertices = door_east
    coords = ((4, 15), (4, 15), (1, 10), (1, 10), (4, 15), (4, 15))
    show_in_inventory = False

class WindowPane(Block):
    name = 'Window Pane'
    coords = ((1, 12), (1, 12), (1, 12))
    vertices = window_pane_south
    solid = False
    occludes = False
    collision = 'mesh'

class WindowPaneSouth(WindowPane):
    name = 'Window Pane South'
    vertices = window_pane_south
    show_in_inventory = False

class WindowPaneWest(WindowPane):
    name = 'Window Pane West'
    vertices = window_pane_west
    show_in_inventory = False

class WindowPaneNorth(WindowPane):
    name = 'Window Pane North'
    vertices = window_pane_north
    show_in_inventory = False

class WindowPaneEast(WindowPane):
    name = 'Window Pane East'
    vertices = window_pane_east
    show_in_inventory = False

class WallTorch(Block):
    name = 'Wall Torch'
    coords = ((3.5-0.125/2, 9.5, 0.125, 0.125), (3.5-0.125/2, 9.0, 0.125, 0.125), (3.5-0.125/2, 9.0, 0.125, 0.625))
    vertices = torch_south
    solid = False
    occludes = False
    glow = 1.0
    collision = 'mesh'
    picker_face = 2

class WallTorchSouth(WallTorch):
    name = 'Wall Torch South'
    vertices = torch_south
    show_in_inventory = False

class WallTorchWest(WallTorch):
    name = 'Wall Torch West'
    vertices = torch_west
    show_in_inventory = False

class WallTorchNorth(WallTorch):
    name = 'Wall Torch North'
    vertices = torch_north
    show_in_inventory = False

class WallTorchEast(WallTorch):
    name = 'Wall Torch East'
    vertices = torch_east
    show_in_inventory = False

class Ladder(Block):
    name = 'Ladder'
    coords = ((4, 4), (4, 4), (3, 10))
    vertices = ladder_south
    solid = False
    occludes = False
    collision = None
    picker_face = 2
    render_all_faces = True

class LadderSouth(Ladder):
    name = 'Ladder South'
    vertices = ladder_south
    show_in_inventory = False

class LadderWest(Ladder):
    name = 'Ladder West'
    vertices = ladder_west
    show_in_inventory = False

class LadderNorth(Ladder):
    name = 'Ladder North'
    vertices = ladder_north
    show_in_inventory = False

class LadderEast(Ladder):
    name = 'Ladder East'
    vertices = ladder_east
    show_in_inventory = False

class Stair(Block):
    name = 'Stair'
    coords = ((4, 15), (4, 15), (4, 15))
    vertices = stair_south
    occludes = False
    collision = 'mesh'

class StairSouth(Stair):
    name = 'Stair South'
    vertices = stair_south
    show_in_inventory = False

class StairWest(Stair):
    name = 'Stair West'
    vertices = stair_west
    show_in_inventory = False

class StairNorth(Stair):
    name = 'Stair North'
    vertices = stair_north
    show_in_inventory = False

class StairEast(Stair):
    name = 'Stair East'
    vertices = stair_east
    show_in_inventory = False

class StairSouthUpsideDown(Stair):
    name = 'Stair South Upside Down'
    vertices = stair_south_ud
    show_in_inventory = False

class StairWestUpsideDown(Stair):
    name = 'Stair West Upside Down'
    vertices = stair_west_ud
    show_in_inventory = False

class StairNorthUpsideDown(Stair):
    name = 'Stair North Upside Down'
    vertices = stair_north_ud
    show_in_inventory = False

class StairEastUpsideDown(Stair):
    name = 'Stair East Upside Down'
    vertices = stair_east_ud
    show_in_inventory = False

class CobbleStair(Stair):
    name = 'Cobble Stair'
    coords = ((0, 14), (0, 14), (0, 14))
    occludes = False

class CobbleStairSouth(CobbleStair):
    name = 'Cobble Stair South'
    vertices = stair_south
    show_in_inventory = False

class CobbleStairWest(CobbleStair):
    name = 'Cobble Stair West'
    vertices = stair_west
    show_in_inventory = False

class CobbleStairNorth(CobbleStair):
    name = 'Cobble Stair North'
    vertices = stair_north
    show_in_inventory = False

class CobbleStairEast(CobbleStair):
    name = 'Cobble Stair East'
    vertices = stair_east
    show_in_inventory = False

class CobbleStairSouthUpsideDown(CobbleStair):
    name = 'Cobble Stair South Upside Down'
    vertices = stair_south_ud
    show_in_inventory = False

class CobbleStairWestUpsideDown(CobbleStair):
    name = 'Cobble Stair West Upside Down'
    vertices = stair_west_ud
    show_in_inventory = False

class CobbleStairNorthUpsideDown(CobbleStair):
    name = 'Cobble Stair North Upside Down'
    vertices = stair_north_ud
    show_in_inventory = False

class CobbleStairEastUpsideDown(CobbleStair):
    name = 'Cobble Stair East Upside Down'
    vertices = stair_east_ud
    show_in_inventory = False

class DirtStair(Stair):
    name = 'Dirt Stair'
    coords = ((2, 15), (2, 15), (2, 15))
    occludes = False

class DirtStairSouth(DirtStair):
    name = 'Dirt Stair South'
    vertices = stair_south
    show_in_inventory = False

class DirtStairWest(DirtStair):
    name = 'Dirt Stair West'
    vertices = stair_west
    show_in_inventory = False

class DirtStairNorth(DirtStair):
    name = 'Dirt Stair North'
    vertices = stair_north
    show_in_inventory = False

class DirtStairEast(DirtStair):
    name = 'Dirt Stair East'
    vertices = stair_east
    show_in_inventory = False

class DirtStairSouthUpsideDown(DirtStair):
    name = 'Dirt Stair South Upside Down'
    vertices = stair_south_ud
    show_in_inventory = False

class DirtStairWestUpsideDown(DirtStair):
    name = 'Dirt Stair West Upside Down'
    vertices = stair_west_ud
    show_in_inventory = False

class DirtStairNorthUpsideDown(DirtStair):
    name = 'Dirt Stair North Upside Down'
    vertices = stair_north_ud
    show_in_inventory = False

class DirtStairEastUpsideDown(DirtStair):
    name = 'Dirt Stair East Upside Down'
    vertices = stair_east_ud
    show_in_inventory = False

class PlankStair(Stair):
    name = 'Plank Stair'
    coords = ((4, 15), (4, 15), (4, 15))
    occludes = False

class PlankStairSouth(PlankStair):
    name = 'Plank Stair South'
    vertices = stair_south
    show_in_inventory = False

class PlankStairWest(PlankStair):
    name = 'Plank Stair West'
    vertices = stair_west
    show_in_inventory = False

class PlankStairNorth(PlankStair):
    name = 'Plank Stair North'
    vertices = stair_north
    show_in_inventory = False

class PlankStairEast(PlankStair):
    name = 'Plank Stair East'
    vertices = stair_east
    show_in_inventory = False

class PlankStairSouthUpsideDown(PlankStair):
    name = 'Plank Stair South Upside Down'
    vertices = stair_south_ud
    show_in_inventory = False

class PlankStairWestUpsideDown(PlankStair):
    name = 'Plank Stair West Upside Down'
    vertices = stair_west_ud
    show_in_inventory = False

class PlankStairNorthUpsideDown(PlankStair):
    name = 'Plank Stair North Upside Down'
    vertices = stair_north_ud
    show_in_inventory = False

class PlankStairEastUpsideDown(PlankStair):
    name = 'Plank Stair East Upside Down'
    vertices = stair_east_ud
    show_in_inventory = False

class GlassBlock(Block):
    name = 'Glass Block'
    coords = ((1, 12), (1, 12), (1, 12))
    solid = True
    occludes = False

# Explicit ordering keeps block IDs stable and ensures the initial inventory
# starts with grass instead of whichever subclass happens to register first.
BLOCKS = [
    DirtWithGrass,
    Dirt,
    Leaves,
    Sand,
    Brick,
    Stone,
    CobbleStone,
    BetterStone,
    IronBlock,
    Wood,
    Plank,
    PlankSouth,
    PlankWest,
    PlankNorth,
    PlankEast,
    CraftingTable,
    Pumpkin,
    JackOLantern,
    Rose,
    GobbleDeBlock,
    IronOre,
    GoldOre,
    CoalOre,
    DiamondOre,
    RedstoneOre,
    EmeraldOre,
    Bookshelf,
    TNT,
    Cake,
    Water,
    Mushroom,
    Door,
    DoorLowerSouth,
    DoorLowerWest,
    DoorLowerNorth,
    DoorLowerEast,
    DoorUpperSouth,
    DoorUpperWest,
    DoorUpperNorth,
    DoorUpperEast,
    WindowPane,
    WindowPaneSouth,
    WindowPaneWest,
    WindowPaneNorth,
    WindowPaneEast,
    WallTorch,
    WallTorchSouth,
    WallTorchWest,
    WallTorchNorth,
    WallTorchEast,
    Ladder,
    LadderSouth,
    LadderWest,
    LadderNorth,
    LadderEast,
    Stair,
    StairSouth,
    StairWest,
    StairNorth,
    StairEast,
    StairSouthUpsideDown,
    StairWestUpsideDown,
    StairNorthUpsideDown,
    StairEastUpsideDown,
    CobbleStair,
    CobbleStairSouth,
    CobbleStairWest,
    CobbleStairNorth,
    CobbleStairEast,
    CobbleStairSouthUpsideDown,
    CobbleStairWestUpsideDown,
    CobbleStairNorthUpsideDown,
    CobbleStairEastUpsideDown,
    DirtStair,
    DirtStairSouth,
    DirtStairWest,
    DirtStairNorth,
    DirtStairEast,
    DirtStairSouthUpsideDown,
    DirtStairWestUpsideDown,
    DirtStairNorthUpsideDown,
    DirtStairEastUpsideDown,
    PlankStair,
    PlankStairSouth,
    PlankStairWest,
    PlankStairNorth,
    PlankStairEast,
    PlankStairSouthUpsideDown,
    PlankStairWestUpsideDown,
    PlankStairNorthUpsideDown,
    PlankStairEastUpsideDown,
    GlassBlock,
]
i = 1
BLOCK_ID = {}
for x in BLOCKS:
    BLOCK_ID[x.name] = i
    i+=1
BLOCK_NORMALS = numpy.array(FACES)
BLOCK_COLORS = numpy.array([white] + [x.colors for x in BLOCKS])
BLOCK_TEXTURES = numpy.array([tex_coords((0,0),(0,0),(0,0))] + [tex_coords(*x.coords) for x in BLOCKS],dtype = numpy.float32)/4
_raw_vertices = [cb_v] + [x.vertices for x in BLOCKS]
_face_counts = [0]
for verts in _raw_vertices[1:]:
    _face_counts.append(int(verts.reshape(-1, 12).shape[0]))
MAX_FACES = max(max(_face_counts), 6)
BLOCK_FACE_COUNT = numpy.array(_face_counts, dtype=numpy.uint8)
BLOCK_VERTICES = numpy.zeros((len(BLOCKS) + 1, MAX_FACES, 12), dtype=numpy.float32)
for i, verts in enumerate(_raw_vertices):
    count = BLOCK_FACE_COUNT[i] if i < len(BLOCK_FACE_COUNT) else 0
    if count <= 0:
        continue
    reshaped = verts.reshape(-1, 12).astype(numpy.float32)
    BLOCK_VERTICES[i, :count] = reshaped[:count]
    if count < MAX_FACES:
        BLOCK_VERTICES[i, count:] = reshaped[count - 1]
BLOCK_FACE_DIR = numpy.zeros((len(BLOCKS) + 1, MAX_FACES), dtype=numpy.uint8)
for i, count in enumerate(BLOCK_FACE_COUNT):
    if count <= 0:
        continue
    dirs = numpy.array([0, 1, 2, 3, 4, 5], dtype=numpy.uint8)
    use = min(count, 6)
    BLOCK_FACE_DIR[i, :use] = dirs[:use]
BLOCK_SOLID = numpy.array([False]+[x.solid for x in BLOCKS], dtype = numpy.uint8)
BLOCK_OCCLUDES = numpy.array([False]+[getattr(x,'occludes', True) for x in BLOCKS], dtype = numpy.uint8)
BLOCK_OCCLUDES_SAME = numpy.array([False]+[getattr(x,'occludes_same', False) for x in BLOCKS], dtype = numpy.uint8)
BLOCK_RENDER_ALL = numpy.array([False]+[getattr(x,'render_all_faces', False) for x in BLOCKS], dtype = numpy.uint8)
# Per-block collision AABBs in world space (block-local [0,1] coords).
_scaled_verts = (0.5 * BLOCK_VERTICES).reshape(len(BLOCKS) + 1, MAX_FACES, 4, 3)
_mesh_aabb_min = _scaled_verts.min(axis=(1, 2))
_mesh_aabb_max = _scaled_verts.max(axis=(1, 2))
# Collision matches rendering: centered X/Z, base-aligned Y (0..1).
_mesh_aabb_min[:, 1] += 0.5
_mesh_aabb_max[:, 1] += 0.5
_full_cube_min = numpy.array([-0.5, 0.0, -0.5], dtype=numpy.float32)
_full_cube_max = numpy.array([0.5, 1.0, 0.5], dtype=numpy.float32)
BLOCK_RENDER_OFFSET = numpy.array([0.0, 0.5, 0.0], dtype=numpy.float32)
BLOCK_COLLIDES = numpy.zeros(len(BLOCKS) + 1, dtype=numpy.uint8)
BLOCK_COLLISION_MIN = numpy.zeros((len(BLOCKS) + 1, 3), dtype=numpy.float32)
BLOCK_COLLISION_MAX = numpy.zeros((len(BLOCKS) + 1, 3), dtype=numpy.float32)
for i, block in enumerate([None] + BLOCKS):
    if i == 0:
        continue
    collision = getattr(block, 'collision', None)
    if collision is None:
        if getattr(block, 'solid', False):
            collision = 'full'
        else:
            continue
    BLOCK_COLLIDES[i] = 1
    if collision == 'mesh':
        BLOCK_COLLISION_MIN[i] = _mesh_aabb_min[i]
        BLOCK_COLLISION_MAX[i] = _mesh_aabb_max[i]
    else:
        BLOCK_COLLISION_MIN[i] = _full_cube_min
        BLOCK_COLLISION_MAX[i] = _full_cube_max
# Preferred HUD picker face per block id (0 is air).
BLOCK_PICKER_FACE = numpy.array([0] + [getattr(x, 'picker_face', 0) for x in BLOCKS], dtype=numpy.uint8)
# Per-block face glow (0-1) applied after lighting; lets emissive blocks look bright even with low emission.
BLOCK_GLOW = numpy.array([0.0] + [getattr(x, 'glow', 0.0) for x in BLOCKS], dtype=numpy.float32)
BLOCK_INVENTORY = [x.name for x in BLOCKS if getattr(x, 'show_in_inventory', True)]
LADDER_IDS = {
    BLOCK_ID['Ladder'],
    BLOCK_ID['Ladder South'],
    BLOCK_ID['Ladder West'],
    BLOCK_ID['Ladder North'],
    BLOCK_ID['Ladder East'],
}

# Orientation indices (XZ around Y).
ORIENT_SOUTH = 0  # +Z
ORIENT_WEST = 1   # -X
ORIENT_NORTH = 2  # -Z
ORIENT_EAST = 3   # +X

LADDER_ORIENT = {
    BLOCK_ID['Ladder South']: ORIENT_SOUTH,
    BLOCK_ID['Ladder West']: ORIENT_WEST,
    BLOCK_ID['Ladder North']: ORIENT_NORTH,
    BLOCK_ID['Ladder East']: ORIENT_EAST,
}

STAIR_BASE_IDS = set()
STAIR_ORIENTED_IDS = {}
STAIR_ORIENTED_UP_IDS = {}
STAIR_IDS = set()
STAIR_UPSIDE_IDS = set()
STAIR_ORIENT = {}

def _register_stair(base_name, south, west, north, east, south_ud, west_ud, north_ud, east_ud):
    base_id = BLOCK_ID[base_name]
    oriented = [BLOCK_ID[south], BLOCK_ID[west], BLOCK_ID[north], BLOCK_ID[east]]
    oriented_up = [BLOCK_ID[south_ud], BLOCK_ID[west_ud], BLOCK_ID[north_ud], BLOCK_ID[east_ud]]
    STAIR_BASE_IDS.add(base_id)
    STAIR_ORIENTED_IDS[base_id] = oriented
    STAIR_ORIENTED_UP_IDS[base_id] = oriented_up
    STAIR_IDS.update(oriented)
    STAIR_IDS.update(oriented_up)
    STAIR_UPSIDE_IDS.update(oriented_up)
    for _orient, _bid in zip([ORIENT_SOUTH, ORIENT_WEST, ORIENT_NORTH, ORIENT_EAST], oriented):
        STAIR_ORIENT[_bid] = _orient
    for _orient, _bid in zip([ORIENT_SOUTH, ORIENT_WEST, ORIENT_NORTH, ORIENT_EAST], oriented_up):
        STAIR_ORIENT[_bid] = _orient

_register_stair(
    'Stair',
    'Stair South', 'Stair West', 'Stair North', 'Stair East',
    'Stair South Upside Down', 'Stair West Upside Down', 'Stair North Upside Down', 'Stair East Upside Down',
)
_register_stair(
    'Cobble Stair',
    'Cobble Stair South', 'Cobble Stair West', 'Cobble Stair North', 'Cobble Stair East',
    'Cobble Stair South Upside Down', 'Cobble Stair West Upside Down', 'Cobble Stair North Upside Down', 'Cobble Stair East Upside Down',
)
_register_stair(
    'Dirt Stair',
    'Dirt Stair South', 'Dirt Stair West', 'Dirt Stair North', 'Dirt Stair East',
    'Dirt Stair South Upside Down', 'Dirt Stair West Upside Down', 'Dirt Stair North Upside Down', 'Dirt Stair East Upside Down',
)
_register_stair(
    'Plank Stair',
    'Plank Stair South', 'Plank Stair West', 'Plank Stair North', 'Plank Stair East',
    'Plank Stair South Upside Down', 'Plank Stair West Upside Down', 'Plank Stair North Upside Down', 'Plank Stair East Upside Down',
)

def _rotate_dir(dir_idx, turns):
    if dir_idx <= 1:
        return dir_idx
    turns %= 4
    for _ in range(turns):
        if dir_idx == 4:
            dir_idx = 3
        elif dir_idx == 3:
            dir_idx = 5
        elif dir_idx == 5:
            dir_idx = 2
        elif dir_idx == 2:
            dir_idx = 4
    return dir_idx

def _rotate_dirs(dirs, turns):
    return numpy.array([_rotate_dir(int(d), turns) for d in dirs], dtype=numpy.uint8)

_STAIR_TURNS = {
    ORIENT_SOUTH: 0,
    ORIENT_EAST: 1,
    ORIENT_NORTH: 2,
    ORIENT_WEST: 3,
}
_STAIR_DIRS_BASE = numpy.array(STAIR_FACE_DIRS, dtype=numpy.uint8)
_STAIR_DIRS_UP = numpy.array(
    [1 if d == 0 else 0 if d == 1 else d for d in _STAIR_DIRS_BASE],
    dtype=numpy.uint8,
)
for _orient, _bid in zip(
    [ORIENT_SOUTH, ORIENT_WEST, ORIENT_NORTH, ORIENT_EAST],
    STAIR_ORIENTED_IDS[BLOCK_ID['Stair']],
):
    turns = _STAIR_TURNS[_orient]
    BLOCK_FACE_DIR[_bid, :len(_STAIR_DIRS_BASE)] = _rotate_dirs(_STAIR_DIRS_BASE, turns)
for _orient, _bid in zip(
    [ORIENT_SOUTH, ORIENT_WEST, ORIENT_NORTH, ORIENT_EAST],
    STAIR_ORIENTED_UP_IDS[BLOCK_ID['Stair']],
):
    turns = _STAIR_TURNS[_orient]
    BLOCK_FACE_DIR[_bid, :len(_STAIR_DIRS_UP)] = _rotate_dirs(_STAIR_DIRS_UP, turns)
for _base_id in STAIR_BASE_IDS:
    if _base_id == BLOCK_ID['Stair']:
        continue
    for _orient, _bid in zip(
        [ORIENT_SOUTH, ORIENT_WEST, ORIENT_NORTH, ORIENT_EAST],
        STAIR_ORIENTED_IDS[_base_id],
    ):
        turns = _STAIR_TURNS[_orient]
        BLOCK_FACE_DIR[_bid, :len(_STAIR_DIRS_BASE)] = _rotate_dirs(_STAIR_DIRS_BASE, turns)
    for _orient, _bid in zip(
        [ORIENT_SOUTH, ORIENT_WEST, ORIENT_NORTH, ORIENT_EAST],
        STAIR_ORIENTED_UP_IDS[_base_id],
    ):
        turns = _STAIR_TURNS[_orient]
        BLOCK_FACE_DIR[_bid, :len(_STAIR_DIRS_UP)] = _rotate_dirs(_STAIR_DIRS_UP, turns)

def _rotate_stair_bounds(min_v, max_v, orient):
    min_x, min_y, min_z = min_v
    max_x, max_y, max_z = max_v
    if orient == ORIENT_SOUTH:
        return min_v, max_v
    if orient == ORIENT_EAST:
        return (
            numpy.array([min_z, min_y, -max_x], dtype=numpy.float32),
            numpy.array([max_z, max_y, -min_x], dtype=numpy.float32),
        )
    if orient == ORIENT_WEST:
        return (
            numpy.array([-max_z, min_y, min_x], dtype=numpy.float32),
            numpy.array([-min_z, max_y, max_x], dtype=numpy.float32),
        )
    return (
        numpy.array([-max_x, min_y, -max_z], dtype=numpy.float32),
        numpy.array([-min_x, max_y, -min_z], dtype=numpy.float32),
    )

_STAIR_BASE_BOXES = (
    (numpy.array([-0.5, 0.0, -0.5], dtype=numpy.float32), numpy.array([0.5, 0.5, 0.0], dtype=numpy.float32)),
    (numpy.array([-0.5, 0.5, 0.0], dtype=numpy.float32), numpy.array([0.5, 1.0, 0.5], dtype=numpy.float32)),
)
_STAIR_BASE_BOXES_UP = (
    (numpy.array([-0.5, 0.5, -0.5], dtype=numpy.float32), numpy.array([0.5, 1.0, 0.0], dtype=numpy.float32)),
    (numpy.array([-0.5, 0.0, 0.0], dtype=numpy.float32), numpy.array([0.5, 0.5, 0.5], dtype=numpy.float32)),
)

def _stair_boxes_for(orient, upside):
    boxes = _STAIR_BASE_BOXES_UP if upside else _STAIR_BASE_BOXES
    return [_rotate_stair_bounds(min_v, max_v, orient) for min_v, max_v in boxes]

STAIR_COLLISION_BOXES = {}
for _base_id in STAIR_BASE_IDS:
    for _orient, _bid in zip(
        [ORIENT_SOUTH, ORIENT_WEST, ORIENT_NORTH, ORIENT_EAST],
        STAIR_ORIENTED_IDS[_base_id],
    ):
        STAIR_COLLISION_BOXES[_bid] = _stair_boxes_for(_orient, upside=False)
    for _orient, _bid in zip(
        [ORIENT_SOUTH, ORIENT_WEST, ORIENT_NORTH, ORIENT_EAST],
        STAIR_ORIENTED_UP_IDS[_base_id],
    ):
        STAIR_COLLISION_BOXES[_bid] = _stair_boxes_for(_orient, upside=True)

ORIENTED_BLOCK_IDS = {}
WALL_MOUNTED_BLOCK_IDS = set()

def _register_oriented(base_name, south, west, north, east, wall_mounted=False):
    base_id = BLOCK_ID[base_name]
    ORIENTED_BLOCK_IDS[base_id] = [
        BLOCK_ID[south],
        BLOCK_ID[west],
        BLOCK_ID[north],
        BLOCK_ID[east],
    ]
    if wall_mounted:
        WALL_MOUNTED_BLOCK_IDS.add(base_id)

_register_oriented('Plank', 'Plank South', 'Plank West', 'Plank North', 'Plank East')
_register_oriented('Door', 'Door Lower South', 'Door Lower West', 'Door Lower North', 'Door Lower East')
_register_oriented('Window Pane', 'Window Pane South', 'Window Pane West', 'Window Pane North', 'Window Pane East')
_register_oriented('Wall Torch', 'Wall Torch South', 'Wall Torch West', 'Wall Torch North', 'Wall Torch East', wall_mounted=True)
_register_oriented('Ladder', 'Ladder South', 'Ladder West', 'Ladder North', 'Ladder East', wall_mounted=True)

DOOR_BASE_IDS = {BLOCK_ID['Door']}
DOOR_LOWER_IDS = {
    BLOCK_ID['Door Lower South'],
    BLOCK_ID['Door Lower West'],
    BLOCK_ID['Door Lower North'],
    BLOCK_ID['Door Lower East'],
}
DOOR_UPPER_IDS = {
    BLOCK_ID['Door Upper South'],
    BLOCK_ID['Door Upper West'],
    BLOCK_ID['Door Upper North'],
    BLOCK_ID['Door Upper East'],
}
DOOR_LOWER_TO_UPPER = {
    BLOCK_ID['Door Lower South']: BLOCK_ID['Door Upper South'],
    BLOCK_ID['Door Lower West']: BLOCK_ID['Door Upper West'],
    BLOCK_ID['Door Lower North']: BLOCK_ID['Door Upper North'],
    BLOCK_ID['Door Lower East']: BLOCK_ID['Door Upper East'],
}
DOOR_UPPER_TO_LOWER = {
    BLOCK_ID['Door Upper South']: BLOCK_ID['Door Lower South'],
    BLOCK_ID['Door Upper West']: BLOCK_ID['Door Lower West'],
    BLOCK_ID['Door Upper North']: BLOCK_ID['Door Lower North'],
    BLOCK_ID['Door Upper East']: BLOCK_ID['Door Lower East'],
}
# Toggle pairs for open/close (rotate 90 degrees around Y).
DOOR_LOWER_TOGGLE = {
    BLOCK_ID['Door Lower South']: BLOCK_ID['Door Lower West'],
    BLOCK_ID['Door Lower West']: BLOCK_ID['Door Lower South'],
    BLOCK_ID['Door Lower North']: BLOCK_ID['Door Lower East'],
    BLOCK_ID['Door Lower East']: BLOCK_ID['Door Lower North'],
}
DOOR_UPPER_TOGGLE = {
    BLOCK_ID['Door Upper South']: BLOCK_ID['Door Upper West'],
    BLOCK_ID['Door Upper West']: BLOCK_ID['Door Upper South'],
    BLOCK_ID['Door Upper North']: BLOCK_ID['Door Upper East'],
    BLOCK_ID['Door Upper East']: BLOCK_ID['Door Upper North'],
}

# Faces that should have horizontal UV flips (per block id).
DOOR_UV_FLIP_FACES = {
    # BLOCK_ID['Door Lower South']: (2, 3),
    # BLOCK_ID['Door Upper South']: (2, 3),
    # BLOCK_ID['Door Lower North']: (4, 5),
    # BLOCK_ID['Door Upper North']: (4, 5),
    # BLOCK_ID['Door Lower West']: (4, 5),
    # BLOCK_ID['Door Upper West']: (4, 5),
    # BLOCK_ID['Door Lower East']: (2, 3),
    # BLOCK_ID['Door Upper East']: (2, 3),
    BLOCK_ID['Door Lower South']: (5, ),
    BLOCK_ID['Door Upper South']: (5, ),
    BLOCK_ID['Door Lower North']: (4, ),
    BLOCK_ID['Door Upper North']: (4, ),
    BLOCK_ID['Door Lower West']: (2, ),
    BLOCK_ID['Door Upper West']: (2, ),
    BLOCK_ID['Door Lower East']: (3, ),
    BLOCK_ID['Door Upper East']: (3, ),
}

# Pre-flip UVs for door faces to avoid per-mesh updates.
BLOCK_TEXTURES_FLIPPED = BLOCK_TEXTURES.copy()
_tex_view = BLOCK_TEXTURES_FLIPPED.reshape(len(BLOCKS) + 1, 7, 4, 2)
for _bid, _faces in DOOR_UV_FLIP_FACES.items():
    for _face in _faces:
        _u0 = _tex_view[_bid, _face, 0, 0].copy()
        _u1 = _tex_view[_bid, _face, 1, 0].copy()
        _tex_view[_bid, _face, 0, 0] = _u1
        _tex_view[_bid, _face, 1, 0] = _u0
        _tex_view[_bid, _face, 2, 0] = _u0
        _tex_view[_bid, _face, 3, 0] = _u1

# Block light emitters (0-1 brightness). Defined here so IDs stay in sync.
BLOCK_LIGHT_LEVELS = {
    BLOCK_ID["Jack O'Lantern"]: 1.0,
    BLOCK_ID["Mushroom"]: 0.25,
    BLOCK_ID["Wall Torch"]: 1.0,
    BLOCK_ID["Wall Torch South"]: 1.0,
    BLOCK_ID["Wall Torch West"]: 1.0,
    BLOCK_ID["Wall Torch North"]: 1.0,
    BLOCK_ID["Wall Torch East"]: 1.0,
}
