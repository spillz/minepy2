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
)

TEXTURE_PATH = 'texture_fv.png'

gr = [50,150,70]
white = numpy.tile(numpy.array([255,255,255]),6*4).reshape(6,3*4)
green = numpy.tile(numpy.array(gr),6*4).reshape(6,3*4)
grass_top = numpy.array([77,244,44]*4+[255,255,255]*5*4).reshape(6,3*4)
water_blue = numpy.tile(numpy.array(WATER_COLOR),6*4).reshape(6,3*4)


class Block(object):
    name = None
    coords = None
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
    name = 'BetterStone'
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
    coords = ((3, 9), (3, 9), (3, 9))
    vertices = torch_south
    solid = False
    occludes = False
    glow = 1.0
    collision = 'mesh'

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
]
i = 1
BLOCK_ID = {}
for x in BLOCKS:
    BLOCK_ID[x.name] = i
    i+=1
BLOCK_NORMALS = numpy.array(FACES)
BLOCK_COLORS = numpy.array([white] + [x.colors for x in BLOCKS])
BLOCK_TEXTURES = numpy.array([tex_coords((0,0),(0,0),(0,0))] + [tex_coords(*x.coords) for x in BLOCKS],dtype = numpy.float32)/4
BLOCK_VERTICES = numpy.array([cb_v]+[x.vertices for x in BLOCKS])
def _scale_partial_uvs(block_textures, block_vertices):
    uv = block_textures.copy()
    uv_view = uv.reshape(len(BLOCKS) + 1, 7, 4, 2)
    verts = block_vertices.reshape(len(BLOCKS) + 1, 6, 4, 3)
    edge_u = numpy.linalg.norm(verts[:, :, 1, :] - verts[:, :, 0, :], axis=2)
    edge_v = numpy.linalg.norm(verts[:, :, 3, :] - verts[:, :, 0, :], axis=2)
    frac_u = numpy.clip(edge_u / 2.0, 0.0, 1.0)
    frac_v = numpy.clip(edge_v / 2.0, 0.0, 1.0)

    u0 = uv_view[:, :6, 0, 0]
    u1 = uv_view[:, :6, 1, 0]
    v0 = uv_view[:, :6, 0, 1]
    v2 = uv_view[:, :6, 2, 1]
    center_u = (u0 + u1) * 0.5
    center_v = (v0 + v2) * 0.5
    half_u = (u1 - u0) * 0.5 * frac_u
    half_v = (v2 - v0) * 0.5 * frac_v

    u_min = center_u - half_u
    u_max = center_u + half_u
    v_min = center_v - half_v
    v_max = center_v + half_v
    uv_view[:, :6, 0, 0] = u_min
    uv_view[:, :6, 3, 0] = u_min
    uv_view[:, :6, 1, 0] = u_max
    uv_view[:, :6, 2, 0] = u_max
    uv_view[:, :6, 0, 1] = v_min
    uv_view[:, :6, 1, 1] = v_min
    uv_view[:, :6, 2, 1] = v_max
    uv_view[:, :6, 3, 1] = v_max
    return uv

BLOCK_TEXTURES = _scale_partial_uvs(BLOCK_TEXTURES, BLOCK_VERTICES)
BLOCK_SOLID = numpy.array([False]+[x.solid for x in BLOCKS], dtype = numpy.uint8)
BLOCK_OCCLUDES = numpy.array([False]+[getattr(x,'occludes', True) for x in BLOCKS], dtype = numpy.uint8)
BLOCK_OCCLUDES_SAME = numpy.array([False]+[getattr(x,'occludes_same', False) for x in BLOCKS], dtype = numpy.uint8)
BLOCK_RENDER_ALL = numpy.array([False]+[getattr(x,'render_all_faces', False) for x in BLOCKS], dtype = numpy.uint8)
# Per-block collision AABBs in world space (block-local [0,1] coords).
_scaled_verts = (0.5 * BLOCK_VERTICES).reshape(len(BLOCKS) + 1, 6, 4, 3)
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

# Orientation indices (XZ around Y).
ORIENT_SOUTH = 0  # +Z
ORIENT_WEST = 1   # -X
ORIENT_NORTH = 2  # -Z
ORIENT_EAST = 3   # +X

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
