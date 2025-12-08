import numpy as np
import time
import pyglet

import noise
from config import SECTOR_SIZE

cb_v = np.array([
        [-1,+1,-1, -1,+1,+1, +1,+1,+1, +1,+1,-1],  # top
        [-1,-1,-1, +1,-1,-1, +1,-1,+1, -1,-1,+1],  # bottom
        [-1,-1,-1, -1,-1,+1, -1,+1,+1, -1,+1,-1],  # left
        [+1,-1,+1, +1,-1,-1, +1,+1,-1, +1,+1,+1],  # right
        [-1,-1,+1, +1,-1,+1, +1,+1,+1, -1,+1,+1],  # front
        [+1,-1,-1, -1,-1,-1, -1,+1,-1, +1,+1,-1],  # back
],dtype = np.float32)

c = 1
cb_v_half = np.array([
        [-1,+0,-1, -1,+0,+1, +1,+0,+1, +1,+0,-1],  # top
        [-1,-1,-1, +1,-1,-1, +1,-1,+1, -1,-1,+1],  # bottom
        [-c,-1,-1, -c,-1,+1, -c,+1,+1, -c,+1,-1],  # left
        [+c,-1,+1, +c,-1,-1, +c,+1,-1, +c,+1,+1],  # right
        [-1,-1,+c, +1,-1,+c, +1,+1,+c, -1,+1,+c],  # front
        [+1,-1,-c, -1,-1,-c, -1,+1,-c, +1,+1,-c],  # back
],dtype = np.float32)

c = 14.0/16
cb_v_cake = np.array([
        [-1,+0,-1, -1,+0,+1, +1,+0,+1, +1,+0,-1],  # top
        [-1,-1,-1, +1,-1,-1, +1,-1,+1, -1,-1,+1],  # bottom
        [-c,-1,-1, -c,-1,+1, -c,+1,+1, -c,+1,-1],  # left
        [+c,-1,+1, +c,-1,-1, +c,+1,-1, +c,+1,+1],  # right
        [-1,-1,+c, +1,-1,+c, +1,+1,+c, -1,+1,+c],  # front
        [+1,-1,-c, -1,-1,-c, -1,+1,-c, +1,+1,-c],  # back
],dtype = np.float32)

de_v = np.array([
        [0]*12,
        [0]*12,
        [-1,-1,+1, +1,-1,-1, +1,+1,-1, -1,+1,+1], 
        [+1,-1,-1, -1,-1,+1, -1,+1,+1, +1,+1,-1],
        [-1,-1,-1, +1,-1,+1, +1,+1,+1, -1,+1,-1], 
        [+1,-1,+1, -1,-1,-1, -1,+1,-1, +1,+1,+1],
],dtype = np.float32)

def cube_v(pos,n):
    return n*cb_v+np.tile(pos,4)
def cube_v2(pos,n):
    return (n*cb_v)+np.tile(pos,4)[:,np.newaxis,:]

def deco_v(pos,n):
    return n*de_v+np.tile(pos,4)
def deco_v2(pos,n):
    return (n*de_v)+np.tile(pos,4)[:,np.newaxis,:]


def tex_coord(x, y, n=4):
    """ Return the bounding vertices of the texture square.

    """
    m = 1.0 / n
    dx = x * m
    dy = y * m
    return [dx, dy, dx + m, dy, dx + m, dy + m, dx, dy + m]


def tex_coords(*sides): #top, bottom, 
    """ Return a list of the texture squares for the top, bottom and side.

    """
#    top = tex_coord(*top)
#    bottom = tex_coord(*bottom)
    result = []
#    result.append(top)
#    result.append(bottom)
    i=6
    for s in sides:
        result.append(tex_coord(*s))
        i-=1
    while i>=0:
        result.append(tex_coord(*sides[-1]))
        i-=1
    return result

FACES = [
    ( 0, 1, 0), #up
    ( 0,-1, 0), #down
    (-1, 0, 0), #left
    ( 1, 0, 0), #right
    ( 0, 0, 1), #forward
    ( 0, 0,-1), #back
]

import numpy

import numpy

import numpy

def compute_vertex_ao(solid_full, core_shape, ao_strength):
    """
    Return per-vertex AO factors (darkening only).
    
    Logic: Orthogonal Neighbor Count.
    - 2 Neighbors (Corner) = 1.0 * strength
    - 1 Neighbor (Wall)    = 0.5 * strength
    - 0 Neighbors (Flat)   = 0.0
    """
    try:
        import config
        if not getattr(config, 'AO_ENABLED', True):
            return None
    except Exception:
        pass

    if ao_strength <= 1e-5:
        return None

    sx, sy, sz = core_shape

    # --- 1. HANDLE SIZE MISMATCH ---
    curr_x, curr_y, curr_z = solid_full.shape
    
    if curr_x > sx + 2:
        start = (curr_x - (sx + 2)) // 2
        solid_full = solid_full[start : start + sx + 2, :, :]
    
    if curr_z > sz + 2:
        start = (curr_z - (sz + 2)) // 2
        solid_full = solid_full[:, :, start : start + sz + 2]

    # Pad Y explicitly to allow looking at layers above/below
    solid_pad = numpy.pad(solid_full, ((0, 0), (1, 1), (0, 0)), mode='constant', constant_values=False)

import numpy

def compute_vertex_ao(solid_full, core_shape, ao_strength, block_mask=None):
    """
    Return per-vertex AO factors (darkening only).

    If block_mask is provided (flattened or shaped to core_shape), we only
    compute AO for those cells and return an array of shape (N, 6, 4) matching
    the masked order. Otherwise a full grid of shape (sx, sy, sz, 6, 4) is returned.
    """
    try:
        import config
        if not getattr(config, 'AO_ENABLED', True):
            return None
    except Exception:
        pass

    if ao_strength <= 1e-5:
        return None

    sx, sy, sz = core_shape

    # Pad Y explicitly to allow looking at layers above/below.
    solid_pad = numpy.pad(solid_full, ((0, 0), (1, 1), (0, 0)), mode='constant', constant_values=False)

    if block_mask is not None:
        mask_reshaped = block_mask.reshape(core_shape)
        coords = numpy.argwhere(mask_reshaped)
    else:
        coords = numpy.indices(core_shape).reshape(3, -1).transpose(1, 0)

    if coords.size == 0:
        return None

    x = coords[:, 0] + 1  # offset for x padding already present in solid_full
    y = coords[:, 1] + 1  # offset for y padding we added here
    z = coords[:, 2] + 1  # offset for z padding already present in solid_full

    def ao_factor(side_a, side_b):
        """Orthogonal neighbor occlusion: 0, 0.5, or 1.0 strength."""
        neighbor_count = side_a.astype(numpy.float32) + side_b.astype(numpy.float32)
        return 1.0 - ao_strength * (neighbor_count / 2.0)

    ao = numpy.ones((len(coords), 6, 4), dtype=numpy.float32)

    # Top (+Y)
    xm = solid_pad[x - 1, y + 1, z]
    xp = solid_pad[x + 1, y + 1, z]
    zm = solid_pad[x, y + 1, z - 1]
    zp = solid_pad[x, y + 1, z + 1]
    ao[:, 0, 0] = ao_factor(xm, zm)
    ao[:, 0, 1] = ao_factor(xm, zp)
    ao[:, 0, 2] = ao_factor(xp, zp)
    ao[:, 0, 3] = ao_factor(xp, zm)

    # Bottom (-Y)
    xm = solid_pad[x - 1, y - 1, z]
    xp = solid_pad[x + 1, y - 1, z]
    zm = solid_pad[x, y - 1, z - 1]
    zp = solid_pad[x, y - 1, z + 1]
    ao[:, 1, 0] = ao_factor(xm, zm)
    ao[:, 1, 1] = ao_factor(xp, zm)
    ao[:, 1, 2] = ao_factor(xp, zp)
    ao[:, 1, 3] = ao_factor(xm, zp)

    # Left (-X)
    ym = solid_pad[x - 1, y - 1, z]
    yp = solid_pad[x - 1, y + 1, z]
    zm = solid_pad[x - 1, y, z - 1]
    zp = solid_pad[x - 1, y, z + 1]
    ao[:, 2, 0] = ao_factor(ym, zm)
    ao[:, 2, 1] = ao_factor(ym, zp)
    ao[:, 2, 2] = ao_factor(yp, zp)
    ao[:, 2, 3] = ao_factor(yp, zm)

    # Right (+X)
    ym = solid_pad[x + 1, y - 1, z]
    yp = solid_pad[x + 1, y + 1, z]
    zm = solid_pad[x + 1, y, z - 1]
    zp = solid_pad[x + 1, y, z + 1]
    ao[:, 3, 0] = ao_factor(ym, zp)
    ao[:, 3, 1] = ao_factor(ym, zm)
    ao[:, 3, 2] = ao_factor(yp, zm)
    ao[:, 3, 3] = ao_factor(yp, zp)

    # Front (+Z)
    xm = solid_pad[x - 1, y, z + 1]
    xp = solid_pad[x + 1, y, z + 1]
    ym = solid_pad[x, y - 1, z + 1]
    yp = solid_pad[x, y + 1, z + 1]
    ao[:, 4, 0] = ao_factor(xm, ym)
    ao[:, 4, 1] = ao_factor(xp, ym)
    ao[:, 4, 2] = ao_factor(xp, yp)
    ao[:, 4, 3] = ao_factor(xm, yp)

    # Back (-Z)
    xm = solid_pad[x - 1, y, z - 1]
    xp = solid_pad[x + 1, y, z - 1]
    ym = solid_pad[x, y - 1, z - 1]
    yp = solid_pad[x, y + 1, z - 1]
    ao[:, 5, 0] = ao_factor(xp, ym)
    ao[:, 5, 1] = ao_factor(xm, ym)
    ao[:, 5, 2] = ao_factor(xm, yp)
    ao[:, 5, 3] = ao_factor(xp, yp)

    # Apply floor and gamma.
    try:
        import config
        ao_min = getattr(config, 'AO_MIN', 0.0)
        ao_gamma = getattr(config, 'AO_GAMMA', 1.0)
    except Exception:
        ao_min = 0.0
        ao_gamma = 1.0

    ao = numpy.clip(ao, ao_min, 1.0)
    if abs(ao_gamma - 1.0) > 1e-6:
        ao = ao ** ao_gamma

    if block_mask is None:
        return ao.reshape(sx, sy, sz, 6, 4)
    return ao

def normalize(position):
    """ Accepts `position` of arbitrary precision and returns the block
    containing that position.

    Parameters
    ----------
    position : tuple of len 3

    Returns
    -------
    block_position : tuple of ints of len 3

    """
    x, y, z = position
    x, y, z = (int(round(x)), int(round(y)), int(round(z)))
    return (x, y, z)


def sectorize(position):
    """ Returns a tuple representing the sector for the given `position`.

    Parameters
    ----------
    position : tuple of len 3

    Returns
    -------
    sector : tuple of len 3

    """
    x, y, z = normalize(position)
    x, y, z = x // SECTOR_SIZE, y // SECTOR_SIZE, z // SECTOR_SIZE
    return (x*SECTOR_SIZE, 0, z*SECTOR_SIZE)

## monkey patch IndirectArrayRegion.__setitem__ to make it quicker for numpy arrays (pyglet 1.x only)
try:
    orig_indirect_array_region_setitem = pyglet.graphics.vertexbuffer.IndirectArrayRegion.__setitem__
    def numpy__setitem__(self, index, value):
        if isinstance(value, np.ndarray) and isinstance(index, slice) \
              and index.start is None and index.stop is None and index.step is None:
            arr = np.ctypeslib.as_array(self.region.array)
            for i in range(self.count):
                arr[i::self.stride] = value[i::self.count]
            return
        orig_indirect_array_region_setitem(self, index, value)
    pyglet.graphics.vertexbuffer.IndirectArrayRegion.__setitem__ = numpy__setitem__
except AttributeError:
    # IndirectArrayRegion not present in pyglet 2.x; skip the monkey patch
    pass
