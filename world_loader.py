'''
world_loader.py -- manages client side terrain generation, world data caching, and syncing of terrain with multi-player server
'''

# standard library imports
import math
import itertools
import time
import numpy
import select
try:
    import cPickle as pickle
except ImportError:
    import pickle
import multiprocessing.connection
import socket
import sys

# local imports
import config
from config import SECTOR_SIZE, SECTOR_HEIGHT, LOADED_SECTORS, SERVER_IP, SERVER_PORT, LOADER_IP, LOADER_PORT
from util import normalize, sectorize, FACES, cube_v, cube_v2, compute_vertex_ao
from blocks import BLOCK_VERTICES, BLOCK_COLORS, BLOCK_NORMALS, BLOCK_TEXTURES, BLOCK_ID, BLOCK_SOLID, BLOCK_OCCLUDES, BLOCK_OCCLUDES_SAME, TEXTURE_PATH, BLOCK_LIGHT_LEVELS
import mapgen

import logging
logging.basicConfig(level = logging.INFO)
def loader_log(msg, *args):
    logging.log(logging.INFO, 'LOADER: '+msg, *args)

SECTOR_ARRAY = numpy.indices((SECTOR_SIZE, SECTOR_HEIGHT, SECTOR_SIZE)).transpose(1, 2, 3, 0)
SH = SECTOR_ARRAY.shape
SECTOR_GRID = SECTOR_ARRAY.reshape((SH[0]*SH[1]*SH[2],3))
WATER = BLOCK_ID['Water']

class WorldLoader(object):
    def __init__(self, client_pipe, server_pipe):
        self.client_pipe = client_pipe
        self.server_pipe = server_pipe
        self.db = None
        if self.server_pipe is None:
            import world_db
            self.db = world_db.World()
            self.world_seed = self.db.get_seed()
        else:
            self.server_pipe.send(['l_get_seed',[]])
            msg, (self.world_seed,) = self.server_pipe.recv()
            assert(msg == 'l_seed')
        self.pos = None
        self.blocks = numpy.zeros((SECTOR_SIZE+2,SECTOR_HEIGHT,SECTOR_SIZE+2),dtype='u2') #blocks of a sector not the whole world
        self.vt_data = None #vertex data for the current sector
        self._loader_loop()

    def _loader_loop(self):
        '''
        Receives request for terrain and vertices for sectors of the map from the client
        will check server for changed blocks. Current implementation is pretty dumb because
        it blocks at each client and server request and expects a single task at a time.
        
        TODO: load light across sector boundaries (potentially across four sectors for each loaded sector)
        
        When a player moves, we can drop sectors out of range and add new ones. For example:
         ++++++
        -OOOOO+
        -OOOOO+
        -OOOOO+
        -OOOOO+
        ------
        0 unchanged
        - sectors to drop
        + sectors to add        
        '''
        import mapgen
        import select
        loader_log('loader loop started')
        cpipe = self.client_pipe
        spipe = self.server_pipe
        mapgen.initialize_map_generator(seed = self.world_seed)
        if getattr(config, 'USE_EXPERIMENTAL_BIOME_GEN', False):
            mapgen.initialize_biome_map_generator(seed=self.world_seed)
        while True:
            try:
                msg, data = cpipe.recv()
                loader_log('received from client %s', msg)
            except:
                loader_log('unexpected error reading from client pipe, exiting')
                return
            if msg == 'quit':
                loader_log('terminated by client')
                return
            if msg == 'sector_blocks':
                self.pos = data[0]
                sector_block_delta = None
                if spipe is not None:
                    loader_log('sending block request to server')
                    spipe.send(('l_get_sector_blocks', [self.pos]))
                    loader_log('getting block response from server')
                    msg, data = spipe.recv()
                    loader_log('got block response from server')
                    assert(msg == 'l_sector_blocks_changed')
                    spos, sector_block_delta = data
                    assert(spos == self.pos)
                else:
                    sector_block_delta = self.db.get_sector_data(self.pos)
                self._initialize(self.pos, sector_block_delta)
                self._calc_vertex_data(self.pos)
                t0 = time.perf_counter()
                payload = pickle.dumps(['sector_blocks',[self.pos, self.blocks, self.vt_data, self.light]],-1)
                t_dump = (time.perf_counter() - t0) * 1000.0
                t1 = time.perf_counter()
                cpipe.send_bytes(payload)
                t_send = (time.perf_counter() - t1) * 1000.0
                loader_log('pickle.dumps+send sector %s: %.1fms dump, %.1fms send, %d bytes', self.pos, t_dump, t_send, len(payload))
            if msg == 'set_block':
                # data may include an optional client token at the end
                if len(data) == 4:
                    notify_server, pos, block_id, sector_data = data
                    token = None
                else:
                    notify_server, pos, block_id, sector_data, token = data
                sector_result = []
                for spos, blocks in sector_data:
                    self.blocks = blocks
                    self.set_block(pos, spos, block_id)
                    self._calc_vertex_data(spos)
                    sector_result.append((spos, self.blocks, self.vt_data, self.light))
                t0 = time.perf_counter()
                payload = pickle.dumps(['sector_blocks2',sector_result, token],-1)
                t_dump = (time.perf_counter() - t0) * 1000.0
                t1 = time.perf_counter()
                cpipe.send_bytes(payload)
                t_send = (time.perf_counter() - t1) * 1000.0
                loader_log('pickle.dumps+send set_block batch: %.1fms dump, %.1fms send, %d bytes', t_dump, t_send, len(payload))
                if spipe is not None:
                    if notify_server:
                        spipe.send(('set_block', [pos, block_id]))
                else:
                    self.db.set_block(pos, block_id)

    def _initialize(self, position, sector_block_delta):
        """ Initialize the sector by procedurally generating terrain using
        simplex noise.

        """
        if getattr(config, 'USE_EXPERIMENTAL_BIOME_GEN', False):
            self.blocks = mapgen.generate_biome_sector(position, None, None)
        else:
            self.blocks = mapgen.generate_sector(position, None, None)
        if sector_block_delta is not None:
            for p in sector_block_delta:
                self.blocks[p] = sector_block_delta[p]


    def _calc_exposed_faces(self):
        # Face exposure booleans: up, down, left, right, front, back
        exposed_faces = numpy.zeros(self.blocks.shape + (6,), dtype=bool)

        # Helper: neighbor occludes if it is solid or if it is the same type as a same-occluding block (e.g., leaves).
        def neighbor_occ_mask(cur_block, neighbor_block):
            solid_occ = BLOCK_SOLID[neighbor_block] != 0
            same_occ = (BLOCK_OCCLUDES_SAME[neighbor_block] != 0) & (neighbor_block == cur_block)
            return solid_occ | same_occ

        # up: neighbor at y+1
        cur = self.blocks[:,:-1,:]
        neighbor = self.blocks[:,1:,:]
        neighbor_occ = neighbor_occ_mask(cur, neighbor)
        exposed_faces[:,:-1,:,0] = ~neighbor_occ
        # down: neighbor at y-1
        cur = self.blocks[:,1:,:]
        neighbor = self.blocks[:,:-1,:]
        neighbor_occ = neighbor_occ_mask(cur, neighbor)
        exposed_faces[:,1:,:,1] = ~neighbor_occ
        # left (-x)
        cur = self.blocks[1:,:,:]
        neighbor = self.blocks[:-1,:,:]
        neighbor_occ = neighbor_occ_mask(cur, neighbor)
        exposed_faces[1:,:,:,2] = ~neighbor_occ
        # right (+x)
        cur = self.blocks[:-1,:,:]
        neighbor = self.blocks[1:,:,:]
        neighbor_occ = neighbor_occ_mask(cur, neighbor)
        exposed_faces[:-1,:,:,3] = ~neighbor_occ
        # forward (+z)
        cur = self.blocks[:,:,:-1]
        neighbor = self.blocks[:,:,1:]
        neighbor_occ = neighbor_occ_mask(cur, neighbor)
        exposed_faces[:,:,:-1,4] = ~neighbor_occ
        # back (-z)
        cur = self.blocks[:,:,1:]
        neighbor = self.blocks[:,:,:-1]
        neighbor_occ = neighbor_occ_mask(cur, neighbor)
        exposed_faces[:,:,1:,5] = ~neighbor_occ

        solid = (self.blocks > 0) & (self.blocks != WATER)
        self.exposed_faces = exposed_faces & solid[..., None]

        # Air mask reused for lighting
        air = (BLOCK_SOLID[self.blocks] == 0)

        # Flood-fill lighting (skylight + optional block emitters), vectorized relaxation
        light = numpy.zeros(self.blocks.shape, dtype=numpy.float32)
        decay = config.LIGHT_DECAY
        top_y = self.blocks.shape[1] - 1
        # Skylight vertical pass (no decay straight down until blocked)
        for x in range(self.blocks.shape[0]):
            for z in range(self.blocks.shape[2]):
                for y in range(top_y, -1, -1):
                    if air[x, y, z]:
                        light[x, y, z] = 1.0
                    else:
                        break
        # Block light emitters
        if BLOCK_LIGHT_LEVELS:
            for bid, lvl in BLOCK_LIGHT_LEVELS.items():
                emitters = numpy.argwhere(self.blocks == bid)
                for ex, ey, ez in emitters:
                    light[ex, ey, ez] = max(light[ex, ey, ez], float(lvl))

        # Relax until convergence or max iterations
        diag_decay = decay * math.sqrt(2.0)
        corner_decay = decay * math.sqrt(3.0)
        for _ in range(64):
            neighbor_max = numpy.zeros_like(light)
            # Axis neighbors
            neighbor_max[:-1,:,:] = numpy.maximum(neighbor_max[:-1,:,:], light[1:,:,:] - decay)   # +x
            neighbor_max[1:,:,:]  = numpy.maximum(neighbor_max[1:,:,:],  light[:-1,:,:] - decay)  # -x
            neighbor_max[:,:-1,:] = numpy.maximum(neighbor_max[:,:-1,:], light[:,1:,:] - decay)   # +y
            neighbor_max[:,1:,:]  = numpy.maximum(neighbor_max[:,1:,:],  light[:,:-1,:] - decay)  # -y
            neighbor_max[:,:,:-1] = numpy.maximum(neighbor_max[:,:,:-1], light[:,:,1:] - decay)   # +z
            neighbor_max[:,:,1:]  = numpy.maximum(neighbor_max[:,:,1:],  light[:,:,:-1] - decay)  # -z
            # Edge diagonals (sqrt(2) cost)
            neighbor_max[:-1,:-1,:] = numpy.maximum(neighbor_max[:-1,:-1,:], light[1:,1:,:] - diag_decay)
            neighbor_max[:-1,1:,:]  = numpy.maximum(neighbor_max[:-1,1:,:],  light[1:,:-1,:] - diag_decay)
            neighbor_max[1:,:-1,:]  = numpy.maximum(neighbor_max[1:,:-1,:],  light[:-1,1:,:] - diag_decay)
            neighbor_max[1:,1:,:]   = numpy.maximum(neighbor_max[1:,1:,:],   light[:-1,:-1,:] - diag_decay)

            neighbor_max[:-1,:,:-1] = numpy.maximum(neighbor_max[:-1,:,:-1], light[1:,:,1:] - diag_decay)
            neighbor_max[:-1,:,1:]  = numpy.maximum(neighbor_max[:-1,:,1:],  light[1:,:,:-1] - diag_decay)
            neighbor_max[1:,:,:-1]  = numpy.maximum(neighbor_max[1:,:,:-1],  light[:-1,:,1:] - diag_decay)
            neighbor_max[1:,:,1:]   = numpy.maximum(neighbor_max[1:,:,1:],   light[:-1,:,:-1] - diag_decay)

            neighbor_max[:,:-1,:-1] = numpy.maximum(neighbor_max[:,:-1,:-1], light[:,1:,1:] - diag_decay)
            neighbor_max[:,1:,:-1]  = numpy.maximum(neighbor_max[:,1:,:-1],  light[:,:-1,1:] - diag_decay)
            neighbor_max[:,:-1,1:]  = numpy.maximum(neighbor_max[:,:-1,1:],  light[:,1:,:-1] - diag_decay)
            neighbor_max[:,1:,1:]   = numpy.maximum(neighbor_max[:,1:,1:],   light[:,:-1,:-1] - diag_decay)
            # Corner diagonals (sqrt(3) cost)
            neighbor_max[:-1,:-1,:-1] = numpy.maximum(neighbor_max[:-1,:-1,:-1], light[1:,1:,1:] - corner_decay)
            neighbor_max[:-1,:-1,1:]  = numpy.maximum(neighbor_max[:-1,:-1,1:],  light[1:,1:,:-1] - corner_decay)
            neighbor_max[:-1,1:,:-1]  = numpy.maximum(neighbor_max[:-1,1:,:-1],  light[1:,:-1,1:] - corner_decay)
            neighbor_max[:-1,1:,1:]   = numpy.maximum(neighbor_max[:-1,1:,1:],   light[1:,:-1,:-1] - corner_decay)

            neighbor_max[1:,:-1,:-1] = numpy.maximum(neighbor_max[1:,:-1,:-1], light[:-1,1:,1:] - corner_decay)
            neighbor_max[1:,:-1,1:]  = numpy.maximum(neighbor_max[1:,:-1,1:],  light[:-1,1:,:-1] - corner_decay)
            neighbor_max[1:,1:,:-1]  = numpy.maximum(neighbor_max[1:,1:,:-1],  light[:-1,:-1,1:] - corner_decay)
            neighbor_max[1:,1:,1:]   = numpy.maximum(neighbor_max[1:,1:,1:],   light[:-1,:-1,:-1] - corner_decay)

            new_light = numpy.where(air, numpy.maximum(light, neighbor_max), 0.0)
            if numpy.array_equal(new_light, light):
                break
            light = new_light
        self.light = light

        ##TODO: For more even light, calculate a separate light value for the 24 vertices not the 6 faces
        exposed_light = numpy.zeros(self.blocks.shape+(6,),dtype=numpy.float32)
        exposed_light[:,:-1,:,0] = light[:,1:,:] # up
        exposed_light[:,1:,:,1] = light[:,:-1,:] # down
        exposed_light[1:,:,:,2] = light[:-1,:,:] # left
        exposed_light[:-1,:,:,3] = light[1:,:,:] # right
        exposed_light[:,:,:-1,4] = light[:,:,1:] # front
        exposed_light[:,:,1:,5] = light[:,:,:-1] # back
        self.exposed_light = exposed_light

    def _calc_vertex_data(self,position):
        self._calc_exposed_faces()
        # interior (skip padding) shapes: (SECTOR_SIZE, SECTOR_HEIGHT, SECTOR_SIZE, 6)
        exposed_faces = self.exposed_faces[1:-1,:,1:-1]
        exposed_light = self.exposed_light[1:-1,:,1:-1]

        sx, sy, sz, _ = exposed_faces.shape
        face_mask = exposed_faces.reshape(sx*sy*sz, 6)
        light_flat = exposed_light.reshape(sx*sy*sz, 6)

        block_mask = face_mask.any(axis=1)
        v = numpy.array([], dtype=numpy.float32)
        t = numpy.array([], dtype=numpy.float32)
        n = numpy.array([], dtype=numpy.float32)
        c = numpy.array([], dtype=numpy.float32)
        count = 0
        if block_mask.any():
            pos = SECTOR_GRID[block_mask] + position  # (N,3)
            face_mask = face_mask[block_mask]
            light_flat = light_flat[block_mask]

            b = self.blocks[1:-1,:,1:-1].reshape(sx*sy*sz)[block_mask]

            verts = (0.5*BLOCK_VERTICES[b].reshape(len(b),6,4,3) + pos[:,None,None,:]).astype(numpy.float32)
            tex = BLOCK_TEXTURES[b][:,:6].reshape(len(b),6,4,2).astype(numpy.float32)
            light = light_flat[:, :, None, None]  # (N,6,1,1)
            colors_base = BLOCK_COLORS[b][:,:6].reshape(len(b),6,4,3).astype(numpy.float32)
            ambient = config.AMBIENT_LIGHT
            colors_lit = colors_base*(ambient + (1.0-ambient)*light)

            ao = None
            if getattr(config, 'AO_ENABLED', True):
                ao = compute_vertex_ao(
                    BLOCK_SOLID[self.blocks].astype(bool),
                    (sx, sy, sz),
                    getattr(config, 'AO_STRENGTH', 0.0),
                    block_mask=block_mask,
                )
                if ao is not None:
                    ao_flat = numpy.where(face_mask[..., None], ao, 1.0)
                    colors_lit = colors_lit * ao_flat[..., None]
            colors = numpy.clip(colors_lit, 0, 255)
            normals = numpy.broadcast_to(BLOCK_NORMALS[None,:,None,:], (len(b),6,4,3)).astype(numpy.float32)

            v = verts[face_mask].reshape(-1,3).ravel()
            t = tex[face_mask].reshape(-1,2).ravel()
            n = normals[face_mask].reshape(-1,3).ravel()
            c = colors[face_mask].reshape(-1,3).ravel()
            count = len(v)//3
            if getattr(config, 'AO_DEBUG', False) and not getattr(self, '_ao_debug_once_vt', False):
                self._ao_debug_once_vt = True
                try:
                    print("[AO] vt colors min/max sample:", float(c.min()), float(c.max()), "sample first 12:", c[:12])
                    print("[AO] vt verts sample first 9:", v[:9])
                except Exception as e:
                    print("[AO] vt debug error", e)

        # Water geometry: exposed faces of water blocks (only to air), duplicated for visibility above/below.
        water_blocks = (self.blocks == WATER)
        water_exposed = numpy.zeros(self.blocks.shape + (6,), dtype=bool)
        neighbor = self.blocks[:,1:,:]
        air = (neighbor != WATER) & (BLOCK_SOLID[neighbor] == 0)
        water_exposed[:,:-1,:,0] = water_blocks[:,:-1,:] & air
        neighbor = self.blocks[:,:-1,:]
        air = (neighbor != WATER) & (BLOCK_SOLID[neighbor] == 0)
        water_exposed[:,1:,:,1] = water_blocks[:,1:,:] & air
        neighbor = self.blocks[:-1,:,:]
        air = (neighbor != WATER) & (BLOCK_SOLID[neighbor] == 0)
        water_exposed[1:,:,:,2] = water_blocks[1:,:,:] & air
        neighbor = self.blocks[1:,:,:]
        air = (neighbor != WATER) & (BLOCK_SOLID[neighbor] == 0)
        water_exposed[:-1,:,:,3] = water_blocks[:-1,:,:] & air
        neighbor = self.blocks[:,:,1:]
        air = (neighbor != WATER) & (BLOCK_SOLID[neighbor] == 0)
        water_exposed[:,:,:-1,4] = water_blocks[:,:,:-1] & air
        neighbor = self.blocks[:,:,:-1]
        air = (neighbor != WATER) & (BLOCK_SOLID[neighbor] == 0)
        water_exposed[:,:,1:,5] = water_blocks[:,:,1:] & air

        water_exposed = water_exposed[1:-1,:,1:-1]
        w_face_mask = water_exposed.reshape(sx*sy*sz, 6)
        water_mask = w_face_mask.any(axis=1)
        water_data = None
        if water_mask.any():
            pos_w = SECTOR_GRID[water_mask] + position
            face_mask_w = w_face_mask[water_mask]
            b = numpy.full(len(pos_w), WATER, dtype=numpy.int32)
            verts = (0.5*BLOCK_VERTICES[b].reshape(len(b),6,4,3) + pos_w[:,None,None,:]).astype(numpy.float32)
            tex = BLOCK_TEXTURES[b][:,:6].reshape(len(b),6,4,2).astype(numpy.float32)
            normals = numpy.broadcast_to(BLOCK_NORMALS[None,:,None,:], (len(b),6,4,3)).astype(numpy.float32)
            colors = BLOCK_COLORS[b][:,:6].reshape(len(b),6,4,3).astype(numpy.float32)

            face_verts = verts[face_mask_w].reshape(-1,4,3)
            face_tex = tex[face_mask_w].reshape(-1,4,2)
            face_norm = normals[face_mask_w].reshape(-1,4,3)
            face_col = colors[face_mask_w].reshape(-1,4,3)

            wv = face_verts.reshape(-1,3).ravel()
            wtcoords = face_tex.reshape(-1,2).ravel()
            wn = face_norm.reshape(-1,3).ravel()
            wc = face_col.reshape(-1,3).ravel()
            water_count = len(wv)//3
            water_data = (water_count, wv, wtcoords, wn, wc)

        solid_data = (count, v, t, n, c)
        self.vt_data = {'solid': solid_data, 'water': water_data}

    def get_block(self, position, sector_position):
        pos = position - numpy.array(sector_position)
        if len(pos.shape)>1:
            pos = pos.T
            return self.blocks[pos[0]+1,pos[1],pos[2]+1]
        return self.blocks[pos[0]+1,pos[1],pos[2]+1]

    def set_block(self, position, sector_position, val):
        pos = position - numpy.array(sector_position)
        if len(pos.shape)>1:
            pos = pos.T
            self.blocks[pos[0]+1, pos[1], pos[2]+1] = val
        self.blocks[pos[0]+1, pos[1], pos[2]+1] = val

##TODO: Move to world_proxy so that we don't need to import the module
##and its dependencies in the main client process (probably doesn't
##really matter on Linux because of the way that fork works)
def _start_loader(client_pipe, server_pipe):
    WorldLoader(client_pipe, server_pipe)

def start_loader(server_pipe = None):
    pipe, _pipe = multiprocessing.Pipe()
    process = multiprocessing.Process(target = _start_loader, args = (_pipe, server_pipe))
    process.start()
    return pipe

