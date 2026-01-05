'''
world_loader.py -- manages client side terrain generation, world data caching, and syncing of terrain with multi-player server
'''

# standard library imports
import time
import numpy
import pickle
import multiprocessing
from collections import OrderedDict

# local imports
import config
from config import (
    SECTOR_SIZE,
    SECTOR_HEIGHT,
    LOADED_SECTORS,
    SERVER_IP,
    SERVER_PORT,
    LOADER_IP,
    LOADER_PORT,
    MAP_GEN_SECTOR_SIZE,
)
from util import normalize, sectorize, FACES
from blocks import BLOCK_ID
import mapgen

import logutil

def loader_log(msg, *args):
    if args:
        msg = msg % args
    logutil.log("LOADER", msg)

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
        self.blocks = numpy.zeros((SECTOR_SIZE, SECTOR_HEIGHT, SECTOR_SIZE), dtype='u2') #blocks of a sector not the whole world
        self.vt_data = None #vertex data for the current sector
        self.light = None
        self._sector_cache = OrderedDict()
        self._cache_limit = max(0, int(getattr(config, 'LOADER_CACHE_SECTORS', 64)))
        self._map_gen_sector_size = int(getattr(config, 'MAP_GEN_SECTOR_SIZE', SECTOR_SIZE))
        if self._map_gen_sector_size <= 0:
            self._map_gen_sector_size = SECTOR_SIZE
        if self._map_gen_sector_size % SECTOR_SIZE != 0:
            loader_log(
                'MAP_GEN_SECTOR_SIZE (%d) not divisible by SECTOR_SIZE (%d); '
                'batching may be misaligned',
                self._map_gen_sector_size, SECTOR_SIZE,
            )
        self._batch_size = max(1, self._map_gen_sector_size // SECTOR_SIZE)
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
                self.light = None
                self.vt_data = None
                t0 = time.perf_counter()
                payload = pickle.dumps(['sector_blocks',[self.pos, self.blocks, self.vt_data, self.light]],-1)
                t_dump = (time.perf_counter() - t0) * 1000.0
                t1 = time.perf_counter()
                cpipe.send_bytes(payload)
                t_send = (time.perf_counter() - t1) * 1000.0
                loader_log('pickle.dumps+send sector %s: %.1fms dump, %.1fms send, %d bytes', self.pos, t_dump, t_send, len(payload))
            if msg == 'get_seed':
                cpipe.send(['seed', self.world_seed])
            if msg == 'set_block':
                # data may include an optional client token at the end
                if len(data) == 4:
                    notify_server, pos, block_id, token = data
                    sector_data = None
                elif len(data) == 5:
                    notify_server, pos, block_id, sector_data, token = data
                else:
                    notify_server, pos, block_id, sector_data = data
                    token = None
                sector_result = []
                if sector_data:
                    for spos, blocks in sector_data:
                        self.blocks = blocks
                        self.set_block(pos, spos, block_id)
                        self.light = None
                        self.vt_data = None
                        sector_result.append((spos, self.blocks, self.vt_data, self.light))
                    t0 = time.perf_counter()
                    payload = pickle.dumps(['sector_blocks2', sector_result, token], -1)
                    t_dump = (time.perf_counter() - t0) * 1000.0
                    t1 = time.perf_counter()
                    cpipe.send_bytes(payload)
                    t_send = (time.perf_counter() - t1) * 1000.0
                    loader_log('pickle.dumps+send set_block batch: %.1fms dump, %.1fms send, %d bytes', t_dump, t_send, len(payload))
                else:
                    if self.db is not None:
                        self.db.set_block(pos, block_id)
                    cpipe.send(['set_block_ack', [pos, block_id, token]])
                if spipe is not None:
                    if notify_server:
                        spipe.send(('set_block', [pos, block_id]))
                else:
                    if sector_data:
                        self.db.set_block(pos, block_id)
            if msg == 'set_blocks':
                if len(data) == 3:
                    notify_server, updates, token = data
                    sector_data = None
                elif len(data) == 4:
                    notify_server, updates, sector_data, token = data
                else:
                    notify_server, updates, sector_data = data
                    token = None
                sector_result = []
                if sector_data:
                    for spos, blocks in sector_data:
                        self.blocks = blocks
                        for pos, block_id in updates:
                            self.set_block(pos, spos, block_id)
                        self.light = None
                        self.vt_data = None
                        sector_result.append((spos, self.blocks, self.vt_data, self.light))
                    t0 = time.perf_counter()
                    payload = pickle.dumps(['sector_blocks2', sector_result, token], -1)
                    t_dump = (time.perf_counter() - t0) * 1000.0
                    t1 = time.perf_counter()
                    cpipe.send_bytes(payload)
                    t_send = (time.perf_counter() - t1) * 1000.0
                    loader_log('pickle.dumps+send set_blocks batch: %.1fms dump, %.1fms send, %d bytes', t_dump, t_send, len(payload))
                else:
                    if self.db is not None:
                        for pos, block_id in updates:
                            self.db.set_block(pos, block_id)
                    cpipe.send(['set_blocks_ack', [updates, token]])
                if spipe is not None:
                    if notify_server:
                        for pos, block_id in updates:
                            spipe.send(('set_block', [pos, block_id]))
                else:
                    if sector_data:
                        for pos, block_id in updates:
                            self.db.set_block(pos, block_id)

    def _initialize(self, position, sector_block_delta):
        """ Initialize the sector by procedurally generating terrain using
        simplex noise.

        """
        self.blocks = self._get_sector_blocks(position, sector_block_delta)

    def _cache_get(self, position):
        blocks = self._sector_cache.get(position)
        if blocks is None:
            return None
        self._sector_cache.move_to_end(position)
        return blocks

    def _cache_put(self, position, blocks):
        if self._cache_limit <= 0:
            return
        self._sector_cache[position] = blocks
        self._sector_cache.move_to_end(position)
        while len(self._sector_cache) > self._cache_limit:
            self._sector_cache.popitem(last=False)

    def _batch_origin(self, position):
        x, _, z = position
        base_x = (x // self._map_gen_sector_size) * self._map_gen_sector_size
        base_z = (z // self._map_gen_sector_size) * self._map_gen_sector_size
        return (base_x, 0, base_z)

    def _fill_cache_for_batch(self, batch_origin):
        if getattr(config, 'USE_EXPERIMENTAL_BIOME_GEN', False):
            batch = mapgen.generate_biome_sector(batch_origin, None, None)
        else:
            batch = mapgen.generate_sector(batch_origin, None, None)
        step = SECTOR_SIZE
        limit = self._map_gen_sector_size
        for ox in range(0, limit, step):
            for oz in range(0, limit, step):
                pos = (batch_origin[0] + ox, 0, batch_origin[2] + oz)
                if pos in self._sector_cache:
                    continue
                sector_blocks = numpy.ascontiguousarray(
                    batch[ox:ox + step, :, oz:oz + step]
                )
                self._cache_put(pos, sector_blocks)

    def _get_sector_blocks(self, position, sector_block_delta):
        base = self._cache_get(position)
        if base is None:
            batch_origin = self._batch_origin(position)
            self._fill_cache_for_batch(batch_origin)
            base = self._cache_get(position)
            if base is None:
                loader_log('failed to cache sector %s after batch gen', position)
                base = numpy.zeros((SECTOR_SIZE, SECTOR_HEIGHT, SECTOR_SIZE), dtype='u2')
        if sector_block_delta is None:
            return base
        blocks = base.copy()
        for p in sector_block_delta:
            x, y, z = p
            if 0 <= x < SECTOR_SIZE and 0 <= y < SECTOR_HEIGHT and 0 <= z < SECTOR_SIZE:
                blocks[x, y, z] = sector_block_delta[p]
        return blocks

    def get_block(self, position, sector_position):
        pos = position - numpy.array(sector_position)
        if len(pos.shape)>1:
            pos = pos.T
            x = pos[0]
            y = pos[1]
            z = pos[2]
            mask = (
                (x >= 0) & (x < SECTOR_SIZE)
                & (y >= 0) & (y < SECTOR_HEIGHT)
                & (z >= 0) & (z < SECTOR_SIZE)
            )
            out = numpy.zeros(x.shape, dtype=self.blocks.dtype)
            if mask.any():
                out[mask] = self.blocks[x[mask], y[mask], z[mask]]
            return out
        x = int(pos[0])
        y = int(pos[1])
        z = int(pos[2])
        if x < 0 or x >= SECTOR_SIZE or y < 0 or y >= SECTOR_HEIGHT or z < 0 or z >= SECTOR_SIZE:
            return 0
        return self.blocks[x, y, z]

    def set_block(self, position, sector_position, val):
        pos = position - numpy.array(sector_position)
        if len(pos.shape)>1:
            pos = pos.T
            x = pos[0]
            y = pos[1]
            z = pos[2]
            mask = (
                (x >= 0) & (x < SECTOR_SIZE)
                & (y >= 0) & (y < SECTOR_HEIGHT)
                & (z >= 0) & (z < SECTOR_SIZE)
            )
            if mask.any():
                self.blocks[x[mask], y[mask], z[mask]] = val
            return
        x = int(pos[0])
        y = int(pos[1])
        z = int(pos[2])
        if x < 0 or x >= SECTOR_SIZE or y < 0 or y >= SECTOR_HEIGHT or z < 0 or z >= SECTOR_SIZE:
            return
        self.blocks[x, y, z] = val

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
