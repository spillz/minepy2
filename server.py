# standard library imports
import time
import numpy
import select
import pickle
import msocket
import socket
import logutil
import sys
import traceback
import config

from players import Player, ClientPlayer
import config
import world_db
    
def get_network_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.connect(('<broadcast>', 0))
    return s.getsockname()[0]


def start_server(ip, port):
    config.SERVER_IP = ip
    config.SERVER_PORT = port
    Server()
        
class ServerConnectionHandler(object):
    '''
    Handles the low level connection handling details of the multiplayer server
    '''
    def __init__(self):
        server_log = getattr(config, "SERVER_LOG_FILE_PATH", "log-server.txt")
        if server_log:
            config.LOG_FILE_PATH = server_log
            config.LOG_FILE_APPEND = True
        logutil.log("SERVER", f"starting server at {config.SERVER_IP}:{config.SERVER_PORT}")
        print(f"SERVER: listening on {config.SERVER_IP}:{config.SERVER_PORT}")
        self.listener = msocket.Listener(config.SERVER_IP, config.SERVER_PORT)
        self.players = []
        self.fn_dict = {}
        self.host_id = None
        self.host_name = None

    def register_function(self, name, fn):
        self.fn_dict[name]=fn

    def call_function(self,name,*args):
        return self.fn_dict[name](*args)

    def connections(self):
        return [p.conn for p in self.players]

    def connections_with_comms(self):
        try: #socket version is non-blocking so we need to check for incomplete sends
            return [p.conn for p in self.players if len(p.comms_queue)>0 or p.conn.unfinished_send()]
        except AttributeError: #multiprocessing version is blocking
            return [p.conn for p in self.players if len(p.comms_queue)>0]

    def player_from_connection(self, conn):
        for p in self.players:
            if conn == p.conn:
                return p

    def accept_connection(self):
        conn = self.listener.accept()
        player = Player(conn)
        self.players.append(player)
        return player

    def serve(self):
        alive = True
        while alive:
            try:
                r,w,x = select.select([self.listener] + self.connections(), self.connections_with_comms(), [], 0.5)
            except KeyboardInterrupt:
                logutil.log("SERVER", "received keyboard interrupt", level="WARN")
                break
            accept_new = True
            for p in list(self.players):
                if p.conn in r:
                    logutil.log("SERVER", f"r select for {p.id} {p.name}", level="DEBUG")
                    accept_new = False
                    try:
                        result = p.conn.recv()
                        if result is not None:
                            msg, data = result
                            logutil.log("SERVER", f"received {msg} from player {p.id} ({p.name})")
                            logutil.log("SERVER", f"data {data}", level="DEBUG")
                            if msg == 'quit':
                                logutil.log("SERVER", f"player {p.id} requested disconnect", level="INFO")
                                print(f"SERVER: player {p.id} disconnected")
                                if hasattr(self, "server_owner"):
                                    self.server_owner._request_player_state(p)
                                p.conn.close()
                                if p in self.players:
                                    self.players.remove(p)
                                self.queue_for_all_players(p, 'other_player_leave', p.id)
                                if p.id == self.host_id:
                                    if hasattr(self, "server_owner"):
                                        self.server_owner._request_entity_snapshot()
                                        self.host_id = None
                                        self.host_name = None
                                        self.server_owner._assign_new_host()
                            else:
                                try:
                                    self.call_function(msg, p, *data)
                                except Exception as ex:
                                    traceback.print_exc()
                    except EOFError:
                        #TODO: Could allow a few retries before dropping the player
                        logutil.log("SERVER", f"disconnect EOF for player {p.id} ({p.name})", level="WARN")
                        print(f"SERVER: player {p.id} disconnected")
                        p.conn.close()
                        if p in self.players:
                            self.players.remove(p)
                        self.queue_for_all_players(p, 'other_player_leave', p.id)
                        if p.id == self.host_id:
                            if hasattr(self, "server_owner"):
                                self.server_owner._request_entity_snapshot()
                                self.host_id = None
                                self.host_name = None
                                self.server_owner._assign_new_host()
            for p in self.players:
                if p.conn in w:
                    logutil.log("SERVER", f"w select for {p.id} {p.name}", level="DEBUG")
                    self.dispatch_top_message(p)
            if accept_new and self.listener in r:
                p = self.accept_connection()
                logutil.log("SERVER", f"connected new player id {p.id}")
                print(f"SERVER: player {p.id} connected from {getattr(p.conn, '_addr', 'client')}")
                if self.host_id is None:
                    self.host_id = p.id
                    self.host_name = p.name
                    self.queue_for_all_players(p, 'host_assign', self.host_id)
                    if getattr(self.server_owner, "last_entity_snapshot", None):
                        self.queue_for_player(p, 'entity_seed_snapshot', self.server_owner.last_entity_snapshot)
                self.queue_for_player(p, 'connected', ClientPlayer(p), [ClientPlayer(ap) for ap in self.players])
                self.queue_for_others(p, 'other_player_join', ClientPlayer(p))
                if self.host_id is not None and self.host_id != p.id:
                    host = next((pl for pl in self.players if pl.id == self.host_id), None)
                    if host is not None:
                        self.queue_for_player(host, 'entity_request_snapshot', p.id)
        self.listener.close()

    ##TODO: queue calls should collapse similar calls (e.g. multiple block adds in the same sector)
    ##TODO: prioritize calls
    def queue_for_player(self, player, message, *data):
        player.comms_queue.append([message, player.id, data])

    def queue_for_others(self, player, message, *data):
        for p in self.players:
            if p != player:
                p.comms_queue.append([message, player.id, data])

    def queue_for_all_players(self, player, message, *data):
        for p in self.players:
            p.comms_queue.append([message, player.id, data])

    def other_players(self, player):
        return [p for p in self.players if p != player]

    def dispatch_top_message(self, player):
        try: #socket version is non-blocking so we need to check for incomplete sends
            if player.conn.unfinished_send():
                if not player.conn.continue_send():
                    return
        except AttributeError: #multiprocessing version is blocking
            pass
        logutil.log("SERVER", f"sending {player.comms_queue[0][0]} to {player.id} ({player.name})")
        #player.conn.send_bytes(pickle.dumps(player.comms_queue.pop(0), -1))
        player.conn.send(player.comms_queue.pop(0))

class Server(object):
    '''
    minepy Multiplayer Server
    manages connections from players and handles data requests

    Maintains the following databases
        block information (delta from what the terrain generator produces)
        player information (unique id/name, location, velocity)

    Server Messages (client must have handlers for these)
        connected(players)
            notifies the player that just connect with a list of all `players`
        other_player_join(player)
            notifies all other players that `player` has joined
        player_set_name(player, name)
            notifies all players that `player` is using name `name`
        player_set_position(player, position)
            notifies all other players that `player` is at `position`
        player_set_block(player, position, block)
            notifies all players that `player` has set `block` at `position`
        updated_sector_blocks(sector_pos, sector_blocks_delta)
            sends `sector_blocks_delta` to the player that requested it
    '''
    def __init__(self):
        self.handler = ServerConnectionHandler()
        self.handler.server_owner = self
        self.world_db = world_db.World()
        print(f"SERVER: ready seed={self.world_db.get_seed()}")
        self.player_state = {}
        self.last_entity_snapshot = self._load_entity_snapshot()
        #TODO: could use a decorator to avoid explicit registration, though I think this is more readable
        self.handler.register_function('set_name',self.set_name)
        self.handler.register_function('set_position',self.set_position)
        self.handler.register_function('set_block',self.set_block)
        self.handler.register_function('sector_blocks',self.sector_blocks)
        self.handler.register_function('l_get_sector_blocks',self.l_get_sector_blocks)
        self.handler.register_function('l_get_seed',self.l_get_seed)
        self.handler.register_function('entity_snapshot', self.entity_snapshot)
        self.handler.register_function('entity_update', self.entity_update)
        self.handler.register_function('entity_spawn', self.entity_spawn)
        self.handler.register_function('entity_despawn', self.entity_despawn)
        try:
            self.handler.serve()
        except KeyboardInterrupt:
            logutil.log("SERVER", "received keyboard interrupt", level="WARN")
            logutil.log("SERVER", "shutting down")
            self._request_all_player_state()
            self._request_entity_snapshot()
            #TODO: Notify players that we're shutting down

    def _request_all_player_state(self):
        for player in list(self.handler.players):
            self._request_player_state(player)

    def _assign_new_host(self):
        if not self.handler.players:
            self.handler.host_id = None
            self.handler.host_name = None
            return
        new_host = min(self.handler.players, key=lambda p: p.id)
        self.handler.host_id = new_host.id
        self.handler.host_name = new_host.name
        print(f"SERVER: host reassigned to {new_host.id} ({new_host.name})")
        self.handler.queue_for_all_players(new_host, 'host_assign', self.handler.host_id)
        if self.last_entity_snapshot:
            self.handler.queue_for_player(new_host, 'entity_seed_snapshot', self.last_entity_snapshot)
        self.handler.queue_for_player(new_host, 'entity_request_snapshot', None)

    def _request_player_state(self, player):
        if player is None:
            return
        self.handler.queue_for_player(player, 'player_state_request')
        self.handler.dispatch_top_message(player)

    def _request_entity_snapshot(self):
        if self.handler.host_id is None:
            self.last_entity_snapshot = None
            self._save_entity_snapshot(None)
            return
        host = next((pl for pl in self.handler.players if pl.id == self.handler.host_id), None)
        if host is None:
            self.last_entity_snapshot = None
            self._save_entity_snapshot(None)
            return
        self.handler.queue_for_player(host, 'entity_request_snapshot', None)
        self.handler.dispatch_top_message(host)
        deadline = time.perf_counter() + 0.5
        while time.perf_counter() < deadline:
            try:
                result = host.conn.recv()
            except Exception:
                break
            if result is None:
                continue
            msg, data = result
            if msg == 'entity_snapshot':
                try:
                    self.entity_snapshot(host, data[0] if isinstance(data, (list, tuple)) and data else data)
                except Exception:
                    pass
                return
        self.last_entity_snapshot = None
        self._save_entity_snapshot(None)

    def set_name(self, player, name):
        '''
        sets the unique name for the player
        player's client must wait for the confirmation player_set_name message from
        the server before assuming the name
        '''
        used = False
        for op in self.handler.other_players(player):
            if op.name == name:
                used = True
        if not used:
            old_name = player.name
            player.name = name
            print(f"SERVER: player {player.id} name set to {player.name}")
            if old_name in self.player_state:
                self.player_state[player.name] = self.player_state.pop(old_name)
            if player.id == self.handler.host_id:
                self.handler.host_name = player.name
            elif self.handler.host_name and player.name == self.handler.host_name:
                self.handler.host_id = player.id
                print(f"SERVER: host reassigned to {player.id} ({player.name})")
                self.handler.queue_for_all_players(player, 'host_assign', self.handler.host_id)
                self.handler.queue_for_player(player, 'entity_request_snapshot', None)
        self.handler.queue_for_all_players(player, 'player_set_name', player.id, player.name)
        state = self._load_player_state(player.name)
        if state:
            self.handler.queue_for_player(player, 'player_spawn_state', state)

    def set_position(self, player, position=None, rotation=None, state=None):
        '''
        sets the `position` tuple for the `player`
        confirmation is not required, player_set_position message is
        broadcast only to other players
        '''
        if isinstance(position, dict) and state is None:
            state = position
            position = state.get("pos")
            rotation = state.get("rot")
        if state is None:
            state = {}
        if position is not None:
            player.position = position
        if rotation is not None:
            player.rotation = rotation
        self.handler.queue_for_others(player, 'player_set_position', player.id, player.position, player.rotation)
        if player.name:
            full_state = dict(state)
            full_state.setdefault("pos", player.position)
            full_state.setdefault("rot", player.rotation)
            full_state.setdefault("name", player.name)
            self._save_player_state(player.name, full_state)

    def set_block(self, player, position, block):
        '''
        sets the block at `position` to the id given by `block`
        player's client must wait for the confirmation `player_set_name` message from
        the server before assuming the block has been set (to avoid synchronization issues
        when multiple players set the same block)
        '''
        self.world_db.set_block(position, block)
        self.handler.queue_for_others(player, 'player_set_block', position, block)

    def sector_blocks(self, player, sector_pos):
        '''
        request by `player` for the changed blocks in `sector_pos`
        data will be sent with the message `sector_blocks`
        '''
        blocks = self.world_db.get_sector_data(sector_pos)
        self.handler.queue_for_player(player, 'sector_blocks_changed', sector_pos, blocks)

    def l_get_sector_blocks(self, player, sector_pos):
        '''
        request by `player`'s loader for the changed blocks in `sector_pos`
        data will be sent with the message `sector_blocks`
        '''
        blocks = self.world_db.get_sector_data(sector_pos)
        self.handler.queue_for_player(player, 'l_sector_blocks_changed', sector_pos, blocks)

    def l_get_seed(self, player):
        self.handler.queue_for_player(player, 'l_seed', self.world_db.get_seed())

    def _is_host(self, player):
        return player is not None and player.id == self.handler.host_id

    def entity_snapshot(self, player, payload):
        if not self._is_host(player):
            return
        self.last_entity_snapshot = payload
        self._save_entity_snapshot(payload)
        self.handler.queue_for_others(player, 'entity_snapshot', payload)

    def entity_update(self, player, payload):
        if not self._is_host(player):
            return
        self.handler.queue_for_others(player, 'entity_update', payload)

    def entity_spawn(self, player, payload):
        if not self._is_host(player):
            return
        self.handler.queue_for_others(player, 'entity_spawn', payload)

    def entity_despawn(self, player, payload):
        if not self._is_host(player):
            return
        self.handler.queue_for_others(player, 'entity_despawn', payload)

    def _load_player_state(self, name):
        try:
            return self.world_db.db.get(f"player_state:{name}")
        except KeyError:
            return None

    def _save_player_state(self, name, state):
        try:
            self.world_db.db.put(f"player_state:{name}", state)
        except Exception:
            pass

    def _load_entity_snapshot(self):
        try:
            return self.world_db.db.get("entity_snapshot")
        except KeyError:
            return None

    def _save_entity_snapshot(self, payload):
        try:
            self.world_db.db.put("entity_snapshot", payload)
        except Exception:
            pass


if __name__ == '__main__':
    #TODO: use a config file for server settings
    #TODO: use argparse module to override default server settings
    config.SERVER_IP = 'localhost'
    if len(sys.argv)>1:
        if sys.argv[1] == 'LAN':
            config.SERVER_IP = get_network_ip()
        elif ':' in sys.argv[1]:
            host, port = sys.argv[1].split(':', 1)
            config.SERVER_IP = host
            try:
                config.SERVER_PORT = int(port)
            except ValueError:
                pass
        else:
            config.SERVER_IP = sys.argv[1]
    s = Server()
