from world_db import SectorDB

GLOBAL_KEY = "global_entities"
_store = SectorDB("world.db")
_cache = None

def _load_all():
    global _cache
    if _cache is None:
        try:
            _cache = _store.get(GLOBAL_KEY)
        except KeyError:
            _cache = {}
    return _cache

def load_entity_state(name):
    return _load_all().get(name)

def save_entity_state(name, state):
    global _cache
    data = _load_all()
    data[name] = state
    _store.put(GLOBAL_KEY, data)
    _cache = data
