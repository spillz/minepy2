# Developer Guide

## Overview
This codebase is a pyglet-driven voxel world with numpy-heavy data processing. Pyglet owns the window, input, OpenGL state, shader program binding, and the per-frame render loop. Numpy backs nearly all world data and mesh generation: terrain is generated into large arrays, lighting/AO are computed in bulk, and vertex/texture/color streams are assembled from vectorized operations for speed.

## Modules (Python only)
- `main.py`: Pyglet window, input handling, camera math, game loop, entity update/animation, and render ordering.
- `config.py`: Global constants and toggles for world sizes, lighting, terrain generation, and rendering.
- `blocks.py`: Block definitions, textures, colors, occlusion rules, and ID/metadata arrays.
- `mapgen.py`: Procedural terrain/biome generation, rivers/roads, decorations, and water filling.
- `noise.py`: Simplex noise implementation (2D/3D/4D) used by terrain generation.
- `util.py`: Geometry helpers (cube vertices, texture coords), AO computation, and shared math utilities.
- `renderer.py`: Entity mesh construction and rendering, animation interpolation, and snake segment batching.
- `shaders.py`: GLSL shader sources and the block shader program creation.
- `world_loader.py`: Background loader process for sector generation, lighting, mesh data, and server sync.
- `world_proxy.py`: Client-side world model, sector cache, mesh upload, water pass, collision, and rendering.
- `world_db.py`: LMDB-backed storage for sector deltas and world seed.
- `world_entity_store.py`: Persist/load entity state in `world.db`.
- `entity.py`: BaseEntity physics and collision integration plus network serialization helpers.
- `players.py`: Server-side player records and lightweight client view.
- `server.py`: Multiplayer server loop, message routing, and block updates.
- `server_connection.py`: Client-side server bridge between client/loader and multiplayer server.
- `msocket.py`: Transport abstraction (multiprocessing or sockets) and UDP discovery helpers.
- `tests/__init__.py`: Test package marker.

## Entities package
- `entities/player.py`: Humanoid model definition and player entity movement/animation logic.
- `entities/snake.py`: Snake model plus segmented follow behavior and path history.
- `entities/snail.py`: Snail model with slow wander/follow logic.
- `entities/seagull.py`: Seagull model with flying wander/circle behavior.
- `entities/tetrapod.py`: Generic quadruped model builder and base tetrapod entity behavior.
- `entities/dog.py`: Dog model built from tetrapod and simple wandering AI.
- `entities/__init__.py`: Package marker.

## Tests
Tests live in `tests/`. We use `pytest`.

Install dependency:

    pip install pytest

Run diagnostics (bash):

    python -m pytest -vv -s tests/test_mapgen_diagnostics.py

Run diagnostics (PowerShell):

    python -m pytest -vv -s tests/test_mapgen_diagnostics.py

Optional: dump seam maps for every test case (PowerShell):

    $env:MAPGEN_DUMP="1"; python -m pytest -vv -s tests/test_mapgen_diagnostics.py
