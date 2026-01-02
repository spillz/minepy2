import os
import sys
import time
import types

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config


def _install_pyglet_stub():
    if "pyglet" in sys.modules:
        return

    pyglet_stub = types.ModuleType("pyglet")
    pyglet_stub.options = {}

    image_stub = types.ModuleType("pyglet.image")
    graphics_stub = types.ModuleType("pyglet.graphics")
    math_stub = types.ModuleType("pyglet.math")
    gl_stub = types.ModuleType("pyglet.gl")

    class TextureGroup:
        pass

    class Mat4:
        pass

    graphics_stub.TextureGroup = TextureGroup
    math_stub.Mat4 = Mat4

    pyglet_stub.image = image_stub
    pyglet_stub.graphics = graphics_stub
    pyglet_stub.math = math_stub
    pyglet_stub.gl = gl_stub

    sys.modules.update({
        "pyglet": pyglet_stub,
        "pyglet.image": image_stub,
        "pyglet.graphics": graphics_stub,
        "pyglet.math": math_stub,
        "pyglet.gl": gl_stub,
    })


def _build_tile(mapgen, seed, sector_pos, sector_size, sector_height):
    mapgen.initialize_map_generator(seed=seed)
    tile = np.zeros((sector_size * 3, sector_height, sector_size * 3), dtype="u2")
    base_x, _, base_z = sector_pos
    for dz in (-1, 0, 1):
        for dx in (-1, 0, 1):
            pos = (base_x + dx * sector_size, 0, base_z + dz * sector_size)
            sector_blocks = mapgen.generate_sector(pos, None, None)
            x0 = (dx + 1) * sector_size
            z0 = (dz + 1) * sector_size
            tile[x0:x0 + sector_size, :, z0:z0 + sector_size] = sector_blocks
    return tile


def _run_light(world_proxy, tile, center, use_bfs, debug_light_grids=False):
    prev = getattr(config, "LIGHT_PROPAGATION_BFS", False)
    config.LIGHT_PROPAGATION_BFS = use_bfs
    try:
        return world_proxy.ModelProxy._build_mesh_job(
            blocks=center,
            light_blocks=tile,
            position=(0, 0, 0),
            ao_strength=0.0,
            ao_enabled=False,
            ambient=0.0,
            incoming_sky={},
            incoming_torch={},
            internal_sky_floor=None,
            internal_sky_side=world_proxy.EMPTY_LIGHT_LIST,
            internal_torch=world_proxy.EMPTY_LIGHT_LIST,
            recompute_internal=True,
            build_mesh=False,
            debug_light_grids=debug_light_grids,
        )
    finally:
        config.LIGHT_PROPAGATION_BFS = prev


def _print_light_map(seed, label, tile_shape, result, use_updates=False):
    torch_grid = result.get("debug_torch_grid")
    sky_grid = result.get("debug_sky_grid")
    sky_direct = result.get("debug_sky_direct_mask")
    torch_sources = result.get("debug_torch_sources")
    torch_updates = result.get("debug_torch_updates")
    sky_updates = result.get("debug_sky_updates")

    if torch_grid is None:
        torch_any = np.zeros(tile_shape, dtype=bool)
    else:
        torch_any = torch_grid > 0
    if sky_grid is None:
        sky_any = np.zeros(tile_shape, dtype=bool)
    else:
        sky_any = sky_grid > 0
    if sky_direct is None:
        sky_direct = np.zeros(tile_shape, dtype=bool)
    if torch_sources is None:
        torch_sources = np.zeros(tile_shape, dtype=bool)

    if use_updates and sky_updates is not None:
        sky_spread = sky_updates
    else:
        sky_spread = sky_any & ~sky_direct
    if use_updates and torch_updates is not None:
        torch_spread = torch_updates
    else:
        torch_spread = torch_any & ~torch_sources

    torch_source_2d = torch_sources.any(axis=1)
    torch_spread_2d = torch_spread.any(axis=1)
    sky_direct_2d = sky_direct.any(axis=1)
    if sky_grid is not None:
        sky_dim = sky_grid.shape[1]
        sky_spread_level = (sky_grid > 0) & (sky_grid < sky_dim)
    else:
        sky_spread_level = sky_spread
    sky_spread_2d = sky_spread_level.any(axis=1)

    lit_cells = int(np.count_nonzero(torch_any | sky_any))
    if sky_grid is not None:
        sky_spread_cells = int(np.count_nonzero((sky_grid > 0) & (sky_grid < sky_grid.shape[1])))
    else:
        sky_spread_cells = 0
    sky_sources = int(np.count_nonzero(sky_direct))

    width, _, depth = tile_shape
    grid = np.full((width, depth), ".", dtype="<U1")
    grid[sky_spread_2d] = "~"
    grid[sky_direct_2d] = "="
    grid[torch_spread_2d] = "+"
    grid[torch_source_2d] = "*"

    print(
        "seed=%d %s lit_cells=%d sky_sources=%d sky_spread_cells=%d"
        % (seed, label, lit_cells, sky_sources, sky_spread_cells)
    )
    for z in range(depth):
        print("".join(grid[:, z]))


def test_light_propagation_bfs_matches_dense():
    _install_pyglet_stub()
    import blocks
    import mapgen
    import world_proxy
    from config import SECTOR_HEIGHT, SECTOR_SIZE

    seed = 1337
    sector_pos = (0, 0, 0)
    tile = _build_tile(mapgen, seed, sector_pos, SECTOR_SIZE, SECTOR_HEIGHT)
    sx = SECTOR_SIZE
    torch_id = blocks.BLOCK_ID["Wall Torch"]
    torch_pos = (sx + sx // 2, SECTOR_HEIGHT // 2, sx + sx // 2)
    tile[torch_pos] = torch_id
    center = tile[sx:sx * 2, :, sx:sx * 2].copy()

    dense = _run_light(world_proxy, tile, center, use_bfs=False)
    bfs = _run_light(world_proxy, tile, center, use_bfs=True)

    assert np.array_equal(dense["sky_floor"], bfs["sky_floor"])
    assert world_proxy._light_list_equal(dense["sky_side"], bfs["sky_side"])
    assert world_proxy._light_list_equal(dense["torch_side"], bfs["torch_side"])
    for offset in world_proxy.NEIGHBOR_OFFSETS_8:
        assert world_proxy._light_list_equal(dense["outgoing_sky"][offset], bfs["outgoing_sky"][offset])
        assert world_proxy._light_list_equal(dense["outgoing_torch"][offset], bfs["outgoing_torch"][offset])


def test_light_propagation_speed_comparison():
    _install_pyglet_stub()
    import blocks
    import mapgen
    import world_proxy
    from config import SECTOR_HEIGHT, SECTOR_SIZE

    seeds = [1337 + (i * 17) for i in range(10)]
    runs_per_seed = 3
    dense_times = []
    bfs_times = []
    dense_torch_times = []
    bfs_torch_times = []
    dense_sky_times = []
    bfs_sky_times = []
    sector_pos = (0, 0, 0)
    sx = SECTOR_SIZE
    torch_id = blocks.BLOCK_ID["Wall Torch"]

    for seed in seeds:
        tile = _build_tile(mapgen, seed, sector_pos, SECTOR_SIZE, SECTOR_HEIGHT)
        torch_pos = (sx + sx // 2, SECTOR_HEIGHT // 2, sx + sx // 2)
        tile[torch_pos] = torch_id
        center = tile[sx:sx * 2, :, sx:sx * 2].copy()

        debug_relax = _run_light(world_proxy, tile, center, use_bfs=False, debug_light_grids=True)
        _print_light_map(seed, "relax", tile.shape, debug_relax, use_updates=False)
        debug_bfs = _run_light(world_proxy, tile, center, use_bfs=True, debug_light_grids=True)
        _print_light_map(seed, "bfs", tile.shape, debug_bfs, use_updates=True)

        _run_light(world_proxy, tile, center, use_bfs=False)
        _run_light(world_proxy, tile, center, use_bfs=True)

        start = time.perf_counter()
        dense_torch_ms = 0.0
        dense_sky_ms = 0.0
        for _ in range(runs_per_seed):
            result = _run_light(world_proxy, tile, center, use_bfs=False)
            dense_torch_ms += result.get("light_torch_ms", 0.0)
            dense_sky_ms += result.get("light_sky_ms", 0.0)
        dense_times.append(time.perf_counter() - start)
        dense_torch_times.append(dense_torch_ms / 1000.0)
        dense_sky_times.append(dense_sky_ms / 1000.0)

        start = time.perf_counter()
        bfs_torch_ms = 0.0
        bfs_sky_ms = 0.0
        for _ in range(runs_per_seed):
            result = _run_light(world_proxy, tile, center, use_bfs=True)
            bfs_torch_ms += result.get("light_torch_ms", 0.0)
            bfs_sky_ms += result.get("light_sky_ms", 0.0)
        bfs_times.append(time.perf_counter() - start)
        bfs_torch_times.append(bfs_torch_ms / 1000.0)
        bfs_sky_times.append(bfs_sky_ms / 1000.0)

    dense_total = sum(dense_times)
    bfs_total = sum(bfs_times)
    ratios = [
        (bfs / dense) if dense else float("inf")
        for dense, bfs in zip(dense_times, bfs_times)
    ]
    torch_total_dense = sum(dense_torch_times)
    torch_total_bfs = sum(bfs_torch_times)
    torch_ratios = [
        (bfs / dense) if dense else float("inf")
        for dense, bfs in zip(dense_torch_times, bfs_torch_times)
    ]
    sky_total_dense = sum(dense_sky_times)
    sky_total_bfs = sum(bfs_sky_times)
    sky_ratios = [
        (bfs / dense) if dense else float("inf")
        for dense, bfs in zip(dense_sky_times, bfs_sky_times)
    ]
    ratio = bfs_total / dense_total if dense_total else float("inf")
    ratio_best = min(ratios) if ratios else float("inf")
    ratio_worst = max(ratios) if ratios else float("inf")
    ratio_avg = (sum(ratios) / len(ratios)) if ratios else float("inf")
    ratio_median = float(np.median(ratios)) if ratios else float("inf")
    torch_ratio = torch_total_bfs / torch_total_dense if torch_total_dense else float("inf")
    torch_best = min(torch_ratios) if torch_ratios else float("inf")
    torch_worst = max(torch_ratios) if torch_ratios else float("inf")
    torch_avg = (sum(torch_ratios) / len(torch_ratios)) if torch_ratios else float("inf")
    torch_median = float(np.median(torch_ratios)) if torch_ratios else float("inf")
    sky_ratio = sky_total_bfs / sky_total_dense if sky_total_dense else float("inf")
    sky_best = min(sky_ratios) if sky_ratios else float("inf")
    sky_worst = max(sky_ratios) if sky_ratios else float("inf")
    sky_avg = (sum(sky_ratios) / len(sky_ratios)) if sky_ratios else float("inf")
    sky_median = float(np.median(sky_ratios)) if sky_ratios else float("inf")
    print(
        "light_speed seeds=%d runs_per_seed=%d dense_total=%.6fs bfs_total=%.6fs "
        "ratio=%.2fx best=%.2fx worst=%.2fx avg=%.2fx median=%.2fx "
        "torch_dense=%.6fs torch_bfs=%.6fs torch_ratio=%.2fx torch_best=%.2fx "
        "torch_worst=%.2fx torch_avg=%.2fx torch_median=%.2fx "
        "sky_dense=%.6fs sky_bfs=%.6fs sky_ratio=%.2fx sky_best=%.2fx "
        "sky_worst=%.2fx sky_avg=%.2fx sky_median=%.2fx"
        % (
            len(seeds),
            runs_per_seed,
            dense_total,
            bfs_total,
            ratio,
            ratio_best,
            ratio_worst,
            ratio_avg,
            ratio_median,
            torch_total_dense,
            torch_total_bfs,
            torch_ratio,
            torch_best,
            torch_worst,
            torch_avg,
            torch_median,
            sky_total_dense,
            sky_total_bfs,
            sky_ratio,
            sky_best,
            sky_worst,
            sky_avg,
            sky_median,
        )
    )


def _run_speed_benchmark():
    _install_pyglet_stub()
    import blocks
    import mapgen
    import world_proxy
    from config import SECTOR_HEIGHT, SECTOR_SIZE

    seeds = [1337 + (i * 17) for i in range(10)]
    runs_per_seed = 3
    dense_times = []
    bfs_times = []
    dense_torch_times = []
    bfs_torch_times = []
    dense_sky_times = []
    bfs_sky_times = []
    sector_pos = (0, 0, 0)
    sx = SECTOR_SIZE
    torch_id = blocks.BLOCK_ID["Wall Torch"]

    for seed in seeds:
        tile = _build_tile(mapgen, seed, sector_pos, SECTOR_SIZE, SECTOR_HEIGHT)
        torch_pos = (sx + sx // 2, SECTOR_HEIGHT // 2, sx + sx // 2)
        tile[torch_pos] = torch_id
        center = tile[sx:sx * 2, :, sx:sx * 2].copy()

        debug_relax = _run_light(world_proxy, tile, center, use_bfs=False, debug_light_grids=True)
        _print_light_map(seed, "relax", tile.shape, debug_relax, use_updates=False)
        debug_bfs = _run_light(world_proxy, tile, center, use_bfs=True, debug_light_grids=True)
        _print_light_map(seed, "bfs", tile.shape, debug_bfs, use_updates=True)

        _run_light(world_proxy, tile, center, use_bfs=False)
        _run_light(world_proxy, tile, center, use_bfs=True)

        start = time.perf_counter()
        dense_torch_ms = 0.0
        dense_sky_ms = 0.0
        for _ in range(runs_per_seed):
            result = _run_light(world_proxy, tile, center, use_bfs=False)
            dense_torch_ms += result.get("light_torch_ms", 0.0)
            dense_sky_ms += result.get("light_sky_ms", 0.0)
        dense_times.append(time.perf_counter() - start)
        dense_torch_times.append(dense_torch_ms / 1000.0)
        dense_sky_times.append(dense_sky_ms / 1000.0)

        start = time.perf_counter()
        bfs_torch_ms = 0.0
        bfs_sky_ms = 0.0
        for _ in range(runs_per_seed):
            result = _run_light(world_proxy, tile, center, use_bfs=True)
            bfs_torch_ms += result.get("light_torch_ms", 0.0)
            bfs_sky_ms += result.get("light_sky_ms", 0.0)
        bfs_times.append(time.perf_counter() - start)
        bfs_torch_times.append(bfs_torch_ms / 1000.0)
        bfs_sky_times.append(bfs_sky_ms / 1000.0)

    dense_total = sum(dense_times)
    bfs_total = sum(bfs_times)
    ratios = [
        (bfs / dense) if dense else float("inf")
        for dense, bfs in zip(dense_times, bfs_times)
    ]
    torch_total_dense = sum(dense_torch_times)
    torch_total_bfs = sum(bfs_torch_times)
    torch_ratios = [
        (bfs / dense) if dense else float("inf")
        for dense, bfs in zip(dense_torch_times, bfs_torch_times)
    ]
    sky_total_dense = sum(dense_sky_times)
    sky_total_bfs = sum(bfs_sky_times)
    sky_ratios = [
        (bfs / dense) if dense else float("inf")
        for dense, bfs in zip(dense_sky_times, bfs_sky_times)
    ]
    ratio = bfs_total / dense_total if dense_total else float("inf")
    ratio_best = min(ratios) if ratios else float("inf")
    ratio_worst = max(ratios) if ratios else float("inf")
    ratio_avg = (sum(ratios) / len(ratios)) if ratios else float("inf")
    ratio_median = float(np.median(ratios)) if ratios else float("inf")
    torch_ratio = torch_total_bfs / torch_total_dense if torch_total_dense else float("inf")
    torch_best = min(torch_ratios) if torch_ratios else float("inf")
    torch_worst = max(torch_ratios) if torch_ratios else float("inf")
    torch_avg = (sum(torch_ratios) / len(torch_ratios)) if torch_ratios else float("inf")
    torch_median = float(np.median(torch_ratios)) if torch_ratios else float("inf")
    sky_ratio = sky_total_bfs / sky_total_dense if sky_total_dense else float("inf")
    sky_best = min(sky_ratios) if sky_ratios else float("inf")
    sky_worst = max(sky_ratios) if sky_ratios else float("inf")
    sky_avg = (sum(sky_ratios) / len(sky_ratios)) if sky_ratios else float("inf")
    sky_median = float(np.median(sky_ratios)) if sky_ratios else float("inf")
    print(
        "light_speed seeds=%d runs_per_seed=%d dense_total=%.6fs bfs_total=%.6fs "
        "ratio=%.2fx best=%.2fx worst=%.2fx avg=%.2fx median=%.2fx "
        "torch_dense=%.6fs torch_bfs=%.6fs torch_ratio=%.2fx torch_best=%.2fx "
        "torch_worst=%.2fx torch_avg=%.2fx torch_median=%.2fx "
        "sky_dense=%.6fs sky_bfs=%.6fs sky_ratio=%.2fx sky_best=%.2fx "
        "sky_worst=%.2fx sky_avg=%.2fx sky_median=%.2fx"
        % (
            len(seeds),
            runs_per_seed,
            dense_total,
            bfs_total,
            ratio,
            ratio_best,
            ratio_worst,
            ratio_avg,
            ratio_median,
            torch_total_dense,
            torch_total_bfs,
            torch_ratio,
            torch_best,
            torch_worst,
            torch_avg,
            torch_median,
            sky_total_dense,
            sky_total_bfs,
            sky_ratio,
            sky_best,
            sky_worst,
            sky_avg,
            sky_median,
        )
    )


if __name__ == "__main__":
    _run_speed_benchmark()
