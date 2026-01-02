import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
import mapgen
from config import SECTOR_SIZE, SECTOR_HEIGHT
from blocks import BLOCK_SOLID, BLOCK_ID

DUMP_MAPS = os.environ.get("MAPGEN_DUMP", "").strip() in ("1", "true", "yes", "on")
DUMP_DIR = os.environ.get("MAPGEN_DUMP_DIR", os.path.join(ROOT, "tests", "mapgen_dumps"))


def _init_generator(seed, use_biome):
    if use_biome:
        mapgen.biome_generator = None
        mapgen.initialize_biome_map_generator(seed=seed)
        return mapgen.biome_generator
    mapgen.initialize_map_generator(seed=seed)
    return None


def _gen_sector(position, use_biome):
    if use_biome:
        if mapgen.biome_generator is None:
            mapgen.initialize_biome_map_generator(seed=12345)
        return mapgen.generate_biome_sector(position, None, None)
    if not hasattr(mapgen, "noise1"):
        mapgen.initialize_map_generator(seed=12345)
    return mapgen.generate_sector(position, None, None)


def _edge_mismatch_counts(edge_a, edge_b, tol=0.0):
    diff = np.abs(edge_a.astype(np.float32) - edge_b.astype(np.float32))
    mism = diff > tol
    return int(np.count_nonzero(mism)), int(edge_a.size)


def _edge_stats(edge_a, edge_b):
    diff = np.abs(edge_a.astype(np.float32) - edge_b.astype(np.float32))
    return {
        "max": float(diff.max(initial=0.0)),
        "mean": float(diff.mean()),
        "p95": float(np.percentile(diff, 95)),
    }


def _diff_stats(diff):
    diff = diff.astype(np.float32)
    return {
        "max": float(diff.max(initial=0.0)),
        "mean": float(diff.mean()),
        "p95": float(np.percentile(diff, 95)),
        "p99": float(np.percentile(diff, 99)),
    }


def _column_heights(blocks):
    # blocks shape: (X,Y,Z)
    solid = BLOCK_SOLID[blocks]
    any_solid = solid.any(axis=1)
    rev = solid[:, ::-1, :]
    top_from_rev = np.argmax(rev, axis=1)
    heights = (SECTOR_HEIGHT - 1) - top_from_rev
    return np.where(any_solid, heights, -1)


def _format_height_map(h_map, bad_mask=None, title=None):
    lines = []
    if title:
        lines.append(title)
    for z in range(h_map.shape[1]):
        row = []
        for x in range(h_map.shape[0]):
            val = int(round(float(h_map[x, z])))
            if val < 0:
                cell = "--"
            else:
                val = min(255, val)
                cell = f"{val:02X}"
            if bad_mask is not None and bad_mask[x, z]:
                cell = f"\x1b[31m{cell}\x1b[0m"
            row.append(cell)
        lines.append(" ".join(row))
    return "\n".join(lines)


def _print_seam_map(height_o, height_x, height_z, bad_x, bad_z, title):
    size = SECTOR_SIZE * 2
    h_map = np.full((size, size), 0.0, dtype=np.float32)
    h_map[:SECTOR_SIZE, :SECTOR_SIZE] = height_o
    h_map[SECTOR_SIZE:, :SECTOR_SIZE] = height_x
    h_map[:SECTOR_SIZE, SECTOR_SIZE:] = height_z
    bad_mask = np.zeros((size, size), dtype=bool)
    # X seam: between O (x=SECTOR_SIZE-1) and X (x=SECTOR_SIZE)
    bad_mask[SECTOR_SIZE - 1, :SECTOR_SIZE] |= bad_x
    bad_mask[SECTOR_SIZE, :SECTOR_SIZE] |= bad_x
    # Z seam: between O (z=SECTOR_SIZE-1) and Z (z=SECTOR_SIZE)
    bad_mask[:SECTOR_SIZE, SECTOR_SIZE - 1] |= bad_z
    bad_mask[:SECTOR_SIZE, SECTOR_SIZE] |= bad_z
    print(_format_height_map(h_map, bad_mask=bad_mask, title=title))


def _dump_seam_map(height_o, height_x, height_z, bad_x, bad_z, name):
    if not DUMP_MAPS:
        return
    os.makedirs(DUMP_DIR, exist_ok=True)
    size = SECTOR_SIZE * 2
    h_map = np.full((size, size), 0.0, dtype=np.float32)
    h_map[:SECTOR_SIZE, :SECTOR_SIZE] = height_o
    h_map[SECTOR_SIZE:, :SECTOR_SIZE] = height_x
    h_map[:SECTOR_SIZE, SECTOR_SIZE:] = height_z
    bad_mask = np.zeros((size, size), dtype=bool)
    bad_mask[SECTOR_SIZE - 1, :SECTOR_SIZE] |= bad_x
    bad_mask[SECTOR_SIZE, :SECTOR_SIZE] |= bad_x
    bad_mask[:SECTOR_SIZE, SECTOR_SIZE - 1] |= bad_z
    bad_mask[:SECTOR_SIZE, SECTOR_SIZE] |= bad_z
    path = os.path.join(DUMP_DIR, f"{name}.txt")
    with open(path, "w", encoding="ascii") as f:
        f.write(_format_height_map(h_map, bad_mask=bad_mask, title=name))
        f.write("\n")


def test_heightfield_seams():
    use_biome = bool(getattr(config, "USE_EXPERIMENTAL_BIOME_GEN", False))
    if not use_biome:
        return
    max_extra = {"mean": 1.5, "p95": 3.0, "p99": 4.0}
    warn_ratio = 0.01

    rng = np.random.RandomState(1337)
    offsets = list(rng.randint(-6, 7, size=(10, 2)) * SECTOR_SIZE)
    seeds = list(rng.randint(1, 1_000_000, size=10))

    def check_once(gen, offset_x, offset_z, seed_tag):
        pos_o = (int(offset_x), 0, int(offset_z))
        pos_x = (int(offset_x + SECTOR_SIZE), 0, int(offset_z))
        pos_z = (int(offset_x), 0, int(offset_z + SECTOR_SIZE))

        elev_o, _, _ = gen._height_field(pos_o)
        elev_x, _, _ = gen._height_field(pos_x)
        elev_z, _, _ = gen._height_field(pos_z)

        edge_x_o = elev_o[SECTOR_SIZE - 1, :]
        edge_x_x = elev_x[0, :]
        edge_z_o = elev_o[:, SECTOR_SIZE - 1]
        edge_z_z = elev_z[:, 0]

        seam_x = np.abs(edge_x_o - edge_x_x)
        seam_z = np.abs(edge_z_o - edge_z_z)

        # Compare against global adjacent diffs in the stitched map.
        map_x = np.concatenate([elev_o, elev_x], axis=0)
        map_z = np.concatenate([elev_o, elev_z], axis=1)
        diff_all_x = np.abs(map_x[1:, :] - map_x[:-1, :])
        diff_all_z = np.abs(map_z[:, 1:] - map_z[:, :-1])

        x_stats = _diff_stats(seam_x)
        z_stats = _diff_stats(seam_z)
        x_all = _diff_stats(diff_all_x)
        z_all = _diff_stats(diff_all_z)
        bad_x = seam_x > (x_all["p99"] + max_extra["p99"])
        bad_z = seam_z > (z_all["p99"] + max_extra["p99"])

        if x_stats["p95"] > x_all["p99"] + max_extra["p95"] * warn_ratio / 0.01:
            print(
                "height seam X warn:",
                f"max={x_stats['max']:.2f} mean={x_stats['mean']:.2f} p95={x_stats['p95']:.2f} "
                f"all_p99={x_all['p99']:.2f} "
                f"seed={seed_tag} off=({offset_x},{offset_z})",
            )
        if z_stats["p95"] > z_all["p99"] + max_extra["p95"] * warn_ratio / 0.01:
            print(
                "height seam Z warn:",
                f"max={z_stats['max']:.2f} mean={z_stats['mean']:.2f} p95={z_stats['p95']:.2f} "
                f"all_p99={z_all['p99']:.2f} "
                f"seed={seed_tag} off=({offset_x},{offset_z})",
            )

        assert x_stats["mean"] <= x_all["mean"] + max_extra["mean"], (
            f"height seam X mean {x_stats['mean']:.2f} all={x_all['mean']:.2f} seed={seed_tag} off=({offset_x},{offset_z})"
        )
        assert x_stats["p95"] <= x_all["p99"] + max_extra["p95"], (
            f"height seam X p95 {x_stats['p95']:.2f} all_p99={x_all['p99']:.2f} seed={seed_tag} off=({offset_x},{offset_z})"
        )
        assert x_stats["max"] <= x_all["p99"] + max_extra["p99"], (
            f"height seam X max {x_stats['max']:.2f} all_p99={x_all['p99']:.2f} seed={seed_tag} off=({offset_x},{offset_z})"
        )

        assert z_stats["mean"] <= z_all["mean"] + max_extra["mean"], (
            f"height seam Z mean {z_stats['mean']:.2f} all={z_all['mean']:.2f} seed={seed_tag} off=({offset_x},{offset_z})"
        )
        assert z_stats["p95"] <= z_all["p99"] + max_extra["p95"], (
            f"height seam Z p95 {z_stats['p95']:.2f} all_p99={z_all['p99']:.2f} seed={seed_tag} off=({offset_x},{offset_z})"
        )
        assert z_stats["max"] <= z_all["p99"] + max_extra["p99"], (
            f"height seam Z max {z_stats['max']:.2f} all_p99={z_all['p99']:.2f} seed={seed_tag} off=({offset_x},{offset_z})"
        )

        if x_stats["p95"] > x_all["p99"] + max_extra["p95"] or z_stats["p95"] > z_all["p99"] + max_extra["p95"]:
            _print_seam_map(
                elev_o,
                elev_x,
                elev_z,
                bad_x=bad_x,
                bad_z=bad_z,
                title=(
                    "height seam map (O,X,Z), bad cells in red "
                    f"seed={seed_tag} off=({offset_x},{offset_z})"
                ),
            )
        _dump_seam_map(
            elev_o,
            elev_x,
            elev_z,
            bad_x=bad_x,
            bad_z=bad_z,
            name=f"height_seam_seed{seed_tag}_off{offset_x}_{offset_z}",
        )

    gen = _init_generator(seed=12345, use_biome=True)
    for offset_x, offset_z in offsets:
        check_once(gen, offset_x, offset_z, seed_tag=12345)

    for seed in seeds:
        gen = _init_generator(seed=seed, use_biome=True)
        check_once(gen, 0, 0, seed_tag=seed)


def test_blocks_seams():
    use_biome = bool(getattr(config, "USE_EXPERIMENTAL_BIOME_GEN", False))
    max_ratio = 0.02
    warn_ratio = 0.01
    height_tol = 1
    max_mismatches = 3

    rng = np.random.RandomState(7331)
    offsets = list(rng.randint(-6, 7, size=(10, 2)) * SECTOR_SIZE)
    seeds = list(rng.randint(1, 1_000_000, size=10))

    def check_once(offset_x, offset_z, seed_tag):
        pos_o = (int(offset_x), 0, int(offset_z))
        pos_x = (int(offset_x + SECTOR_SIZE), 0, int(offset_z))
        pos_z = (int(offset_x), 0, int(offset_z + SECTOR_SIZE))

        sec_o = _gen_sector(pos_o, use_biome)
        sec_x = _gen_sector(pos_x, use_biome)
        sec_z = _gen_sector(pos_z, use_biome)

        h_o = _column_heights(sec_o)
        h_x = _column_heights(sec_x)
        h_z = _column_heights(sec_z)


        edge_x_o = h_o[SECTOR_SIZE - 1, :]
        edge_x_x = h_x[0, :]
        edge_z_o = h_o[:, SECTOR_SIZE - 1]
        edge_z_z = h_z[:, 0]

        diff_x = np.abs(edge_x_o - edge_x_x)
        diff_z = np.abs(edge_z_o - edge_z_z)

        map_x = np.concatenate([h_o, h_x], axis=0)
        map_z = np.concatenate([h_o, h_z], axis=1)
        diff_all_x = np.abs(map_x[1:, :] - map_x[:-1, :])
        diff_all_z = np.abs(map_z[:, 1:] - map_z[:, :-1])
        x_mism, x_total = _edge_mismatch_counts(edge_x_o, edge_x_x, tol=height_tol)
        z_mism, z_total = _edge_mismatch_counts(edge_z_o, edge_z_z, tol=height_tol)

        x_ratio = x_mism / x_total
        z_ratio = z_mism / z_total
        x_stats = _diff_stats(diff_x)
        z_stats = _diff_stats(diff_z)
        x_all = _diff_stats(diff_all_x)
        z_all = _diff_stats(diff_all_z)
        if x_ratio > warn_ratio:
            print(
                "block seam X warn:",
                f"{x_mism}/{x_total} ({x_ratio:.4f}) tol={height_tol} seed={seed_tag} off=({offset_x},{offset_z})",
            )
        if z_ratio > warn_ratio:
            print(
                "block seam Z warn:",
                f"{z_mism}/{z_total} ({z_ratio:.4f}) tol={height_tol} seed={seed_tag} off=({offset_x},{offset_z})",
            )
        assert x_ratio <= max_ratio or x_mism <= max_mismatches, (
            f"block seam X {x_mism}/{x_total} ({x_ratio:.4f}) tol={height_tol} seed={seed_tag} off=({offset_x},{offset_z})"
        )
        assert z_ratio <= max_ratio or z_mism <= max_mismatches, (
            f"block seam Z {z_mism}/{z_total} ({z_ratio:.4f}) tol={height_tol} seed={seed_tag} off=({offset_x},{offset_z})"
        )
        assert x_stats["p95"] <= x_all["p99"] + 3.0, (
            f"block seam X p95 {x_stats['p95']:.2f} all_p99={x_all['p99']:.2f} seed={seed_tag} off=({offset_x},{offset_z})"
        )
        assert z_stats["p95"] <= z_all["p99"] + 3.0, (
            f"block seam Z p95 {z_stats['p95']:.2f} all_p99={z_all['p99']:.2f} seed={seed_tag} off=({offset_x},{offset_z})"
        )

        if x_ratio > max_ratio or z_ratio > max_ratio:
            bad_x = diff_x > height_tol
            bad_z = diff_z > height_tol
            _print_seam_map(
                h_o,
                h_x,
                h_z,
                bad_x=bad_x,
                bad_z=bad_z,
                title=(
                    "block seam height map (O,X,Z), bad cells in red "
                    f"seed={seed_tag} off=({offset_x},{offset_z})"
                ),
            )
        _dump_seam_map(
            h_o,
            h_x,
            h_z,
            bad_x=diff_x > height_tol,
            bad_z=diff_z > height_tol,
            name=f"block_seam_seed{seed_tag}_off{offset_x}_{offset_z}",
        )

    _init_generator(seed=12345, use_biome=use_biome)
    for offset_x, offset_z in offsets:
        check_once(offset_x, offset_z, seed_tag=12345)

    for seed in seeds:
        _init_generator(seed=seed, use_biome=use_biome)
        check_once(0, 0, seed_tag=seed)


def test_mapgen_determinism():
    use_biome = bool(getattr(config, "USE_EXPERIMENTAL_BIOME_GEN", False))
    seed = 424242

    _init_generator(seed=seed, use_biome=use_biome)
    sec_a = _gen_sector((0, 0, 0), use_biome)
    sec_b = _gen_sector((SECTOR_SIZE, 0, 0), use_biome)

    _init_generator(seed=seed, use_biome=use_biome)
    sec_a2 = _gen_sector((0, 0, 0), use_biome)
    sec_b2 = _gen_sector((SECTOR_SIZE, 0, 0), use_biome)

    sample = [
        (0, 0, 0),
        (SECTOR_SIZE - 1, 0, SECTOR_SIZE - 1),
        (SECTOR_SIZE // 2, 10, SECTOR_SIZE // 2),
        (SECTOR_SIZE // 2, SECTOR_SIZE // 2, SECTOR_SIZE // 3),
    ]
    for x, y, z in sample:
        assert sec_a[x, y, z] == sec_a2[x, y, z]
        assert sec_b[x, y, z] == sec_b2[x, y, z]


def test_no_empty_columns():
    use_biome = bool(getattr(config, "USE_EXPERIMENTAL_BIOME_GEN", False))
    seed = 12345
    _init_generator(seed=seed, use_biome=use_biome)
    rng = np.random.RandomState(2024)
    offsets = list(rng.randint(-6, 7, size=(10, 2)) * SECTOR_SIZE)
    seeds = list(rng.randint(1, 1_000_000, size=10))

    def check_sector(sec, seed_tag, offset_x, offset_z, label):
        heights = _column_heights(sec)
        if (heights < 0).any():
            count = int((heights < 0).sum())
            bad = heights < 0
            _print_seam_map(
                heights,
                heights,
                heights,
                bad_x=bad[SECTOR_SIZE - 1, :],
                bad_z=bad[:, SECTOR_SIZE - 1],
                title=f"empty column map seed={seed_tag} off=({offset_x},{offset_z})",
            )
            _dump_seam_map(
                heights,
                heights,
                heights,
                bad_x=bad[SECTOR_SIZE - 1, :],
                bad_z=bad[:, SECTOR_SIZE - 1],
                name=f"empty_columns_seed{seed_tag}_off{offset_x}_{offset_z}",
            )
            raise AssertionError(
                f"empty columns in {label}: {count} seed={seed_tag} off=({offset_x},{offset_z})"
            )

    for offset_x, offset_z in offsets:
        pos = (int(offset_x), 0, int(offset_z))
        sec = _gen_sector(pos, use_biome)
        check_sector(sec, seed, offset_x, offset_z, label="offset")

    for s in seeds:
        _init_generator(seed=s, use_biome=use_biome)
        sec = _gen_sector((0, 0, 0), use_biome)
        check_sector(sec, s, 0, 0, label="seed")


def test_block_geometry_shape_and_values():
    use_biome = bool(getattr(config, "USE_EXPERIMENTAL_BIOME_GEN", False))
    _init_generator(seed=12345, use_biome=use_biome)
    blocks = _gen_sector((0, 0, 0), use_biome)
    assert blocks.shape == (SECTOR_SIZE, SECTOR_HEIGHT, SECTOR_SIZE)
    assert np.issubdtype(blocks.dtype, np.integer)
    if np.issubdtype(blocks.dtype, np.floating):
        assert np.isfinite(blocks).all()
    max_id = max(int(v) for v in BLOCK_ID.values())
    assert int(blocks.min()) >= 0
    assert int(blocks.max()) <= max_id
