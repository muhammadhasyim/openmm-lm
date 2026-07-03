"""Microbenchmarks for ML neighbor-list build and packing."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def _water_coords(n_atoms: int, box_ang: float, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_mol = n_atoms // 3
    pos = rng.random((n_mol, 3)) * box_ang
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    oh = 0.9572
    for m in range(n_mol):
        c = pos[m]
        coords[3 * m] = c
        coords[3 * m + 1] = c + np.array([oh, 0.0, 0.0])
        coords[3 * m + 2] = c + np.array([0.0, oh, 0.0])
    cell = np.eye(3, dtype=np.float64) * box_ang
    return coords, cell


def _bench_aimnet2(coord_np: np.ndarray, cell_np: np.ndarray, repeats: int) -> dict:
    from openmmml.aimnet2_nblist import build_aimnet2_neighbor_payload, nblists_torch_pbc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coord = torch.tensor(coord_np, dtype=torch.float32, device=device)
    cell = torch.tensor(cell_np, dtype=torch.float32, device=device)

    class _Model:
        cutoff = 5.0
        cutoff_lr = float("inf")

    model = _Model()

    t0 = time.perf_counter()
    for _ in range(repeats):
        build_aimnet2_neighbor_payload(model, coord, cell=cell)
    build_s = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        nblists_torch_pbc(coord, cell, 5.0)
    pbc_s = (time.perf_counter() - t0) / repeats

    return {"aimnet2_payload_s": build_s, "aimnet2_pbc_sr_s": pbc_s}


def _bench_sparse(coord_np: np.ndarray, cell_np: np.ndarray, cutoff: float, repeats: int) -> dict:
    from openmmml.neighbors import build_sparse_edges_numpy

    t0 = time.perf_counter()
    for _ in range(repeats):
        build_sparse_edges_numpy(coord_np, cutoff, cell_np)
    return {"sparse_edges_s": (time.perf_counter() - t0) / repeats}


def _bench_skin(cache_repeats: int = 100) -> dict:
    from openmmml.neighbor_cache import NeighborCache

    cache = NeighborCache(skin_ang=0.3)
    pos0 = torch.zeros(9, 3)
    cache.update(pos0, {"edge_index": torch.zeros(2, 1, dtype=torch.long)})
    t0 = time.perf_counter()
    rebuilds = 0
    for step in range(cache_repeats):
        pos = pos0 + 0.001 * step
        if cache.needs_rebuild(pos):
            rebuilds += 1
            cache.update(pos, cache.get())
    return {
        "skin_steps": cache_repeats,
        "skin_rebuilds": rebuilds,
        "skin_check_s": (time.perf_counter() - t0) / cache_repeats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ML neighbor builds")
    parser.add_argument("--sizes", default="9,300,900", help="Comma-separated atom counts")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--box-nm", type=float, default=2.321)
    args = parser.parse_args()

    box_ang = args.box_nm * 10.0
    print(f"device={torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"repeats={args.repeats}, box={args.box_nm:.3f} nm")

    for n in [int(x) for x in args.sizes.split(",") if x.strip()]:
        coord, cell = _water_coords(n, box_ang)
        print(f"\n=== N={n} ===")
        aim = _bench_aimnet2(coord, cell, args.repeats)
        sparse = _bench_sparse(coord, cell, cutoff=5.0, repeats=args.repeats)
        for k, v in {**aim, **sparse}.items():
            print(f"  {k}: {v*1000:.2f} ms")

    skin = _bench_skin()
    print(f"\n=== skin cache (N=9) ===")
    print(f"  rebuilds in {skin['skin_steps']} steps: {skin['skin_rebuilds']}")
    print(f"  skin_check_s: {skin['skin_check_s']*1e6:.2f} us")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
