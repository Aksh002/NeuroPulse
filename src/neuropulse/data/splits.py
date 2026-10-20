"""Engine-level partitions and fixed truncated validation endpoints."""

import math

import numpy as np

from neuropulse.data.readers import Engines


def split_engines(
    engines: Engines, fraction: float = 0.8, seed: int = 42
) -> tuple[list[int], list[int]]:
    ids = np.array(sorted(engines), dtype=np.int64)
    count = math.floor(len(ids) * fraction)
    if not 0 < fraction < 1 or count == 0 or count == len(ids):
        raise ValueError("Split requires nonempty training and validation engine sets")
    ids = np.random.default_rng(seed).permutation(ids)
    return sorted(ids[:count].tolist()), sorted(ids[count:].tolist())


def validation_prefixes(engines: Engines, ids: list[int], seed: int = 142) -> dict[int, int]:
    rng = np.random.default_rng(seed)
    endpoints: dict[int, int] = {}
    for engine in sorted(ids):
        terminal = len(engines[engine])
        lower = max(1, math.ceil(0.5 * terminal))
        if lower >= terminal:
            raise ValueError(f"Engine {engine} lacks a valid truncated endpoint")
        endpoints[engine] = int(rng.integers(lower, terminal))
    return endpoints
