"""Cover every residue once while providing flanking context to ESMC."""

import random

from .common import LENGTH_SCALE, MAX_RESIDUES, stable_seed


def make_chunks(length, core_size=512, halo=64, offset=0):
    """Return zero-based, end-exclusive core and context intervals."""
    if length < 1 or core_size < 1 or halo < 0:
        raise ValueError("Length/core size must be positive and halo must be nonnegative")
    if core_size + 2 * halo > MAX_RESIDUES:
        raise ValueError("Core plus both context flanks exceeds the ESMC residue budget")
    if not 0 <= offset < core_size:
        raise ValueError("Offset must be between zero and core_size - 1")
    chunks, start = [], 0
    while start < length:
        size = core_size - offset if start == 0 else core_size
        end = min(length, start + size)
        chunks.append({"start": start, "end": end,
                       "context_start": max(0, start - halo),
                       "context_end": min(length, end + halo)})
        start = end
    return chunks


def position_features(chunks, length):
    return [[c["start"] / length, c["end"] / length,
             (c["end"] - c["start"]) / LENGTH_SCALE,
             float(c["start"] == 0), float(c["end"] == length)] for c in chunks]


def make_views(length, sample_id, seed=42, sizes=(128, 256, 512, 768), halo=64):
    """Keep the deployment view first, then unique augmented partitions."""
    rng = random.Random(stable_seed(seed, sample_id))
    views, seen = [], set()
    for size, offset in [(512, 0)] + [(s, rng.randrange(max(1, s // 2))) for s in sizes]:
        chunks = make_chunks(length, size, halo, offset)
        signature = tuple((c["start"], c["end"]) for c in chunks)
        if signature not in seen:
            views.append({"core_size": size, "offset": offset, "chunks": chunks})
            seen.add(signature)
    return views
