"""Cover every residue once while providing flanking context to ESMC."""

import random

from .common import LENGTH_SCALE, MAX_RESIDUES, stable_seed


def make_chunks(length, core_size=512, halo=64, offset=0):
    """Return zero-based, end-exclusive core and context intervals."""
    if length < 1 or core_size < 1 or halo < 0:
        raise ValueError("Length/core size must be positive and halo must be nonnegative")
    # ESMC sees the core plus its context flanks, so budget for their combined size.
    if core_size + 2 * halo > MAX_RESIDUES:
        raise ValueError("Core plus both context flanks exceeds the ESMC residue budget")
    if not 0 <= offset < core_size:
        raise ValueError("Offset must be between zero and core_size - 1")
    chunks, start = [], 0
    while start < length:
        # Shorten only the first core to shift all subsequent boundaries for augmentation.
        size = core_size - offset if start == 0 else core_size
        end = min(length, start + size)
        # Cores form a disjoint cover. Context intervals can overlap, but their
        # extra residues provide context only and are excluded from pooled means.
        chunks.append({"start": start, "end": end,
                       "context_start": max(0, start - halo),
                       "context_end": min(length, end + halo)})
        start = end
    return chunks


def position_features(chunks, length):
    # Features: relative start/end, core length in 512-aa units, and terminal flags.
    # Continuous coordinates support chunk counts not observed during training.
    return [[c["start"] / length, c["end"] / length,
             (c["end"] - c["start"]) / LENGTH_SCALE,
             float(c["start"] == 0), float(c["end"] == length)] for c in chunks]


def make_views(length, sample_id, seed=42, sizes=(128, 256, 512, 768), halo=64, mode="multiscale"):
    """Keep the deployment view first, then unique augmented partitions."""
    if mode not in {"fixed", "multiscale"}:
        raise ValueError("Chunk view mode must be fixed or multiscale")
    if mode == "fixed":
        return [{"core_size": 512, "offset": 0, "chunks": make_chunks(length, 512, halo, 0)}]
    # Tie augmentation to the sample ID so preparation order cannot change its views.
    rng = random.Random(stable_seed(seed, sample_id))
    views, seen = [], set()
    for size, offset in [(512, 0)] + [(s, rng.randrange(max(1, s // 2))) for s in sizes]:
        chunks = make_chunks(length, size, halo, offset)
        # Different settings can produce identical partitions on short proteins;
        # keep each partition once so duplicate views do not change sampling weights.
        signature = tuple((c["start"], c["end"]) for c in chunks)
        if signature not in seen:
            views.append({"core_size": size, "offset": offset, "chunks": chunks})
            seen.add(signature)
    return views
