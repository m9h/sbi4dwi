"""
Rebalance a FORCE simulations dict to a target fibre-count distribution.

Background: dipy's :func:`generate_force_simulations` samples fibre
fractions from ``Dirichlet(2, 1, 1)``, which produces a library that is
~10 % single-fibre / 20 % two-fibre / **70 % three-fibre**. For
single-strand-dominated targets (DiSCo phantom, dilute white-matter
regions), the 70 % 3-fibre bias drives false-positive multi-fibre
tractography (audit finding #4 in doc 004 §17.7).

This module subsamples an existing library — without re-generating —
to match a user-specified ``{1: p, 2: q, 3: r}`` distribution
(``p + q + r = 1``). Within each fibre-count class, sampling is
random-without-replacement, so the within-class parameter coverage
is preserved.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def rebalance_force_library(
    sims: Dict[str, np.ndarray],
    target_fractions: Dict[int, float],
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Subsample a FORCE library to match target fibre-count fractions.

    Parameters
    ----------
    sims
        Loaded simulations dict (e.g. from ``load_force_simulations``).
        Must contain ``num_fibers`` plus arrays whose leading dimension
        equals the library size N.
    target_fractions
        Mapping ``{fibre_count: target_fraction}``. Must sum to 1.0.
        Example: ``{1: 0.8, 2: 0.1, 3: 0.1}`` → Dirichlet(8,1,1) mean.
    seed
        RNG seed for subsampling.

    Returns
    -------
    A new dict with the same keys but a smaller leading dimension. If
    one fibre-count class has fewer entries available than implied by
    the target, the rebalancer caps that class at all-available and
    scales the others down proportionally so the final fractions still
    match ``target_fractions`` (rather than over-sampling with
    replacement).
    """
    if not target_fractions:
        raise ValueError("target_fractions cannot be empty")
    if abs(sum(target_fractions.values()) - 1.0) > 1e-6:
        raise ValueError(
            f"target_fractions must sum to 1.0, got {sum(target_fractions.values())}"
        )

    nf = np.asarray(sims["num_fibers"])
    rng = np.random.default_rng(seed)

    # Available counts per fibre class
    avail = {k: int((nf == k).sum()) for k in target_fractions}

    # Find the binding class: which target * scale ≤ available?
    # Largest scale = total library size; bound by min(avail[k] / target_fractions[k])
    max_scale = min(
        avail[k] / p for k, p in target_fractions.items() if p > 0
    )
    total = int(np.floor(max_scale))
    take = {k: int(round(total * p)) for k, p in target_fractions.items()}
    # Adjust rounding so total matches
    diff = total - sum(take.values())
    if diff != 0:
        k_max = max(take, key=take.get)
        take[k_max] += diff

    # Subsample without replacement per class
    keep = []
    for k, n in take.items():
        idx_k = np.where(nf == k)[0]
        chosen = rng.choice(idx_k, size=n, replace=False)
        keep.append(chosen)
    keep = np.concatenate(keep)
    rng.shuffle(keep)

    out = {}
    for key, arr in sims.items():
        out[key] = np.asarray(arr)[keep]
    return out
