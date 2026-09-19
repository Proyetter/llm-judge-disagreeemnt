#!/usr/bin/env python3
"""
bootstrap_utils.py
---------------------
Canonical pair-level cluster bootstrap utilities, shared by
analyze_confidence_v2.py and compare_splits.py.

WHY "CLUSTER" BOOTSTRAP
------------------------
Each pair contributes 6 judgments (3 AB + 3 BA), and those 6 are NOT
independent observations -- they're 6 looks at the same item. Resampling
individual judgments would pretend you have more independent data than you
do, and understate every confidence interval. The correct unit to resample
is the PAIR (the item), keeping all of its judgments together as one
cluster. That's what every function here does.

Also includes a generic rank-based AUROC (consistent with the convention
used throughout this project: for instability targets, score = -confidence,
so that LOWER confidence correctly predicts the positive class).

This module is meant for NEW scripts going forward. Earlier scripts
(analyze_decomposition.py, analyze_pair_classes.py, analyze_order_effect.py)
already have their own tested, working inline versions of similar logic and
are left alone -- there is no correctness reason to touch code that already
passed its own unit tests.
"""

import numpy as np


def cluster_bootstrap(items, stat_fn, n_boot=2000, seed=0):
    """items: list of per-pair records (any type, as long as stat_fn accepts
    a list of them). Resamples ITEMS (pairs) with replacement, n_boot times,
    and returns the array of bootstrap statistic values (for percentile CIs)."""
    rng = np.random.default_rng(seed)
    n = len(items)
    idx = np.arange(n)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        sample = [items[i] for i in rng.choice(idx, size=n, replace=True)]
        vals[b] = stat_fn(sample)
    return vals


def bootstrap_ci(items, stat_fn, n_boot=2000, seed=0, alpha=0.05):
    """Convenience wrapper: returns (point_estimate, lo, hi) where point_estimate
    is stat_fn computed on the FULL (unresampled) data -- not the bootstrap
    mean, which can differ slightly from the true statistic by bootstrap noise
    (this distinction mattered enough to cause a real bug earlier in this
    project; always use the direct computation for the point estimate, and
    the bootstrap ONLY for the interval)."""
    point = stat_fn(items)
    boot_vals = cluster_bootstrap(items, stat_fn, n_boot, seed)
    lo, hi = np.percentile(boot_vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def bootstrap_diff_ci(items_a, items_b, stat_fn, n_boot=2000, seed=0, alpha=0.05):
    """CI for the DIFFERENCE stat_fn(items_a) - stat_fn(items_b), resampling
    each group independently. Useful for comparing e.g. GPT-split vs
    Claude-split values for the same model."""
    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed + 1)
    na, nb = len(items_a), len(items_b)
    idx_a, idx_b = np.arange(na), np.arange(nb)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa = [items_a[j] for j in rng_a.choice(idx_a, size=na, replace=True)]
        sb = [items_b[j] for j in rng_b.choice(idx_b, size=nb, replace=True)]
        diffs[i] = stat_fn(sa) - stat_fn(sb)
    point = stat_fn(items_a) - stat_fn(items_b)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def compute_auroc(scores, labels):
    """Rank-based AUROC (Mann-Whitney U formulation), no sklearn dependency.
    labels: 0/1 (or False/True). scores: higher score should predict label=1.
    Returns None if labels contain only one class (AUROC undefined).

    CONVENTION used throughout this project: for "does low confidence predict
    instability" questions, pass scores = [-c for c in confidences], so that
    lower confidence produces a higher score and 0.5/1.0 keep their usual
    meaning (0.5 = uninformative, 1.0 = perfect)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    cursor = 1
    while i < len(sorted_scores):
        j = i
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (cursor + (cursor + (j - i) - 1)) / 2.0
        ranks[order[i:j]] = avg_rank
        cursor += (j - i)
        i = j

    sum_ranks_pos = ranks[labels].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
