"""Select important layers from a probing run's ``*_complementarity.csv``.

Turns the pairwise complementarity maps (error correlation + concat-SRCC gain)
into concrete layer sets to feed ``train.py --fusion_layers``. Emits several
selections so you can run the controlled comparison in one place:

  * ``last_layer``            -- vanilla ``get_image_features`` baseline.
  * ``all_layers``            -- every tapped layer (the "just use everything" baseline).
  * ``top_k_srcc``            -- top-k by *marginal* SRCC (ignores redundancy).
  * ``cluster_representative``-- one layer per redundancy band (err-corr >= thresh).
  * ``greedy_complementary``  -- greedily add the layer most complementary to the
                                 current set (max of the *minimum* pairwise gain).

The claim to test: ``greedy_complementary`` / ``cluster_representative`` match or
beat ``all_layers`` with far fewer layers, and beat ``top_k_srcc`` (i.e.
complementarity matters, not just marginal strength).

NOTE: these are *pairwise* heuristics -- pairwise gain can't see set-level
redundancy, so treat the sets as candidates and let training be the judge.

Usage:
    python select_layers.py probe_out/KADID10K_.../ --k 4
    python select_layers.py path/to/KADID10K_complementarity.csv --k 5 --err_thresh 0.9
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np


def load_complementarity(path):
    """Return ``(layers, srcc, errcorr, gain, idx)`` from a complementarity CSV."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    srcc = {}
    for r in rows:
        srcc[int(r["layer_a"])] = float(r["srcc_a"])
        srcc[int(r["layer_b"])] = float(r["srcc_b"])
    layers = sorted(srcc)
    idx = {l: i for i, l in enumerate(layers)}
    n = len(layers)
    ec, gain = np.eye(n), np.zeros((n, n))
    for r in rows:
        a, b = idx[int(r["layer_a"])], idx[int(r["layer_b"])]
        ec[a, b] = ec[b, a] = float(r["err_corr"])
        gain[a, b] = gain[b, a] = float(r["concat_gain"])
    return layers, srcc, ec, gain, idx


def top_k_srcc(layers, srcc, k):
    return sorted(sorted(layers, key=lambda l: -srcc[l])[:k])


def greedy_complementary(layers, srcc, gain, idx, k, eps):
    """Seed with the best single layer; add the candidate whose *minimum* pairwise
    concat-gain against the current set is largest (must help beyond every member).
    Stops early when the best candidate's gain drops below ``eps``."""
    sel = [max(layers, key=lambda l: srcc[l])]
    while len(sel) < k:
        best, best_g = None, -1e9
        for c in layers:
            if c in sel:
                continue
            g = min(gain[idx[c], idx[s]] for s in sel)  # complementary to ALL selected
            if g > best_g:
                best_g, best = g, c
        if best is None or best_g < eps:
            break
        sel.append(best)
    return sorted(sel)


def cluster_representative(layers, srcc, ec, idx, thresh):
    """One representative per redundancy band: take the highest-SRCC unassigned
    layer as a rep, absorb every layer with err-corr >= thresh to it, repeat."""
    reps, assigned = [], set()
    for L in sorted(layers, key=lambda l: -srcc[l]):
        if L in assigned:
            continue
        reps.append(L)
        for o in layers:
            if ec[idx[L], idx[o]] >= thresh:
                assigned.add(o)
    return sorted(reps)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a run dir (containing *_complementarity.csv) or the CSV itself")
    ap.add_argument("--k", type=int, default=4, help="target size for top_k / greedy")
    ap.add_argument("--err_thresh", type=float, default=0.9,
                    help="error-correlation above which two layers count as redundant")
    ap.add_argument("--gain_eps", type=float, default=0.005,
                    help="stop greedy when the best added layer's min gain < this")
    ap.add_argument("--fusion_type", default="adaptive",
                    help="fusion_type to print in the example train.py commands")
    ap.add_argument("--dataset", default="KADID10K", help="dataset for the example commands")
    args = ap.parse_args()

    csv_path = args.path
    if os.path.isdir(csv_path):
        hits = glob.glob(os.path.join(csv_path, "*_complementarity.csv"))
        if not hits:
            raise SystemExit(f"no *_complementarity.csv found in {csv_path}")
        csv_path = hits[0]
    layers, srcc, ec, gain, idx = load_complementarity(csv_path)

    sets = {
        "last_layer": [max(layers)],
        "all_layers": list(layers),
        "top_k_srcc": top_k_srcc(layers, srcc, args.k),
        "cluster_representative": cluster_representative(layers, srcc, ec, idx, args.err_thresh),
        "greedy_complementary": greedy_complementary(layers, srcc, gain, idx, args.k, args.gain_eps),
    }

    print(f"Source: {csv_path}\n")
    ranked = sorted(layers, key=lambda l: -srcc[l])
    print("per-layer SRCC (desc):")
    print("  " + "  ".join(f"L{l}={srcc[l]:.3f}" for l in ranked) + "\n")
    for name, s in sets.items():
        print(f"  {name:24s} ({len(s):2d}): {','.join(map(str, s))}")

    print("\nControlled comparison -- same fusion_type, vary only the layer set:")
    for name, s in sets.items():
        layer_arg = ",".join(map(str, s))
        if name == "last_layer":
            print(f"  # {name} (vanilla baseline)\n"
                  f"  python train.py --dataset {args.dataset} --fusion_type none")
        else:
            print(f"  # {name}\n"
                  f"  python train.py --dataset {args.dataset} "
                  f"--fusion_type {args.fusion_type} --fusion_layers {layer_arg}")

    out = os.path.join(os.path.dirname(csv_path) or ".", "selected_layers.json")
    with open(out, "w") as f:
        json.dump(sets, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
