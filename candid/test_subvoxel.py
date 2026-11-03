#!/usr/bin/env python3
# Copyright (c) 2026 Ibrahim Alroudhan. All rights reserved. See ../LICENSE.
"""test_subvoxel.py — shows the sub-voxel (marching-cubes) distance removes the
~one-voxel overestimate of the voxel-centre method, on a known-answer phantom.
Run: python test_subvoxel.py"""
import numpy as np
from geometry import min_distance, min_distance_subvoxel, compute_features

SP = (0.3, 0.3, 0.3)

results = []
def check(name, cond, detail=""):
    results.append(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


shape = (40, 40, 40)
tooth = np.zeros(shape, bool); tooth[10:30, 10:30, 5:15] = True    # x 5..14
canal = np.zeros(shape, bool); canal[10:30, 10:30, 25:35] = True   # x 25..34
# 10 empty voxels between -> TRUE physical gap = 10 * 0.3 = 3.0 mm
TRUE_GAP = 3.0

d_vc, _ = min_distance(tooth, canal, SP)
d_sv, _ = min_distance_subvoxel(tooth, canal, SP)
print(f"true gap {TRUE_GAP} mm | voxel-centre {d_vc:.3f} | sub-voxel {d_sv:.3f}")

check("sub-voxel close to true gap", abs(d_sv - TRUE_GAP) <= 0.12, f"{d_sv:.3f} vs {TRUE_GAP}")
check("voxel-centre overestimates", d_vc > d_sv + 0.15, f"vc {d_vc:.3f} > sv {d_sv:.3f}")
check("sub-voxel more accurate than voxel-centre",
      abs(d_sv - TRUE_GAP) < abs(d_vc - TRUE_GAP))

# overlap still returns 0 on both
tooth2 = np.zeros(shape, bool); tooth2[10:30, 10:30, 5:20] = True
canal2 = np.zeros(shape, bool); canal2[10:30, 10:30, 18:35] = True
check("sub-voxel overlap -> 0", min_distance_subvoxel(tooth2, canal2, SP)[0] == 0.0)

# compute_features records the method and uses sub-voxel by default
feat = compute_features(tooth, canal, SP)
check("features default to sub-voxel", feat.distance_method == "subvoxel", feat.distance_method)
check("feature distance ~ true gap", abs(feat.min_distance_mm - TRUE_GAP) <= 0.12,
      f"{feat.min_distance_mm}")
feat_vc = compute_features(tooth, canal, SP, subvoxel=False)
check("voxel-centre mode selectable", feat_vc.distance_method == "voxel-centre")

passed = sum(results)
print(f"\n{passed}/{len(results)} checks passed")
raise SystemExit(0 if passed == len(results) else 1)
