#!/usr/bin/env python3
# Copyright (c) 2026 Ibrahim Alroudhan. All rights reserved. See ../LICENSE.
"""
test_geometry.py — synthetic phantoms with KNOWN geometry to verify geometry.py.
Run: python test_geometry.py
"""
import numpy as np
from geometry import (min_distance, close_approach_length, configuration,
                      count_roots, canal_position, compute_features, AxisFrame,
                      assess_impaction)

SPACING = (0.3, 0.3, 0.3)  # mm, isotropic


def cuboid(shape, z, y, x):
    m = np.zeros(shape, bool)
    m[z[0]:z[1], y[0]:y[1], x[0]:x[1]] = True
    return m


def approx(a, b, tol):
    return abs(a - b) <= tol


results = []
def check(name, cond, detail=""):
    results.append((name, cond, detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


shape = (60, 60, 60)

# --- Test 1: known gap distance ------------------------------------------------
# tooth block and canal block separated by a 10-voxel gap along x -> 10*0.3=3.0mm
tooth = cuboid(shape, (20, 40), (20, 40), (10, 25))
canal = cuboid(shape, (20, 40), (20, 40), (35, 50))
d, closest = min_distance(tooth, canal, SPACING)
# surfaces are at x=24 (tooth) and x=35 (canal): gap = 11 vox = 3.3mm
check("min_distance ~ known gap", approx(d, 3.3, 0.35), f"got {d:.2f} mm")

# --- Test 2: contact / overlap -> distance 0 ----------------------------------
tooth2 = cuboid(shape, (20, 40), (20, 40), (10, 30))
canal2 = cuboid(shape, (20, 40), (20, 40), (28, 45))  # overlaps at x 28-30
d2, _ = min_distance(tooth2, canal2, SPACING)
check("overlap -> distance 0", approx(d2, 0.0, 1e-6), f"got {d2:.2f}")
check("configuration overlap -> perforated",
      configuration(tooth2, canal2, d2, SPACING) == "perforated")

# --- Test 3: separated configuration ------------------------------------------
check("configuration gap -> separated",
      configuration(tooth, canal, d, SPACING) == "separated",
      f"got {configuration(tooth, canal, d, SPACING)}")

# --- Test 4: close-approach length along a parallel canal ----------------------
# canal runs along z for 30 voxels, all ~3.3mm from tooth (> 2mm) -> length 0
len_far = close_approach_length(tooth, canal, SPACING, close_thresh_mm=2.0)
check("close-approach length 0 when gap>thresh", approx(len_far, 0.0, 0.31),
      f"got {len_far:.2f} mm")
# bring canal to 1 voxel gap over a 24-voxel z-run -> length ~ 24*0.3 = 7.2mm
canal_near = cuboid(shape, (12, 36), (20, 40), (25, 40))
tooth_near = cuboid(shape, (12, 36), (20, 40), (10, 24))  # surface x=23, canal x=25 -> ~0.6mm
len_near = close_approach_length(tooth_near, canal_near, SPACING, close_thresh_mm=2.0)
check("close-approach length ~ run length", approx(len_near, 7.2, 1.2),
      f"got {len_near:.2f} mm")

# --- Test 5: canal position buccal vs lingual ---------------------------------
# buccal axis = +x. Canal on +x side of tooth -> buccal; on -x side -> lingual.
frame = AxisFrame(buccal=(0, 0, 1), superior=(-1, 0, 0))
pos_buccal = canal_position(tooth, canal, min_distance(tooth, canal, SPACING)[1], SPACING, frame)
check("canal on +x classified buccal", pos_buccal == "buccal", f"got {pos_buccal}")
canal_ling = cuboid(shape, (20, 40), (20, 40), (0, 6))   # -x side
d5, c5 = min_distance(tooth, canal_ling, SPACING)
pos_ling = canal_position(tooth, canal_ling, c5, SPACING, frame)
check("canal on -x classified lingual", pos_ling == "lingual", f"got {pos_ling}")

# --- Test 6: root counting -----------------------------------------------------
# Realistic phantom in a bigger volume: crown + two thick, separated roots
# (each well above the 40 mm³ volume floor). Roots separated by a 2.4 mm gap so
# a tilted copy (below) keeps them distinct under nearest-neighbour rotation.
rshape = (80, 80, 80)
def make_two_root(tilt_deg=0):
    tr = np.zeros(rshape, bool)
    tr[8:28, 26:52, 26:54] = True                # crown (bulkier end)
    tr[28:60, 26:52, 26:38] = True               # root 1
    tr[28:60, 26:52, 46:58] = True               # root 2 (gap x=38-46 -> 2.4 mm)
    if tilt_deg:
        from scipy.ndimage import rotate
        tr = rotate(tr.astype(float), tilt_deg, axes=(0, 2), order=0, reshape=True) > 0.5
    return tr

n = count_roots(make_two_root(), SPACING)
check("root count = 2 (axis-aligned)", n == 2, f"got {n}")
# impaction-tilt case: the tooth's long axis is ~30° off the scan z-axis.
# The old z-slab method fails this; the PCA-axis method must still return 2.
n_tilt = count_roots(make_two_root(tilt_deg=30), SPACING)
check("root count = 2 (tilted 30°)", n_tilt == 2, f"got {n_tilt}")
# single solid root -> 1 (no spurious over-segmentation)
single = np.zeros(rshape, bool); single[10:50, 30:44, 30:44] = True
check("single root -> 1", count_roots(single, SPACING) == 1,
      f"got {count_roots(single, SPACING)}")

# --- Test 6b: impaction (M3 <-> M2) relationship ------------------------------
# two adjacent molar blocks touching -> contact; pulled apart -> close/separated
m3 = cuboid(shape, (20, 40), (20, 40), (20, 30))
m2_touch = cuboid(shape, (20, 40), (20, 40), (30, 40))   # abuts m3 at x=30 -> contact
imp_c = assess_impaction(m3, m2_touch, SPACING, contact_mm=0.5, close_mm=2.0)
check("M3-M2 abutting -> contact", imp_c.m2_contact and imp_c.m2_relation == "contact",
      f"{imp_c.m2_min_distance_mm}mm {imp_c.m2_relation}")
m2_close = cuboid(shape, (20, 40), (20, 40), (34, 44))    # ~1.2mm gap -> close
imp_cl = assess_impaction(m3, m2_close, SPACING, contact_mm=0.5, close_mm=2.0)
check("M3-M2 small gap -> close (not contact)",
      (not imp_cl.m2_contact) and imp_cl.m2_relation == "close",
      f"{imp_cl.m2_min_distance_mm}mm {imp_cl.m2_relation}")
m2_far = cuboid(shape, (20, 40), (20, 40), (45, 55))      # ~4.5mm gap -> separated
imp_f = assess_impaction(m3, m2_far, SPACING, contact_mm=0.5, close_mm=2.0)
check("M3-M2 wide gap -> separated", imp_f.m2_relation == "separated",
      f"{imp_f.m2_min_distance_mm}mm {imp_f.m2_relation}")

# --- Test 7: end-to-end feature vector ----------------------------------------
feat = compute_features(tooth, canal, SPACING, ct=None, frame=frame,
                        contact_mm=0.5, close_mm=2.0)
check("end-to-end contact False for 3.3mm gap", feat.contact is False)
check("end-to-end grade slot empty pre-rule", feat.risk_grade == "")
print("\nFeature vector (Test 1 geometry):")
for k, v in feat.as_dict().items():
    print(f"   {k:22s}: {v}")

# --- summary -------------------------------------------------------------------
passed = sum(1 for _, c, _ in results if c)
print(f"\n{passed}/{len(results)} checks passed")
raise SystemExit(0 if passed == len(results) else 1)
