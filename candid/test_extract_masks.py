#!/usr/bin/env python3
# Copyright (c) 2026 Ibrahim Alroudhan. All rights reserved. See ../LICENSE.
"""test_extract_masks.py — verifies label extraction preserves geometry and picks
the right voxels, using a synthetic multi-label volume. Needs SimpleITK.
Run: python test_extract_masks.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    print("[skip] SimpleITK not installed — extraction test needs it.")
    sys.exit(0)

from extract_masks import list_labels, extract, assemble_mesial_mandibular_tooth

results = []
def check(name, cond, detail=""):
    results.append(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    shape = (40, 40, 40)
    spacing_xyz = (0.3, 0.3, 0.3)

    # multi-label phantom: 2 = mandible, 4 = lower teeth, 5 = canal
    seg = np.zeros(shape, np.uint8)
    seg[5:35, 5:35, 5:35] = 2
    seg[10:20, 10:20, 10:20] = 4       # teeth block (1000 voxels)
    seg[10:20, 10:20, 25:30] = 5       # canal block (500 voxels)
    img = sitk.GetImageFromArray(seg)
    img.SetSpacing(spacing_xyz)
    img.SetOrigin((1.0, 2.0, 3.0))     # non-trivial geometry to test preservation
    seg_path = tmp / "seg.nii.gz"
    sitk.WriteImage(img, str(seg_path))

    # list labels
    check("lists non-zero labels", list_labels(seg_path) == [2, 4, 5],
          str(list_labels(seg_path)))

    # extract canal (label 5)
    canal_out = tmp / "canal.nii.gz"
    n = extract(seg_path, [5], canal_out)
    check("canal voxel count correct", n == 10 * 10 * 5, f"got {n}")
    ci = sitk.ReadImage(str(canal_out))
    carr = sitk.GetArrayFromImage(ci)
    check("canal mask is binary", set(np.unique(carr)).issubset({0, 1}))
    check("canal geometry preserved (spacing)",
          all(abs(a - b) < 1e-4 for a, b in zip(ci.GetSpacing(), spacing_xyz)),
          str(ci.GetSpacing()))
    check("canal geometry preserved (origin)", ci.GetOrigin() == (1.0, 2.0, 3.0))
    check("canal matches source region", np.array_equal(carr.astype(bool), (seg == 5)))

    # extract teeth (label 4)
    teeth_out = tmp / "teeth.nii.gz"
    n2 = extract(seg_path, [4], teeth_out)
    check("teeth voxel count correct", n2 == 10 * 10 * 10, f"got {n2}")

    # merge multiple labels (2 + 4)
    merged_out = tmp / "merged.nii.gz"
    n3 = extract(seg_path, [2, 4], merged_out)
    check("merged count = union", n3 == int(((seg == 2) | (seg == 4)).sum()), f"got {n3}")

    # missing label -> 0 voxels, no crash
    empty_out = tmp / "empty.nii.gz"
    n4 = extract(seg_path, [99], empty_out)
    check("absent label -> 0 voxels, no crash", n4 == 0)

# --------------------------------------------------------------------------- #
# Second-molar assembly: two-arch phantom (maxillary must be rejected; the split
# lower second molar must be assembled whole across its two labels).
# --------------------------------------------------------------------------- #
def make_two_arch(tmp, with_second_molar=True):
    """Build a synthetic two-arch segmentation at 0.3 mm isotropic.
      label 1  = mandible bone slab (lower arch)
      label 3  = canal (excluded)
      label 34 = lower third molar (target)
      label 33 + 42 = lower second molar, SPLIT across two labels (contiguous)
      label 26 = maxillary antagonist tooth, across the bite gap (must be rejected)
    """
    shape = (60, 60, 60)
    seg = np.zeros(shape, np.uint8)
    seg[40:54, 8:52, 8:54] = 1                 # mandible bone (lower)
    seg[33:38, 24:36, 14:20] = 3               # canal, embedded in bone (excluded)
    seg[30:40, 24:36, 42:52] = 34              # lower third molar (touches bone top)
    if with_second_molar:
        seg[30:40, 24:36, 28:34] = 33          # second molar, mesial half
        seg[30:40, 24:36, 34:40] = 42          # second molar, distal half (contiguous)
    seg[6:16, 24:36, 28:40] = 26               # maxillary antagonist, across the gap
    img = sitk.GetImageFromArray(seg)
    img.SetSpacing((0.3, 0.3, 0.3))
    img.SetOrigin((0.0, 0.0, 0.0))
    p = tmp / ("two_arch.nii.gz" if with_second_molar else "two_arch_missing.nii.gz")
    sitk.WriteImage(img, str(p))
    return p


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    seg_path = make_two_arch(tmp)
    excl = [1, 3, 34]      # mandible + canal + third molar
    mask, ref, c, spans = assemble_mesial_mandibular_tooth(
        seg_path, third_molar_label=34, mandible_labels=[1], exclude_labels=excl)

    check("second molar spans both split labels", spans == [33, 42], str(spans))
    check("maxillary antagonist (26) rejected", 26 not in spans)
    check("third molar (34) not absorbed", 34 not in spans)
    # whole tooth assembled = both halves, 10*12*12 = 1440 voxels
    check("assembled second molar voxel count", int(mask.sum()) == 1440,
          f"got {int(mask.sum())}")
    # centroid must be in the lower arch (near the teeth z-band, not the maxillary band)
    seg_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path)))
    check("second molar mask disjoint from maxillary tooth",
          not (mask & (seg_arr == 26)).any())

    # missing second molar -> only maxillary tooth remains -> must raise, not return 26
    seg_missing = make_two_arch(tmp, with_second_molar=False)
    raised = False
    try:
        assemble_mesial_mandibular_tooth(seg_missing, 34, mandible_labels=[1],
                                         exclude_labels=[1, 3, 34])
    except ValueError:
        raised = True
    check("no mandibular neighbour -> ValueError (no wrong-arch fallback)", raised)


passed = sum(results)
print(f"\n{passed}/{len(results)} checks passed")
raise SystemExit(0 if passed == len(results) else 1)
