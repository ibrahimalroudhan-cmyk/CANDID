#!/usr/bin/env python3
# Copyright (c) 2026 Ibrahim Alroudhan. All rights reserved. See ../LICENSE.
"""
extract_masks.py  —  Stage 2.5 glue for CANDID.

Pulls binary tooth / canal masks out of a multi-label segmentation into the two
inputs the later stages expect. Source-agnostic: give it the segmentation file and
the label integer(s) for the structure you want. Works for:

  * DentalSegmentator  (mandibular canal = one label; teeth grouped as
                        "lower teeth" = one label)
  * ToothSeg           (per-FDI teeth, e.g. 38 / 48 as their own labels)
  * full ToothFairy2   (left/right canals + per-FDI teeth as separate labels)

The output geometry (spacing / origin / direction) is copied from the input, so
the extracted masks stay co-registered with the scan and with each other.

Confirm the label integers from each model's own files first:
    python extract_masks.py --seg seg.nii.gz --list

Examples
--------
  # canal from a DentalSegmentator output (confirm the canal id with --list):
  python extract_masks.py --seg dental_seg.nii.gz --label 5 --out data/case001/case001_canal.nii.gz

  # third molar 38 from a ToothSeg / full-model output:
  python extract_masks.py --seg tooth_seg.nii.gz --label 38 --out data/case001/case001_38.nii.gz

  # a bilateral canal label works fine for distance to one tooth — the nearest
  # canal surface is picked automatically. To merge several ids, list them:
  python extract_masks.py --seg seg.nii.gz --label 3 4 --out canal.nii.gz

Note: a merged bilateral canal is acceptable for the minimum-distance measurement
because the nearer canal dominates; split left/right only if you need the
canal_position label to be unambiguous per side.

Second-molar discovery (mandibular voxel region)
------------------------------------------------
The second molar (FDI 37/47) has no reliable label — the model may split it across
several tooth labels (e.g. 33/42).  assemble_mesial_mandibular_tooth() finds it as a
voxel region inside the mandible bone envelope (which excludes all maxillary teeth),
then takes the in-arch tooth region adjacent to the third molar:

  python extract_masks.py --seg seg.nii.gz --find-mesial-to 34 \\
         --mandible 1 2 --exclude 1 2 3 4 43 --out work/case001/case001_37.nii.gz

This is robust to impacted / tilted third molars, where an anterior-axis or
nearest-centroid heuristic would return the wrong-arch antagonist tooth.

Config-driven shortcut
----------------------
  python extract_masks.py --config config.yaml \\
         --seg work/case001/nnunet_out/case.nii.gz --side left --out-dir work/case001/masks/

Writes canal.nii.gz, third_molar.nii.gz, second_molar.nii.gz, mandible.nii.gz.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _load(path):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return img, arr


# --------------------------------------------------------------------------- #
# Second-molar (FDI 37/47) discovery by mandibular voxel region
# --------------------------------------------------------------------------- #
# Thresholds (mm / mm³), named so they are auditable and easy to retune.
_ENVELOPE_MM   = 2.5    # tooth voxels within this distance of mandible bone = "in-arch"
_MIN_TOOTH_MM3 = 30.0   # discard connected fragments smaller than this (specks, spurs)
_ADJ_MM        = 3.0    # second molar must abut the third molar within this distance
_REUNITE_MM    = 6.0    # generous mandibular restriction used to reunite carved fragments


def assemble_mesial_mandibular_tooth(seg_path, third_molar_label: int,
                                     mandible_labels: list, exclude_labels: list) -> tuple:
    """Assemble the mask of the mandibular tooth immediately mesial to the third molar.

    Finds FDI 37 (or 47) as a *voxel region*, not a single label — the model may
    split the second molar across several unreliable tooth labels (e.g. 33/42), so
    trusting one integer is unsafe.  The arch is disambiguated geometrically via the
    mandible bone envelope, which excludes all maxillary teeth (antagonist + upper
    molars) regardless of how they are labelled.  This is robust to impacted, tilted
    third molars where an anterior-axis / nearest-centroid heuristic fails.

    Algorithm:
      1. Build the mandible envelope: the true mm distance transform from the merged
         mandible mask (mandible_labels), thresholded at _ENVELOPE_MM.  A distance
         transform (not binary_dilation iterations) is required at anisotropic native
         CBCT spacing, where iteration-based dilation over-grows along the thick axis
         and can bridge the occlusal gap between the arches.
      2. Candidate cloud = all tooth voxels (every non-excluded, non-third-molar label)
         intersected with the envelope.  Maxillary teeth fall outside and drop out.
      3. Connected-component the candidate cloud (full 26-connectivity); discard
         fragments below _MIN_TOOTH_MM3 (partial-volume specks, isolated spurs).
      4. Keep components whose surface is within _ADJ_MM of the third molar; the
         nearest such component is the body of the second molar.
      5. Reunite the whole tooth: use that body as a seed and recover any same-tooth
         fragments the envelope carved off, by keeping connected components of the
         un-carved mandibular tooth cloud (distance ≤ _REUNITE_MM from bone) that
         overlap the seed.  Tooth voxels of the third molar are explicitly removed so
         a touching second molar can never absorb it.

    Returns (mask_bool, img, centroid_lps, spanned_labels).
    Raises ValueError if the third molar is absent or no adjacent mandibular tooth
    is found (rather than silently returning a maxillary or wrong-side label).
    """
    from scipy import ndimage
    img, arr = _load(seg_path)
    spacing   = np.array(img.GetSpacing())               # (x, y, z) mm — LPS
    origin    = np.array(img.GetOrigin())                # LPS mm
    direction = np.array(img.GetDirection()).reshape(3, 3)
    sampling  = spacing[::-1]                             # (z, y, x) for EDT / arrays
    vox_mm3   = float(np.prod(spacing))
    struct    = ndimage.generate_binary_structure(3, 3)  # full 26-connectivity

    third = (arr == int(third_molar_label))
    if not third.any():
        raise ValueError(f"third molar label {third_molar_label} not found in segmentation")

    mandible = np.isin(arr, [int(m) for m in mandible_labels])
    if not mandible.any():
        raise ValueError(f"mandible labels {mandible_labels} not found in segmentation")

    def centroid_lps(mask):
        mean_xyz = np.argwhere(mask).mean(axis=0)[::-1]  # (z,y,x) → (x,y,z)
        return origin + direction @ (spacing * mean_xyz)

    # 1. mandible envelope (mm distance transform, arch-safe at anisotropic spacing)
    dist_bone = ndimage.distance_transform_edt(~mandible, sampling=sampling)
    envelope  = dist_bone <= _ENVELOPE_MM

    # 2. candidate tooth cloud, restricted to the mandibular arch
    skip = set(int(x) for x in exclude_labels) | {0, int(third_molar_label)}
    tooth_cloud = np.isin(arr, [int(v) for v in np.unique(arr) if int(v) not in skip])
    candidates = tooth_cloud & envelope

    # 3. connected components; drop sub-tooth fragments
    cc, n = ndimage.label(candidates, structure=struct)
    dist_third = ndimage.distance_transform_edt(~third, sampling=sampling)

    # 4. nearest component that abuts the third molar
    best = None  # (dist_to_third, component_mask)
    for i in range(1, n + 1):
        cm = (cc == i)
        if cm.sum() * vox_mm3 < _MIN_TOOTH_MM3:
            continue
        d = float(dist_third[cm].min())
        if d <= _ADJ_MM and (best is None or d < best[0]):
            best = (d, cm)

    if best is None:
        raise ValueError(
            f"no mandibular tooth found within {_ADJ_MM} mm of third molar "
            f"{third_molar_label}. Checked {n} in-arch components after excluding "
            f"{sorted(skip)}. The second molar may be missing/agenetic, or the "
            f"mandible labels {mandible_labels} may be wrong for this segmentation."
        )
    d_body, body = best

    # 5. reunite same-tooth fragments carved off by the envelope; never absorb the 3rd molar
    mand_cloud = tooth_cloud & (dist_bone <= _REUNITE_MM)
    cc2, _ = ndimage.label(mand_cloud, structure=struct)
    seed_ids = set(int(v) for v in np.unique(cc2[body]) if v != 0)
    mask = np.isin(cc2, list(seed_ids)) & ~third

    spans = sorted(int(v) for v in np.unique(arr[mask]) if v != 0)
    return mask, img, centroid_lps(mask), spans


def _write_mask(mask, ref_img, out_path):
    """Write a boolean mask as a binary NIfTI, copying geometry from ref_img."""
    import SimpleITK as sitk
    out = sitk.GetImageFromArray(mask.astype(np.uint8))
    out.CopyInformation(ref_img)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(out_path))
    return int(mask.sum())


def list_labels(path):
    """Return the sorted non-zero label integers present in a segmentation."""
    _, arr = _load(path)
    return sorted(int(v) for v in np.unique(arr) if v != 0)


def extract(path, label_ids, out_path):
    """Write a binary NIfTI where voxels equal any of label_ids, preserving the
    input geometry. Returns the number of voxels written."""
    import SimpleITK as sitk
    img, arr = _load(path)
    ids = [int(x) for x in label_ids]
    mask = np.isin(arr, ids).astype(np.uint8)
    n = int(mask.sum())
    if n == 0:
        print(f"WARNING: no voxels found for labels {ids} in {path} "
              f"(present: {sorted(int(v) for v in np.unique(arr) if v != 0)})",
              file=sys.stderr)
    out = sitk.GetImageFromArray(mask)
    out.CopyInformation(img)  # spacing / origin / direction preserved
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(out_path))
    return n


def main():
    ap = argparse.ArgumentParser(description="Extract binary masks from a multi-label segmentation.")
    ap.add_argument("--seg", help="multi-label segmentation NIfTI")
    ap.add_argument("--label", nargs="*", type=int, help="label integer(s) to extract")
    ap.add_argument("--out", help="output binary NIfTI path")
    ap.add_argument("--list", action="store_true", help="just list the labels present and exit")
    ap.add_argument("--find-mesial-to", type=int, metavar="LABEL",
                    help="assemble the mandibular tooth mesial-adjacent to third-molar LABEL")
    ap.add_argument("--mandible", nargs="*", type=int, default=[1, 2],
                    help="mandible bone label integer(s) for the arch envelope (default: 1 2)")
    ap.add_argument("--exclude", nargs="*", type=int, default=[],
                    help="label integers to skip during the mesial search")
    ap.add_argument("--config", metavar="CONFIG_YAML",
                    help="read label map from config.yaml and extract all standard masks")
    ap.add_argument("--side", choices=["left", "right"],
                    help="which side to process when using --config (default: left)")
    ap.add_argument("--out-dir", metavar="DIR",
                    help="output directory when using --config")
    args = ap.parse_args()

    # ── config-driven batch extraction ──────────────────────────────────────
    if args.config:
        import yaml
        if not args.seg:
            sys.exit("ERROR: --seg is required with --config")
        cfg = yaml.safe_load(Path(args.config).read_text())
        lbl = cfg["labels"]
        side = args.side or "left"
        canal_lbl     = lbl["iac_left"]  if side == "left" else lbl["iac_right"]
        molar3_lbl    = lbl["tooth_38"]  if side == "left" else lbl["tooth_48"]
        mandible_lbls = lbl["mandible"]
        all_known     = list(mandible_lbls) + [lbl["iac_left"], lbl["iac_right"],
                                               lbl["tooth_38"], lbl["tooth_48"]]
        out_dir = Path(args.out_dir or ".")
        extract(args.seg, [canal_lbl],   out_dir / "canal.nii.gz")
        print(f"wrote canal.nii.gz  (label {canal_lbl})")
        extract(args.seg, [molar3_lbl],  out_dir / "third_molar.nii.gz")
        print(f"wrote third_molar.nii.gz  (label {molar3_lbl})")
        extract(args.seg, mandible_lbls, out_dir / "mandible.nii.gz")
        print(f"wrote mandible.nii.gz  (labels {mandible_lbls})")
        # Second molar is assembled as a mandibular voxel region (may span several
        # unreliable labels), disambiguated from maxillary teeth by the bone envelope.
        mask, ref, c, spans = assemble_mesial_mandibular_tooth(
            args.seg, molar3_lbl, mandible_labels=mandible_lbls, exclude_labels=all_known)
        n = _write_mask(mask, ref, out_dir / "second_molar.nii.gz")
        print(f"wrote second_molar.nii.gz  (mandibular region spanning labels {spans}, "
              f"centroid {c.round(1)} mm, {n} voxels)")
        return

    # ── mandibular mesial-tooth assembly ────────────────────────────────────
    if args.find_mesial_to:
        if not args.seg:
            sys.exit("ERROR: --seg is required with --find-mesial-to")
        mask, ref, c, spans = assemble_mesial_mandibular_tooth(
            args.seg, args.find_mesial_to,
            mandible_labels=args.mandible, exclude_labels=args.exclude or [])
        print(f"mesial mandibular tooth: spans labels {spans}, centroid {c.round(1)} mm, "
              f"{int(mask.sum())} voxels")
        if args.out:
            n = _write_mask(mask, ref, args.out)
            print(f"wrote {args.out}  ({n} voxels)")
        return

    # ── standard single-label extraction ────────────────────────────────────
    if not args.seg:
        sys.exit("ERROR: --seg is required")
    if args.list:
        print(f"labels present in {args.seg}: {list_labels(args.seg)}")
        return
    if not args.label or not args.out:
        sys.exit("ERROR: need --label and --out (or use --list to inspect).")

    n = extract(args.seg, args.label, args.out)
    print(f"wrote {args.out}  ({n} voxels, labels {args.label})")


if __name__ == "__main__":
    main()
