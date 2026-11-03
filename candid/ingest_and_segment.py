#!/usr/bin/env python3
# Copyright (c) 2026 Ibrahim Alroudhan. All rights reserved. See ../LICENSE.
"""
ingest_and_segment.py  —  Stage 1 (ingest/normalise) + Stage 2 (segmentation)
of the automated third-molar / IAC pipeline.

What it does
------------
1. Reads a CBCT DICOM series into a 3D volume.
2. Resamples to isotropic voxels and records the spacing (this sets the
   resolution-limited contact threshold used downstream).
3. Writes the volume in nnU-Net input format and runs the imported model.
4. Loads the predicted segmentation and extracts the inferior alveolar
   canal(s) and the target third molar(s) (FDI 38 / 48).
5. Runs a first-pass quality check (per-label voxel counts, presence,
   bounding boxes) and writes a QC report.

This is a SCAFFOLD. The segmentation call and the label ids are integration
points — validate on a SINGLE scan before batch use. Run only inside the
approved environment when the input is patient data.

Usage
-----
    export nnUNet_results=/models/nnUNet_results
    python ingest_and_segment.py --config config.yaml
    python ingest_and_segment.py --config config.yaml --dry-run   # skip the model call
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_dicom_series(folder: str):
    """Read a single CBCT DICOM series folder into a SimpleITK image."""
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(folder)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {folder}")
    if len(series_ids) > 1:
        print(f"[warn] {len(series_ids)} series present; using the first. "
              "Split multi-series studies beforehand.")
    files = reader.GetGDCMSeriesFileNames(folder, series_ids[0])
    reader.SetFileNames(files)
    img = reader.Execute()
    print(f"[ingest] size={img.GetSize()} spacing={tuple(round(s,3) for s in img.GetSpacing())} mm")
    return img


def resample_isotropic(img, spacing_mm: float):
    """Resample to isotropic voxels; linear for intensity images."""
    import SimpleITK as sitk
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()
    new_spacing = [float(spacing_mm)] * 3
    new_size = [int(round(osz * ospc / spacing_mm)) for osz, ospc in zip(orig_size, orig_spacing)]
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing(new_spacing)
    rs.SetSize(new_size)
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetOutputDirection(img.GetDirection())
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(-1000)
    out = rs.Execute(img)
    print(f"[resample] -> size={out.GetSize()} spacing={new_spacing} mm")
    return out


def run_nnunet(in_dir: Path, out_dir: Path, model: dict):
    """Call nnU-Net v2 inference. Requires nnUNet_results to be exported."""
    cmd = [
        "nnUNetv2_predict",
        "-i", str(in_dir),
        "-o", str(out_dir),
        "-d", str(model["dataset_id"]),
        "-c", str(model["configuration"]),
        "-f", str(model["folds"]),
        "-device", str(model.get("device", "cpu")),  # cpu | mps (M1 native) | cuda
    ]
    print("[segment] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def label_qc(seg_path: Path, labels: dict) -> dict:
    """Per-label presence, voxel count and bounding box for the target structures."""
    import SimpleITK as sitk
    import numpy as np
    seg = sitk.ReadImage(str(seg_path))
    arr = sitk.GetArrayFromImage(seg)  # z, y, x
    spacing = seg.GetSpacing()
    report = {"spacing_mm": [round(s, 4) for s in spacing], "structures": {}}
    for name, lid in labels.items():
        if not lid:  # 0 / unset
            report["structures"][name] = {"label_id": lid, "status": "UNSET — fill config"}
            continue
        mask = arr == int(lid)
        n = int(mask.sum())
        entry = {"label_id": int(lid), "voxels": n, "present": n > 0}
        if n > 0:
            zs, ys, xs = np.where(mask)
            entry["bbox_vox"] = [int(xs.min()), int(ys.min()), int(zs.min()),
                                 int(xs.max()), int(ys.max()), int(zs.max())]
            entry["volume_mm3"] = round(n * spacing[0] * spacing[1] * spacing[2], 1)
        report["structures"][name] = entry
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="skip the segmentation model call")
    args = ap.parse_args()

    cfg = load_config(args.config)
    work = Path(cfg["paths"]["work_dir"])
    nn_in = work / "nnunet_in"
    nn_out = work / "nnunet_out"
    for d in (work, nn_in, nn_out):
        d.mkdir(parents=True, exist_ok=True)

    # Stage 1 — ingest + normalise
    import SimpleITK as sitk
    img = read_dicom_series(cfg["paths"]["dicom_series"])
    iso = resample_isotropic(img, cfg["preprocess"]["target_spacing_mm"])
    # nnU-Net v2 expects <CASE>_0000.nii.gz for a single-channel input
    case_in = nn_in / "case_0000.nii.gz"
    sitk.WriteImage(iso, str(case_in))
    print(f"[write] {case_in}")

    # Stage 2 — segmentation (imported model)
    if args.dry_run:
        print("[dry-run] skipping nnU-Net; expecting an existing seg for QC if present.")
    else:
        if not os.environ.get("nnUNet_results"):
            sys.exit("ERROR: export nnUNet_results before running (see README).")
        run_nnunet(nn_in, nn_out, cfg["model"])

    # QC on the predicted segmentation
    seg_path = nn_out / "case.nii.gz"
    if seg_path.exists():
        report = label_qc(seg_path, cfg["labels"])
        missing = [k for k, v in report["structures"].items()
                   if v.get("status") or not v.get("present", False)]
        report["all_targets_present"] = len(missing) == 0
        report["missing_or_unset"] = missing
        with open(work / "segmentation_qc.json", "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        if missing:
            print(f"[qc] CHECK: missing/unset target labels: {missing}")
        else:
            print("[qc] all target structures present.")
    else:
        print(f"[qc] no segmentation at {seg_path} (dry-run, or model not yet run).")


if __name__ == "__main__":
    main()
