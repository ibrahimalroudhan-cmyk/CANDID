#!/usr/bin/env python3
# Copyright (c) 2026 Ibrahim Alroudhan. All rights reserved. See ../LICENSE.
"""
geometry.py  —  Stage 3 of the third-molar / IAC pipeline.

Turns the binary masks produced by segmentation (Stage 2) into the ten
per-tooth variables defined in the Measurement & Grading Protocol. Each
variable is its own function so it can be unit-tested in isolation.

Inputs throughout:
    tooth : np.ndarray (bool)   segmented third-molar mask
    canal : np.ndarray (bool)   segmented inferior alveolar canal mask
    spacing : (sz, sy, sx)      voxel spacing in mm, matching array axes (z, y, x)
    ct    : np.ndarray (float)  the resampled CT volume (only cortical integrity needs it)

Axis convention: arrays are (z=superior-inferior, y=?, x=?). The buccal-lingual
and vertical axes are passed in explicitly (see AxisFrame) rather than assumed,
because they depend on how the volume was oriented in Stage 1.

NOTE: distances use `spacing`, so every output is in millimetres. Nothing here
assumes isotropic voxels, though Stage 1 resamples to isotropic.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from scipy import ndimage


@dataclass
class AxisFrame:
    """Unit vectors (in z,y,x order) defining anatomical directions for this scan.
    Supplied by Stage 1 orientation, not guessed here."""
    buccal: tuple = (0.0, 0.0, 1.0)     # points toward buccal
    superior: tuple = (-1.0, 0.0, 0.0)  # points toward crown/occlusal (‑z = up here)


@dataclass
class ToothCanalFeatures:
    min_distance_mm: float
    contact: bool
    canal_position: str            # buccal | lingual | inferior | inter-radicular
    cortical_integrity: str        # intact | thinned | interrupted | unknown
    close_approach_len_mm: float
    configuration: str             # separated | grooved | perforated
    n_roots: int
    risk_grade: str                # filled downstream
    distance_method: str = "subvoxel"   # subvoxel (marching cubes) | voxel-centre
    # provenance / QC
    closest_point_vox: tuple = (0, 0, 0)

    def as_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _surface_coords(mask: np.ndarray, spacing) -> np.ndarray:
    """World-mm coordinates (z,y,x) of surface voxels of a binary mask."""
    if not mask.any():
        return np.empty((0, 3))
    eroded = ndimage.binary_erosion(mask)
    surface = mask & ~eroded
    idx = np.argwhere(surface)
    return idx * np.asarray(spacing)


def _all_coords_mm(mask: np.ndarray, spacing) -> np.ndarray:
    idx = np.argwhere(mask)
    return idx * np.asarray(spacing)


# --------------------------------------------------------------------------- #
# individual variables
# --------------------------------------------------------------------------- #
def min_distance(tooth, canal, spacing):
    """Minimum surface-to-surface distance (mm) and the closest tooth voxel,
    using voxel-CENTRE surfaces. Kept for comparison / sensitivity analysis;
    note it overestimates a true gap by up to one voxel. Prefer
    min_distance_subvoxel for the reported measurement."""
    if (tooth & canal).any():
        overlap = np.argwhere(tooth & canal)
        c = overlap[len(overlap) // 2]
        return 0.0, tuple(int(v) for v in c)
    from scipy.spatial import cKDTree
    t_surf = _surface_coords(tooth, spacing)
    c_surf = _surface_coords(canal, spacing)
    if len(t_surf) == 0 or len(c_surf) == 0:
        return float("nan"), (0, 0, 0)
    tree = cKDTree(c_surf)
    d, _ = tree.query(t_surf, k=1)
    j = int(np.argmin(d))
    closest_mm = t_surf[j]
    closest_idx = tuple(int(round(closest_mm[a] / spacing[a])) for a in range(3))
    return float(d[j]), closest_idx


def _mc_surface(mask, spacing):
    """Sub-voxel surface vertices (mm) via marching cubes at the 0.5 isolevel.
    Pads by one voxel so masks touching the array border still close."""
    from skimage.measure import marching_cubes
    m = np.pad(mask.astype(float), 1)
    try:
        verts, _faces, _n, _v = marching_cubes(m, level=0.5, spacing=tuple(spacing))
    except (RuntimeError, ValueError):
        return None
    if len(verts) == 0:
        return None
    return verts - np.asarray(spacing)  # undo the 1-voxel pad offset


def min_distance_subvoxel(tooth, canal, spacing):
    """Minimum surface-to-surface distance (mm) using SUB-VOXEL marching-cubes
    surfaces. Places boundaries at true half-voxel edges, removing the ~one-voxel
    overestimate of the voxel-centre method. This is the reported measurement.
    Falls back to voxel-centre if marching cubes cannot build a surface."""
    if (tooth & canal).any():
        overlap = np.argwhere(tooth & canal)
        c = overlap[len(overlap) // 2]
        return 0.0, tuple(int(v) for v in c)
    from scipy.spatial import cKDTree
    tv = _mc_surface(tooth, spacing)
    cv = _mc_surface(canal, spacing)
    if tv is None or cv is None:
        return min_distance(tooth, canal, spacing)  # graceful fallback
    tree = cKDTree(cv)
    d, _ = tree.query(tv, k=1)
    j = int(np.argmin(d))
    closest_mm = tv[j]
    closest_idx = tuple(int(round(closest_mm[a] / spacing[a])) for a in range(3))
    return float(d[j]), closest_idx


def is_contact(distance_mm, contact_thresh_mm):
    return bool(distance_mm <= contact_thresh_mm)


def canal_position(tooth, canal, closest_idx, spacing, frame: AxisFrame):
    """Classify where the canal sits relative to the tooth at the closest point:
    buccal / lingual / inferior / inter-radicular.
    Uses the LOCAL direction from the closest tooth point to the nearest canal
    voxel (not the canal centroid), so a distant canal bend cannot skew it."""
    from scipy.spatial import cKDTree
    c_vox = np.argwhere(canal)
    if len(c_vox) == 0:
        return "inter-radicular"
    c_mm = c_vox * np.asarray(spacing)
    p_mm = np.array(closest_idx, dtype=float) * np.asarray(spacing)
    _, j = cKDTree(c_mm).query(p_mm, k=1)
    vec = c_mm[j] - p_mm
    n = np.linalg.norm(vec)
    if n == 0:
        return "inter-radicular"
    vec = vec / n
    buccal_comp = float(np.dot(vec, np.array(frame.buccal, dtype=float)))   # + buccal / - lingual
    vert_comp = float(np.dot(vec, np.array(frame.superior, dtype=float)))   # + superior / - inferior
    if vert_comp > 0.6 and abs(buccal_comp) < 0.4:
        return "inter-radicular"
    if vert_comp < -0.5 and abs(buccal_comp) < 0.5:
        return "inferior"
    return "buccal" if buccal_comp >= 0 else "lingual"


def cortical_integrity(ct, canal, tooth, closest_idx, spacing,
                       intact_hu=1000.0, thin_frac=0.5):
    """Assess the corticated canal wall on the tooth-facing aspect at the closest
    point. Samples a short profile from the canal wall toward the tooth and looks
    for a high-density (cortical) ridge. Returns intact | thinned | interrupted.
    Requires the CT volume; returns 'unknown' if not supplied."""
    if ct is None:
        return "unknown"
    canal_com = np.array(ndimage.center_of_mass(canal))
    p = np.array(closest_idx, dtype=float)
    direction = (p - canal_com)
    dn = np.linalg.norm(direction)
    if dn == 0:
        return "unknown"
    direction = direction / dn
    # sample a few voxels straddling the wall
    samples = []
    for step in np.linspace(-2, 2, 9):
        q = np.round(canal_com + direction * step / min(spacing) * 1.0).astype(int)
        if np.all(q >= 0) and np.all(q < np.array(ct.shape)):
            samples.append(ct[tuple(q)])
    if not samples:
        return "unknown"
    peak = float(np.max(samples))
    if peak >= intact_hu:
        return "intact"
    if peak >= intact_hu * thin_frac:
        return "thinned"
    return "interrupted"


def close_approach_length(tooth, canal, spacing, close_thresh_mm):
    """Length (mm) of canal over which it lies within `close_thresh_mm` of the tooth.
    Approximated as: (# canal voxels within threshold) projected onto the canal's
    long axis via its principal direction."""
    from scipy.spatial import cKDTree
    t_surf = _surface_coords(tooth, spacing)
    c_vox = np.argwhere(canal)
    if len(t_surf) == 0 or len(c_vox) == 0:
        return 0.0
    c_mm = c_vox * np.asarray(spacing)
    tree = cKDTree(t_surf)
    d, _ = tree.query(c_mm, k=1)
    near = c_mm[d <= close_thresh_mm]
    if len(near) < 2:
        return 0.0
    # principal axis of the near segment; extent along it = length
    centered = near - near.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vh[0]
    return float(proj.max() - proj.min())


def configuration(tooth, canal, distance_mm, spacing):
    """separated | grooved | perforated.
    perforated : canal lies within the tooth root envelope (masks overlap, or
                 canal voxels enclosed by the tooth convex hull region).
    grooved    : canal indents the root surface (very close + concavity) but not enclosed.
    separated  : clear gap."""
    if (tooth & canal).any():
        return "perforated"
    # enclosed test: fill tooth holes; if canal sits inside filled-but-not-solid region
    filled = ndimage.binary_fill_holes(tooth)
    enclosed = filled & ~tooth & canal
    if enclosed.any():
        return "perforated"
    if distance_mm <= 1.0:
        return "grooved"
    return "separated"


def count_roots(tooth, spacing, min_peak_sep_mm=3.0, min_root_mm3=40.0,
                apical_fraction=0.55, max_roots=4):
    """Count roots along the tooth's OWN long axis, robust to impaction tilt.

    The old version slabbed along the scan z-axis, which is wrong for a tilted /
    impacted third molar (its long axis can be ~60-90° off z).  Here the long axis
    is found by PCA, the apical portion is taken along it, and roots are counted as:
      1. connected components of the apical mass (handles roots already separated in
         the mask), with a resolution-independent volume floor (min_root_mm3); else
      2. if the apical mass is a single fused component, a distance-transform
         watershed splits it — the segmentation often fuses adjacent roots into one
         blob (only a thin isthmus between them), which no connected-component count
         can resolve.  Watershed seeds are well-separated distance-transform peaks
         (>= min_peak_sep_mm apart); basins below min_root_mm3 are discarded.

    Volumes are in mm³ so the result is resolution-independent.  Capped at
    max_roots (a third molar realistically has 1-4 roots).
    """
    if not tooth.any():
        return 0
    sp = np.asarray(spacing, float)
    vox_mm3 = float(np.prod(sp))
    struct = ndimage.generate_binary_structure(3, 3)

    # tooth long axis via PCA (robust to how the scan was oriented / impaction tilt)
    vox = np.argwhere(tooth)
    mm_c = vox * sp
    mm_c = mm_c - mm_c.mean(0)
    _u, _s, vh = np.linalg.svd(mm_c, full_matrices=False)
    t = mm_c @ vh[0]                                     # coronal <-> apical coordinate

    # apical portion = the end away from the crown; the crown is the bulkier end.
    nb = 14
    edges = np.linspace(t.min(), t.max(), nb + 1)
    prof = [((t >= edges[i]) & (t < edges[i + 1])).sum() for i in range(nb)]
    crown_high = np.mean(prof[nb // 2:]) > np.mean(prof[:nb // 2])
    q = apical_fraction * 100.0
    cut = np.percentile(t, q) if crown_high else np.percentile(t, 100.0 - q)
    apical_sel = (t < cut) if crown_high else (t > cut)
    ap = np.zeros_like(tooth)
    ap[tuple(vox[apical_sel].T)] = True
    if ap.sum() * vox_mm3 < min_root_mm3:
        return 1

    # 1) roots already separated in the mask -> connected components (deterministic)
    lbl, n = ndimage.label(ap, structure=struct)
    sizes = ndimage.sum(np.ones_like(lbl), lbl, index=range(1, n + 1)) * vox_mm3
    cc = int((np.asarray(sizes) >= min_root_mm3).sum())
    if cc >= 2:
        return min(cc, max_roots)

    # 2) fused mass -> split via distance-transform watershed
    try:
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
    except ImportError:
        return 1                                         # graceful: report a lower bound
    dt = ndimage.gaussian_filter(ndimage.distance_transform_edt(ap, sampling=sp), 1.0)
    footprint = max(1, int(round(min_peak_sep_mm / float(sp.min()))))
    coords = peak_local_max(dt, min_distance=footprint, labels=ap,
                            threshold_abs=0.4 * float(dt.max()), exclude_border=False)
    if len(coords) <= 1:
        return 1
    markers = np.zeros_like(ap, dtype=int)
    for k, c in enumerate(coords, 1):
        markers[tuple(c)] = k
    ws = watershed(-dt, markers, mask=ap)
    basins = [(ws == i).sum() * vox_mm3 for i in range(1, ws.max() + 1)]
    return min(max(sum(1 for b in basins if b >= min_root_mm3), 1), max_roots)


# --------------------------------------------------------------------------- #
# third-molar ↔ second-molar relationship (impaction / surgical difficulty)
# --------------------------------------------------------------------------- #
@dataclass
class ImpactionFeatures:
    """Third-molar ↔ second-molar (FDI 38↔37 / 48↔47) relationship.

    This is a SURGICAL-DIFFICULTY / impaction signal and is deliberately kept
    separate from the nerve-injury risk grade, which stays purely IAC-driven.
    Contact means the third molar abuts the distal of the second molar."""
    m2_min_distance_mm: float
    m2_contact: bool
    m2_relation: str               # contact | close | separated
    distance_method: str = "subvoxel"
    closest_point_vox: tuple = (0, 0, 0)

    def as_dict(self):
        return asdict(self)


def assess_impaction(third_molar, second_molar, spacing,
                     contact_mm=0.5, close_mm=2.0, subvoxel=True) -> ImpactionFeatures:
    """Measure the third-molar ↔ second-molar relationship.

    Contact (the third molar abutting the second molar) indicates a mesioangular /
    horizontal impaction pattern, which drives distal bone removal and crown
    sectioning to disimpact without levering on the second molar. This is a
    surgical-difficulty signal only — it does NOT feed the nerve-injury risk grade."""
    if subvoxel:
        d, closest = min_distance_subvoxel(third_molar, second_molar, spacing)
        method = "subvoxel"
    else:
        d, closest = min_distance(third_molar, second_molar, spacing)
        method = "voxel-centre"
    contact = is_contact(d, contact_mm)
    if contact:
        relation = "contact"
    elif d == d and d <= close_mm:   # d==d guards NaN
        relation = "close"
    else:
        relation = "separated"
    return ImpactionFeatures(
        m2_min_distance_mm=round(d, 3),
        m2_contact=contact,
        m2_relation=relation,
        distance_method=method,
        closest_point_vox=tuple(int(v) for v in closest),
    )


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def compute_features(tooth, canal, spacing, ct=None,
                     frame: Optional[AxisFrame] = None,
                     contact_mm=0.5, close_mm=2.0, subvoxel=True) -> ToothCanalFeatures:
    frame = frame or AxisFrame()
    if subvoxel:
        d, closest = min_distance_subvoxel(tooth, canal, spacing)
        method = "subvoxel"
    else:
        d, closest = min_distance(tooth, canal, spacing)
        method = "voxel-centre"
    contact = is_contact(d, contact_mm)
    pos = canal_position(tooth, canal, closest, spacing, frame)
    cort = cortical_integrity(ct, canal, tooth, closest, spacing)
    length = close_approach_length(tooth, canal, spacing, close_mm)
    config = configuration(tooth, canal, d, spacing)
    nroots = count_roots(tooth, spacing)
    return ToothCanalFeatures(
        min_distance_mm=round(d, 3),
        contact=contact,
        canal_position=pos,
        cortical_integrity=cort,
        close_approach_len_mm=round(length, 2),
        configuration=config,
        n_roots=nroots,
        risk_grade="",  # assigned downstream
        distance_method=method,
        closest_point_vox=tuple(int(v) for v in closest),
    )
