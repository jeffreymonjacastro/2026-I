import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def md(cell_id, text):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": text}


def code(cell_id, text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": text.strip() + "\n",
    }


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_kernel(version, cells, metadata):
    out = ROOT / "kaggle" / version / "input"
    out.mkdir(parents=True, exist_ok=True)
    (out / "main.ipynb").write_text(json.dumps(nb(cells), indent=1) + "\n", encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


COMMON_SETUP = r'''
import json, math, os, random, re, shutil, subprocess, sys, time, traceback, zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
    from ultralytics import YOLO

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print({"python": sys.version.split()[0], "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
'''


COMMON_HELPERS = r'''
def read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def write_rgb(path, rgb, quality=92):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, quality=quality)

def normalize_image_id(name):
    stem = Path(str(name)).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    return re.sub(r"_+", "_", stem).strip("_")

def extract_holdout_original(filename):
    parts = filename.split("__", 2)
    if len(parts) != 3 or len(parts[0]) != 3 or not parts[0].isdigit():
        return None
    return parts[2]

def find_plantvillage_root(input_root):
    candidates = [
        input_root / "plantdisease" / "PlantVillage",
        input_root / "datasets" / "emmarex" / "plantdisease" / "PlantVillage",
    ]
    for c in candidates:
        if c.exists():
            return c
    for c in input_root.glob("**/PlantVillage"):
        if c.is_dir():
            return c
    raise FileNotFoundError("PlantVillage root not found")

def find_holdout_root(input_root, categories):
    candidates = [
        input_root / "cv-pc5-v3-plantvillage-segmentation-holdout",
        input_root / "datasets" / "jeffreyamc" / "cv-pc5-v3-plantvillage-segmentation-holdout",
    ]
    for c in candidates:
        if c.exists() and all((c / x).exists() for x in categories):
            return c
    for c in input_root.glob("**"):
        if c.is_dir() and all((c / x).exists() for x in categories):
            return c
    raise FileNotFoundError("Holdout root not found")

def split_stratified(df, group_col="source_class", train_ratio=0.70, val_ratio=0.15):
    chunks = []
    for _, g in df.groupby(group_col, sort=True):
        g = g.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        n = len(g)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))
        else:
            n_train, n_val = max(1, n - 1), 0
        g.loc[: n_train - 1, "split"] = "train"
        if n_val:
            g.loc[n_train : n_train + n_val - 1, "split"] = "val"
        g.loc[n_train + n_val :, "split"] = "test"
        chunks.append(g)
    return pd.concat(chunks, ignore_index=True)

def fill_holes(mask):
    mask = (mask > 0).astype(np.uint8) * 255
    h, w = mask.shape
    flood = mask.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)

def reconstruct_from_seed(seed, candidate, iterations=48):
    seed = (seed > 0).astype(np.uint8) * 255
    candidate = (candidate > 0).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    cur = cv2.bitwise_and(seed, candidate)
    for _ in range(iterations):
        nxt = cv2.bitwise_and(cv2.dilate(cur, kernel, iterations=1), candidate)
        if np.array_equal(nxt, cur):
            break
        cur = nxt
    return cur

def keep_reasonable_components(mask, max_components=3, min_area_ratio=0.004):
    h, w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return np.zeros_like(mask)
    min_area = max(64, int(min_area_ratio * h * w))
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
    clean = np.zeros_like(mask)
    for lab_id in order[:max_components]:
        if stats[lab_id, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == lab_id] = 255
    if clean.sum() == 0:
        clean[labels == int(order[0])] = 255
    return clean

def leaf_tissue_shape_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    bb = rgb[:, :, 2].astype(np.int16)
    exg = cv2.normalize((2 * g - r - bb).astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    exg_otsu = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0
    sat_otsu = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0
    chroma = cv2.threshold(cv2.absdiff(a, b), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0

    green = (h >= 22) & (h <= 100) & (s >= 18) & (v >= 25)
    yellow = (h >= 12) & (h <= 48) & (s >= 18) & (v >= 45)
    brown = (((h <= 24) | (h >= 165)) & (s >= 18) & (v >= 22) & (v <= 220))
    dark_leaf_like = (v <= 105) & (s >= 15) & (gray >= 12)

    # Keep the v4-style foreground as the anchor. Broad chroma/Otsu masks are
    # useful only as seed evidence; if used as candidates they can absorb gray
    # textured backgrounds and train YOLO on bad pseudo-labels.
    seed_bool = (green | yellow | (exg_otsu & (s >= 18)) | (chroma & sat_otsu & (s >= 20))) & (v >= 20)
    seed = seed_bool.astype(np.uint8) * 255
    seed_neighborhood = cv2.dilate(seed, np.ones((21, 21), np.uint8), iterations=2) > 0
    candidate_bool = green | yellow | brown | (dark_leaf_like & seed_neighborhood)
    candidate = (candidate_bool & (v >= 12)).astype(np.uint8) * 255

    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, kernel3, iterations=1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel5, iterations=2)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel3, iterations=1)
    seed = keep_reasonable_components(seed, max_components=4, min_area_ratio=0.002)
    mask = reconstruct_from_seed(seed, candidate, iterations=64)
    seed_fallback = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel5, iterations=3)
    seed_fallback = fill_holes(keep_reasonable_components(seed_fallback, max_components=3, min_area_ratio=0.004))
    if mask.sum() == 0:
        mask = seed_fallback
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=3)
    before_fill_area = int((mask > 0).sum())
    mask = fill_holes(mask)
    mask = keep_reasonable_components(mask, max_components=3, min_area_ratio=0.004)
    mask = cv2.medianBlur(mask, 5)
    # If reconstruction still leaked into the background, fall back to the
    # conservative anchor mask rather than publishing a false full-image leaf.
    simple_fg = mask > 0
    border = np.zeros_like(simple_fg, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    area_ratio = float(simple_fg.mean())
    border_touch = float(np.logical_and(simple_fg, border).sum() / max(1, border.sum()))
    if area_ratio > 0.88 or (area_ratio > 0.60 and border_touch > 0.55):
        mask = cv2.medianBlur(seed_fallback, 5)
        before_fill_area = int((mask > 0).sum())
        mask = fill_holes(mask)
        mask = keep_reasonable_components(mask, max_components=3, min_area_ratio=0.004)
        mask = cv2.medianBlur(mask, 5)
    after_fill_area = int((mask > 0).sum())
    return mask, before_fill_area, after_fill_area

def mask_quality(mask, before_fill_area=None, after_fill_area=None):
    h, w = mask.shape
    fg = mask > 0
    area = int(fg.sum())
    area_ratio = float(area / max(1, h * w))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    comp_count = max(0, n - 1)
    border = np.zeros_like(fg, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    border_touch_ratio = float(np.logical_and(fg, border).sum() / max(1, border.sum()))
    if area:
        ys, xs = np.where(fg)
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
        bbox_area = int((x2 - x1 + 1) * (y2 - y1 + 1))
        bbox_fill_ratio = float(area / max(1, bbox_area))
        bbox_image_ratio = float(bbox_area / max(1, h * w))
    else:
        bbox_fill_ratio = 0.0
        bbox_image_ratio = 0.0
    if before_fill_area is None or after_fill_area is None:
        hole_fill_ratio = 0.0
    else:
        hole_fill_ratio = float(max(0, after_fill_area - before_fill_area) / max(1, after_fill_area))
    reasons = []
    if area == 0:
        reasons.append("empty_mask")
    if area_ratio < 0.025:
        reasons.append("small_mask_area")
    if area_ratio > 0.88:
        reasons.append("large_mask_area")
    if comp_count > 4:
        reasons.append("many_components")
    if border_touch_ratio > 0.80 and area_ratio > 0.60:
        reasons.append("excessive_border_touch")
    if border_touch_ratio > 0.55 and bbox_fill_ratio > 0.90 and area_ratio > 0.55:
        reasons.append("border_touch_background_like")
    if bbox_fill_ratio > 0.97 and bbox_image_ratio > 0.60:
        reasons.append("background_like_solid_region")
    if bbox_fill_ratio < 0.18 and area_ratio > 0.03:
        reasons.append("fragmented_sparse_shape")
    return {
        "mask_area_ratio": area_ratio,
        "component_count": int(comp_count),
        "border_touch_ratio": border_touch_ratio,
        "hole_fill_ratio": hole_fill_ratio,
        "bbox_fill_ratio": bbox_fill_ratio,
        "bbox_image_ratio": bbox_image_ratio,
        "weak_reasons": ";".join(reasons),
        "accepted": len(reasons) == 0,
    }

def masks_to_yolo_lines(mask, class_id=0, min_area_ratio=0.0015, epsilon_ratio=0.0025, max_contours=8):
    h, w = mask.shape
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    lines, min_area = [], max(16, h * w * min_area_ratio)
    for cnt in contours[:max_contours]:
        if cv2.contourArea(cnt) < min_area:
            continue
        eps = max(1.0, epsilon_ratio * cv2.arcLength(cnt, True))
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        coords = []
        for x, y in approx:
            coords.extend([float(np.clip(x / w, 0, 1)), float(np.clip(y / h, 0, 1))])
        lines.append(str(class_id) + " " + " ".join(f"{v:.6f}" for v in coords))
    return lines

def label_file_to_mask(label_path, shape):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    p = Path(label_path)
    if not p.exists():
        return mask
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        vals = [float(v) for v in parts[1:]]
        pts = [[int(round(x * w)), int(round(y * h))] for x, y in zip(vals[0::2], vals[1::2])]
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return mask

def result_to_mask(result, shape):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    masks = getattr(result, "masks", None)
    if masks is None:
        return mask
    data = getattr(masks, "data", None)
    if data is not None and len(data):
        for arr in data.detach().cpu().numpy():
            m = (arr > 0.5).astype(np.uint8) * 255
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = cv2.bitwise_or(mask, m)
    elif getattr(masks, "xy", None) is not None:
        for poly in masks.xy:
            if poly is not None and len(poly) >= 3:
                cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 255)
    return mask

def mask_iou(a, b):
    aa, bb = a > 0, b > 0
    union = np.logical_or(aa, bb).sum()
    return 1.0 if union == 0 else float(np.logical_and(aa, bb).sum() / union)

def overlay_mask(rgb, mask, color=(40, 180, 80), alpha=0.45):
    out = rgb.copy()
    c = np.array(color, dtype=np.uint8)
    out[mask > 0] = (out[mask > 0] * (1 - alpha) + c * alpha).astype(np.uint8)
    edges = cv2.Canny((mask > 0).astype(np.uint8) * 255, 50, 150)
    out[edges > 0] = np.array([255, 40, 40], dtype=np.uint8)
    return out

def side_by_side(images):
    h = min(img.shape[0] for img in images)
    resized = []
    for img in images:
        if img.shape[0] != h:
            w = int(img.shape[1] * h / img.shape[0])
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        resized.append(img)
    return np.concatenate(resized, axis=1)

def zip_dir(folder, zip_path):
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=folder)
    return zip_path

def metrics_from_results(metrics):
    values = {}
    for attr in ["box", "seg"]:
        obj = getattr(metrics, attr, None)
        if obj is None:
            continue
        for key in ["mp", "mr", "map50", "map", "map75"]:
            val = getattr(obj, key, None)
            if val is not None:
                try:
                    values[f"pseudo_{attr}_{key}"] = float(val)
                except Exception:
                    pass
    return values

def assert_only_class_zero(label_root):
    bad = []
    for p in Path(label_root).glob("**/*.txt"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip() and line.split()[0] != "0":
                bad.append(str(p))
                break
    if bad:
        raise AssertionError(f"Non-zero class ids in labels: {bad[:5]}")
'''


V7_PATHS = r'''
INPUT_ROOT = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")
CATEGORIES = ["healthy_leaves","small_diseased_regions","large_diseased_regions","simple_backgrounds","complex_backgrounds","multiple_leaves","partial_occlusions"]
MIN_ACCEPT_RATE = 0.80
MIN_ACCEPTED_IMAGES = 5000

PREPARED_DIR = WORKING_DIR / "prepared_leaf_yolo_dataset"
PREVIEW_DIR = WORKING_DIR / "leaf_pseudo_labels_preview"
QUAL_HOLDOUT_DIR = WORKING_DIR / "qualitative_leaf_holdout"
QUANT_DIR = WORKING_DIR / "quantitative_metrics"
RUNS_DIR = WORKING_DIR / "runs_leaf"
SUMMARY_PATH = WORKING_DIR / "run_summary.json"
for p in [PREPARED_DIR, PREVIEW_DIR/"accepted", PREVIEW_DIR/"rejected", PREVIEW_DIR/"qa_failures", QUAL_HOLDOUT_DIR, QUANT_DIR, RUNS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
for c in CATEGORIES:
    (QUAL_HOLDOUT_DIR / "by_category" / c).mkdir(parents=True, exist_ok=True)
progress = {
    "status": "running",
    "stage": "start",
    "task": "leaf segmentation",
    "pseudo_label_source": "v7 heuristic leaf tissue + shape; no MobileSAM truth",
    "holdout_policy": "67 holdout images excluded from train/val/test; qualitative only",
    "qa_gate": {"min_accept_rate": MIN_ACCEPT_RATE, "min_accepted_images": MIN_ACCEPTED_IMAGES},
    "artifacts": [],
}
def save_progress():
    SUMMARY_PATH.write_text(json.dumps(progress, indent=2), encoding="utf-8")
save_progress()
'''


DISCOVER_INPUTS = r'''
progress["stage"] = "discover_inputs"; save_progress()
plant_root = find_plantvillage_root(INPUT_ROOT)
holdout_root = find_holdout_root(INPUT_ROOT, CATEGORIES)
holdout_records, holdout_norm_ids = [], set()
for category in CATEGORIES:
    for p in sorted((holdout_root / category).glob("*.jpg")):
        original = extract_holdout_original(p.name)
        if original is None:
            raise ValueError(f"Bad holdout filename: {p.name}")
        norm = normalize_image_id(original)
        holdout_norm_ids.add(norm)
        holdout_records.append({"category": category, "holdout_filename": p.name, "original_filename": original, "normalized_original_id": norm, "path": str(p)})
holdout_df = pd.DataFrame(holdout_records)
if len(holdout_df) != 67:
    raise AssertionError(f"Expected 67 holdout images, found {len(holdout_df)}")
holdout_manifest_path = WORKING_DIR / "holdout_manifest.csv"
holdout_df.to_csv(holdout_manifest_path, index=False)

records, excluded = [], []
for class_dir in sorted([p for p in plant_root.iterdir() if p.is_dir()]):
    is_healthy = "healthy" in class_dir.name.lower()
    paths = []
    for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.png"]:
        paths.extend(class_dir.glob(ext))
    for p in sorted(paths):
        norm = normalize_image_id(p.name)
        rec = {"source_class": class_dir.name, "filename": p.name, "normalized_id": norm, "is_healthy": is_healthy, "path": str(p)}
        (excluded if norm in holdout_norm_ids else records).append(rec)
source_df = pd.DataFrame(records)
if set(source_df["normalized_id"]) & holdout_norm_ids:
    raise AssertionError("Holdout leaked into candidate source set")
progress.update({
    "plant_root": str(plant_root),
    "holdout_root": str(holdout_root),
    "holdout_count": int(len(holdout_df)),
    "excluded_from_plantvillage": int(len(excluded)),
    "source_images_after_holdout_exclusion": int(len(source_df)),
})
progress["artifacts"].append(str(holdout_manifest_path))
save_progress()
print({k: progress[k] for k in ["holdout_count", "excluded_from_plantvillage", "source_images_after_holdout_exclusion"]})
'''


V7_LABELS = r'''
progress["stage"] = "leaf_pseudo_label_qa"; save_progress()
all_rows, accepted_rows, rejection_rows = [], [], []
preview_counts = {"accepted": 0, "rejected": 0}
for idx, row in enumerate(source_df.itertuples(index=False), start=1):
    p = Path(row.path)
    rgb = read_rgb(p)
    mask, before_fill_area, after_fill_area = leaf_tissue_shape_mask(rgb)
    q = mask_quality(mask, before_fill_area, after_fill_area)
    lines = masks_to_yolo_lines(mask, class_id=0, min_area_ratio=0.003, max_contours=4) if q["accepted"] else []
    if q["accepted"] and not lines:
        q["accepted"] = False
        q["weak_reasons"] = (q["weak_reasons"] + ";empty_polygon").strip(";")
    rec = {
        "source_class": row.source_class,
        "filename": row.filename,
        "normalized_id": row.normalized_id,
        "is_healthy": bool(row.is_healthy),
        "source_path": str(p),
        "polygon_count": len(lines),
        **q,
    }
    all_rows.append(rec)
    if q["accepted"]:
        accepted_rows.append(rec)
        bucket = "accepted"
    else:
        rejection_rows.append(rec)
        bucket = "rejected"
    if preview_counts[bucket] < 120:
        out = PREVIEW_DIR / bucket / f"{idx:06d}__{row.source_class}__{p.stem}.jpg"
        color = (40, 200, 90) if bucket == "accepted" else (255, 160, 30)
        write_rgb(out, side_by_side([rgb, overlay_mask(rgb, mask, color)]))
        preview_counts[bucket] += 1
    if idx % 500 == 0:
        pd.DataFrame(all_rows).to_csv(QUANT_DIR / "leaf_pseudo_label_inventory_partial.csv", index=False)
        progress["leaf_qa_processed"] = idx
        save_progress()
        print({"leaf_qa_processed": idx, "accepted": len(accepted_rows), "rejected": len(rejection_rows)})

inventory_df = pd.DataFrame(all_rows)
accepted_df = pd.DataFrame(accepted_rows)
rejections_df = pd.DataFrame(rejection_rows)
inventory_path = QUANT_DIR / "leaf_pseudo_label_inventory.csv"
rejections_path = QUANT_DIR / "leaf_pseudo_label_rejections.csv"
qa_path = QUANT_DIR / "leaf_qa_summary.json"
inventory_df.to_csv(inventory_path, index=False)
rejections_df.to_csv(rejections_path, index=False)
accept_rate = float(len(accepted_df) / max(1, len(inventory_df)))
qa_summary = {
    "total_candidates": int(len(inventory_df)),
    "accepted": int(len(accepted_df)),
    "rejected": int(len(rejections_df)),
    "accept_rate": accept_rate,
    "rejection_reasons": {k: int(v) for k, v in Counter(";".join(rejections_df.get("weak_reasons", pd.Series(dtype=str)).dropna()).split(";")).items() if k},
    "min_accept_rate": MIN_ACCEPT_RATE,
    "min_accepted_images": MIN_ACCEPTED_IMAGES,
}
qa_path.write_text(json.dumps(qa_summary, indent=2), encoding="utf-8")
progress["leaf_qa_summary"] = qa_summary
progress["artifacts"].extend([str(inventory_path), str(rejections_path), str(qa_path), str(PREVIEW_DIR)])
if accept_rate < MIN_ACCEPT_RATE or len(accepted_df) < MIN_ACCEPTED_IMAGES:
    progress["status"] = "failed_qa"
    progress["stage"] = "leaf_qa_failed_no_training"
    progress["qa_failure_reason"] = "acceptance below configured threshold; training skipped"
    save_progress()
else:
    progress["stage"] = "leaf_qa_passed"
    save_progress()
print(qa_summary)
'''


V7_PREPARE_TRAIN = r'''
if progress.get("status") != "failed_qa":
    progress["stage"] = "prepare_leaf_yolo_dataset"; save_progress()
    accepted_df = split_stratified(accepted_df, "source_class", 0.70, 0.15)
    accepted_df["row_id"] = [f"leafv7_{i:06d}" for i in range(len(accepted_df))]
    if set(accepted_df["normalized_id"]) & holdout_norm_ids:
        raise AssertionError("Holdout leaked into v7 train/val/test")
    for split in ["train", "val", "test"]:
        (PREPARED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (PREPARED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    prep_rows = []
    for idx, row in enumerate(accepted_df.itertuples(index=False), start=1):
        p = Path(row.source_path)
        rgb = read_rgb(p)
        mask, _, _ = leaf_tissue_shape_mask(rgb)
        lines = masks_to_yolo_lines(mask, class_id=0, min_area_ratio=0.003, max_contours=4)
        out_name = f"{row.row_id}__{row.source_class}__{p.stem}.jpg"
        image_out = PREPARED_DIR / "images" / row.split / out_name
        label_out = PREPARED_DIR / "labels" / row.split / (Path(out_name).stem + ".txt")
        shutil.copy2(p, image_out)
        label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        prep_rows.append({**row._asdict(), "image_path": str(image_out), "label_path": str(label_out), "polygon_count": len(lines)})
        if idx % 1000 == 0:
            print({"prepared": idx, "total": len(accepted_df)})
    split_manifest = QUANT_DIR / "leaf_training_split_manifest.csv"
    pd.DataFrame(prep_rows).to_csv(split_manifest, index=False)
    data_yaml = PREPARED_DIR / "data.yaml"
    data_yaml.write_text(f"path: {PREPARED_DIR}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: leaf\n", encoding="utf-8")
    assert_only_class_zero(PREPARED_DIR / "labels")
    progress["split_counts"] = {k: int(v) for k, v in accepted_df["split"].value_counts().to_dict().items()}
    progress["artifacts"].extend([str(split_manifest), str(data_yaml)])
    save_progress()
    print(progress["split_counts"])
'''


V7_TRAIN = r'''
if progress.get("status") != "failed_qa":
    progress["stage"] = "train_leaf_yolo"; save_progress()
    model = YOLO("yolo11n-seg.pt")
    train_result = model.train(data=str(PREPARED_DIR / "data.yaml"), epochs=25, imgsz=640, batch=16, workers=2, seed=SEED, device=DEVICE, project=str(RUNS_DIR), name="yolo11n_leaf_v7", exist_ok=True, patience=8, verbose=False)
    train_dir = Path(getattr(train_result, "save_dir", RUNS_DIR / "yolo11n_leaf_v7"))
    best_leaf = WORKING_DIR / "best_leaf_model.pt"
    if (train_dir / "weights" / "best.pt").exists():
        shutil.copy2(train_dir / "weights" / "best.pt", best_leaf)
    trained = YOLO(str(best_leaf if best_leaf.exists() else train_dir / "weights" / "last.pt"))
    metric_outputs = {}
    for split in ["train", "test"]:
        m = trained.val(data=str(PREPARED_DIR / "data.yaml"), split=split, imgsz=640, batch=16, device=DEVICE, project=str(RUNS_DIR), name=f"eval_{split}", exist_ok=True, verbose=False)
        vals = metrics_from_results(m)
        (QUANT_DIR / f"leaf_{split}_pseudo_metrics.json").write_text(json.dumps(vals, indent=2), encoding="utf-8")
        pd.DataFrame([vals]).to_csv(QUANT_DIR / f"leaf_{split}_pseudo_metrics.csv", index=False)
        metric_outputs[split] = vals
    progress["train_test_pseudo_metrics"] = metric_outputs
    progress["artifacts"].extend([str(best_leaf), str(train_dir)])
    save_progress()
    print(metric_outputs)
'''


V7_HOLDOUT = r'''
progress["stage"] = "qualitative_leaf_holdout"; save_progress()
holdout_rows = []
trained = YOLO(str(WORKING_DIR / "best_leaf_model.pt")) if progress.get("status") != "failed_qa" and (WORKING_DIR / "best_leaf_model.pt").exists() else None
for rec in holdout_df.itertuples(index=False):
    p = Path(rec.path)
    rgb = read_rgb(p)
    pseudo_mask, before_fill_area, after_fill_area = leaf_tissue_shape_mask(rgb)
    q = mask_quality(pseudo_mask, before_fill_area, after_fill_area)
    if trained is not None:
        pred = trained.predict(source=rgb, imgsz=640, device=DEVICE, conf=0.25, verbose=False)[0]
        pred_mask = result_to_mask(pred, rgb.shape[:2])
    else:
        pred_mask = np.zeros(pseudo_mask.shape, dtype=np.uint8)
    panel = side_by_side([rgb, overlay_mask(rgb, pseudo_mask, (40, 200, 90)), overlay_mask(rgb, pred_mask, (60, 120, 255))])
    out_path = QUAL_HOLDOUT_DIR / "by_category" / rec.category / f"{Path(rec.holdout_filename).stem}_orig_pseudo_pred.jpg"
    write_rgb(out_path, panel)
    holdout_rows.append({"category": rec.category, "holdout_filename": rec.holdout_filename, "original_filename": rec.original_filename, **q, "pred_mask_area_ratio": float((pred_mask > 0).mean()), "pseudo_mask_iou_unseen_holdout": mask_iou(pseudo_mask, pred_mask), "qualitative_path": str(out_path)})
holdout_csv = WORKING_DIR / "holdout_leaf_qualitative_metrics.csv"
pd.DataFrame(holdout_rows).to_csv(holdout_csv, index=False)
progress["unseen_holdout_leaf_results"] = {"count": int(len(holdout_rows)), "mean_pseudo_iou_for_reference_only": float(pd.DataFrame(holdout_rows)["pseudo_mask_iou_unseen_holdout"].mean()), "note": "Holdout is unseen and qualitative-only; this IoU is reference only."}
progress["artifacts"].append(str(holdout_csv))
save_progress()
print(progress["unseen_holdout_leaf_results"])
'''


V7_FINALIZE = r'''
progress["stage"] = "finalize"; save_progress()
for folder, zip_name in [(PREPARED_DIR, "prepared_leaf_yolo_dataset.zip"), (PREVIEW_DIR, "leaf_pseudo_labels_preview.zip"), (QUAL_HOLDOUT_DIR, "qualitative_leaf_holdout.zip"), (QUANT_DIR, "quantitative_metrics.zip")]:
    if folder.exists():
        z = zip_dir(folder, WORKING_DIR / zip_name)
        progress["artifacts"].append(str(z))
if progress.get("status") != "failed_qa":
    progress["status"] = "complete"
for bulky in [PREPARED_DIR, PREVIEW_DIR, QUAL_HOLDOUT_DIR, RUNS_DIR]:
    if bulky.exists():
        shutil.rmtree(bulky)
progress["stage"] = "done"
progress["artifacts"] = sorted(set(progress["artifacts"]))
save_progress()
print(json.dumps({"status": progress["status"], "stage": progress["stage"], "artifact_count": len(progress["artifacts"])}, indent=2))
'''


V8_PATHS = r'''
INPUT_ROOT = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")
CATEGORIES = ["healthy_leaves","small_diseased_regions","large_diseased_regions","simple_backgrounds","complex_backgrounds","multiple_leaves","partial_occlusions"]
MIN_ACCEPTED_DISEASE_POSITIVES = 1000
MAX_HEALTHY_NEGATIVES_PER_CLASS = None

PREPARED_DIR = WORKING_DIR / "prepared_disease_yolo_dataset"
INPUTS_PREVIEW_DIR = WORKING_DIR / "disease_inputs_preview"
PSEUDO_PREVIEW_DIR = WORKING_DIR / "disease_pseudo_labels_preview"
QUAL_HOLDOUT_DIR = WORKING_DIR / "qualitative_disease_holdout"
QUAL_TEST_DIR = WORKING_DIR / "qualitative_disease_results"
QUANT_DIR = WORKING_DIR / "quantitative_metrics"
RUNS_DIR = WORKING_DIR / "runs_disease"
SUMMARY_PATH = WORKING_DIR / "run_summary.json"
for p in [PREPARED_DIR, INPUTS_PREVIEW_DIR, PSEUDO_PREVIEW_DIR, QUAL_HOLDOUT_DIR, QUAL_TEST_DIR, QUANT_DIR, RUNS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
for c in CATEGORIES:
    (QUAL_HOLDOUT_DIR / "by_category" / c).mkdir(parents=True, exist_ok=True)
progress = {
    "status": "running",
    "stage": "start",
    "task": "disease_region segmentation",
    "holdout_policy": "67 holdout images excluded from train/val/test; qualitative only",
    "leaf_source_policy": "prefer v7 best_leaf_model.pt; fallback to v7 heuristic; never use v5/v6 leaf",
    "artifacts": [],
}
def save_progress():
    SUMMARY_PATH.write_text(json.dumps(progress, indent=2), encoding="utf-8")
save_progress()
'''


V8_DISCOVER = r'''
progress["stage"] = "discover_inputs"; save_progress()
plant_root = find_plantvillage_root(INPUT_ROOT)
holdout_root = find_holdout_root(INPUT_ROOT, CATEGORIES)
leaf_candidates = [p for p in INPUT_ROOT.glob("**/best_leaf_model.pt") if "v7-leaf-heuristic-yolo-seg" in str(p)]
leaf_model_path = leaf_candidates[0] if leaf_candidates else None
leaf_model = YOLO(str(leaf_model_path)) if leaf_model_path else None
leaf_model_source = "v7_yolo_leaf" if leaf_model_path else "v7_heuristic_fallback"

holdout_records, holdout_norm_ids = [], set()
for category in CATEGORIES:
    for p in sorted((holdout_root / category).glob("*.jpg")):
        original = extract_holdout_original(p.name)
        if original is None:
            raise ValueError(f"Bad holdout filename: {p.name}")
        norm = normalize_image_id(original)
        holdout_norm_ids.add(norm)
        holdout_records.append({"category": category, "holdout_filename": p.name, "original_filename": original, "normalized_original_id": norm, "path": str(p)})
holdout_df = pd.DataFrame(holdout_records)
if len(holdout_df) != 67:
    raise AssertionError(f"Expected 67 holdout images, found {len(holdout_df)}")
holdout_manifest_path = WORKING_DIR / "holdout_manifest.csv"
holdout_df.to_csv(holdout_manifest_path, index=False)

records, excluded = [], []
for class_dir in sorted([p for p in plant_root.iterdir() if p.is_dir()]):
    is_healthy = "healthy" in class_dir.name.lower()
    paths = []
    for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.png"]:
        paths.extend(class_dir.glob(ext))
    for p in sorted(paths):
        norm = normalize_image_id(p.name)
        rec = {"source_class": class_dir.name, "filename": p.name, "normalized_id": norm, "is_healthy": is_healthy, "path": str(p)}
        (excluded if norm in holdout_norm_ids else records).append(rec)
source_df = pd.DataFrame(records)
if set(source_df["normalized_id"]) & holdout_norm_ids:
    raise AssertionError("Holdout leaked into disease candidate source set")
progress.update({
    "plant_root": str(plant_root),
    "holdout_root": str(holdout_root),
    "holdout_count": int(len(holdout_df)),
    "excluded_from_plantvillage": int(len(excluded)),
    "source_images_after_holdout_exclusion": int(len(source_df)),
    "leaf_mask_source": leaf_model_source,
    "leaf_model_path": str(leaf_model_path) if leaf_model_path else None,
})
progress["artifacts"].append(str(holdout_manifest_path))
save_progress()
print({k: progress[k] for k in ["holdout_count", "excluded_from_plantvillage", "leaf_mask_source", "leaf_model_path"]})
'''


V8_DISEASE_HELPERS = r'''
def leaf_mask_for_image(rgb):
    if leaf_model is not None:
        try:
            res = leaf_model.predict(source=rgb, imgsz=640, device=DEVICE, conf=0.25, verbose=False)[0]
            mask = result_to_mask(res, rgb.shape[:2])
            q = mask_quality(mask)
            if mask.sum() > 0 and q["accepted"]:
                return mask, "v7_yolo_leaf", ""
            fallback, _, _ = leaf_tissue_shape_mask(rgb)
            return fallback, "v7_heuristic_fallback", "v7 yolo mask failed QA"
        except Exception as exc:
            fallback, _, _ = leaf_tissue_shape_mask(rgb)
            return fallback, "v7_heuristic_fallback", repr(exc)
    fallback, _, _ = leaf_tissue_shape_mask(rgb)
    return fallback, "v7_heuristic_fallback", "no v7 leaf model mounted"

def isolate_leaf(rgb, leaf_mask):
    out = rgb.copy()
    out[leaf_mask == 0] = 0
    return out

def black_background_ratio(rgb, leaf_mask):
    outside = leaf_mask == 0
    if outside.sum() == 0:
        return 0.0
    return float(((rgb[:, :, 0] < 3) & (rgb[:, :, 1] < 3) & (rgb[:, :, 2] < 3) & outside).sum() / outside.sum())

def disease_mask_from_leaf(isolated_rgb, leaf_mask, is_healthy=False):
    if is_healthy:
        return np.zeros(leaf_mask.shape, dtype=np.uint8), ["healthy_negative"]
    hsv = cv2.cvtColor(isolated_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(isolated_rgb, cv2.COLOR_RGB2LAB)
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)
    inside = leaf_mask > 0
    if inside.sum() == 0:
        return np.zeros(leaf_mask.shape, dtype=np.uint8), ["empty_leaf_mask"]
    lab_inside = lab[inside].astype(np.float32)
    med = np.median(lab_inside, axis=0)
    dist = np.linalg.norm(lab.astype(np.float32) - med.reshape(1, 1, 3), axis=2)
    dist_thr = max(14.0, float(np.percentile(dist[inside], 90)))
    brown_necrotic = (((h <= 24) | (h >= 165)) & (s >= 28) & (v >= 18) & (v <= 205))
    yellow_chlorotic = ((h >= 16) & (h <= 48) & (s >= 24) & (v >= 70))
    dark_internal = ((l <= np.percentile(l[inside], 16)) & (s >= 18) & (v >= 18))
    lab_outlier = dist >= dist_thr
    mask = (inside & (brown_necrotic | yellow_chlorotic | dark_internal | lab_outlier)).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    clean = np.zeros(mask.shape, dtype=np.uint8)
    min_area = max(12, int(0.0008 * inside.sum()))
    for lab_id in range(1, n):
        if stats[lab_id, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == lab_id] = 255
    area_leaf = float((clean > 0).sum() / max(1, inside.sum()))
    comp_count = cv2.connectedComponents((clean > 0).astype(np.uint8), 8)[0] - 1
    edge = cv2.Canny((leaf_mask > 0).astype(np.uint8) * 255, 50, 150) > 0
    edge_touch = float(np.logical_and(clean > 0, cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0).sum() / max(1, (clean > 0).sum()))
    reasons = []
    severe = []
    if area_leaf == 0:
        reasons.append("empty_disease_mask_nonhealthy")
        severe.append("empty_disease_mask_nonhealthy")
    if 0 < area_leaf < 0.0015:
        reasons.append("tiny_disease_mask")
    if area_leaf > 0.45:
        reasons.append("large_disease_mask")
        severe.append("large_disease_mask")
    if comp_count > 25:
        reasons.append("many_components")
        severe.append("many_components")
    if edge_touch > 0.70 and area_leaf > 0.02:
        reasons.append("mostly_leaf_edge")
        severe.append("mostly_leaf_edge")
    return clean, reasons + [f"severe:{x}" for x in severe]
'''


V8_PREPARE = r'''
progress["stage"] = "disease_pseudo_label_qa"; save_progress()
rows, accepted_rows, rejected_rows = [], [], []
preview_counts = Counter()
for idx, row in enumerate(source_df.itertuples(index=False), start=1):
    p = Path(row.path)
    rgb = read_rgb(p)
    leaf_mask, leaf_source, leaf_error = leaf_mask_for_image(rgb)
    isolated = isolate_leaf(rgb, leaf_mask)
    dmask, weak = disease_mask_from_leaf(isolated, leaf_mask, bool(row.is_healthy))
    severe = [x for x in weak if x.startswith("severe:")]
    lines = [] if bool(row.is_healthy) else masks_to_yolo_lines(dmask, class_id=0, min_area_ratio=0.0006, max_contours=20)
    accepted = bool(row.is_healthy) or (len(lines) > 0 and not severe)
    rec = {
        "source_class": row.source_class,
        "filename": row.filename,
        "normalized_id": row.normalized_id,
        "is_healthy": bool(row.is_healthy),
        "source_path": str(p),
        "leaf_mask_source": leaf_source,
        "leaf_error": leaf_error,
        "polygon_count": len(lines),
        "disease_area_ratio_image": float((dmask > 0).mean()),
        "disease_area_ratio_leaf": float((dmask > 0).sum() / max(1, (leaf_mask > 0).sum())),
        "black_background_ratio": black_background_ratio(isolated, leaf_mask),
        "weak_reasons": ";".join(weak),
        "accepted": accepted,
    }
    rows.append(rec)
    if accepted:
        accepted_rows.append(rec)
        bucket = "accepted"
    else:
        rejected_rows.append(rec)
        bucket = "rejected"
    if preview_counts[bucket] < 90:
        write_rgb(INPUTS_PREVIEW_DIR / bucket / f"{idx:06d}__{row.source_class}__isolated.jpg", isolated)
        write_rgb(PSEUDO_PREVIEW_DIR / bucket / f"{idx:06d}__{row.source_class}__disease_overlay.jpg", overlay_mask(isolated, dmask, (255, 180, 40)))
        preview_counts[bucket] += 1
    if idx % 500 == 0:
        pd.DataFrame(rows).to_csv(QUANT_DIR / "disease_pseudo_label_inventory_partial.csv", index=False)
        progress["disease_qa_processed"] = idx
        save_progress()
        print({"disease_qa_processed": idx, "accepted": len(accepted_rows), "rejected": len(rejected_rows)})

inventory_df = pd.DataFrame(rows)
accepted_df = pd.DataFrame(accepted_rows)
rejections_df = pd.DataFrame(rejected_rows)
inventory_path = QUANT_DIR / "disease_pseudo_label_inventory.csv"
rejections_path = QUANT_DIR / "disease_pseudo_label_rejections.csv"
qa_path = QUANT_DIR / "disease_qa_summary.json"
inventory_df.to_csv(inventory_path, index=False)
rejections_df.to_csv(rejections_path, index=False)
positive_count = int(((accepted_df.get("is_healthy", pd.Series(dtype=bool)) == False) & (accepted_df.get("polygon_count", pd.Series(dtype=int)) > 0)).sum()) if len(accepted_df) else 0
qa_summary = {
    "total_candidates": int(len(inventory_df)),
    "accepted": int(len(accepted_df)),
    "rejected": int(len(rejections_df)),
    "positive_disease_labels": positive_count,
    "healthy_negative_labels": int((accepted_df.get("is_healthy", pd.Series(dtype=bool)) == True).sum()) if len(accepted_df) else 0,
    "min_accepted_disease_positives": MIN_ACCEPTED_DISEASE_POSITIVES,
    "leaf_mask_source_summary": {k: int(v) for k, v in inventory_df["leaf_mask_source"].value_counts().to_dict().items()},
    "rejection_reasons": {k: int(v) for k, v in Counter(";".join(rejections_df.get("weak_reasons", pd.Series(dtype=str)).dropna()).split(";")).items() if k},
}
qa_path.write_text(json.dumps(qa_summary, indent=2), encoding="utf-8")
progress["disease_qa_summary"] = qa_summary
progress["artifacts"].extend([str(inventory_path), str(rejections_path), str(qa_path), str(INPUTS_PREVIEW_DIR), str(PSEUDO_PREVIEW_DIR)])
if positive_count < MIN_ACCEPTED_DISEASE_POSITIVES:
    progress["status"] = "failed_qa"
    progress["stage"] = "disease_qa_failed_no_training"
    progress["qa_failure_reason"] = "not enough accepted disease positives; training skipped"
else:
    progress["stage"] = "disease_qa_passed"
save_progress()
print(qa_summary)
'''


V8_PREPARE_DATASET = r'''
if progress.get("status") != "failed_qa":
    progress["stage"] = "prepare_disease_yolo_dataset"; save_progress()
    accepted_df = split_stratified(accepted_df, "source_class", 0.70, 0.15)
    accepted_df["row_id"] = [f"diseasev8_{i:06d}" for i in range(len(accepted_df))]
    if set(accepted_df["normalized_id"]) & holdout_norm_ids:
        raise AssertionError("Holdout leaked into v8 train/val/test")
    for split in ["train", "val", "test"]:
        (PREPARED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (PREPARED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    prep_rows = []
    for idx, row in enumerate(accepted_df.itertuples(index=False), start=1):
        p = Path(row.source_path)
        rgb = read_rgb(p)
        leaf_mask, leaf_source, leaf_error = leaf_mask_for_image(rgb)
        isolated = isolate_leaf(rgb, leaf_mask)
        dmask, weak = disease_mask_from_leaf(isolated, leaf_mask, bool(row.is_healthy))
        lines = [] if bool(row.is_healthy) else masks_to_yolo_lines(dmask, class_id=0, min_area_ratio=0.0006, max_contours=20)
        out_name = f"{row.row_id}__{row.source_class}__{p.stem}.jpg"
        image_out = PREPARED_DIR / "images" / row.split / out_name
        label_out = PREPARED_DIR / "labels" / row.split / (Path(out_name).stem + ".txt")
        write_rgb(image_out, isolated)
        label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        prep_rows.append({**row._asdict(), "image_path": str(image_out), "label_path": str(label_out), "leaf_mask_source_runtime": leaf_source, "black_background_ratio_runtime": black_background_ratio(isolated, leaf_mask), "polygon_count_runtime": len(lines)})
        if idx % 1000 == 0:
            print({"prepared": idx, "total": len(accepted_df)})
    split_manifest = QUANT_DIR / "disease_training_split_manifest.csv"
    pd.DataFrame(prep_rows).to_csv(split_manifest, index=False)
    data_yaml = PREPARED_DIR / "data.yaml"
    data_yaml.write_text(f"path: {PREPARED_DIR}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: disease_region\n", encoding="utf-8")
    assert_only_class_zero(PREPARED_DIR / "labels")
    progress["split_counts"] = {k: int(v) for k, v in accepted_df["split"].value_counts().to_dict().items()}
    progress["artifacts"].extend([str(split_manifest), str(data_yaml)])
    save_progress()
    print(progress["split_counts"])
'''


V8_TRAIN = r'''
if progress.get("status") != "failed_qa":
    progress["stage"] = "train_disease_yolo"; save_progress()
    model = YOLO("yolo11n-seg.pt")
    train_result = model.train(data=str(PREPARED_DIR / "data.yaml"), epochs=25, imgsz=640, batch=16, workers=2, seed=SEED, device=DEVICE, project=str(RUNS_DIR), name="yolo11n_disease_v8", exist_ok=True, patience=8, verbose=False)
    train_dir = Path(getattr(train_result, "save_dir", RUNS_DIR / "yolo11n_disease_v8"))
    best_disease = WORKING_DIR / "best_disease_model.pt"
    if (train_dir / "weights" / "best.pt").exists():
        shutil.copy2(train_dir / "weights" / "best.pt", best_disease)
    trained = YOLO(str(best_disease if best_disease.exists() else train_dir / "weights" / "last.pt"))
    metric_outputs = {}
    for split in ["train", "test"]:
        m = trained.val(data=str(PREPARED_DIR / "data.yaml"), split=split, imgsz=640, batch=16, device=DEVICE, project=str(RUNS_DIR), name=f"eval_{split}", exist_ok=True, verbose=False)
        vals = metrics_from_results(m)
        (QUANT_DIR / f"disease_{split}_pseudo_metrics.json").write_text(json.dumps(vals, indent=2), encoding="utf-8")
        pd.DataFrame([vals]).to_csv(QUANT_DIR / f"disease_{split}_pseudo_metrics.csv", index=False)
        metric_outputs[split] = vals
    progress["train_test_pseudo_metrics"] = metric_outputs
    progress["artifacts"].extend([str(best_disease), str(train_dir)])
    save_progress()
    print(metric_outputs)
'''


V8_HOLDOUT = r'''
progress["stage"] = "qualitative_disease_holdout"; save_progress()
trained = YOLO(str(WORKING_DIR / "best_disease_model.pt")) if progress.get("status") != "failed_qa" and (WORKING_DIR / "best_disease_model.pt").exists() else None
holdout_rows = []
for rec in holdout_df.itertuples(index=False):
    p = Path(rec.path)
    rgb = read_rgb(p)
    leaf_mask, leaf_source, leaf_error = leaf_mask_for_image(rgb)
    isolated = isolate_leaf(rgb, leaf_mask)
    is_holdout_healthy = (rec.category == "healthy_leaves") or ("healthy" in rec.holdout_filename.lower())
    pseudo, weak = disease_mask_from_leaf(isolated, leaf_mask, is_holdout_healthy)
    if trained is not None:
        pred = trained.predict(source=isolated, imgsz=640, device=DEVICE, conf=0.25, verbose=False)[0]
        pred_mask = result_to_mask(pred, isolated.shape[:2])
    else:
        pred_mask = np.zeros(pseudo.shape, dtype=np.uint8)
    panel = side_by_side([rgb, isolated, overlay_mask(isolated, pseudo, (255, 180, 40)), overlay_mask(isolated, pred_mask, (80, 140, 255))])
    out_path = QUAL_HOLDOUT_DIR / "by_category" / rec.category / f"{Path(rec.holdout_filename).stem}_orig_isolated_pseudo_pred.jpg"
    write_rgb(out_path, panel)
    holdout_rows.append({"category": rec.category, "holdout_filename": rec.holdout_filename, "original_filename": rec.original_filename, "leaf_mask_source": leaf_source, "leaf_error": leaf_error, "pseudo_disease_area_ratio": float((pseudo > 0).mean()), "pred_disease_area_ratio": float((pred_mask > 0).mean()), "pseudo_disease_iou_unseen_holdout": mask_iou(pseudo, pred_mask), "weak_reasons": ";".join(weak), "qualitative_path": str(out_path)})
holdout_csv = WORKING_DIR / "holdout_disease_qualitative_metrics.csv"
pd.DataFrame(holdout_rows).to_csv(holdout_csv, index=False)
progress["unseen_holdout_disease_results"] = {"count": int(len(holdout_rows)), "mean_pseudo_iou_for_reference_only": float(pd.DataFrame(holdout_rows)["pseudo_disease_iou_unseen_holdout"].mean()), "note": "Holdout is unseen and qualitative-only; this IoU is reference only."}
progress["artifacts"].append(str(holdout_csv))
save_progress()
print(progress["unseen_holdout_disease_results"])
'''


V8_TEST_QUAL = r'''
if progress.get("status") != "failed_qa":
    progress["stage"] = "qualitative_disease_test"; save_progress()
    trained = YOLO(str(WORKING_DIR / "best_disease_model.pt"))
    sample = accepted_df[accepted_df["split"] == "test"].sample(n=min(100, int((accepted_df["split"] == "test").sum())), random_state=SEED)
    for row in sample.itertuples(index=False):
        rgb = read_rgb(row.source_path)
        leaf_mask, _, _ = leaf_mask_for_image(rgb)
        isolated = isolate_leaf(rgb, leaf_mask)
        pseudo = label_file_to_mask(PREPARED_DIR / "labels" / "test" / (f"{row.row_id}__{row.source_class}__{Path(row.source_path).stem}.txt"), isolated.shape[:2])
        pred = trained.predict(source=isolated, imgsz=640, device=DEVICE, conf=0.25, verbose=False)[0]
        pred_mask = result_to_mask(pred, isolated.shape[:2])
        panel = side_by_side([isolated, overlay_mask(isolated, pseudo, (255, 180, 40)), overlay_mask(isolated, pred_mask, (80, 140, 255))])
        write_rgb(QUAL_TEST_DIR / f"{row.row_id}_pseudo_pred.jpg", panel)
    progress["artifacts"].append(str(QUAL_TEST_DIR))
    save_progress()
'''


V8_FINALIZE = r'''
progress["stage"] = "finalize"; save_progress()
for folder, zip_name in [
    (PREPARED_DIR, "prepared_disease_yolo_dataset.zip"),
    (INPUTS_PREVIEW_DIR, "disease_inputs_preview.zip"),
    (PSEUDO_PREVIEW_DIR, "disease_pseudo_labels_preview.zip"),
    (QUAL_HOLDOUT_DIR, "qualitative_disease_holdout.zip"),
    (QUAL_TEST_DIR, "qualitative_disease_results.zip"),
    (QUANT_DIR, "quantitative_metrics.zip"),
]:
    if folder.exists():
        z = zip_dir(folder, WORKING_DIR / zip_name)
        progress["artifacts"].append(str(z))
if progress.get("status") != "failed_qa":
    progress["status"] = "complete"
for bulky in [PREPARED_DIR, INPUTS_PREVIEW_DIR, PSEUDO_PREVIEW_DIR, QUAL_HOLDOUT_DIR, QUAL_TEST_DIR, RUNS_DIR]:
    if bulky.exists():
        shutil.rmtree(bulky)
progress["stage"] = "done"
progress["artifacts"] = sorted(set(progress["artifacts"]))
save_progress()
print(json.dumps({"status": progress["status"], "stage": progress["stage"], "artifact_count": len(progress["artifacts"])}, indent=2))
'''


v7_cells = [
    md("intro", "# PC5 v7 - Leaf YOLO-Seg With Clean Heuristic Pseudo-Labels\nTrain one class, `leaf`, using a leaf tissue + shape heuristic. MobileSAM is not used as pseudo-label truth. Holdout is excluded from train/val/test and used only for unseen qualitative review."),
    md("setup-md", "## Setup\nInstall/check Ultralytics and configure deterministic GPU execution."),
    code("setup", COMMON_SETUP),
    md("paths-md", "## Paths And Outputs\nCreate Kaggle working folders, QA gates, and run summary."),
    code("paths", V7_PATHS),
    md("helpers-md", "## Shared Helpers\nDataset discovery, holdout exclusion, leaf mask heuristic, polygon conversion, overlays, metrics, and zipping."),
    code("helpers", COMMON_HELPERS),
    md("discover-md", "## Discover Inputs And Exclude Holdout\nBuild holdout manifest and remove matching original PlantVillage files before any split."),
    code("discover", DISCOVER_INPUTS),
    md("qa-md", "## Leaf Pseudo-Label QA\nGenerate full-leaf masks using leaf tissue + shape, save accepted/rejected previews, and gate training."),
    code("qa", V7_LABELS),
    md("prepare-md", "## Prepare YOLO Dataset\nUse accepted masks only, split 70/15/15, and write YOLO-seg labels."),
    code("prepare", V7_PREPARE_TRAIN),
    md("train-md", "## Train And Quantitative Pseudo Metrics\nTrain YOLO leaf and save separate train/test pseudo metrics."),
    code("train", V7_TRAIN),
    md("holdout-md", "## Unseen Qualitative Holdout\nSave original, heuristic pseudo-mask, and YOLO prediction for all 67 holdout images."),
    code("holdout", V7_HOLDOUT),
    md("final-md", "## Finalize\nZip deliverables and write final run summary."),
    code("finalize", V7_FINALIZE),
]


v8_cells = [
    md("intro", "# PC5 v8 - Disease Region YOLO-Seg From Clean Isolated Leaves\nTrain one class, `disease_region`, using leaves isolated by v7 leaf model or v7 heuristic fallback. Holdout is excluded from train/val/test and used only for unseen qualitative review."),
    md("setup-md", "## Setup\nInstall/check Ultralytics and configure deterministic GPU execution."),
    code("setup", COMMON_SETUP),
    md("paths-md", "## Paths And Outputs\nCreate Kaggle working folders, QA gates, and run summary."),
    code("paths", V8_PATHS),
    md("helpers-md", "## Shared Helpers\nDataset discovery, holdout exclusion, v7 leaf mask heuristic, polygon conversion, overlays, metrics, and zipping."),
    code("helpers", COMMON_HELPERS),
    md("discover-md", "## Discover Inputs And v7 Leaf Source\nBuild holdout manifest, exclude holdout, and mount v7 leaf model if available."),
    code("discover", V8_DISCOVER),
    md("disease-helpers-md", "## Disease Pseudo-Label Helpers\nIsolate leaves with black background and detect disease candidates only inside the leaf mask."),
    code("disease-helpers", V8_DISEASE_HELPERS),
    md("qa-md", "## Disease Pseudo-Label QA\nGenerate disease masks, keep healthy negatives, reject weak positive labels, and gate training."),
    code("qa", V8_PREPARE),
    md("prepare-md", "## Prepare Disease YOLO Dataset\nUse accepted samples only and write isolated black-background inputs plus YOLO-seg labels."),
    code("prepare", V8_PREPARE_DATASET),
    md("train-md", "## Train And Quantitative Pseudo Metrics\nTrain YOLO disease and save separate train/test pseudo metrics."),
    code("train", V8_TRAIN),
    md("holdout-md", "## Unseen Qualitative Disease Holdout\nSave original, isolated leaf, pseudo disease mask, and YOLO prediction for all 67 holdout images."),
    code("holdout", V8_HOLDOUT),
    md("qual-test-md", "## Qualitative Test Results\nSave prediction panels on a quantitative test sample."),
    code("qual-test", V8_TEST_QUAL),
    md("final-md", "## Finalize\nZip deliverables and write final run summary."),
    code("finalize", V8_FINALIZE),
]


common_meta = {
    "kernel_type": "notebook",
    "code_file": "main.ipynb",
    "language": "python",
    "is_private": True,
    "enable_gpu": True,
    "machine_shape": "NvidiaTeslaT4",
    "enable_internet": True,
    "dataset_sources": [
        "emmarex/plantdisease",
        "jeffreyamc/cv-pc5-v3-plantvillage-segmentation-holdout",
    ],
}

v7_meta = {
    **common_meta,
    "id": "jeffreyamc/cv-pc5-v7-leaf-heuristic-yolo-seg",
    "title": "cv-pc5-v7-leaf-heuristic-yolo-seg",
}
v8_meta = {
    **common_meta,
    "id": "jeffreyamc/cv-pc5-v8-disease-heuristic-yolo-seg",
    "title": "cv-pc5-v8-disease-heuristic-yolo-seg",
    "kernel_sources": ["jeffreyamc/cv-pc5-v7-leaf-heuristic-yolo-seg"],
}


write_kernel("v7", v7_cells, v7_meta)
write_kernel("v8", v8_cells, v8_meta)
print("generated v7/v8 notebooks")
