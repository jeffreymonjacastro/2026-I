import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V7_NOTEBOOK = ROOT / "kaggle" / "v7" / "input" / "main.ipynb"


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
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def notebook_cell_source(path, cell_id):
    data = json.loads(path.read_text(encoding="utf-8"))
    for cell in data["cells"]:
        if cell.get("id") == cell_id:
            return "".join(cell.get("source", []))
    raise KeyError(f"Cell id not found: {cell_id}")


def write_kernel(version, cells, metadata):
    out = ROOT / "kaggle" / version / "input"
    out.mkdir(parents=True, exist_ok=True)
    (out / "main.ipynb").write_text(json.dumps(nb(cells), indent=1) + "\n", encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


SETUP = notebook_cell_source(V7_NOTEBOOK, "setup")
HELPERS = notebook_cell_source(V7_NOTEBOOK, "helpers")
DISCOVER = notebook_cell_source(V7_NOTEBOOK, "discover")


PATHS = r'''
INPUT_ROOT = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")
CATEGORIES = ["healthy_leaves","small_diseased_regions","large_diseased_regions","simple_backgrounds","complex_backgrounds","multiple_leaves","partial_occlusions"]
MIN_ACCEPTED_DISEASE_POSITIVES = 500
MIN_HEALTHY_NEGATIVES = 1000

PREPARED_DIR = WORKING_DIR / "prepared_multiclass_yolo_dataset"
PREVIEW_DIR = WORKING_DIR / "multiclass_pseudo_labels_preview"
REAL_HOLDOUT_DIR = WORKING_DIR / "real_holdout_predictions"
HOLDOUT_DIAG_DIR = WORKING_DIR / "holdout_diagnostic_reference"
QUAL_TEST_DIR = WORKING_DIR / "qualitative_multiclass_results"
QUANT_DIR = WORKING_DIR / "quantitative_metrics"
RUNS_DIR = WORKING_DIR / "runs_multiclass"
SUMMARY_PATH = WORKING_DIR / "run_summary.json"

for p in [
    PREPARED_DIR,
    PREVIEW_DIR / "accepted",
    PREVIEW_DIR / "rejected",
    PREVIEW_DIR / "healthy_leaf_only",
    REAL_HOLDOUT_DIR,
    HOLDOUT_DIAG_DIR,
    QUAL_TEST_DIR,
    QUANT_DIR,
    RUNS_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)
for c in CATEGORIES:
    (REAL_HOLDOUT_DIR / "by_category" / c).mkdir(parents=True, exist_ok=True)
    (HOLDOUT_DIAG_DIR / "by_category" / c).mkdir(parents=True, exist_ok=True)

progress = {
    "status": "running",
    "stage": "start",
    "task": "multiclass leaf + disease_region segmentation",
    "pseudo_label_source": "leaf heuristic v7 + conservative disease heuristic; no YOLO/SAM pseudo-label source",
    "holdout_policy": "67 holdout images excluded from train/val/test; real qualitative predictions only",
    "classes": {"0": "leaf", "1": "disease_region"},
    "qa_gate": {
        "min_accepted_disease_positives": MIN_ACCEPTED_DISEASE_POSITIVES,
        "min_healthy_negatives": MIN_HEALTHY_NEGATIVES,
    },
    "artifacts": [],
}

def save_progress():
    SUMMARY_PATH.write_text(json.dumps(progress, indent=2), encoding="utf-8")

save_progress()
'''


MULTICLASS_HELPERS = r'''
FOCAL_DISEASE_HINTS = [
    "bacterial_spot",
    "early_blight",
    "late_blight",
    "septoria",
    "target_spot",
    "leaf_mold",
    "spider_mites",
]
DIFFUSE_DISEASE_HINTS = ["yellowleaf", "yellow_leaf", "curl_virus", "mosaic"]

def normalized_class_name(name):
    return normalize_image_id(str(name))

def disease_policy(source_class):
    n = normalized_class_name(source_class)
    if any(x in n for x in DIFFUSE_DISEASE_HINTS):
        return "exclude_diffuse"
    if any(x in n for x in FOCAL_DISEASE_HINTS):
        return "focal_positive"
    return "unknown_diseased_exclude"

def disease_area_max_for_class(source_class):
    n = normalized_class_name(source_class)
    if "late_blight" in n or "leaf_mold" in n:
        return 0.35
    return 0.20

def bbox_from_mask(mask, pad_ratio=0.10):
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, w - 1, h - 1)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    pad = int(max(x2 - x1 + 1, y2 - y1 + 1) * pad_ratio)
    return (max(0, x1 - pad), max(0, y1 - pad), min(w - 1, x2 + pad), min(h - 1, y2 + pad))

def crop_rgb_mask(rgb, mask, bbox):
    x1, y1, x2, y2 = bbox
    return rgb[y1:y2 + 1, x1:x2 + 1].copy(), mask[y1:y2 + 1, x1:x2 + 1].copy()

def paste_crop_mask(crop_mask, full_shape, bbox):
    out = np.zeros(full_shape, dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    out[y1:y2 + 1, x1:x2 + 1] = crop_mask[: y2 - y1 + 1, : x2 - x1 + 1]
    return out

def disease_mask_conservative(crop_rgb, crop_leaf_mask, source_class):
    inside = crop_leaf_mask > 0
    leaf_area = int(inside.sum())
    if leaf_area == 0:
        return np.zeros(crop_leaf_mask.shape, dtype=np.uint8), {
            "accepted": False,
            "weak_reasons": "empty_leaf_mask",
            "disease_area_ratio_leaf": 0.0,
            "component_count": 0,
            "class_area_max": disease_area_max_for_class(source_class),
        }

    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)
    lab_inside = lab[inside].astype(np.float32)
    med = np.median(lab_inside, axis=0)
    dist = np.linalg.norm(lab.astype(np.float32) - med.reshape(1, 1, 3), axis=2)
    dist_inside = dist[inside]
    local_l = cv2.GaussianBlur(l.astype(np.float32), (0, 0), 5)
    contrast = np.abs(l.astype(np.float32) - local_l)
    contrast_thr = max(7.0, float(np.percentile(contrast[inside], 92)))

    brown_necrotic = (((h <= 26) | (h >= 165)) & (s >= 28) & (v >= 18) & (v <= 215))
    dark_internal = ((l <= np.percentile(l[inside], 14)) & (s >= 14) & (v >= 14))
    lab_outlier_strong = (dist >= max(18.0, float(np.percentile(dist_inside, 94)))) & (s >= 12)
    contrast_seed = (contrast >= contrast_thr) & (s >= 18) & ((v <= np.percentile(v[inside], 55)) | brown_necrotic)
    seeds = inside & (brown_necrotic | dark_internal | lab_outlier_strong | contrast_seed)

    yellow = ((h >= 16) & (h <= 48) & (s >= 24) & (v >= 70))
    lab_outlier_moderate = dist >= max(10.0, float(np.percentile(dist_inside, 80)))
    seed_neighborhood = cv2.dilate(seeds.astype(np.uint8) * 255, np.ones((7, 7), np.uint8), iterations=2) > 0
    candidate = inside & (seeds | (yellow & seed_neighborhood & lab_outlier_moderate))

    k3 = np.ones((3, 3), np.uint8)
    candidate_u8 = (candidate.astype(np.uint8) * 255)
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, k3, iterations=1)
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, k3, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((candidate_u8 > 0).astype(np.uint8), 8)
    clean = np.zeros(candidate_u8.shape, dtype=np.uint8)
    min_area = max(10, int(0.0005 * leaf_area))
    max_area = disease_area_max_for_class(source_class) * leaf_area
    leaf_edge = cv2.Canny((crop_leaf_mask > 0).astype(np.uint8) * 255, 50, 150) > 0
    leaf_edge_dil = cv2.dilate(leaf_edge.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    component_rejections = Counter()

    for lab_id in range(1, n):
        comp = labels == lab_id
        area = int(comp.sum())
        if area < min_area:
            component_rejections["component_too_small"] += 1
            continue
        if area > max_area:
            component_rejections["component_too_large"] += 1
            continue
        seed_ratio = float(seeds[comp].mean()) if area else 0.0
        if seed_ratio < 0.04:
            component_rejections["low_necrotic_seed_support"] += 1
            continue
        edge_touch = float(np.logical_and(comp, leaf_edge_dil).sum() / max(1, area))
        if edge_touch > 0.82 and area / leaf_area > 0.02 and seed_ratio < 0.16:
            component_rejections["mostly_leaf_edge"] += 1
            continue
        clean[comp] = 255

    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, k3, iterations=1)
    area_leaf = float((clean > 0).sum() / max(1, leaf_area))
    comp_count = cv2.connectedComponents((clean > 0).astype(np.uint8), 8)[0] - 1
    class_max = disease_area_max_for_class(source_class)
    reasons = []
    if area_leaf == 0:
        reasons.append("empty_disease_mask")
    if 0 < area_leaf < 0.001:
        reasons.append("tiny_disease_mask")
    if area_leaf > class_max:
        reasons.append("large_disease_mask")
    if comp_count > 50:
        reasons.append("many_components")
    if component_rejections:
        reasons.extend([f"component_reject_{k}:{v}" for k, v in sorted(component_rejections.items())])

    severe = [r for r in reasons if r in {"empty_disease_mask", "tiny_disease_mask", "large_disease_mask", "many_components"}]
    return clean, {
        "accepted": len(severe) == 0,
        "weak_reasons": ";".join(reasons),
        "disease_area_ratio_leaf": area_leaf,
        "component_count": int(comp_count),
        "class_area_max": class_max,
    }

def label_lines_to_masks(lines, shape):
    h, w = shape
    masks = {0: np.zeros((h, w), dtype=np.uint8), 1: np.zeros((h, w), dtype=np.uint8)}
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls = int(parts[0])
        vals = [float(v) for v in parts[1:]]
        pts = [[int(round(x * w)), int(round(y * h))] for x, y in zip(vals[0::2], vals[1::2])]
        if cls in masks and len(pts) >= 3:
            cv2.fillPoly(masks[cls], [np.array(pts, dtype=np.int32)], 255)
    return masks

def assert_multiclass_labels(label_root, manifest=None):
    bad = []
    for p in Path(label_root).glob("**/*.txt"):
        text = p.read_text(encoding="utf-8").splitlines()
        classes = []
        for line in text:
            if not line.strip():
                continue
            cls = line.split()[0]
            if cls not in {"0", "1"}:
                bad.append((str(p), "bad_class", cls))
            classes.append(cls)
        if "0" not in classes:
            bad.append((str(p), "missing_leaf_class", ""))
    if manifest is not None and len(manifest):
        for row in manifest.itertuples(index=False):
            if bool(row.is_healthy) and int(row.disease_polygon_count) != 0:
                bad.append((str(row.label_path), "healthy_has_disease_label", ""))
            if (not bool(row.is_healthy)) and str(row.sample_type) == "leaf_disease" and int(row.disease_polygon_count) == 0:
                bad.append((str(row.label_path), "positive_missing_disease_label", ""))
    if bad:
        raise AssertionError(f"Multiclass label validation failed: {bad[:5]}")

def class_result_masks(result, shape):
    h, w = shape
    out = {0: np.zeros((h, w), dtype=np.uint8), 1: np.zeros((h, w), dtype=np.uint8)}
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None:
        return out
    data = getattr(masks, "data", None)
    if data is None or len(data) == 0:
        return out
    if boxes is not None and getattr(boxes, "cls", None) is not None:
        cls_vals = boxes.cls.detach().cpu().numpy().astype(int).tolist()
    else:
        cls_vals = [0] * len(data)
    for arr, cls_id in zip(data.detach().cpu().numpy(), cls_vals):
        if cls_id not in out:
            continue
        m = (arr > 0.5).astype(np.uint8) * 255
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        out[cls_id] = cv2.bitwise_or(out[cls_id], m)
    return out

def overlay_multiclass(rgb, leaf_mask, disease_mask, alpha=0.45):
    out = rgb.copy()
    if leaf_mask is not None:
        out = overlay_mask(out, leaf_mask, (40, 200, 90), alpha=alpha)
    if disease_mask is not None:
        out = overlay_mask(out, disease_mask, (80, 140, 255), alpha=0.55)
    return out

def metrics_from_results_multiclass(metrics):
    values = metrics_from_results(metrics)
    names = getattr(metrics, "names", None)
    if names is not None:
        values["names"] = {str(k): str(v) for k, v in dict(names).items()}
    for attr in ["box", "seg"]:
        obj = getattr(metrics, attr, None)
        maps = getattr(obj, "maps", None) if obj is not None else None
        if maps is not None:
            arr = np.array(maps).astype(float).ravel().tolist()
            for i, val in enumerate(arr):
                key = str(names.get(i, i)) if names is not None and hasattr(names, "get") else str(i)
                values[f"pseudo_{attr}_map_class_{key}"] = float(val)
    return values
'''


QA_CELL = r'''
progress["stage"] = "multiclass_pseudo_label_qa"; save_progress()
rows, accepted_rows, rejected_rows = [], [], []
preview_counts = Counter()

for idx, row in enumerate(source_df.itertuples(index=False), start=1):
    p = Path(row.path)
    rgb = read_rgb(p)
    leaf_mask, before_fill_area, after_fill_area = leaf_tissue_shape_mask(rgb)
    leaf_q = mask_quality(leaf_mask, before_fill_area, after_fill_area)
    bbox = bbox_from_mask(leaf_mask, pad_ratio=0.10)
    crop_rgb, crop_leaf = crop_rgb_mask(rgb, leaf_mask, bbox)
    leaf_lines = masks_to_yolo_lines(crop_leaf, class_id=0, min_area_ratio=0.003, max_contours=4) if leaf_q["accepted"] else []
    is_healthy = bool(row.is_healthy)
    policy = "healthy_leaf_only" if is_healthy else disease_policy(row.source_class)
    disease_mask = np.zeros(crop_leaf.shape, dtype=np.uint8)
    disease_stats = {
        "accepted": False,
        "weak_reasons": "",
        "disease_area_ratio_leaf": 0.0,
        "component_count": 0,
        "class_area_max": disease_area_max_for_class(row.source_class),
    }
    disease_lines = []
    reasons = []
    accepted = False
    sample_type = "rejected"

    if not leaf_q["accepted"] or not leaf_lines:
        reasons.append(("leaf_mask_rejected;" + leaf_q.get("weak_reasons", "")).strip(";"))
    elif is_healthy:
        accepted = True
        sample_type = "healthy_leaf_only"
    elif policy == "focal_positive":
        disease_mask, disease_stats = disease_mask_conservative(crop_rgb, crop_leaf, row.source_class)
        disease_lines = masks_to_yolo_lines(disease_mask, class_id=1, min_area_ratio=0.0004, max_contours=50) if disease_stats["accepted"] else []
        if disease_stats["accepted"] and disease_lines:
            accepted = True
            sample_type = "leaf_disease"
        else:
            reasons.append(("disease_rejected;" + disease_stats.get("weak_reasons", "")).strip(";"))
    else:
        reasons.append(policy)

    rec = {
        "source_class": row.source_class,
        "filename": row.filename,
        "normalized_id": row.normalized_id,
        "is_healthy": is_healthy,
        "source_path": str(p),
        "policy": policy,
        "sample_type": sample_type,
        "accepted": accepted,
        "bbox": ",".join(str(int(v)) for v in bbox),
        "crop_h": int(crop_rgb.shape[0]),
        "crop_w": int(crop_rgb.shape[1]),
        "leaf_polygon_count": len(leaf_lines),
        "disease_polygon_count": len(disease_lines),
        "leaf_mask_area_ratio_crop": float((crop_leaf > 0).mean()),
        "leaf_quality_reasons": leaf_q.get("weak_reasons", ""),
        "disease_area_ratio_leaf": float(disease_stats["disease_area_ratio_leaf"]),
        "disease_component_count": int(disease_stats["component_count"]),
        "disease_class_area_max": float(disease_stats["class_area_max"]),
        "disease_weak_reasons": disease_stats.get("weak_reasons", ""),
        "rejection_reasons": ";".join([x for x in reasons if x]),
    }
    rows.append(rec)
    if accepted:
        accepted_rows.append(rec)
        bucket = "healthy_leaf_only" if is_healthy else "accepted"
    else:
        rejected_rows.append(rec)
        bucket = "rejected"

    if preview_counts[bucket] < 140:
        if bucket == "healthy_leaf_only":
            panel = side_by_side([crop_rgb, overlay_mask(crop_rgb, crop_leaf, (40, 200, 90))])
        elif bucket == "accepted":
            panel = side_by_side([
                crop_rgb,
                overlay_mask(crop_rgb, crop_leaf, (40, 200, 90)),
                overlay_mask(crop_rgb, disease_mask, (80, 140, 255)),
                overlay_multiclass(crop_rgb, crop_leaf, disease_mask),
            ])
        else:
            panel = side_by_side([
                crop_rgb,
                overlay_mask(crop_rgb, crop_leaf, (255, 160, 30)),
                overlay_mask(crop_rgb, disease_mask, (255, 80, 80)),
            ])
        write_rgb(PREVIEW_DIR / bucket / f"{idx:06d}__{row.source_class}__{p.stem}.jpg", panel)
        preview_counts[bucket] += 1

    if idx % 500 == 0:
        pd.DataFrame(rows).to_csv(QUANT_DIR / "multiclass_pseudo_label_inventory_partial.csv", index=False)
        progress["multiclass_qa_processed"] = idx
        save_progress()
        print({"processed": idx, "accepted": len(accepted_rows), "rejected": len(rejected_rows)})

inventory_df = pd.DataFrame(rows)
accepted_df = pd.DataFrame(accepted_rows)
rejections_df = pd.DataFrame(rejected_rows)
inventory_path = QUANT_DIR / "multiclass_pseudo_label_inventory.csv"
rejections_path = QUANT_DIR / "multiclass_pseudo_label_rejections.csv"
qa_path = QUANT_DIR / "multiclass_qa_summary.json"
inventory_df.to_csv(inventory_path, index=False)
rejections_df.to_csv(rejections_path, index=False)

positive_count = int(((accepted_df.get("sample_type", pd.Series(dtype=str)) == "leaf_disease")).sum()) if len(accepted_df) else 0
healthy_count = int(((accepted_df.get("sample_type", pd.Series(dtype=str)) == "healthy_leaf_only")).sum()) if len(accepted_df) else 0
qa_summary = {
    "total_candidates": int(len(inventory_df)),
    "accepted": int(len(accepted_df)),
    "rejected": int(len(rejections_df)),
    "leaf_disease_positive_samples": positive_count,
    "healthy_leaf_only_samples": healthy_count,
    "min_accepted_disease_positives": MIN_ACCEPTED_DISEASE_POSITIVES,
    "min_healthy_negatives": MIN_HEALTHY_NEGATIVES,
    "accepted_by_type": {str(k): int(v) for k, v in accepted_df.get("sample_type", pd.Series(dtype=str)).value_counts().to_dict().items()},
    "rejection_reasons": {k: int(v) for k, v in Counter(";".join(rejections_df.get("rejection_reasons", pd.Series(dtype=str)).fillna("").astype(str)).split(";")).items() if k},
}
qa_path.write_text(json.dumps(qa_summary, indent=2), encoding="utf-8")
progress["multiclass_qa_summary"] = qa_summary
progress["artifacts"].extend([str(inventory_path), str(rejections_path), str(qa_path), str(PREVIEW_DIR)])
if positive_count < MIN_ACCEPTED_DISEASE_POSITIVES or healthy_count < MIN_HEALTHY_NEGATIVES:
    progress["status"] = "failed_qa"
    progress["stage"] = "multiclass_qa_failed_no_training"
    progress["qa_failure_reason"] = "not enough accepted disease positives or healthy negatives"
else:
    progress["stage"] = "multiclass_qa_passed"
save_progress()
print(qa_summary)
'''


PREPARE_CELL = r'''
if progress.get("status") != "failed_qa":
    progress["stage"] = "prepare_multiclass_yolo_dataset"; save_progress()
    accepted_df = split_stratified(accepted_df, "source_class", 0.70, 0.15)
    accepted_df["row_id"] = [f"multi9_{i:06d}" for i in range(len(accepted_df))]
    if set(accepted_df["normalized_id"]) & holdout_norm_ids:
        raise AssertionError("Holdout leaked into v9 train/val/test")
    for split in ["train", "val", "test"]:
        (PREPARED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (PREPARED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    prep_rows = []
    for idx, row in enumerate(accepted_df.itertuples(index=False), start=1):
        p = Path(row.source_path)
        rgb = read_rgb(p)
        leaf_mask, _, _ = leaf_tissue_shape_mask(rgb)
        bbox = bbox_from_mask(leaf_mask, pad_ratio=0.10)
        crop_rgb, crop_leaf = crop_rgb_mask(rgb, leaf_mask, bbox)
        leaf_lines = masks_to_yolo_lines(crop_leaf, class_id=0, min_area_ratio=0.003, max_contours=4)
        disease_mask = np.zeros(crop_leaf.shape, dtype=np.uint8)
        disease_lines = []
        if not bool(row.is_healthy) and str(row.sample_type) == "leaf_disease":
            disease_mask, disease_stats = disease_mask_conservative(crop_rgb, crop_leaf, row.source_class)
            disease_lines = masks_to_yolo_lines(disease_mask, class_id=1, min_area_ratio=0.0004, max_contours=50)
        lines = leaf_lines + disease_lines
        out_name = f"{row.row_id}__{row.source_class}__{p.stem}.jpg"
        image_out = PREPARED_DIR / "images" / row.split / out_name
        label_out = PREPARED_DIR / "labels" / row.split / (Path(out_name).stem + ".txt")
        write_rgb(image_out, crop_rgb)
        label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        prep_rows.append({
            **row._asdict(),
            "image_path": str(image_out),
            "label_path": str(label_out),
            "bbox_runtime": ",".join(str(int(v)) for v in bbox),
            "leaf_polygon_count_runtime": len(leaf_lines),
            "disease_polygon_count_runtime": len(disease_lines),
            "has_leaf_label": len(leaf_lines) > 0,
            "has_disease_label": len(disease_lines) > 0,
        })
        if idx % 1000 == 0:
            print({"prepared": idx, "total": len(accepted_df)})

    prep_df = pd.DataFrame(prep_rows)
    split_manifest = QUANT_DIR / "multiclass_training_split_manifest.csv"
    prep_df.to_csv(split_manifest, index=False)
    data_yaml = PREPARED_DIR / "data.yaml"
    data_yaml.write_text(f"path: {PREPARED_DIR}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: leaf\n  1: disease_region\n", encoding="utf-8")
    assert_multiclass_labels(PREPARED_DIR / "labels", prep_df)
    progress["split_counts"] = {k: int(v) for k, v in accepted_df["split"].value_counts().to_dict().items()}
    progress["prepared_counts"] = {
        "total": int(len(prep_df)),
        "leaf_only": int((prep_df["sample_type"] == "healthy_leaf_only").sum()),
        "leaf_disease": int((prep_df["sample_type"] == "leaf_disease").sum()),
    }
    progress["artifacts"].extend([str(split_manifest), str(data_yaml)])
    save_progress()
    print(progress["split_counts"])
'''


TRAIN_CELL = r'''
if progress.get("status") != "failed_qa":
    progress["stage"] = "train_multiclass_yolo"; save_progress()
    model = YOLO("yolo11n-seg.pt")
    train_result = model.train(
        data=str(PREPARED_DIR / "data.yaml"),
        epochs=25,
        imgsz=640,
        batch=16,
        workers=2,
        seed=SEED,
        device=DEVICE,
        project=str(RUNS_DIR),
        name="yolo11n_multiclass_v9",
        exist_ok=True,
        patience=8,
        verbose=False,
    )
    train_dir = Path(getattr(train_result, "save_dir", RUNS_DIR / "yolo11n_multiclass_v9"))
    best_model = WORKING_DIR / "best_multiclass_model.pt"
    if (train_dir / "weights" / "best.pt").exists():
        shutil.copy2(train_dir / "weights" / "best.pt", best_model)
    trained = YOLO(str(best_model if best_model.exists() else train_dir / "weights" / "last.pt"))
    metric_outputs = {}
    for split in ["train", "test"]:
        m = trained.val(
            data=str(PREPARED_DIR / "data.yaml"),
            split=split,
            imgsz=640,
            batch=16,
            device=DEVICE,
            project=str(RUNS_DIR),
            name=f"eval_{split}",
            exist_ok=True,
            verbose=False,
        )
        vals = metrics_from_results_multiclass(m)
        (QUANT_DIR / f"{split}_pseudo_metrics.json").write_text(json.dumps(vals, indent=2), encoding="utf-8")
        pd.DataFrame([vals]).to_csv(QUANT_DIR / f"{split}_pseudo_metrics.csv", index=False)
        metric_outputs[split] = vals
    progress["train_test_pseudo_metrics"] = metric_outputs
    progress["artifacts"].extend([str(best_model), str(train_dir)])
    save_progress()
    print(metric_outputs)
'''


HEALTHY_METRICS_CELL = r'''
if progress.get("status") != "failed_qa":
    progress["stage"] = "healthy_false_positive_metrics"; save_progress()
    trained = YOLO(str(WORKING_DIR / "best_multiclass_model.pt"))
    healthy_test = prep_df[(prep_df["split"] == "test") & (prep_df["sample_type"] == "healthy_leaf_only")]
    rows = []
    for row in healthy_test.itertuples(index=False):
        rgb = read_rgb(row.image_path)
        pred = trained.predict(source=rgb, imgsz=640, device=DEVICE, conf=0.25, verbose=False)[0]
        masks = class_result_masks(pred, rgb.shape[:2])
        disease_area = float((masks[1] > 0).mean())
        rows.append({
            "row_id": row.row_id,
            "source_class": row.source_class,
            "image_path": row.image_path,
            "pred_disease_area_ratio": disease_area,
            "has_false_positive": disease_area > 0.001,
        })
    df = pd.DataFrame(rows)
    metrics = {
        "healthy_test_count": int(len(df)),
        "healthy_false_positive_rate_test": float(df["has_false_positive"].mean()) if len(df) else None,
        "mean_pred_disease_area_healthy_test": float(df["pred_disease_area_ratio"].mean()) if len(df) else None,
        "empty_prediction_rate_healthy_test": float((df["pred_disease_area_ratio"] <= 0.001).mean()) if len(df) else None,
    }
    (QUANT_DIR / "healthy_false_positive_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    df.to_csv(QUANT_DIR / "healthy_false_positive_metrics.csv", index=False)
    progress["healthy_false_positive_metrics"] = metrics
    save_progress()
    print(metrics)
'''


HOLDOUT_CELL = r'''
progress["stage"] = "real_holdout_predictions"; save_progress()
trained = YOLO(str(WORKING_DIR / "best_multiclass_model.pt")) if progress.get("status") != "failed_qa" and (WORKING_DIR / "best_multiclass_model.pt").exists() else None
holdout_rows = []
diag_rows = []

for rec in holdout_df.itertuples(index=False):
    p = Path(rec.path)
    rgb = read_rgb(p)
    leaf_mask, _, _ = leaf_tissue_shape_mask(rgb)
    bbox = bbox_from_mask(leaf_mask, pad_ratio=0.10)
    crop_rgb, crop_leaf = crop_rgb_mask(rgb, leaf_mask, bbox)

    if trained is not None:
        pred = trained.predict(source=crop_rgb, imgsz=640, device=DEVICE, conf=0.25, verbose=False)[0]
        crop_masks = class_result_masks(pred, crop_rgb.shape[:2])
    else:
        crop_masks = {0: np.zeros(crop_leaf.shape, dtype=np.uint8), 1: np.zeros(crop_leaf.shape, dtype=np.uint8)}
    full_leaf_pred = paste_crop_mask(crop_masks[0], rgb.shape[:2], bbox)
    full_disease_pred = paste_crop_mask(crop_masks[1], rgb.shape[:2], bbox)

    panel = side_by_side([
        rgb,
        overlay_mask(rgb, full_leaf_pred, (40, 200, 90)),
        overlay_mask(rgb, full_disease_pred, (80, 140, 255)),
        overlay_multiclass(rgb, full_leaf_pred, full_disease_pred),
    ])
    out_path = REAL_HOLDOUT_DIR / "by_category" / rec.category / f"{Path(rec.holdout_filename).stem}_orig_leafpred_diseasepred_combined.jpg"
    write_rgb(out_path, panel)
    holdout_rows.append({
        "category": rec.category,
        "holdout_filename": rec.holdout_filename,
        "original_filename": rec.original_filename,
        "bbox_for_inference": ",".join(str(int(v)) for v in bbox),
        "pred_leaf_area_ratio": float((full_leaf_pred > 0).mean()),
        "pred_disease_area_ratio": float((full_disease_pred > 0).mean()),
        "qualitative_path": str(out_path),
        "note": "real unseen holdout prediction; no pseudo metric",
    })

    is_holdout_healthy = (rec.category == "healthy_leaves") or ("healthy" in rec.holdout_filename.lower())
    diag_disease = np.zeros(crop_leaf.shape, dtype=np.uint8)
    if not is_holdout_healthy and disease_policy(rec.original_filename) == "focal_positive":
        diag_disease, _ = disease_mask_conservative(crop_rgb, crop_leaf, rec.original_filename)
    full_diag_disease = paste_crop_mask(diag_disease, rgb.shape[:2], bbox)
    diag_panel = side_by_side([
        rgb,
        overlay_mask(rgb, leaf_mask, (40, 200, 90)),
        overlay_mask(rgb, full_diag_disease, (255, 180, 40)),
    ])
    diag_path = HOLDOUT_DIAG_DIR / "by_category" / rec.category / f"{Path(rec.holdout_filename).stem}_diagnostic_leaf_diseaseheuristic.jpg"
    write_rgb(diag_path, diag_panel)
    diag_rows.append({
        "category": rec.category,
        "holdout_filename": rec.holdout_filename,
        "diagnostic_path": str(diag_path),
        "note": "diagnostic heuristic reference only; not training, not ground truth, not metric",
    })

holdout_csv = WORKING_DIR / "real_holdout_predictions.csv"
diag_csv = WORKING_DIR / "holdout_diagnostic_reference.csv"
pd.DataFrame(holdout_rows).to_csv(holdout_csv, index=False)
pd.DataFrame(diag_rows).to_csv(diag_csv, index=False)
progress["real_holdout_predictions"] = {"count": int(len(holdout_rows)), "note": "holdout excluded from training and used only for qualitative unseen predictions"}
progress["artifacts"].extend([str(holdout_csv), str(diag_csv)])
save_progress()
print(progress["real_holdout_predictions"])
'''


QUAL_TEST_CELL = r'''
if progress.get("status") != "failed_qa":
    progress["stage"] = "qualitative_multiclass_test"; save_progress()
    trained = YOLO(str(WORKING_DIR / "best_multiclass_model.pt"))
    sample = prep_df[prep_df["split"] == "test"].sample(n=min(120, int((prep_df["split"] == "test").sum())), random_state=SEED)
    for row in sample.itertuples(index=False):
        rgb = read_rgb(row.image_path)
        label_masks = label_file_to_mask(row.label_path, rgb.shape[:2])
        pseudo_leaf = label_masks if isinstance(label_masks, np.ndarray) else label_masks
        masks_from_label = label_lines_to_masks(Path(row.label_path).read_text(encoding="utf-8").splitlines(), rgb.shape[:2])
        pred = trained.predict(source=rgb, imgsz=640, device=DEVICE, conf=0.25, verbose=False)[0]
        pred_masks = class_result_masks(pred, rgb.shape[:2])
        panel = side_by_side([
            rgb,
            overlay_multiclass(rgb, masks_from_label[0], masks_from_label[1]),
            overlay_multiclass(rgb, pred_masks[0], pred_masks[1]),
        ])
        write_rgb(QUAL_TEST_DIR / f"{row.row_id}_pseudo_pred.jpg", panel)
    progress["artifacts"].append(str(QUAL_TEST_DIR))
    save_progress()
'''


FINALIZE_CELL = r'''
progress["stage"] = "finalize"; save_progress()
for folder, zip_name in [
    (PREPARED_DIR, "prepared_multiclass_yolo_dataset.zip"),
    (PREVIEW_DIR, "multiclass_pseudo_labels_preview.zip"),
    (REAL_HOLDOUT_DIR, "real_holdout_predictions.zip"),
    (HOLDOUT_DIAG_DIR, "holdout_diagnostic_reference.zip"),
    (QUAL_TEST_DIR, "qualitative_multiclass_results.zip"),
    (QUANT_DIR, "quantitative_metrics.zip"),
]:
    if folder.exists():
        z = zip_dir(folder, WORKING_DIR / zip_name)
        progress["artifacts"].append(str(z))
if progress.get("status") != "failed_qa":
    progress["status"] = "complete"
for bulky in [PREPARED_DIR, PREVIEW_DIR, REAL_HOLDOUT_DIR, HOLDOUT_DIAG_DIR, QUAL_TEST_DIR, RUNS_DIR]:
    if bulky.exists():
        shutil.rmtree(bulky)
progress["stage"] = "done"
progress["artifacts"] = sorted(set(progress["artifacts"]))
save_progress()
print(json.dumps({"status": progress["status"], "stage": progress["stage"], "artifact_count": len(progress["artifacts"])}, indent=2))
'''


cells = [
    md("intro", "# PC5 v9 - Multiclass Leaf + Disease YOLO-Seg\nTrain one YOLO segmentation model with two classes: `leaf` and `disease_region`. Inputs are RGB leaf crops, not black-background isolated leaves. Holdout is excluded from train/val/test and used only for real unseen qualitative predictions."),
    md("setup-md", "## Setup\nInstall/check Ultralytics and configure deterministic GPU execution."),
    code("setup", SETUP),
    md("paths-md", "## Paths And Outputs\nCreate Kaggle working folders, QA gates, classes, and run summary."),
    code("paths", PATHS),
    md("helpers-md", "## Leaf Helpers\nReuse the v7 leaf tissue + shape heuristic, dataset discovery, polygon conversion, overlays, metrics, and zipping."),
    code("helpers", HELPERS),
    md("discover-md", "## Discover Inputs And Exclude Holdout\nBuild holdout manifest and remove matching original PlantVillage files before any split."),
    code("discover", DISCOVER),
    md("multiclass-helpers-md", "## Multiclass Helpers\nCreate RGB crops, conservative disease masks, multiclass labels, prediction masks, and label validation."),
    code("multiclass-helpers", MULTICLASS_HELPERS),
    md("qa-md", "## Multiclass Pseudo-Label QA\nGenerate leaf and disease pseudo-labels, keep healthy leaf-only samples, reject weak disease positives, and gate training."),
    code("qa", QA_CELL),
    md("prepare-md", "## Prepare Multiclass YOLO Dataset\nUse accepted samples only, split 70/15/15, and write YOLO-seg labels with class 0 and class 1."),
    code("prepare", PREPARE_CELL),
    md("train-md", "## Train And Quantitative Pseudo Metrics\nTrain YOLO multiclass and save separate train/test pseudo metrics."),
    code("train", TRAIN_CELL),
    md("healthy-md", "## Healthy False Positive Metrics\nMeasure disease false positives on healthy test crops."),
    code("healthy-metrics", HEALTHY_METRICS_CELL),
    md("holdout-md", "## Real Unseen Holdout Predictions\nSave original, YOLO leaf prediction, YOLO disease prediction, and combined prediction for all 67 holdout images."),
    code("holdout", HOLDOUT_CELL),
    md("qual-test-md", "## Qualitative Test Results\nSave pseudo-vs-pred panels on a quantitative test sample."),
    code("qual-test", QUAL_TEST_CELL),
    md("final-md", "## Finalize\nZip deliverables and write final run summary."),
    code("finalize", FINALIZE_CELL),
]

metadata = {
    "id": "jeffreyamc/cv-pc5-v9-multiclass-leaf-disease-yolo-seg",
    "title": "cv-pc5-v9-multiclass-leaf-disease-yolo-seg",
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

write_kernel("v9", cells, metadata)
print("generated v9 notebook")
