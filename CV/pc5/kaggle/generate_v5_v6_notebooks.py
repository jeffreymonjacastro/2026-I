import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def cell(kind, cid, src):
    base = {"cell_type": kind, "id": cid, "metadata": {}, "source": src.strip() + "\n"}
    if kind == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


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


COMMON_SETUP = r'''
import importlib.util, json, math, os, random, re, shutil, subprocess, sys, time, traceback
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

if importlib.util.find_spec("ultralytics") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])

from ultralytics import YOLO, SAM

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
})
'''


COMMON_HELPERS = r'''
def read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def write_rgb(path, rgb, quality=92):
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

def heuristic_leaf_mask_and_bbox(rgb, expand=0.08):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)
    r, g, bb = rgb[:, :, 0].astype(np.int16), rgb[:, :, 1].astype(np.int16), rgb[:, :, 2].astype(np.int16)
    exg = cv2.normalize((2 * g - r - bb).astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    exg_mask = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    hsv_mask = ((h >= 15) & (h <= 105) & (s >= 20) & (v >= 25)).astype(np.uint8) * 255
    chroma = cv2.threshold(cv2.absdiff(a, b), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    mask = cv2.bitwise_or(hsv_mask, cv2.bitwise_and(exg_mask, chroma))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.medianBlur(mask, 5)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    clean = np.zeros(mask.shape, dtype=np.uint8)
    hh, ww = mask.shape
    if n > 1:
        order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
        min_area = max(64, int(0.01 * hh * ww))
        for lab_id in order[:4]:
            if stats[lab_id, cv2.CC_STAT_AREA] >= min_area:
                clean[labels == lab_id] = 255
        if clean.sum() == 0 and len(order):
            clean[labels == int(order[0])] = 255
    if clean.sum() == 0:
        clean[:] = 255
    ys, xs = np.where(clean > 0)
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    px, py = int((x2 - x1 + 1) * expand), int((y2 - y1 + 1) * expand)
    return clean, [max(0, x1 - px), max(0, y1 - py), min(ww - 1, x2 + px), min(hh - 1, y2 + py)]

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
                bad.append(str(p)); break
    if bad:
        raise AssertionError(f"Non-zero class ids in labels: {bad[:5]}")
'''


def write_kernel(vdir, metadata, cells):
    inp = ROOT / "kaggle" / vdir / "input"
    out = ROOT / "kaggle" / vdir / "outputs"
    inp.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    (inp / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (inp / "main.ipynb").write_text(json.dumps(nb(cells), indent=1) + "\n", encoding="utf-8")


v5_cells = [
    cell("markdown", "intro", """
# PC5 v5 - Leaf YOLO-Seg with MobileSAM Pseudo-Labels
Train one YOLO segmentation class, `leaf`, on PlantVillage excluding the unseen holdout. MobileSAM creates pseudo-labels from heuristic bbox prompts. Metrics are pseudo-label metrics, never human ground truth. Holdout images are qualitative-only unseen results.
"""),
    cell("markdown", "setup-md", "## Setup\nLoad dependencies, Ultralytics, seed, and GPU state."),
    cell("code", "setup", COMMON_SETUP),
    cell("markdown", "paths-md", "## Paths\nCreate v5 outputs and progress state."),
    cell("code", "paths", r'''
INPUT_ROOT = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")
CATEGORIES = ["healthy_leaves","small_diseased_regions","large_diseased_regions","simple_backgrounds","complex_backgrounds","multiple_leaves","partial_occlusions"]
PREPARED_DIR = WORKING_DIR / "prepared_yolo_dataset"
QUAL_HOLDOUT_DIR = WORKING_DIR / "qualitative_holdout"
QUANT_DIR = WORKING_DIR / "quantitative_metrics"
RUNS_DIR = WORKING_DIR / "runs_leaf"
SUMMARY_PATH = WORKING_DIR / "run_summary.json"
for p in [PREPARED_DIR, QUAL_HOLDOUT_DIR, QUANT_DIR, RUNS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
for c in CATEGORIES:
    (QUAL_HOLDOUT_DIR / "by_category" / c).mkdir(parents=True, exist_ok=True)
progress = {"status":"running","stage":"start","task":"leaf segmentation","pseudo_label_source":"MobileSAM bbox prompts with heuristic fallback","holdout_policy":"67 holdout images excluded from train/val/test; qualitative only","artifacts":[]}
def save_progress():
    SUMMARY_PATH.write_text(json.dumps(progress, indent=2), encoding="utf-8")
save_progress()
'''),
    cell("markdown", "discover-md", "## Discover Inputs\nFind PlantVillage and holdout, parse holdout names, exclude before splitting."),
    cell("code", "discover", COMMON_HELPERS + r'''
progress["stage"] = "discover_inputs"; save_progress()
plant_root = find_plantvillage_root(INPUT_ROOT)
holdout_root = find_holdout_root(INPUT_ROOT, CATEGORIES)
holdout_records, holdout_norm_ids = [], set()
for category in CATEGORIES:
    for p in sorted((holdout_root / category).glob("*.jpg")):
        original = extract_holdout_original(p.name)
        if original is None:
            raise ValueError(f"Bad holdout filename: {p.name}")
        holdout_norm_ids.add(normalize_image_id(original))
        holdout_records.append({"category":category,"holdout_filename":p.name,"original_filename":original,"normalized_original_id":normalize_image_id(original),"path":str(p)})
holdout_df = pd.DataFrame(holdout_records)
if len(holdout_df) != 67:
    raise AssertionError(f"Expected 67 holdout images, found {len(holdout_df)}")
holdout_manifest_path = WORKING_DIR / "holdout_manifest.csv"
holdout_df.to_csv(holdout_manifest_path, index=False)
records, excluded = [], []
for class_dir in sorted([p for p in plant_root.iterdir() if p.is_dir()]):
    for p in sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))):
        norm = normalize_image_id(p.name)
        rec = {"source_class":class_dir.name,"filename":p.name,"normalized_id":norm,"path":str(p)}
        (excluded if norm in holdout_norm_ids else records).append(rec)
plant_df = pd.DataFrame(records)
plant_df = split_stratified(plant_df, "source_class", 0.70, 0.15)
plant_df["row_id"] = [f"leaf_{i:06d}" for i in range(len(plant_df))]
if set(plant_df["normalized_id"]) & holdout_norm_ids:
    raise AssertionError("Holdout leaked into train/val/test")
split_manifest = QUANT_DIR / "training_split_manifest.csv"
plant_df.to_csv(split_manifest, index=False)
progress.update({"plant_root":str(plant_root),"holdout_root":str(holdout_root),"holdout_count":int(len(holdout_df)),"excluded_from_plantvillage":int(len(excluded)),"split_counts":{k:int(v) for k,v in plant_df["split"].value_counts().to_dict().items()}})
progress["artifacts"].extend([str(holdout_manifest_path), str(split_manifest)])
save_progress(); print(progress["split_counts"])
'''),
    cell("markdown", "labels-md", "## MobileSAM Leaf Pseudo-Labels\nGenerate bbox prompts, MobileSAM masks, fallback masks, and YOLO segmentation labels."),
    cell("code", "labels", r'''
progress["stage"] = "pseudo_label_leaf"; save_progress()
for split in ["train","val","test"]:
    (PREPARED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (PREPARED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
sam_model = SAM("mobile_sam.pt")
label_rows, failure_rows = [], []
def sam_leaf_mask(path, rgb, bbox):
    try:
        res = sam_model.predict(str(path), bboxes=[bbox], device=DEVICE, verbose=False)[0]
        mask = result_to_mask(res, rgb.shape[:2])
        if mask.sum() == 0:
            raise ValueError("empty MobileSAM mask")
        return mask, "mobile_sam", ""
    except Exception as exc:
        fallback, _ = heuristic_leaf_mask_and_bbox(rgb)
        return fallback, "fallback_heuristic", repr(exc)
for idx, row in enumerate(plant_df.itertuples(index=False), start=1):
    p = Path(row.path); rgb = read_rgb(p)
    _, bbox = heuristic_leaf_mask_and_bbox(rgb)
    mask, source, error = sam_leaf_mask(p, rgb, bbox)
    lines = masks_to_yolo_lines(mask, class_id=0, min_area_ratio=0.003, max_contours=4)
    out_name = f"{row.row_id}__{row.source_class}__{p.stem}.jpg"
    image_out = PREPARED_DIR / "images" / row.split / out_name
    label_out = PREPARED_DIR / "labels" / row.split / (Path(out_name).stem + ".txt")
    shutil.copy2(p, image_out)
    label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    area = float((mask > 0).mean())
    weak = []
    if not lines: weak.append("empty_polygon")
    if area < 0.03: weak.append("small_leaf_mask")
    if area > 0.95: weak.append("large_leaf_mask")
    if source != "mobile_sam": weak.append("sam_fallback")
    rec = {"row_id":row.row_id,"split":row.split,"source_class":row.source_class,"source_path":str(p),"image_path":str(image_out),"label_path":str(label_out),"mask_source":source,"sam_error":error,"polygon_count":len(lines),"mask_area_ratio":area,"weak_reasons":";".join(weak)}
    label_rows.append(rec)
    if weak: failure_rows.append(rec)
    if idx % 250 == 0:
        pd.DataFrame(label_rows).to_csv(QUANT_DIR / "pseudo_label_inventory_partial.csv", index=False)
        progress["pseudo_labeled"] = idx; save_progress()
        print({"pseudo_labeled":idx,"total":len(plant_df)})
labels_df, failures_df = pd.DataFrame(label_rows), pd.DataFrame(failure_rows)
pseudo_inventory, pseudo_failures = QUANT_DIR / "pseudo_label_inventory.csv", QUANT_DIR / "pseudo_label_failures.csv"
labels_df.to_csv(pseudo_inventory, index=False); failures_df.to_csv(pseudo_failures, index=False)
data_yaml = PREPARED_DIR / "data.yaml"
data_yaml.write_text(f"path: {PREPARED_DIR}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: leaf\n", encoding="utf-8")
assert_only_class_zero(PREPARED_DIR / "labels")
progress["leaf_label_summary"] = {"labels":int(len(labels_df)),"empty_labels":int((labels_df["polygon_count"]==0).sum()),"weak_labels":int(labels_df["weak_reasons"].astype(bool).sum()),"mobile_sam_labels":int((labels_df["mask_source"]=="mobile_sam").sum()),"fallback_labels":int((labels_df["mask_source"]!="mobile_sam").sum())}
progress["artifacts"].extend([str(pseudo_inventory), str(pseudo_failures), str(data_yaml)])
save_progress(); print(progress["leaf_label_summary"])
'''),
    cell("markdown", "train-md", "## Train And Quantitative Metrics\nTrain leaf YOLO and save separate train/test pseudo metrics."),
    cell("code", "train", r'''
progress["stage"] = "train_leaf_yolo"; save_progress()
model = YOLO("yolo11n-seg.pt")
train_result = model.train(data=str(PREPARED_DIR/"data.yaml"), epochs=25, imgsz=640, batch=16, workers=2, seed=SEED, device=DEVICE, project=str(RUNS_DIR), name="yolo11n_leaf", exist_ok=True, patience=8, verbose=False)
train_dir = Path(getattr(train_result, "save_dir", RUNS_DIR/"yolo11n_leaf"))
best_leaf = WORKING_DIR / "best_leaf_model.pt"
if (train_dir/"weights"/"best.pt").exists():
    shutil.copy2(train_dir/"weights"/"best.pt", best_leaf)
trained = YOLO(str(best_leaf if best_leaf.exists() else train_dir/"weights"/"last.pt"))
metric_outputs = {}
for split in ["train","test"]:
    m = trained.val(data=str(PREPARED_DIR/"data.yaml"), split=split, imgsz=640, batch=16, device=DEVICE, project=str(RUNS_DIR), name=f"eval_{split}", exist_ok=True, verbose=False)
    vals = metrics_from_results(m)
    (QUANT_DIR / f"{split}_pseudo_metrics.json").write_text(json.dumps(vals, indent=2), encoding="utf-8")
    pd.DataFrame([vals]).to_csv(QUANT_DIR / f"{split}_pseudo_metrics.csv", index=False)
    metric_outputs[split] = vals
progress["train_test_pseudo_metrics"] = metric_outputs
progress["artifacts"].extend([str(best_leaf), str(train_dir)])
save_progress(); print(metric_outputs)
'''),
    cell("markdown", "holdout-md", "## Unseen Qualitative Holdout\nRun leaf model over excluded holdout only for exposition overlays."),
    cell("code", "holdout", r'''
progress["stage"] = "qualitative_holdout_leaf"; save_progress()
trained = YOLO(str(WORKING_DIR / "best_leaf_model.pt"))
holdout_rows = []
for rec in holdout_df.itertuples(index=False):
    p = Path(rec.path); rgb = read_rgb(p)
    _, bbox = heuristic_leaf_mask_and_bbox(rgb)
    sam_mask, source, error = sam_leaf_mask(p, rgb, bbox)
    pred = trained.predict(source=rgb, imgsz=640, device=DEVICE, conf=0.25, verbose=False)[0]
    pred_mask = result_to_mask(pred, rgb.shape[:2])
    panel = side_by_side([rgb, overlay_mask(rgb, sam_mask, (40,200,90)), overlay_mask(rgb, pred_mask, (60,120,255))])
    out_path = QUAL_HOLDOUT_DIR / "by_category" / rec.category / f"{Path(rec.holdout_filename).stem}_orig_sam_pred.jpg"
    write_rgb(out_path, panel)
    holdout_rows.append({"category":rec.category,"holdout_filename":rec.holdout_filename,"original_filename":rec.original_filename,"sam_source":source,"pseudo_mask_area_ratio":float((sam_mask>0).mean()),"pred_mask_area_ratio":float((pred_mask>0).mean()),"pseudo_mask_iou_unseen_holdout":mask_iou(sam_mask,pred_mask),"qualitative_path":str(out_path)})
holdout_csv = WORKING_DIR / "holdout_leaf_qualitative_metrics.csv"
pd.DataFrame(holdout_rows).to_csv(holdout_csv, index=False)
progress["unseen_holdout_leaf_results"] = {"count":int(len(holdout_rows)),"mean_pseudo_iou_for_reference_only":float(pd.DataFrame(holdout_rows)["pseudo_mask_iou_unseen_holdout"].mean()),"note":"Holdout is unseen and qualitative-only; this IoU is reference only."}
progress["artifacts"].append(str(holdout_csv)); save_progress(); print(progress["unseen_holdout_leaf_results"])
'''),
    cell("markdown", "final-md", "## Finalize\nZip deliverables and remove large folders from published outputs."),
    cell("code", "finalize", r'''
progress["stage"] = "finalize"; save_progress()
for z in [zip_dir(PREPARED_DIR, WORKING_DIR/"prepared_yolo_dataset.zip"), zip_dir(QUAL_HOLDOUT_DIR, WORKING_DIR/"qualitative_holdout.zip"), zip_dir(QUANT_DIR, WORKING_DIR/"quantitative_metrics.zip")]:
    progress["artifacts"].append(str(z))
for bulky in [PREPARED_DIR, QUAL_HOLDOUT_DIR, RUNS_DIR]:
    if bulky.exists(): shutil.rmtree(bulky)
progress["status"], progress["stage"] = "complete", "done"
progress["artifacts"] = sorted(set(progress["artifacts"]))
save_progress()
print(json.dumps({"status":progress["status"],"stage":progress["stage"],"artifact_count":len(progress["artifacts"]),"summary_path":str(SUMMARY_PATH)}, indent=2))
'''),
]


v6_cells = [
    cell("markdown", "intro", """
# PC5 v6 - Disease Region YOLO-Seg from Isolated Leaves
Train one YOLO segmentation class, `disease_region`. Inputs are leaves with black background outside leaf mask from v5 leaf model when mounted, else heuristic fallback. Holdout is excluded from train/val/test and used only for unseen qualitative results. Metrics are pseudo-label metrics, not human ground truth.
"""),
    cell("markdown", "setup-md", "## Setup\nLoad dependencies, Ultralytics, seed, and GPU state."),
    cell("code", "setup", COMMON_SETUP),
    cell("markdown", "paths-md", "## Paths\nCreate v6 outputs and progress state."),
    cell("code", "paths", r'''
INPUT_ROOT = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")
CATEGORIES = ["healthy_leaves","small_diseased_regions","large_diseased_regions","simple_backgrounds","complex_backgrounds","multiple_leaves","partial_occlusions"]
PREPARED_DIR = WORKING_DIR / "prepared_disease_yolo_dataset"
INPUTS_PREVIEW_DIR = WORKING_DIR / "disease_inputs_preview"
PSEUDO_PREVIEW_DIR = WORKING_DIR / "disease_pseudo_labels_preview"
FAILURES_DIR = WORKING_DIR / "disease_label_failures"
QUAL_HOLDOUT_DIR = WORKING_DIR / "qualitative_disease_holdout"
QUAL_TEST_DIR = WORKING_DIR / "qualitative_disease_results"
QUANT_DIR = WORKING_DIR / "quantitative_metrics"
RUNS_DIR = WORKING_DIR / "runs_disease"
SUMMARY_PATH = WORKING_DIR / "run_summary.json"
for p in [PREPARED_DIR, INPUTS_PREVIEW_DIR, PSEUDO_PREVIEW_DIR, FAILURES_DIR, QUAL_HOLDOUT_DIR, QUAL_TEST_DIR, QUANT_DIR, RUNS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
for c in CATEGORIES:
    (QUAL_HOLDOUT_DIR / "by_category" / c).mkdir(parents=True, exist_ok=True)
progress = {"status":"running","stage":"start","task":"disease_region segmentation","holdout_policy":"67 holdout images excluded from train/val/test; qualitative only","artifacts":[]}
def save_progress():
    SUMMARY_PATH.write_text(json.dumps(progress, indent=2), encoding="utf-8")
save_progress()
'''),
    cell("markdown", "discover-md", "## Discover Inputs\nFind PlantVillage, holdout, and v5 leaf model if mounted."),
    cell("code", "discover", COMMON_HELPERS + r'''
progress["stage"] = "discover_inputs"; save_progress()
plant_root = find_plantvillage_root(INPUT_ROOT)
holdout_root = find_holdout_root(INPUT_ROOT, CATEGORIES)
leaf_candidates = list(INPUT_ROOT.glob("**/best_leaf_model.pt"))
leaf_model_path = leaf_candidates[0] if leaf_candidates else None
leaf_model_source = "yolo_leaf" if leaf_model_path else "fallback_heuristic"
leaf_model = YOLO(str(leaf_model_path)) if leaf_model_path else None
holdout_records, holdout_norm_ids = [], set()
for category in CATEGORIES:
    for p in sorted((holdout_root / category).glob("*.jpg")):
        original = extract_holdout_original(p.name)
        if original is None: raise ValueError(f"Bad holdout filename: {p.name}")
        holdout_norm_ids.add(normalize_image_id(original))
        holdout_records.append({"category":category,"holdout_filename":p.name,"original_filename":original,"normalized_original_id":normalize_image_id(original),"path":str(p)})
holdout_df = pd.DataFrame(holdout_records)
if len(holdout_df) != 67: raise AssertionError(f"Expected 67 holdout images, found {len(holdout_df)}")
holdout_manifest_path = WORKING_DIR / "holdout_manifest.csv"
holdout_df.to_csv(holdout_manifest_path, index=False)
records, excluded = [], []
for class_dir in sorted([p for p in plant_root.iterdir() if p.is_dir()]):
    is_healthy = "healthy" in class_dir.name.lower()
    for p in sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))):
        norm = normalize_image_id(p.name)
        rec = {"source_class":class_dir.name,"filename":p.name,"normalized_id":norm,"is_healthy":is_healthy,"path":str(p)}
        (excluded if norm in holdout_norm_ids else records).append(rec)
plant_df = pd.DataFrame(records)
plant_df = split_stratified(plant_df, "source_class", 0.70, 0.15)
plant_df["row_id"] = [f"disease_{i:06d}" for i in range(len(plant_df))]
if set(plant_df["normalized_id"]) & holdout_norm_ids: raise AssertionError("Holdout leaked into train/val/test")
split_manifest = QUANT_DIR / "disease_training_inventory.csv"
plant_df.to_csv(split_manifest, index=False)
progress.update({"plant_root":str(plant_root),"holdout_root":str(holdout_root),"holdout_count":int(len(holdout_df)),"excluded_from_plantvillage":int(len(excluded)),"leaf_mask_source":leaf_model_source,"leaf_model_path":str(leaf_model_path) if leaf_model_path else None,"split_counts":{k:int(v) for k,v in plant_df["split"].value_counts().to_dict().items()},"healthy_count":int(plant_df["is_healthy"].sum()),"diseased_count":int((~plant_df["is_healthy"]).sum())})
progress["artifacts"].extend([str(holdout_manifest_path), str(split_manifest)])
save_progress(); print(progress)
'''),
    cell("markdown", "disease-fns-md", "## Disease Pseudo-Label Functions\nUse leaf isolation and color outlier heuristics inside leaves."),
    cell("code", "disease-fns", r'''
def leaf_mask_for_image(path, rgb):
    if leaf_model is not None:
        try:
            res = leaf_model.predict(source=rgb, imgsz=640, device=DEVICE, conf=0.25, verbose=False)[0]
            mask = result_to_mask(res, rgb.shape[:2])
            if mask.sum() > 0: return mask, "yolo_leaf", ""
            raise ValueError("empty yolo leaf mask")
        except Exception as exc:
            fallback, _ = heuristic_leaf_mask_and_bbox(rgb)
            return fallback, "fallback_heuristic", repr(exc)
    fallback, _ = heuristic_leaf_mask_and_bbox(rgb)
    return fallback, "fallback_heuristic", "no leaf model mounted"

def isolate_leaf(rgb, leaf_mask):
    out = rgb.copy(); out[leaf_mask == 0] = 0; return out

def disease_mask_from_leaf(rgb, leaf_mask, is_healthy=False):
    if is_healthy:
        return np.zeros(leaf_mask.shape, dtype=np.uint8), ["healthy_negative"]
    hsv, lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    h,s,v = cv2.split(hsv); l,a,b = cv2.split(lab)
    inside = leaf_mask > 0
    if inside.sum() == 0: return np.zeros(leaf_mask.shape, dtype=np.uint8), ["empty_leaf_mask"]
    med = np.median(lab[inside].astype(np.float32), axis=0)
    dist = np.linalg.norm(lab.astype(np.float32) - med.reshape(1,1,3), axis=2)
    thr = float(np.percentile(dist[inside], 88))
    brown = (((h <= 25) | (h >= 160)) & (s >= 35) & (v <= 185))
    yellow = ((h >= 20) & (h <= 45) & (s >= 35) & (v >= 80))
    dark = ((l <= np.percentile(l[inside], 18)) & (s >= 20))
    outlier = dist >= max(12.0, thr)
    mask = (inside & (brown | yellow | dark | outlier)).astype(np.uint8) * 255
    k = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    clean = np.zeros(mask.shape, dtype=np.uint8)
    min_area = max(12, int(0.0008 * inside.sum()))
    for lab_id in range(1, n):
        if stats[lab_id, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == lab_id] = 255
    area = float((clean > 0).sum() / max(1, inside.sum()))
    reasons = []
    if area == 0: reasons.append("empty_disease_mask_nonhealthy")
    if 0 < area < 0.002: reasons.append("tiny_disease_mask")
    if area > 0.70: reasons.append("large_disease_mask")
    if cv2.connectedComponents((clean > 0).astype(np.uint8), 8)[0] - 1 > 20: reasons.append("many_components")
    return clean, reasons
'''),
    cell("markdown", "prepare-md", "## Build Disease Dataset\nGenerate black-background inputs, labels, previews, and failure records."),
    cell("code", "prepare", r'''
progress["stage"] = "prepare_disease_dataset"; save_progress()
for split in ["train","val","test"]:
    (PREPARED_DIR/"images"/split).mkdir(parents=True, exist_ok=True)
    (PREPARED_DIR/"labels"/split).mkdir(parents=True, exist_ok=True)
rows, failures, preview_counts = [], [], Counter()
for idx, row in enumerate(plant_df.itertuples(index=False), start=1):
    p = Path(row.path); rgb = read_rgb(p)
    leaf_mask, leaf_source, leaf_error = leaf_mask_for_image(p, rgb)
    isolated = isolate_leaf(rgb, leaf_mask)
    dmask, weak = disease_mask_from_leaf(isolated, leaf_mask, bool(row.is_healthy))
    lines = [] if bool(row.is_healthy) else masks_to_yolo_lines(dmask, class_id=0, min_area_ratio=0.0006, max_contours=20)
    out_name = f"{row.row_id}__{row.source_class}__{p.stem}.jpg"
    image_out = PREPARED_DIR/"images"/row.split/out_name
    label_out = PREPARED_DIR/"labels"/row.split/(Path(out_name).stem + ".txt")
    write_rgb(image_out, isolated); label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if preview_counts[row.split] < 20:
        write_rgb(INPUTS_PREVIEW_DIR/row.split/f"{row.row_id}_isolated.jpg", isolated)
        write_rgb(PSEUDO_PREVIEW_DIR/row.split/f"{row.row_id}_disease_overlay.jpg", overlay_mask(isolated, dmask, (255,180,40)))
        preview_counts[row.split] += 1
    rec = {"row_id":row.row_id,"split":row.split,"source_class":row.source_class,"is_healthy":bool(row.is_healthy),"source_path":str(p),"image_path":str(image_out),"label_path":str(label_out),"leaf_mask_source":leaf_source,"leaf_error":leaf_error,"polygon_count":len(lines),"disease_area_ratio_image":float((dmask>0).mean()),"disease_area_ratio_leaf":float((dmask>0).sum()/max(1,(leaf_mask>0).sum())),"weak_reasons":";".join(weak)}
    rows.append(rec)
    if (not row.is_healthy and (weak or not lines)):
        failures.append(rec)
    if idx % 500 == 0:
        pd.DataFrame(rows).to_csv(QUANT_DIR/"disease_pseudo_label_inventory_partial.csv", index=False)
        progress["disease_pseudo_labeled"] = idx; save_progress(); print({"disease_pseudo_labeled":idx,"total":len(plant_df)})
disease_df, failure_df = pd.DataFrame(rows), pd.DataFrame(failures)
inventory_path, failure_path = QUANT_DIR/"disease_pseudo_label_inventory.csv", QUANT_DIR/"disease_pseudo_label_failures.csv"
disease_df.to_csv(inventory_path, index=False); failure_df.to_csv(failure_path, index=False)
data_yaml = PREPARED_DIR/"data.yaml"
data_yaml.write_text(f"path: {PREPARED_DIR}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: disease_region\n", encoding="utf-8")
assert_only_class_zero(PREPARED_DIR/"labels")
progress["disease_label_summary"] = {"total_images":int(len(disease_df)),"positive_labels":int((disease_df["polygon_count"]>0).sum()),"negative_empty_labels":int((disease_df["polygon_count"]==0).sum()),"healthy_negatives":int(disease_df["is_healthy"].sum()),"weak_labels":int(disease_df["weak_reasons"].astype(bool).sum()),"leaf_mask_source":progress.get("leaf_mask_source")}
progress["artifacts"].extend([str(inventory_path), str(failure_path), str(data_yaml)])
save_progress(); print(progress["disease_label_summary"])
'''),
    cell("markdown", "train-md", "## Train And Quantitative Metrics\nTrain disease YOLO and save separate train/test pseudo metrics."),
    cell("code", "train", r'''
progress["stage"] = "train_disease_yolo"; save_progress()
model = YOLO("yolo11n-seg.pt")
train_result = model.train(data=str(PREPARED_DIR/"data.yaml"), epochs=25, imgsz=640, batch=16, workers=2, seed=SEED, device=DEVICE, project=str(RUNS_DIR), name="yolo11n_disease", exist_ok=True, patience=8, verbose=False)
train_dir = Path(getattr(train_result, "save_dir", RUNS_DIR/"yolo11n_disease"))
best_disease = WORKING_DIR/"best_disease_model.pt"
if (train_dir/"weights"/"best.pt").exists(): shutil.copy2(train_dir/"weights"/"best.pt", best_disease)
trained = YOLO(str(best_disease if best_disease.exists() else train_dir/"weights"/"last.pt"))
metric_outputs = {}
for split in ["train","test"]:
    m = trained.val(data=str(PREPARED_DIR/"data.yaml"), split=split, imgsz=640, batch=16, device=DEVICE, project=str(RUNS_DIR), name=f"eval_{split}", exist_ok=True, verbose=False)
    vals = metrics_from_results(m)
    (QUANT_DIR/f"disease_{split}_pseudo_metrics.json").write_text(json.dumps(vals, indent=2), encoding="utf-8")
    pd.DataFrame([vals]).to_csv(QUANT_DIR/f"disease_{split}_pseudo_metrics.csv", index=False)
    metric_outputs[split] = vals
progress["train_test_pseudo_metrics"] = metric_outputs
progress["artifacts"].extend([str(best_disease), str(train_dir)])
save_progress(); print(metric_outputs)
'''),
    cell("markdown", "qual-md", "## Qualitative Results\nSave unseen holdout overlays and a test sample."),
    cell("code", "qual", r'''
progress["stage"] = "qualitative_disease_results"; save_progress()
trained = YOLO(str(WORKING_DIR/"best_disease_model.pt"))
holdout_rows = []
for rec in holdout_df.itertuples(index=False):
    p = Path(rec.path); rgb = read_rgb(p)
    leaf_mask, leaf_source, leaf_error = leaf_mask_for_image(p, rgb)
    isolated = isolate_leaf(rgb, leaf_mask)
    pseudo, weak = disease_mask_from_leaf(isolated, leaf_mask, "healthy" in rec.original_filename.lower())
    pred = trained.predict(source=isolated, imgsz=640, device=DEVICE, conf=0.25, verbose=False)[0]
    pred_mask = result_to_mask(pred, isolated.shape[:2])
    panel = side_by_side([rgb, isolated, overlay_mask(isolated, pseudo, (255,180,40)), overlay_mask(isolated, pred_mask, (80,140,255))])
    out_path = QUAL_HOLDOUT_DIR/"by_category"/rec.category/f"{Path(rec.holdout_filename).stem}_orig_isolated_pseudo_pred.jpg"
    write_rgb(out_path, panel)
    holdout_rows.append({"category":rec.category,"holdout_filename":rec.holdout_filename,"original_filename":rec.original_filename,"leaf_mask_source":leaf_source,"pseudo_disease_area_ratio":float((pseudo>0).mean()),"pred_disease_area_ratio":float((pred_mask>0).mean()),"pseudo_disease_iou_unseen_holdout":mask_iou(pseudo,pred_mask),"qualitative_path":str(out_path)})
holdout_csv = WORKING_DIR/"holdout_disease_qualitative_metrics.csv"
pd.DataFrame(holdout_rows).to_csv(holdout_csv, index=False)
for row in disease_df[disease_df["split"]=="test"].head(100).itertuples(index=False):
    rgb = read_rgb(row.image_path)
    pseudo = label_file_to_mask(row.label_path, rgb.shape[:2])
    pred = trained.predict(source=rgb, imgsz=640, device=DEVICE, conf=0.25, verbose=False)[0]
    pred_mask = result_to_mask(pred, rgb.shape[:2])
    write_rgb(QUAL_TEST_DIR/f"{row.row_id}_pseudo_pred.jpg", side_by_side([rgb, overlay_mask(rgb,pseudo,(255,180,40)), overlay_mask(rgb,pred_mask,(80,140,255))]))
progress["unseen_holdout_disease_results"] = {"count":int(len(holdout_rows)),"mean_pseudo_iou_for_reference_only":float(pd.DataFrame(holdout_rows)["pseudo_disease_iou_unseen_holdout"].mean()),"note":"Holdout is unseen and qualitative-only; this IoU is reference only."}
progress["artifacts"].append(str(holdout_csv))
save_progress(); print(progress["unseen_holdout_disease_results"])
'''),
    cell("markdown", "final-md", "## Finalize\nZip deliverables and remove large folders from published outputs."),
    cell("code", "finalize", r'''
progress["stage"] = "finalize"; save_progress()
for z in [zip_dir(PREPARED_DIR, WORKING_DIR/"prepared_disease_yolo_dataset.zip"), zip_dir(INPUTS_PREVIEW_DIR, WORKING_DIR/"disease_inputs_preview.zip"), zip_dir(PSEUDO_PREVIEW_DIR, WORKING_DIR/"disease_pseudo_labels_preview.zip"), zip_dir(QUAL_HOLDOUT_DIR, WORKING_DIR/"qualitative_disease_holdout.zip"), zip_dir(QUAL_TEST_DIR, WORKING_DIR/"qualitative_disease_results.zip"), zip_dir(QUANT_DIR, WORKING_DIR/"quantitative_metrics.zip")]:
    progress["artifacts"].append(str(z))
for bulky in [PREPARED_DIR, INPUTS_PREVIEW_DIR, PSEUDO_PREVIEW_DIR, FAILURES_DIR, QUAL_HOLDOUT_DIR, QUAL_TEST_DIR, RUNS_DIR]:
    if bulky.exists(): shutil.rmtree(bulky)
progress["status"], progress["stage"] = "complete", "done"
progress["artifacts"] = sorted(set(progress["artifacts"]))
save_progress()
print(json.dumps({"status":progress["status"],"stage":progress["stage"],"artifact_count":len(progress["artifacts"]),"summary_path":str(SUMMARY_PATH)}, indent=2))
'''),
]


write_kernel("v5", {
    "id": "jeffreyamc/cv-pc5-v5-leaf-sam-yolo-seg",
    "title": "cv-pc5-v5-leaf-sam-yolo-seg",
    "code_file": "main.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": ["emmarex/plantdisease", "jeffreyamc/cv-pc5-v3-plantvillage-segmentation-holdout"],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": []
}, v5_cells)

write_kernel("v6", {
    "id": "jeffreyamc/cv-pc5-v6-disease-yolo-seg",
    "title": "cv-pc5-v6-disease-yolo-seg",
    "code_file": "main.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": ["emmarex/plantdisease", "jeffreyamc/cv-pc5-v3-plantvillage-segmentation-holdout"],
    "competition_sources": [],
    "kernel_sources": ["jeffreyamc/cv-pc5-v5-leaf-sam-yolo-seg"],
    "model_sources": []
}, v6_cells)

print("generated v5/v6 notebooks")
