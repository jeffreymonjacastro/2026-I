import csv
import json
import math
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DATASET_SLUG = "plantdisease"
MAX_PER_CATEGORY = 300
SCORE_MAX_SIDE = 256
CONTACT_THUMB_SIZE = 160
CONTACT_COLUMNS = 20
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CATEGORIES = [
    "healthy_leaves",
    "small_diseased_regions",
    "large_diseased_regions",
    "simple_backgrounds",
    "complex_backgrounds",
    "multiple_leaves",
    "partial_occlusions",
]


def resolve_paths():
    input_root = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path.cwd()
    output_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd().parent / "outputs"
    dataset_root = input_root / DATASET_SLUG
    scan_root = dataset_root / "PlantVillage"

    if not scan_root.exists():
        candidates = []
        candidates.extend(sorted(input_root.glob("*/PlantVillage")))
        candidates.extend(sorted(input_root.rglob("PlantVillage")))
        candidates = [candidate for candidate in candidates if candidate.is_dir()]
        if candidates:
            scan_root = candidates[0]
            dataset_root = scan_root.parent
        else:
            available_inputs = []
            if input_root.exists():
                available_inputs = [path.name for path in sorted(input_root.iterdir())]
            raise FileNotFoundError(
                "Expected PlantVillage directory not found. "
                f"Tried {dataset_root / 'PlantVillage'} and recursive search under {input_root}. "
                f"Available input folders: {available_inputs}"
            )

    selected_dir = output_dir / "selected_dataset"
    sheets_dir = output_dir / "review_contact_sheets"
    selected_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        (selected_dir / category).mkdir(parents=True, exist_ok=True)

    return input_root, dataset_root, scan_root, output_dir, selected_dir, sheets_dir


def slugify(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._")
    return value or "image"


def image_files(scan_root):
    paths = []
    for path in sorted(scan_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(path)
    return paths


def resize_for_scoring(bgr):
    height, width = bgr.shape[:2]
    max_side = max(height, width)
    if max_side <= SCORE_MAX_SIDE:
        return bgr.copy(), 1.0
    scale = SCORE_MAX_SIDE / max_side
    resized = cv2.resize(bgr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def clean_mask(mask, min_area_ratio=0.004):
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel3)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel5)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), 8)
    image_area = mask_u8.shape[0] * mask_u8.shape[1]
    min_area = max(20, int(image_area * min_area_ratio))
    cleaned = np.zeros_like(mask_u8, dtype=np.uint8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned > 0


def leaf_mask_from_image(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    exg = 2 * green - red - blue
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, exg_mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    green_yellow = (hue >= 18) & (hue <= 95) & (sat >= 25) & (val >= 35)
    strong_green = (exg > np.percentile(exg, 60)) & (sat >= 20)
    mask = (exg_mask > 0) | green_yellow | strong_green
    return clean_mask(mask, min_area_ratio=0.006)


def connected_component_stats(mask, min_component_ratio=0.02):
    image_area = mask.shape[0] * mask.shape[1]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    min_area = max(35, int(image_area * min_component_ratio))
    component_areas = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            component_areas.append(area)
    component_areas.sort(reverse=True)
    largest = component_areas[0] if component_areas else 0
    return len(component_areas), largest, component_areas


def disease_mask_from_image(bgr, leaf_mask):
    if int(leaf_mask.sum()) == 0:
        return np.zeros(leaf_mask.shape, dtype=bool)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    light = lab[:, :, 0]
    a_chan = lab[:, :, 1].astype(np.float32)
    b_chan = lab[:, :, 2].astype(np.float32)

    leaf_a = a_chan[leaf_mask]
    leaf_b = b_chan[leaf_mask]
    med_a = float(np.median(leaf_a))
    med_b = float(np.median(leaf_b))
    color_dist = np.sqrt((a_chan - med_a) ** 2 + (b_chan - med_b) ** 2)
    leaf_dist = color_dist[leaf_mask]
    dist_threshold = max(18.0, float(np.percentile(leaf_dist, 82)))

    brown = ((hue <= 24) | (hue >= 165)) & (sat >= 35) & (val <= 205)
    yellow = (hue >= 20) & (hue <= 42) & (sat >= 45) & (val >= 80)
    dark = (val <= 78) & (sat >= 18)
    low_light = light <= max(35, np.percentile(light[leaf_mask], 12))
    outlier = color_dist >= dist_threshold

    raw = leaf_mask & ((brown & outlier) | (yellow & outlier) | dark | (low_light & outlier))
    return clean_mask(raw, min_area_ratio=0.0015)


def component_area_ratios(mask, base_area):
    if base_area <= 0:
        return 0.0, 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, num_labels)]
    areas = [area for area in areas if area >= 8]
    largest = max(areas) if areas else 0
    return largest / base_area, len(areas)


def background_metrics(bgr, leaf_mask):
    background = ~leaf_mask
    if int(background.sum()) < 20:
        return 0.0, 0.0, 0.0

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bg_pixels = rgb[background]
    color_std = float(np.mean(np.std(bg_pixels.astype(np.float32), axis=0)))

    edges = cv2.Canny(gray, 70, 150)
    edge_density = float((edges[background] > 0).mean())

    hist = np.bincount(gray[background].ravel(), minlength=256).astype(np.float64)
    hist = hist / max(hist.sum(), 1.0)
    nonzero = hist[hist > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())
    return color_std, edge_density, entropy


def contour_metrics(leaf_mask):
    mask_u8 = (leaf_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, 0.0, 0.0

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = max(float(w * h), 1.0)
    bbox_fill_ratio = area / bbox_area

    hull = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    solidity = area / hull_area
    convexity_defect_score = max(0.0, min(1.0, 1.0 - solidity))

    image_h, image_w = leaf_mask.shape[:2]
    border_width = max(2, int(round(min(image_h, image_w) * 0.02)))
    border = np.zeros_like(leaf_mask, dtype=bool)
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    border_leaf = int((leaf_mask & border).sum())
    leaf_area = max(int(leaf_mask.sum()), 1)
    touches_edge = x <= border_width or y <= border_width or (x + w) >= image_w - border_width or (y + h) >= image_h - border_width
    border_touch_ratio = min(1.0, border_leaf / max(leaf_area * 0.035, 1.0))
    if touches_edge:
        border_touch_ratio = max(border_touch_ratio, 0.35)

    occlusion_score = (
        0.45 * border_touch_ratio
        + 0.35 * convexity_defect_score
        + 0.20 * max(0.0, 1.0 - bbox_fill_ratio)
    )
    return border_touch_ratio, convexity_defect_score, bbox_fill_ratio, occlusion_score


def analyze_image(path, scan_root):
    bgr_original = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr_original is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    bgr, _ = resize_for_scoring(bgr_original)
    height, width = bgr.shape[:2]
    image_area = height * width

    leaf_mask = leaf_mask_from_image(bgr)
    leaf_area = int(leaf_mask.sum())
    leaf_area_ratio = leaf_area / max(image_area, 1)
    leaf_component_count, largest_leaf_component, _ = connected_component_stats(leaf_mask)
    largest_leaf_component_ratio = largest_leaf_component / max(image_area, 1)

    disease_mask = disease_mask_from_image(bgr, leaf_mask)
    disease_pixels = int(disease_mask.sum())
    disease_ratio = disease_pixels / max(leaf_area, 1)
    largest_disease_component_ratio, disease_component_count = component_area_ratios(disease_mask, leaf_area)

    background_color_std, background_edge_density, background_entropy = background_metrics(bgr, leaf_mask)
    border_touch_ratio, convexity_defect_score, bbox_fill_ratio, occlusion_score = contour_metrics(leaf_mask)

    rel_path = path.relative_to(scan_root).as_posix()
    source_class = path.parent.name
    is_healthy_class = "healthy" in source_class.lower()

    return {
        "source_class": source_class,
        "source_file_name": path.name,
        "source_relative_path": rel_path,
        "source_suffix": path.suffix.lower(),
        "source_size_bytes": path.stat().st_size,
        "image_width": int(bgr_original.shape[1]),
        "image_height": int(bgr_original.shape[0]),
        "is_healthy_class": is_healthy_class,
        "leaf_area_ratio": leaf_area_ratio,
        "leaf_component_count": leaf_component_count,
        "largest_leaf_component_ratio": largest_leaf_component_ratio,
        "leaf_bbox_fill_ratio": bbox_fill_ratio,
        "disease_ratio": disease_ratio,
        "largest_disease_component_ratio": largest_disease_component_ratio,
        "disease_component_count": disease_component_count,
        "background_color_std": background_color_std,
        "background_edge_density": background_edge_density,
        "background_entropy": background_entropy,
        "border_touch_ratio": border_touch_ratio,
        "convexity_defect_score": convexity_defect_score,
        "bbox_fill_ratio": bbox_fill_ratio,
        "occlusion_score": occlusion_score,
    }


def normalize_metric(records, key):
    values = np.array([float(record[key]) for record in records], dtype=np.float32)
    if values.size == 0:
        return {}
    low = float(np.percentile(values, 2))
    high = float(np.percentile(values, 98))
    denom = max(high - low, 1e-6)
    return {
        record["source_relative_path"]: max(0.0, min(1.0, (float(record[key]) - low) / denom))
        for record in records
    }


def add_composite_scores(records):
    norm_color = normalize_metric(records, "background_color_std")
    norm_edges = normalize_metric(records, "background_edge_density")
    norm_entropy = normalize_metric(records, "background_entropy")

    for record in records:
        rel_path = record["source_relative_path"]
        background_complexity = (
            0.35 * norm_color[rel_path]
            + 0.35 * norm_edges[rel_path]
            + 0.30 * norm_entropy[rel_path]
        )
        record["background_complexity_score"] = background_complexity
        record["simple_background_score"] = 1.0 - background_complexity
        record["healthy_score"] = (
            0.45 * record["leaf_area_ratio"]
            + 0.25 * record["leaf_bbox_fill_ratio"]
            + 0.20 * record["simple_background_score"]
            + 0.10 * max(0.0, 1.0 - record["disease_ratio"])
        )
        record["small_disease_score"] = (
            1.0 - abs(record["disease_ratio"] - 0.06) / 0.06
            + 0.15 * min(record["disease_component_count"], 6)
        )
        record["large_disease_score"] = (
            0.70 * record["disease_ratio"]
            + 0.30 * record["largest_disease_component_ratio"]
        )
        record["multiple_leaves_score"] = (
            record["leaf_component_count"]
            + 0.25 * record["largest_leaf_component_ratio"]
        )


def diverse_take(candidates, sort_key, reverse=True, limit=MAX_PER_CATEGORY):
    grouped = defaultdict(list)
    for record in sorted(candidates, key=lambda item: item[sort_key], reverse=reverse):
        grouped[record["source_class"]].append(record)

    selected = []
    seen = set()
    while len(selected) < limit:
        progressed = False
        for source_class in sorted(grouped):
            if grouped[source_class]:
                record = grouped[source_class].pop(0)
                rel_path = record["source_relative_path"]
                if rel_path not in seen:
                    selected.append(record)
                    seen.add(rel_path)
                    progressed = True
                    if len(selected) >= limit:
                        break
        if not progressed:
            break
    return selected


def select_categories(records):
    healthy = [
        record for record in records
        if record["is_healthy_class"] and record["leaf_area_ratio"] >= 0.08 and record["disease_ratio"] <= 0.18
    ]
    diseased = [
        record for record in records
        if not record["is_healthy_class"] and record["leaf_area_ratio"] >= 0.06
    ]
    small_diseased = [
        record for record in diseased
        if 0.01 <= record["disease_ratio"] <= 0.12 and record["disease_component_count"] >= 1
    ]
    large_diseased = [
        record for record in diseased
        if record["disease_ratio"] >= 0.25 or record["largest_disease_component_ratio"] >= 0.15
    ]
    usable_leaf = [record for record in records if record["leaf_area_ratio"] >= 0.06]
    multiple_leaves = [
        record for record in usable_leaf
        if record["leaf_component_count"] >= 2
    ]
    partial_occlusions = [
        record for record in usable_leaf
        if record["occlusion_score"] >= 0.20
    ]

    return {
        "healthy_leaves": diverse_take(healthy, "healthy_score", reverse=True),
        "small_diseased_regions": diverse_take(small_diseased, "small_disease_score", reverse=True),
        "large_diseased_regions": diverse_take(large_diseased, "large_disease_score", reverse=True),
        "simple_backgrounds": diverse_take(usable_leaf, "simple_background_score", reverse=True),
        "complex_backgrounds": diverse_take(usable_leaf, "background_complexity_score", reverse=True),
        "multiple_leaves": diverse_take(multiple_leaves, "multiple_leaves_score", reverse=True),
        "partial_occlusions": diverse_take(partial_occlusions, "occlusion_score", reverse=True),
    }


def copy_selected(scan_root, selected_dir, selections):
    manifest_rows = []
    assignments = defaultdict(list)

    for category, records in selections.items():
        category_dir = selected_dir / category
        for rank, record in enumerate(records, start=1):
            source_path = scan_root / record["source_relative_path"]
            output_name = (
                f"{rank:03d}__{slugify(record['source_class'])}__"
                f"{slugify(Path(record['source_file_name']).stem)}{record['source_suffix']}"
            )
            output_path = category_dir / output_name
            shutil.copy2(source_path, output_path)
            assignments[record["source_relative_path"]].append(category)
            manifest_rows.append(
                {
                    "category": category,
                    "rank": rank,
                    "source_class": record["source_class"],
                    "source_relative_path": record["source_relative_path"],
                    "output_relative_path": output_path.relative_to(selected_dir.parent).as_posix(),
                    "leaf_area_ratio": record["leaf_area_ratio"],
                    "disease_ratio": record["disease_ratio"],
                    "background_complexity_score": record["background_complexity_score"],
                    "occlusion_score": record["occlusion_score"],
                }
            )

    return manifest_rows, assignments


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_contact_sheet(category, rows, selected_dir, sheets_dir):
    if not rows:
        return None
    thumb_w = CONTACT_THUMB_SIZE
    thumb_h = CONTACT_THUMB_SIZE
    label_h = 34
    columns = min(CONTACT_COLUMNS, max(1, len(rows)))
    rows_count = math.ceil(len(rows) / columns)
    sheet = Image.new("RGB", (columns * thumb_w, rows_count * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, row in enumerate(rows):
        x = (index % columns) * thumb_w
        y = (index // columns) * (thumb_h + label_h)
        image_path = selected_dir.parent / row["output_relative_path"]
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                offset_x = x + (thumb_w - image.width) // 2
                offset_y = y + (thumb_h - image.height) // 2
                sheet.paste(image, (offset_x, offset_y))
        except Exception as exc:
            draw.text((x + 4, y + 4), f"read error: {exc}", fill=(180, 0, 0), font=font)

        label = f"{row['rank']:03d} {row['source_class']}"
        draw.text((x + 4, y + thumb_h + 3), label[:28], fill=(0, 0, 0), font=font)
        draw.text((x + 4, y + thumb_h + 17), f"d={float(row['disease_ratio']):.2f} o={float(row['occlusion_score']):.2f}", fill=(70, 70, 70), font=font)

    output_path = sheets_dir / f"{category}.jpg"
    sheet.save(output_path, quality=92)
    return output_path


def zip_directory(source_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir.parent).as_posix())


def runtime_summary():
    summary = {
        "gpu_requested": "NvidiaTeslaT4",
        "cuda_available": False,
        "cuda_device_name": None,
        "opencv_version": cv2.__version__,
    }
    try:
        import torch

        summary["cuda_available"] = bool(torch.cuda.is_available())
        if summary["cuda_available"]:
            summary["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        summary["torch_probe_error"] = str(exc)
    return summary


def main():
    input_root, dataset_root, scan_root, output_dir, selected_dir, sheets_dir = resolve_paths()
    print(f"Input root: {input_root}")
    print(f"Dataset root: {dataset_root}")
    print(f"Scan root: {scan_root}")
    print(f"Output dir: {output_dir}")
    print(f"GPU requested: NvidiaTeslaT4")

    paths = image_files(scan_root)
    print(f"Image files found: {len(paths)}")

    records = []
    errors = []
    for index, path in enumerate(paths, start=1):
        try:
            records.append(analyze_image(path, scan_root))
        except Exception as exc:
            errors.append({"path": path.relative_to(scan_root).as_posix(), "error": str(exc)})
        if index % 1000 == 0:
            print(f"Scored {index}/{len(paths)} images")

    add_composite_scores(records)
    selections = select_categories(records)
    manifest_rows, assignments = copy_selected(scan_root, selected_dir, selections)

    for record in records:
        record["assigned_categories"] = "|".join(assignments.get(record["source_relative_path"], []))

    manifest_fields = [
        "category",
        "rank",
        "source_class",
        "source_relative_path",
        "output_relative_path",
        "leaf_area_ratio",
        "disease_ratio",
        "background_complexity_score",
        "occlusion_score",
    ]
    score_fields = [
        "source_class",
        "source_file_name",
        "source_relative_path",
        "assigned_categories",
        "image_width",
        "image_height",
        "is_healthy_class",
        "leaf_area_ratio",
        "leaf_component_count",
        "largest_leaf_component_ratio",
        "leaf_bbox_fill_ratio",
        "disease_ratio",
        "largest_disease_component_ratio",
        "disease_component_count",
        "background_color_std",
        "background_edge_density",
        "background_entropy",
        "background_complexity_score",
        "simple_background_score",
        "border_touch_ratio",
        "convexity_defect_score",
        "bbox_fill_ratio",
        "occlusion_score",
        "healthy_score",
        "small_disease_score",
        "large_disease_score",
        "multiple_leaves_score",
    ]
    write_csv(output_dir / "selected_dataset_manifest.csv", manifest_rows, manifest_fields)
    write_csv(output_dir / "heuristic_scores.csv", records, score_fields)

    sheet_paths = {}
    rows_by_category = defaultdict(list)
    for row in manifest_rows:
        rows_by_category[row["category"]].append(row)
    for category in CATEGORIES:
        sheet_path = create_contact_sheet(category, rows_by_category[category], selected_dir, sheets_dir)
        if sheet_path is not None:
            sheet_paths[category] = sheet_path.relative_to(output_dir).as_posix()

    selected_zip = output_dir / "selected_dataset.zip"
    sheets_zip = output_dir / "review_contact_sheets.zip"
    zip_directory(selected_dir, selected_zip)
    zip_directory(sheets_dir, sheets_zip)

    class_counts = defaultdict(int)
    for record in records:
        class_counts[record["source_class"]] += 1

    category_summary = {}
    for category in CATEGORIES:
        category_rows = rows_by_category[category]
        per_class = defaultdict(int)
        for row in category_rows:
            per_class[row["source_class"]] += 1
        category_summary[category] = {
            "selected_count": len(category_rows),
            "max_requested": MAX_PER_CATEGORY,
            "weak_category": len(category_rows) < MAX_PER_CATEGORY,
            "class_counts": dict(sorted(per_class.items())),
            "contact_sheet": sheet_paths.get(category),
        }

    summary = {
        "input_root": str(input_root),
        "dataset_slug": DATASET_SLUG,
        "dataset_root": str(dataset_root),
        "scan_root": str(scan_root),
        "output_dir": str(output_dir),
        "image_file_count": len(paths),
        "scored_image_count": len(records),
        "error_count": len(errors),
        "errors": errors[:50],
        "max_per_category": MAX_PER_CATEGORY,
        "allow_category_overlap": True,
        "class_counts": dict(sorted(class_counts.items())),
        "categories": category_summary,
        "runtime": runtime_summary(),
        "outputs": {
            "selected_dataset_dir": selected_dir.relative_to(output_dir).as_posix(),
            "selected_dataset_manifest": "selected_dataset_manifest.csv",
            "heuristic_scores": "heuristic_scores.csv",
            "selected_dataset_summary": "selected_dataset_summary.json",
            "selected_dataset_zip": selected_zip.name,
            "review_contact_sheets_dir": sheets_dir.relative_to(output_dir).as_posix(),
            "review_contact_sheets_zip": sheets_zip.name,
        },
        "heuristic_notes": {
            "healthy_vs_diseased": "Derived from PlantVillage class folder names. Visual color heuristics are only used to score lesion size after that split.",
            "complex_backgrounds": "PlantVillage backgrounds are often controlled. This category ranks the most visually complex backgrounds available in the dataset.",
            "partial_occlusions": "PlantVillage has no occlusion label. This category uses border contact, contour irregularity and low bbox fill as review candidates.",
        },
    }
    (output_dir / "selected_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Selected counts:")
    for category, data in category_summary.items():
        print(f"  {category}: {data['selected_count']}")
    print(f"Wrote: {output_dir / 'selected_dataset_manifest.csv'}")
    print(f"Wrote: {output_dir / 'heuristic_scores.csv'}")
    print(f"Wrote: {output_dir / 'selected_dataset_summary.json'}")
    print(f"Wrote: {selected_zip}")
    print(f"Wrote: {sheets_zip}")


if __name__ == "__main__":
    main()
