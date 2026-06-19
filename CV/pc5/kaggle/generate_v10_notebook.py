import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V9_NOTEBOOK = ROOT / "kaggle" / "v9" / "input" / "main.ipynb"


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


SETUP = notebook_cell_source(V9_NOTEBOOK, "setup")
HELPERS = notebook_cell_source(V9_NOTEBOOK, "helpers")
MULTICLASS_HELPERS = notebook_cell_source(V9_NOTEBOOK, "multiclass-helpers")


PATHS = r'''
INPUT_ROOT = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")
OUTPUT_DIR = WORKING_DIR / "selected_inference_outputs"
SUMMARY_PATH = WORKING_DIR / "run_summary.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_IMAGES = [
    "PlantVillage/Pepper__bell___Bacterial_spot/bbda7917-5169-48e8-b041-9ee38aabb29d___JR_B.Spot 3211.JPG",
    "PlantVillage/Pepper__bell___healthy/261f9a0f-eccc-41fb-be35-cabed2531059___JR_HL 8722.JPG",
    "PlantVillage/Potato___Early_blight/b2e21749-afe2-4392-8d5d-23088d37873c___RS_Early.B 9028.JPG",
    "PlantVillage/Potato___Late_blight/ec53a246-df50-44c1-a87c-c8296f25f20c___RS_LB 4717.JPG",
    "PlantVillage/Potato___healthy/2e0b8b4b-e900-408b-b760-730690bbd382___RS_HL 1901.JPG",
    "PlantVillage/Tomato_Bacterial_spot/4d02e3e4-c583-402a-adc0-b258c8a3384f___UF.GRC_BS_Lab Leaf 0287.JPG",
    "PlantVillage/Tomato_Late_blight/7a31833f-807e-4c54-951f-3c8c1d5b78d7___GHLB2 Leaf 8939.JPG",
    "PlantVillage/Tomato__Target_Spot/9f74980b-56b6-4eca-88f8-ca84e3232427___Com.G_TgS_FL 9697.JPG",
    "PlantVillage/Tomato__Tomato_YellowLeaf__Curl_Virus/1d284bff-229f-4fb3-89c6-afdaff9e6f3e___UF.GRC_YLCV_Lab 08577.JPG",
    "PlantVillage/Tomato__Tomato_mosaic_virus/39dc536a-03ba-4e34-9ade-9618715f4b96___PSU_CG 2106.JPG",
]

progress = {
    "status": "running",
    "stage": "start",
    "task": "v10 selected-image inference with v9 multiclass model",
    "model_source": "kernel source jeffreyamc/cv-pc5-v9-multiclass-leaf-disease-yolo-seg/best_multiclass_model.pt",
    "target_count": len(TARGET_IMAGES),
    "output_format": "side-by-side original | segmented overlay with leaf green and disease blue",
    "artifacts": [],
}

def save_progress():
    SUMMARY_PATH.write_text(json.dumps(progress, indent=2), encoding="utf-8")

save_progress()
'''


DISCOVER = r'''
progress["stage"] = "discover_inputs"; save_progress()
plant_root = find_plantvillage_root(INPUT_ROOT)
model_candidates = [p for p in INPUT_ROOT.glob("**/best_multiclass_model.pt")]
if not model_candidates:
    raise FileNotFoundError("best_multiclass_model.pt not found in Kaggle inputs. Check v9 kernel source.")
model_path = model_candidates[0]
model = YOLO(str(model_path))

def resolve_target(rel):
    rel = rel.replace("\\", "/")
    if rel.startswith("PlantVillage/"):
        sub = rel.split("/", 1)[1]
    else:
        sub = rel
    direct = plant_root / sub
    if direct.exists():
        return direct
    filename = Path(sub).name
    matches = list(plant_root.glob(f"**/{filename}"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for {rel}: {matches[:5]}")
    raise FileNotFoundError(f"Target image not found: {rel}")

resolved_targets = []
for rel in TARGET_IMAGES:
    path = resolve_target(rel)
    resolved_targets.append({"requested_path": rel, "resolved_path": str(path), "source_class": path.parent.name, "filename": path.name})

progress.update({
    "plant_root": str(plant_root),
    "model_path": str(model_path),
    "resolved_target_count": len(resolved_targets),
})
save_progress()
print({"plant_root": str(plant_root), "model_path": str(model_path), "targets": len(resolved_targets)})
'''


INFERENCE = r'''
progress["stage"] = "run_selected_inference"; save_progress()

rows = []
for idx, item in enumerate(resolved_targets, start=1):
    image_path = Path(item["resolved_path"])
    rgb = read_rgb(image_path)

    # The v9 model was trained on RGB leaf crops. For full-image outputs, infer on
    # the crop and paste the predicted masks back into the original image frame.
    leaf_mask, _, _ = leaf_tissue_shape_mask(rgb)
    bbox = bbox_from_mask(leaf_mask, pad_ratio=0.10)
    crop_rgb, crop_leaf = crop_rgb_mask(rgb, leaf_mask, bbox)
    pred = model.predict(source=crop_rgb, imgsz=640, device=DEVICE, conf=0.25, verbose=False)[0]
    crop_masks = class_result_masks(pred, crop_rgb.shape[:2])
    full_leaf_pred = paste_crop_mask(crop_masks[0], rgb.shape[:2], bbox)
    full_disease_pred = paste_crop_mask(crop_masks[1], rgb.shape[:2], bbox)

    segmented = overlay_multiclass(rgb, full_leaf_pred, full_disease_pred)
    panel = side_by_side([rgb, segmented])
    safe_class = normalize_image_id(item["source_class"])
    out_name = f"{idx:02d}__{safe_class}__{Path(item['filename']).stem}_original_segmented.jpg"
    out_path = OUTPUT_DIR / out_name
    write_rgb(out_path, panel, quality=95)

    rows.append({
        **item,
        "output_path": str(out_path),
        "bbox_for_inference": ",".join(str(int(v)) for v in bbox),
        "pred_leaf_area_ratio": float((full_leaf_pred > 0).mean()),
        "pred_disease_area_ratio": float((full_disease_pred > 0).mean()),
    })
    print({"done": idx, "file": item["filename"], "output": str(out_path)})

manifest = WORKING_DIR / "selected_inference_manifest.csv"
pd.DataFrame(rows).to_csv(manifest, index=False)
zip_path = zip_dir(OUTPUT_DIR, WORKING_DIR / "selected_inference_outputs.zip")
progress.update({
    "status": "complete",
    "stage": "done",
    "processed_count": len(rows),
    "manifest": str(manifest),
    "outputs_zip": str(zip_path),
})
progress["artifacts"].extend([str(manifest), str(zip_path), str(OUTPUT_DIR)])
save_progress()
print(json.dumps({"status": progress["status"], "processed_count": len(rows), "outputs_zip": str(zip_path)}, indent=2))
'''


cells = [
    md("intro", "# PC5 v10 - Selected Inference With v9 Multiclass Model\nRun inference only on the 10 requested PlantVillage images using `best_multiclass_model.pt` from v9. Output panels are exactly original image next to a combined segmentation overlay: green leaf and blue disease region."),
    md("setup-md", "## Setup\nInstall/check Ultralytics and configure GPU execution."),
    code("setup", SETUP),
    md("paths-md", "## Paths And Target Images\nCreate output folders and list exactly the requested dataset images."),
    code("paths", PATHS),
    md("helpers-md", "## Helpers\nUse v9 helpers for PlantVillage resolution, crop inference, mask pasting, and overlays."),
    code("helpers", HELPERS + "\n" + MULTICLASS_HELPERS),
    md("discover-md", "## Resolve Model And Images\nFind PlantVillage, load the v9 model, and resolve the 10 requested image paths."),
    code("discover", DISCOVER),
    md("infer-md", "## Inference\nRun inference on each crop and save side-by-side original plus combined segmentation overlay."),
    code("inference", INFERENCE),
]

metadata = {
    "id": "jeffreyamc/cv-pc5-v10-selected-multiclass-inference",
    "title": "cv-pc5-v10-selected-multiclass-inference",
    "kernel_type": "notebook",
    "code_file": "main.ipynb",
    "language": "python",
    "is_private": True,
    "enable_gpu": True,
    "machine_shape": "NvidiaTeslaT4",
    "enable_internet": True,
    "dataset_sources": ["emmarex/plantdisease"],
    "kernel_sources": ["jeffreyamc/cv-pc5-v9-multiclass-leaf-disease-yolo-seg"],
}

write_kernel("v10", cells, metadata)
print("generated v10 notebook")
