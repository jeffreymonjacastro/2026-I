"""
preprocess.py
=============
Preprocesamiento de videos de ovejas:
  1. Reducción a 480p
  2. Extracción de N frames uniformemente distribuidos
  3. Detección de oveja con YOLOv8 (bounding box)
  4. Crop cuadrado centrado en la oveja + padding
  5. Guardado como imágenes PNG en una carpeta estructurada

Uso:
    python preprocess.py --video_dir data/train --output_dir data/processed --split train --label_csv data/train.csv
    python preprocess.py --video_dir data/test  --output_dir data/processed --split test
"""

import os
import cv2
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ──────────────────────────────────────────────
# Configuración global
# ──────────────────────────────────────────────
TARGET_HEIGHT = 480  # altura objetivo (mantiene aspect ratio)
NUM_FRAMES = 16  # frames extraídos por video
CROP_SIZE = 224  # tamaño final del crop (cuadrado)
YOLO_CONF = 0.25  # confianza mínima de detección
YOLO_CLASS_ANIMAL = None  # None = usar cualquier detección (la más grande)

# YOLOv8 detecta animales en la clase 16 (COCO), pero como son ovejas
# usaremos la detección más grande de la imagen si no hay clase específica.
COCO_SHEEP_CLASS = 18  # clase "sheep" en COCO


def load_yolo(model_name: str = "yolov8m.pt") -> YOLO:
    """Descarga y carga el modelo YOLOv8."""
    print(f"[YOLO] Cargando modelo {model_name}...")
    model = YOLO(model_name)  # se descarga automáticamente si no existe

    if torch.cuda.is_available():
        print("[INFO] ¡GPU detectada! Moviendo modelo a CUDA.")
        model.to("cuda")
    else:
        print("[WARN] No se detectó GPU. Usando CPU (será lento).")
    return model


def resize_frame(frame: np.ndarray, target_height: int = TARGET_HEIGHT) -> np.ndarray:
    """Redimensiona manteniendo aspect ratio."""
    h, w = frame.shape[:2]
    if h <= target_height:
        return frame
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)


def extract_uniform_frames(
    video_path: str, n_frames: int = NUM_FRAMES
) -> list[np.ndarray]:
    """Extrae N frames uniformemente distribuidos a lo largo del video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = resize_frame(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()

    # Si no se pudieron leer todos, duplicar el último frame
    while len(frames) < n_frames:
        frames.append(
            frames[-1]
            if frames
            else np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
        )

    return frames[:n_frames]


def get_sheep_bbox(model: YOLO, frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Devuelve el bounding box (x1, y1, x2, y2) de la oveja detectada.
    Estrategia:
      1. Buscar clase 'sheep' (COCO class 18)
      2. Si no hay, usar la detección de mayor área (asumimos que el sujeto principal es la oveja)
    """
    results = model(frame, verbose=False, conf=YOLO_CONF)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None

    cls_ids = boxes.cls.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    # Prioridad 1: clase sheep
    sheep_mask = cls_ids == COCO_SHEEP_CLASS
    if sheep_mask.any():
        sheep_boxes = xyxy[sheep_mask]
        sheep_confs = confs[sheep_mask]
        best = sheep_boxes[sheep_confs.argmax()]
        return tuple(best.astype(int))

    # Prioridad 2: detección de mayor área
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    best = xyxy[areas.argmax()]
    return tuple(best.astype(int))


def get_sheep_bboxes_batch(model: YOLO, frames: list[np.ndarray]) -> list[tuple | None]:
    """
    Devuelve una lista de bounding boxes (uno por cada frame).
    Procesa todos los frames a la vez usando Batch Inference.
    """
    # ¡Aquí ocurre la magia del batching! Pasamos la lista completa.
    results = model(frames, verbose=False, conf=YOLO_CONF)

    bboxes = []

    # YOLO nos devuelve una lista de resultados, iteramos sobre ella
    for res in results:
        boxes = res.boxes

        # Si en este frame en particular no detectó nada
        if boxes is None or len(boxes) == 0:
            bboxes.append(None)
            continue

        cls_ids = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        # Prioridad 1: clase sheep (COCO_SHEEP_CLASS = 18)
        sheep_mask = cls_ids == COCO_SHEEP_CLASS
        if sheep_mask.any():
            sheep_boxes = xyxy[sheep_mask]
            sheep_confs = confs[sheep_mask]
            best = sheep_boxes[sheep_confs.argmax()]
            bboxes.append(tuple(best.astype(int)))
        else:
            # Prioridad 2: detección de mayor área
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            best = xyxy[areas.argmax()]
            bboxes.append(tuple(best.astype(int)))

    return bboxes


def crop_sheep(
    frame: np.ndarray, bbox: tuple | None, crop_size: int = CROP_SIZE
) -> np.ndarray:
    """
    Recorta la oveja del frame con padding cuadrado.
    Si no hay bbox, retorna el frame completo redimensionado.
    """
    h, w = frame.shape[:2]

    if bbox is None:
        # Sin detección: resize central del frame completo
        return cv2.resize(frame, (crop_size, crop_size), interpolation=cv2.INTER_AREA)

    x1, y1, x2, y2 = bbox
    # Expansión del bbox un 15% para contexto
    pad_x = int((x2 - x1) * 0.15)
    pad_y = int((y2 - y1) * 0.15)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return cv2.resize(frame, (crop_size, crop_size), interpolation=cv2.INTER_AREA)

    # Hacer cuadrado con padding negro
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    y_off = (side - ch) // 2
    x_off = (side - cw) // 2
    square[y_off : y_off + ch, x_off : x_off + cw] = crop

    return cv2.resize(square, (crop_size, crop_size), interpolation=cv2.INTER_AREA)


def process_video(
    video_path: str,
    yolo_model: YOLO,
    output_dir: str,
    video_id: str,
    n_frames: int = NUM_FRAMES,
    crop_size: int = CROP_SIZE,
) -> bool:
    """
    Procesa un video completo y guarda los frames procesados.
    Detecta automáticamente si usar Batching (GPU) o procesamiento secuencial (CPU).
    """
    frames = extract_uniform_frames(video_path, n_frames)
    if not frames:
        print(f"[WARN] No se pudieron extraer frames de {video_path}")
        return False

    os.makedirs(output_dir, exist_ok=True)

    # 1. Decidir la estrategia dependiendo del Hardware
    if torch.cuda.is_available():
        # === ESTRATEGIA GPU: Procesamiento por Lotes (Batching) ===
        print("CUDAS")
        bboxes = get_sheep_bboxes_batch(yolo_model, frames)
    else:
        # === ESTRATEGIA CPU: Procesamiento Secuencial (Uno por uno) ===
        bboxes = []
        for frame in frames:
            bboxes.append(get_sheep_bbox(yolo_model, frame))

    # 2. Recortar y guardar las imágenes (Operaciones ligeras en CPU)
    # Usamos zip() para emparejar cada frame con su respectivo bounding box
    for i, (frame, bbox) in enumerate(zip(frames, bboxes)):
        crop = crop_sheep(frame, bbox, crop_size)

        # Guardar como PNG (sin pérdida)
        out_path = os.path.join(output_dir, f"frame_{i:02d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocesamiento de videos de ovejas")
    parser.add_argument(
        "--video_dir", required=True, help="Carpeta con los videos (.mov, .mp4, etc.)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Carpeta de salida para los frames procesados",
    )
    parser.add_argument(
        "--split", required=True, choices=["train", "test"], help="Split a procesar"
    )
    parser.add_argument(
        "--label_csv", default=None, help="CSV con etiquetas (solo para train)"
    )
    parser.add_argument(
        "--n_frames", type=int, default=NUM_FRAMES, help="Número de frames por video"
    )
    parser.add_argument("--yolo_model", default="yolov8m.pt", help="Modelo YOLO a usar")
    parser.add_argument("--video_ext", default=".mov", help="Extensión de los videos")
    args = parser.parse_args()

    yolo = load_yolo(args.yolo_model)

    video_dir = Path(args.video_dir)
    output_root = Path(args.output_dir) / args.split

    # Obtener lista de videos
    video_files = sorted(video_dir.glob(f"*{args.video_ext}"))
    if not video_files:
        video_files = sorted(video_dir.glob("*.mp4"))
    print(f"[INFO] Encontrados {len(video_files)} videos en {video_dir}")

    failed = []
    for vf in tqdm(video_files, desc=f"Procesando {args.split}"):
        video_id = vf.stem  # nombre sin extensión
        out_dir = output_root / video_id
        process_video(
            video_path=str(vf),
            yolo_model=yolo,
            output_dir=str(out_dir),
            video_id=video_id,
            n_frames=args.n_frames,
        )

    if failed:
        print(f"[WARN] {len(failed)} videos fallaron: {failed}")

    print(f"[DONE] Frames guardados en {output_root}")

    # Copiar CSV de etiquetas si aplica
    if args.split == "train" and args.label_csv:
        import shutil

        shutil.copy(args.label_csv, Path(args.output_dir) / "train_labels.csv")
        print(f"[INFO] CSV de etiquetas copiado.")


if __name__ == "__main__":
    main()
