import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
LAYER_INDEX = 10
IMAGE_SIZE = 256
GRID_SIZE = 8
DINO_GRID_SIZE = 16
PATCHES_PER_IMAGE = 64

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def load_targets(path: Path) -> list[dict]:
    df = pd.read_csv(path)

    targets = []

    for row in df.itertuples(index=False):
        image_id = str(row.ID)
        patch_indices = [int(x) for x in str(row.target).split()]

        targets.append({
            "image_id": image_id,
            "patch_indices": patch_indices,
        })

    return targets


def index_images(root: Path) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"No existe la carpeta: {root}")

    mapping = {}

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image_id = path.stem

        if image_id in mapping:
            raise ValueError(
                f"Existen varias imágenes con el mismo ID: {image_id}"
            )

        mapping[image_id] = path

    return mapping


def load_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise RuntimeError(f"No se pudo abrir la imagen: {path}") from error

    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"{path.name} tiene tamaño {image.size}; "
            f"se requiere {IMAGE_SIZE}x{IMAGE_SIZE}."
        )

    return image


class DinoVQ:
    def __init__(
        self,
        codebook_path: Path,
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.dtype = (
            torch.float16
            if self.device.type == "cuda"
            else torch.float32
        )

        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=self.dtype,
        )
        self.model.eval()
        self.model.to(self.device)

        self.mean = torch.tensor(
            self.processor.image_mean,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)

        self.std = torch.tensor(
            self.processor.image_std,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)

        codebook = np.load(codebook_path).astype(np.float32)

        if codebook.ndim != 2:
            raise ValueError(
                f"codebook.npy debe ser bidimensional; shape={codebook.shape}"
            )

        norms = np.linalg.norm(codebook, axis=1, keepdims=True)

        if np.any(norms == 0):
            raise ValueError("El codebook contiene centros con norma cero.")

        codebook = codebook / norms

        self.codebook = torch.from_numpy(codebook).to(
            self.device,
            dtype=torch.float32,
        )

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        arrays = [
            np.asarray(image, dtype=np.float32) / 255.0
            for image in images
        ]

        pixels = torch.from_numpy(
            np.stack(arrays, axis=0)
        ).permute(0, 3, 1, 2)

        pixels = pixels.to(self.device, dtype=torch.float32)
        pixels = (pixels - self.mean) / self.std

        return pixels.to(dtype=self.dtype)

    @torch.inference_mode()
    def quantize(self, images: list[Image.Image]) -> np.ndarray:
        pixel_values = self.preprocess(images)

        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states

        if hidden_states is None or LAYER_INDEX >= len(hidden_states):
            raise RuntimeError(
                f"No se pudo obtener hidden_states[{LAYER_INDEX}]."
            )

        hidden = hidden_states[LAYER_INDEX]
        batch_size, sequence_length, embedding_dim = hidden.shape

        num_register_tokens = int(
            getattr(self.model.config, "num_register_tokens", 0) or 0
        )

        start = 1 + num_register_tokens
        spatial = hidden[:, start:, :]

        expected_tokens = DINO_GRID_SIZE * DINO_GRID_SIZE

        if spatial.shape[1] != expected_tokens:
            inferred_start = sequence_length - expected_tokens

            if inferred_start < 1:
                raise RuntimeError(
                    f"Se esperaban {expected_tokens} tokens espaciales, "
                    f"pero la secuencia tiene longitud {sequence_length}."
                )

            spatial = hidden[:, inferred_start:, :]

        spatial = spatial.reshape(
            batch_size,
            DINO_GRID_SIZE,
            DINO_GRID_SIZE,
            embedding_dim,
        )

        pooled = spatial.reshape(
            batch_size,
            GRID_SIZE,
            2,
            GRID_SIZE,
            2,
            embedding_dim,
        ).mean(dim=(2, 4))

        pooled = F.normalize(pooled.float(), p=2, dim=-1)
        pooled = pooled.reshape(batch_size, PATCHES_PER_IMAGE, embedding_dim)

        if embedding_dim != self.codebook.shape[1]:
            raise RuntimeError(
                f"Dimensión DINOv3={embedding_dim}, "
                f"dimensión codebook={self.codebook.shape[1]}."
            )

        similarities = pooled @ self.codebook.T
        codes = similarities.argmax(dim=-1)

        return codes.cpu().numpy().astype(np.int64)


def build_submission(
    images_root: Path,
    target_csv: Path,
    codebook_path: Path,
    output_csv: Path,
    batch_size: int,
    device: str,
) -> None:
    targets = load_targets(target_csv)
    image_paths = index_images(images_root)

    missing = [
        item["image_id"]
        for item in targets
        if item["patch_indices"] and item["image_id"] not in image_paths
    ]

    if missing:
        raise FileNotFoundError(
            "Faltan reconstrucciones para estas imágenes: "
            + ", ".join(missing[:10])
        )

    quantizer = DinoVQ(
        codebook_path=codebook_path,
        device=device,
    )

    records = []
    expected_rows = sum(len(x["patch_indices"]) for x in targets)

    for start in tqdm(
        range(0, len(targets), batch_size),
        desc="Generando submission",
    ):
        batch_info = targets[start:start + batch_size]

        nonempty = [
            item
            for item in batch_info
            if item["patch_indices"]
        ]

        if not nonempty:
            continue

        images = [
            load_image(image_paths[item["image_id"]])
            for item in nonempty
        ]

        try:
            codes = quantizer.quantize(images)
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "Memoria CUDA insuficiente. "
                "Ejecuta nuevamente con un --batch-size menor."
            ) from error

        for row_index, item in enumerate(nonempty):
            image_id = item["image_id"]

            for patch_idx in item["patch_indices"]:
                records.append({
                    "Id": f"{image_id}_{patch_idx}",
                    "Target": int(codes[row_index, patch_idx]),
                })

    submission = pd.DataFrame(records, columns=["Id", "Target"])

    if len(submission) != expected_rows:
        raise RuntimeError(
            f"Se esperaban {expected_rows} filas, "
            f"pero se generaron {len(submission)}."
        )

    if submission["Id"].duplicated().any():
        raise RuntimeError("Se generaron IDs duplicados.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_csv, index=False)

    print(f"Archivo generado: {output_csv}")
    print(f"Filas: {len(submission)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera submission.csv usando DINOv3 y el codebook."
    )

    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("generate"),
    )
    parser.add_argument(
        "--target-csv",
        type=Path,
        default=Path("target.csv"),
    )
    parser.add_argument(
        "--codebook-path",
        type=Path,
        default=Path("codebook.npy"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("submission.csv"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_submission(
        images_root=args.images_root,
        target_csv=args.target_csv,
        codebook_path=args.codebook_path,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()