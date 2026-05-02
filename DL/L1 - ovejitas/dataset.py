"""
dataset.py
==========
Dataset PyTorch para la clasificación de actividad de ovejas.

Cada muestra = 1 video representado como una secuencia de N frames (ya recortados).
Salida del __getitem__: tensor (N, C, H, W) + label int.

Augmentations:
  - Train: flip horizontal, color jitter, random crop, rotación, gaussian blur, cutout
  - Val/Test: solo resize + normalize
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ──────────────────────────────────────────────
# Normalización ImageNet (usada por ViT/DINO)
# ──────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_SIZE = 224


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────

def get_train_transform() -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.1),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        T.RandomGrayscale(p=0.05),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),  # Cutout
    ])


def get_val_transform() -> T.Compose:
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ──────────────────────────────────────────────
# Augmentation temporal consistente
# (el mismo random crop/flip se aplica a TODOS los frames del mismo video)
# ──────────────────────────────────────────────

class TemporalConsistentTransform:
    """
    Aplica la misma transformación espacial aleatoria a todos los frames
    de un clip para mantener consistencia temporal.
    """
    def __init__(self, is_train: bool = True):
        self.is_train = is_train
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.random_erase = T.RandomErasing(p=0.3, scale=(0.02, 0.15))

    def __call__(self, frames: list[Image.Image]) -> torch.Tensor:
        """
        Args:
            frames: lista de PIL Images (N frames)
        Returns:
            Tensor (N, C, H, W)
        """
        if self.is_train:
            # Parámetros aleatorios compartidos para todos los frames
            i, j, h, w = T.RandomResizedCrop.get_params(
                frames[0],
                scale=(0.7, 1.0),
                ratio=(0.85, 1.15),
            )
            flip_h = random.random() < 0.5
            flip_v = random.random() < 0.1
            angle = random.uniform(-20, 20)
            # ColorJitter: parámetros aleatorios
            brightness = random.uniform(0.6, 1.4)
            contrast   = random.uniform(0.6, 1.4)
            saturation = random.uniform(0.7, 1.3)
            hue        = random.uniform(-0.1, 0.1)

        processed = []
        for frame in frames:
            if self.is_train:
                # Crop
                frame = TF.resized_crop(frame, i, j, h, w, (IMG_SIZE, IMG_SIZE))
                # Flip
                if flip_h:
                    frame = TF.hflip(frame)
                if flip_v:
                    frame = TF.vflip(frame)
                # Rotation
                frame = TF.rotate(frame, angle)
                # Color
                frame = TF.adjust_brightness(frame, brightness)
                frame = TF.adjust_contrast(frame, contrast)
                frame = TF.adjust_saturation(frame, saturation)
                frame = TF.adjust_hue(frame, hue)
            else:
                frame = frame.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

            t = self.to_tensor(frame)
            t = self.normalize(t)

            if self.is_train:
                t = self.random_erase(t)

            processed.append(t)

        return torch.stack(processed, dim=0)  # (N, C, H, W)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class SheepActivityDataset(Dataset):
    """
    Args:
        processed_dir: ruta a data/processed/train/ o data/processed/test/
        labels_df: DataFrame con columnas [Id, Target] (None para test)
        n_frames: número de frames por clip
        is_train: si True aplica augmentations fuertes
    """

    def __init__(
        self,
        processed_dir: str,
        labels_df: pd.DataFrame | None = None,
        n_frames: int = 16,
        is_train: bool = True,
    ):
        self.processed_dir = Path(processed_dir)
        self.n_frames = n_frames
        self.is_train = is_train
        self.transform = TemporalConsistentTransform(is_train=is_train)

        # Construir lista de samples
        if labels_df is not None:
            # Train / Val: filtrar solo los que tienen carpeta
            self.samples = []
            for _, row in labels_df.iterrows():
                vid_dir = self.processed_dir / str(row["Id"])
                if vid_dir.exists():
                    self.samples.append((str(row["Id"]), int(row["Target"])))
            print(f"[Dataset] {len(self.samples)} samples cargados (is_train={is_train})")
        else:
            # Test: todas las carpetas
            self.samples = [
                (d.name, -1)
                for d in sorted(self.processed_dir.iterdir())
                if d.is_dir()
            ]
            print(f"[Dataset] {len(self.samples)} videos de test")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frames(self, video_id: str) -> list[Image.Image]:
        vid_dir = self.processed_dir / video_id
        frame_files = sorted(vid_dir.glob("frame_*.png"))

        frames = []
        for ff in frame_files:
            img = Image.open(ff).convert("RGB")
            frames.append(img)

        # Si hay menos frames de los esperados, repetir el último
        while len(frames) < self.n_frames:
            frames.append(frames[-1] if frames else Image.new("RGB", (224, 224)))

        # Si hay más, samplear uniformemente
        if len(frames) > self.n_frames:
            indices = np.linspace(0, len(frames) - 1, self.n_frames, dtype=int)
            frames = [frames[i] for i in indices]

        return frames[:self.n_frames]

    def __getitem__(self, idx: int) -> dict:
        video_id, label = self.samples[idx]
        frames = self._load_frames(video_id)
        clip = self.transform(frames)  # (N, C, H, W)
        return {
            "clip": clip,
            "label": torch.tensor(label, dtype=torch.long),
            "video_id": video_id,
        }


# ──────────────────────────────────────────────
# Mixup
# ──────────────────────────────────────────────

def mixup_batch(
    clips: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 5,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aplica Mixup al batch.
    Returns: clips_mixed, labels_soft (one-hot mixeados)
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = clips.size(0)
    index = torch.randperm(batch_size, device=clips.device)

    clips_mixed = lam * clips + (1 - lam) * clips[index]

    # Etiquetas suaves
    labels_onehot = torch.zeros(batch_size, num_classes, device=clips.device)
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
    labels_mixed = lam * labels_onehot + (1 - lam) * labels_onehot[index]

    return clips_mixed, labels_mixed


# ──────────────────────────────────────────────
# Factory de DataLoaders
# ──────────────────────────────────────────────

def build_dataloaders(
    processed_dir: str,
    label_csv: str,
    n_frames: int = 16,
    batch_size: int = 8,
    val_fraction: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Crea DataLoaders de train y validación con split estratificado.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    df = pd.read_csv(label_csv)
    df.columns = ["Id", "Target"]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(sss.split(df["Id"], df["Target"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    train_ds = SheepActivityDataset(processed_dir, train_df, n_frames, is_train=True)
    val_ds   = SheepActivityDataset(processed_dir, val_df,   n_frames, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[DataLoader] Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader
