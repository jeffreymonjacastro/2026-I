"""
model.py
========
Arquitectura del clasificador de actividades de ovejas.

Diseño:
  - Backbone: ViT-B/16 (pretrained en ImageNet-21k via timm)
    → Solo se descongelan las últimas 4 transformer blocks + head
  - Temporal Pooling: promedio de los embeddings de los N frames
    (alternativa: atención temporal aprendible)
  - Clasificador: MLP con Dropout fuerte
  - Salida: 5 logits (clases 0-4)

Por qué ViT en lugar de CNN:
  - Mejor captura de relaciones globales (postura de la oveja)
  - Pretrained fuerte → menos datos necesarios
  - Los últimos blocks se fine-tunean para la tarea específica
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ──────────────────────────────────────────────
# Módulo de atención temporal (opcional, mejor que mean pool)
# ──────────────────────────────────────────────

class TemporalAttentionPool(nn.Module):
    """
    Aprende a ponderar la importancia de cada frame para la clasificación.
    En lugar de promediar los embeddings, usa un mecanismo de atención.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) — batch de secuencias de embeddings
        Returns:
            (B, D) — embedding del video
        """
        scores = self.attention(x)           # (B, N, 1)
        weights = F.softmax(scores, dim=1)   # (B, N, 1)
        pooled = (weights * x).sum(dim=1)    # (B, D)
        return pooled


# ──────────────────────────────────────────────
# Modelo principal
# ──────────────────────────────────────────────

class SheepActivityClassifier(nn.Module):
    """
    Clasificador de actividad de ovejas basado en ViT.

    Args:
        num_classes: número de clases (5)
        n_frames: número de frames por video (16)
        backbone_name: nombre del modelo timm
        dropout: dropout en el clasificador
        unfreeze_last_n_blocks: cuántos bloques del ViT descongelar
        use_temporal_attention: si True usa atención; si False usa mean pool
    """

    def __init__(
        self,
        num_classes: int = 5,
        n_frames: int = 16,
        backbone_name: str = "vit_base_patch16_224",
        dropout: float = 0.4,
        unfreeze_last_n_blocks: int = 4,
        use_temporal_attention: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n_frames = n_frames

        # ── Backbone ViT ──────────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,       # Sin cabeza de clasificación
            global_pool="token", # Usa el [CLS] token
        )
        embed_dim = self.backbone.embed_dim  # 768 para ViT-B

        # ── Congelar capas ────────────────────────────────────────────
        self._freeze_backbone(unfreeze_last_n_blocks)

        # ── Temporal Pooling ──────────────────────────────────────────
        self.use_temporal_attention = use_temporal_attention
        if use_temporal_attention:
            self.temporal_pool = TemporalAttentionPool(embed_dim)
        # Si no, se usa mean pool en forward

        # ── Clasificador MLP ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(embed_dim // 2, num_classes),
        )

        # Inicialización del clasificador
        self._init_classifier()

    def _freeze_backbone(self, unfreeze_last_n: int):
        """
        Congela todos los parámetros del backbone excepto los últimos N bloques.
        Esto previene overfitting y acelera el entrenamiento.
        """
        # Congelar TODO primero
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Descongelar los últimos N bloques transformer
        blocks = list(self.backbone.blocks)
        for block in blocks[-unfreeze_last_n:]:
            for param in block.parameters():
                param.requires_grad = True

        # Descongelar norm final
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        # Contar parámetros
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Parámetros totales: {total:,} | Entrenables: {trainable:,} ({100*trainable/total:.1f}%)")

    def _init_classifier(self):
        """Inicializa el clasificador con Xavier."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip: (B, N, C, H, W) — batch de clips

        Returns:
            logits: (B, num_classes)
        """
        B, N, C, H, W = clip.shape

        # Procesar todos los frames juntos en un batch aplanado
        frames = clip.view(B * N, C, H, W)           # (B*N, C, H, W)
        embeddings = self.backbone(frames)            # (B*N, D)
        embeddings = embeddings.view(B, N, -1)        # (B, N, D)

        # Temporal pooling
        if self.use_temporal_attention:
            video_emb = self.temporal_pool(embeddings)  # (B, D)
        else:
            video_emb = embeddings.mean(dim=1)          # (B, D)

        # Clasificación
        logits = self.classifier(video_emb)           # (B, num_classes)
        return logits


# ──────────────────────────────────────────────
# Loss con Label Smoothing
# ──────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy con label smoothing.
    Reduce la confianza excesiva del modelo en una clase → menos overfitting.
    """

    def __init__(self, smoothing: float = 0.1, num_classes: int = 5):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) — pueden ser logits o ya one-hot suaves (para Mixup)
            targets: (B,) int o (B, C) float (para Mixup)
        """
        log_probs = F.log_softmax(logits, dim=-1)

        if targets.dim() == 1:
            # Convertir a one-hot suave
            targets_onehot = torch.zeros_like(log_probs)
            targets_onehot.fill_(self.smoothing / (self.num_classes - 1))
            targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        else:
            # Ya viene en formato suave (Mixup)
            targets_onehot = targets
            # Aplicar label smoothing sobre los targets mixeados
            targets_onehot = targets_onehot * (1 - self.smoothing) + self.smoothing / self.num_classes

        loss = -(targets_onehot * log_probs).sum(dim=-1).mean()
        return loss


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

def build_model(
    num_classes: int = 5,
    n_frames: int = 16,
    dropout: float = 0.4,
    unfreeze_last_n_blocks: int = 4,
    device: str = "cuda",
) -> SheepActivityClassifier:
    model = SheepActivityClassifier(
        num_classes=num_classes,
        n_frames=n_frames,
        dropout=dropout,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        use_temporal_attention=True,
    )
    return model.to(device)
