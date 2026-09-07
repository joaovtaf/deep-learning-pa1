"""DSB2018 / BBBC038v1, opcao A do enunciado.

Cada imagem vem numa pasta com images/<id>.png e masks/*.png, uma mascara binaria
por nucleo, sem sobreposicao. A gente junta as mascaras num unico mapa de labels e
guarda em cache .npy, senao cada epoca reabriria dezenas de milhares de PNG e o
gargalo do treino viraria o disco em vez da rede.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import albumentations as A

from .splits import read_image
from .targets import distance_map, semantic_mask, three_class_mask


def load_instance_labels(root, image_id) -> np.ndarray:
    """Junta os PNG de masks/ num mapa de labels 1..n."""
    mask_dir = Path(root) / image_id / "masks"
    files = sorted(mask_dir.glob("*.png"))
    first = np.array(Image.open(files[0]))
    labels = np.zeros(first.shape[:2], dtype=np.int32)
    for i, f in enumerate(files, start=1):
        m = np.array(Image.open(f))
        if m.ndim == 3:
            m = m[..., 0]
        labels[m > 0] = i
    return labels


def cached_labels(root, image_id, cache_dir="data/dsb2018/labels_cache") -> np.ndarray:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{image_id}.npy"
    if path.exists():
        return np.load(path)
    labels = load_instance_labels(root, image_id)
    np.save(path, labels)
    return labels


def train_transform(crop: int):
    """Augmentation geometrica e fotometrica.

    Flip e rot90 sao de graca aqui porque nucleo nao tem orientacao canonica. O
    bloco de brilho/contraste existe porque o dataset mistura fluorescencia e
    histologia e o modelo precisa aguentar as duas.
    """
    return A.Compose(
        [
            A.PadIfNeeded(min_height=crop, min_width=crop, border_mode=0, fill=0, fill_mask=0),
            A.RandomCrop(height=crop, width=crop),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
            A.GaussNoise(p=0.2),
        ]
    )


def pad_to_multiple(img, labels, multiple: int = 32):
    """Val e teste rodam na imagem inteira, e o encoder tem stride 32."""
    h, w = labels.shape
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        labels = np.pad(labels, ((0, ph), (0, pw)), mode="constant")
    return img, labels


class DSB2018Dataset(Dataset):
    def __init__(
        self,
        image_ids,
        root="data/dsb2018/stage1_train",
        split="train",
        crop=128,
        boundary_thickness=2,
        cache_dir="data/dsb2018/labels_cache",
    ):
        self.image_ids = list(image_ids)
        self.root = Path(root)
        self.split = split
        self.crop = int(crop)
        self.boundary_thickness = int(boundary_thickness)
        self.cache_dir = cache_dir
        self.tf = train_transform(self.crop) if split == "train" else None

    def __len__(self):
        return len(self.image_ids)

    def raw(self, idx):
        """Imagem uint8 RGB e labels na resolucao original, sem padding nem crop."""
        image_id = self.image_ids[idx]
        return read_image(self.root, image_id), cached_labels(self.root, image_id, self.cache_dir)

    def __getitem__(self, idx):
        image, labels = self.raw(idx)

        if self.split == "train":
            out = self.tf(image=image, mask=labels)
            image, labels = out["image"], out["mask"].astype(np.int32)
        else:
            image, labels = pad_to_multiple(image, labels, 32)

        # os alvos da trilha A saem do label ja aumentado, nao antes: interpolar um
        # mapa de 3 classes no augment inventaria fronteira que nao existe
        x = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1) / 255.0)
        return {
            "image": x,
            "mask": torch.from_numpy(semantic_mask(labels)[None]),
            "labels": torch.from_numpy(labels.astype(np.int64)),
            "three": torch.from_numpy(three_class_mask(labels, self.boundary_thickness).astype(np.int64)),
            "dist": torch.from_numpy(distance_map(labels)[None]),
            "image_id": self.image_ids[idx],
            "orig_size": torch.tensor(self.raw(idx)[1].shape),
        }
