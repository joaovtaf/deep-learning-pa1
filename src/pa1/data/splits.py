"""Split treino/val/teste estratificado por modalidade (exigencia da secao 2 do PA).

O DSB2018 nao vem com a coluna de modalidade, entao a gente constroi um proxy:
descreve cada imagem com estatistica de cor simples e roda k-means. Na pratica os
clusters separam bem fluorescencia (fundo quase preto, nucleo claro, imagem quase
cinza), brightfield (fundo claro) e histologia (roxo/rosa, saturacao alta). Depois
o split e feito dentro de cada cluster, entao as tres modalidades aparecem nas
mesmas proporcoes nos tres conjuntos.

Se o split fosse aleatorio direto, uma modalidade rara poderia cair quase toda no
treino e o teste ficaria facil, ou o contrario.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def list_image_ids(root) -> list[str]:
    root = Path(root)
    return sorted(p.name for p in root.iterdir() if (p / "images").is_dir())


def read_image(root, image_id) -> np.ndarray:
    """RGB uint8. O DSB2018 salva PNG RGBA, o canal alfa nao serve pra nada aqui."""
    path = Path(root) / image_id / "images" / f"{image_id}.png"
    return np.array(Image.open(path).convert("RGB"))


def image_descriptor(img: np.ndarray) -> np.ndarray:
    """Vetor curto de cor/contraste que serve de proxy pra modalidade."""
    x = img.astype(np.float32) / 255.0
    gray = x.mean(axis=2)
    mx, mn = x.max(axis=2), x.min(axis=2)
    saturation = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return np.array(
        [
            gray.mean(),
            gray.std(),
            saturation.mean(),
            np.median(gray),
            float((gray < 0.25).mean()),   # fundo escuro = fluorescencia
            float((gray > 0.75).mean()),   # fundo claro = brightfield
            x[..., 0].mean() - x[..., 1].mean(),
            x[..., 2].mean() - x[..., 1].mean(),
        ],
        dtype=np.float32,
    )


def modality_clusters(root, n_clusters: int = 4, seed: int = 0, cache: str | None = "modality.json"):
    """Devolve (ids, rotulo de cluster por id). Cacheia porque abrir 670 PNG demora."""
    root = Path(root)
    ids = list_image_ids(root)
    cache_path = root.parent / cache if cache else None

    if cache_path and cache_path.exists():
        saved = json.loads(cache_path.read_text())
        if saved.get("ids") == ids and saved.get("n_clusters") == n_clusters and saved.get("seed") == seed:
            return ids, np.array(saved["clusters"], dtype=int)

    feats = np.stack([image_descriptor(read_image(root, i)) for i in ids])
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(feats)
    clusters = km.labels_.astype(int)

    if cache_path:
        cache_path.write_text(
            json.dumps({"ids": ids, "n_clusters": n_clusters, "seed": seed, "clusters": clusters.tolist()})
        )
    return ids, clusters


def stratified_split(root, fractions=(0.7, 0.15, 0.15), n_clusters: int = 4, seed: int = 0) -> dict[str, list[str]]:
    ids, clusters = modality_clusters(root, n_clusters=n_clusters, seed=seed)
    ids = np.array(ids)
    rng = np.random.default_rng(seed)

    out = {"train": [], "val": [], "test": []}
    f_train, f_val, _ = fractions
    for c in range(int(clusters.max()) + 1):
        group = ids[clusters == c]
        group = group[rng.permutation(len(group))]
        n = len(group)
        n_train = int(round(f_train * n))
        n_val = int(round(f_val * n))
        # cluster minusculo nao pode deixar val ou teste vazio
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_val = min(max(n_val, 1), n - n_train - 1)
        out["train"] += list(group[:n_train])
        out["val"] += list(group[n_train : n_train + n_val])
        out["test"] += list(group[n_train + n_val :])

    return {k: sorted(v) for k, v in out.items()}
