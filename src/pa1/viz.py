"""Plots: mascara de instancia colorida, painel comparativo e grafico de densidade."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def colorize_labels(labels, seed: int = 0, background=(0, 0, 0)):
    """Cor aleatoria por instancia."""
    labels = np.asarray(labels)
    n = int(labels.max())
    rng = np.random.default_rng(seed)
    lut = rng.uniform(0.25, 1.0, size=(n + 1, 3))
    lut[0] = background
    return lut[labels]


def overlay_instances(image, labels, alpha: float = 0.5, seed: int = 0):
    """Instancias coloridas por cima da imagem, que tem que estar em [0,1]."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    colors = colorize_labels(labels, seed=seed)
    mask = (np.asarray(labels) > 0)[..., None]
    return np.clip(np.where(mask, (1 - alpha) * image + alpha * colors, image), 0, 1)


def panel(image, gt_labels, pred_labels, prob=None, title: str = "", path=None):
    """Imagem, ground truth, predicao e opcionalmente o mapa de probabilidade."""
    cols = 3 + (prob is not None)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4.2))
    img = np.asarray(image)
    axes[0].imshow(img, cmap="gray" if img.ndim == 2 else None)
    axes[0].set_title("imagem")
    axes[1].imshow(colorize_labels(gt_labels))
    axes[1].set_title(f"ground truth ({int(np.max(gt_labels))} obj)")
    axes[2].imshow(colorize_labels(pred_labels))
    axes[2].set_title(f"predicao ({int(np.max(pred_labels))} obj)")
    if prob is not None:
        axes[3].imshow(prob, cmap="magma", vmin=0, vmax=1)
        axes[3].set_title("prob de foreground")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def density_plot(density, ap, count_error, path=None, title="Parte 1: o decoder ingenuo piora com a densidade"):
    """Item 5 da Parte 1: mAP e erro de contagem em funcao da densidade de objetos."""
    density = np.asarray(density, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, y, ylabel in ((axes[0], ap, "AP @.50:.95 por imagem"), (axes[1], count_error, "|erro de contagem|")):
        ax.scatter(density, y, s=22, alpha=0.55, edgecolor="none")
        if len(density) > 2 and density.std() > 0:
            coef = np.polyfit(density, y, 1)
            xs = np.linspace(density.min(), density.max(), 50)
            ax.plot(xs, np.polyval(coef, xs), "r--", lw=2, label=f"tendencia, slope={coef[0]:.4f}")
            # media por faixa, senao a tendencia some no meio da nuvem de pontos
            bins = np.linspace(density.min(), density.max() + 1e-6, 8)
            idx = np.digitize(density, bins) - 1
            means = [y[idx == b].mean() if (idx == b).any() else np.nan for b in range(len(bins) - 1)]
            ax.plot(0.5 * (bins[:-1] + bins[1:]), means, "o-", color="k", lw=1.5, ms=4, label="media por faixa")
            ax.legend()
        ax.set_xlabel("densidade (instancias no ground truth)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig
