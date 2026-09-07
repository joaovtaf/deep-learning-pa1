"""Plots: mascara de instancia colorida, paineis comparativos e os graficos das partes 1 a 6."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

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
    if image.max() > 1.5:
        image = image / 255.0
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    colors = colorize_labels(labels, seed=seed)
    mask = (np.asarray(labels) > 0)[..., None]
    return np.clip(np.where(mask, (1 - alpha) * image + alpha * colors, image), 0, 1)


def _show(ax, img, title, cmap=None, vmin=None, vmax=None):
    img = np.asarray(img)
    if img.ndim == 2 and cmap is None:
        cmap = "gray"
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def panel(image, gt_labels, pred_labels, maps=None, title: str = "", path=None):
    """Imagem, ground truth, predicao e os mapas intermediarios que a Parte 5 exige."""
    maps = maps or {}
    cols = 3 + len(maps)
    fig, axes = plt.subplots(1, cols, figsize=(3.4 * cols, 3.8))
    if cols == 1:
        axes = [axes]

    img = np.asarray(image)
    if img.dtype == np.uint8:
        img = img / 255.0
    _show(axes[0], img, "imagem")
    _show(axes[1], colorize_labels(gt_labels), f"ground truth ({int(np.max(gt_labels))} obj)")
    _show(axes[2], colorize_labels(pred_labels), f"predicao ({int(np.max(pred_labels))} obj)")
    for k, (name, m) in enumerate(maps.items()):
        _show(axes[3 + k], m, name, cmap="magma", vmin=0, vmax=1)

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def density_plot(density, ap, count_error, path=None, title="mAP e erro de contagem contra densidade"):
    """Item 5 da Parte 1: quantifica o fracasso em funcao da densidade de objetos."""
    density = np.asarray(density, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, y, ylabel in ((axes[0], np.asarray(ap), "AP @.50:.95 por imagem"),
                          (axes[1], np.asarray(count_error), "|erro de contagem|")):
        ax.scatter(density, y, s=22, alpha=0.55, edgecolor="none")
        if len(density) > 2 and density.std() > 0:
            coef = np.polyfit(density, y, 1)
            xs = np.linspace(density.min(), density.max(), 50)
            ax.plot(xs, np.polyval(coef, xs), "r--", lw=2, label=f"tendencia, slope={coef[0]:.4f}")
            bins = np.linspace(density.min(), density.max() + 1e-6, 8)
            idx = np.digitize(density, bins) - 1
            means = [y[idx == b].mean() if (idx == b).any() else np.nan for b in range(len(bins) - 1)]
            ax.plot(0.5 * (bins[:-1] + bins[1:]), means, "o-", color="k", lw=1.5, ms=4, label="media por faixa")
            ax.legend(fontsize=8)
        ax.set_xlabel("densidade (instancias no ground truth)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def density_comparison(series, path=None, title="baseline contra cabeca de instancias"):
    """Mesma curva de densidade pra dois modelos, que e o lado a lado que a Parte 2 pede."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, (density, ap, cerr) in series.items():
        density = np.asarray(density, dtype=float)
        for ax, y in ((axes[0], np.asarray(ap)), (axes[1], np.asarray(cerr))):
            bins = np.linspace(density.min(), density.max() + 1e-6, 8)
            idx = np.digitize(density, bins) - 1
            means = [y[idx == b].mean() if (idx == b).any() else np.nan for b in range(len(bins) - 1)]
            ax.plot(0.5 * (bins[:-1] + bins[1:]), means, "o-", lw=1.8, ms=4, label=label)
    axes[0].set_ylabel("AP @.50:.95 (media por faixa)")
    axes[1].set_ylabel("|erro de contagem| (media por faixa)")
    for ax in axes:
        ax.set_xlabel("densidade (instancias no ground truth)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def ablation_bars(rows, key="mAP", path=None, title="", xlabel=""):
    """Barras com media +- desvio das 2 seeds, que e o formato que a Parte 3 pede."""
    names = [r["label"] for r in rows]
    means = np.array([r[f"{key}_mean"] for r in rows])
    stds = np.array([r[f"{key}_std"] for r in rows])

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(rows)), 4.2))
    x = np.arange(len(rows))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4878a8", edgecolor="k", linewidth=0.6)
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(key)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def degradation_curve(table, path=None, title="Parte 6: degradacao do mAP com a corrupcao"):
    """table: {nome da corrupcao: (intensidades, mAPs)}, mais a linha do limpo."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name, (sev, vals) in table.items():
        ax.plot(sev, vals, "o-", lw=1.8, ms=5, label=name)
    ax.set_xlabel("intensidade da corrupcao (0 = imagem limpa)")
    ax.set_ylabel("mAP @.50:.95")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def size_vs_receptive_field(diameters, rf_lines, path=None, title="tamanho dos objetos contra campo receptivo"):
    """Histograma de diametro dos objetos com o campo receptivo marcado (Parte 5)."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(diameters, bins=60, color="#7aa6c2", edgecolor="k", linewidth=0.4)
    ax.set_xlabel("diametro equivalente do objeto (px)")
    ax.set_ylabel("numero de instancias")
    ymax = ax.get_ylim()[1]
    colors = ["crimson", "darkgreen", "purple", "orange"]
    for (label, value), c in zip(rf_lines.items(), colors):
        ax.axvline(value, color=c, ls="--", lw=1.6)
        ax.text(value, ymax * 0.92, f" {label} = {value:g} px", color=c, fontsize=8, rotation=90, va="top")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def mosaic_figure(image, gt_labels, variants, path=None, title="Parte 4: inferencia em mosaico"):
    """Mosaico grande com o ground truth e cada estrategia de tiling lado a lado."""
    cols = 2 + len(variants)
    fig, axes = plt.subplots(1, cols, figsize=(4.2 * cols, 4.6))
    _show(axes[0], image if image.dtype != np.uint8 else image / 255.0, "mosaico")
    _show(axes[1], colorize_labels(gt_labels), f"ground truth ({int(gt_labels.max())} obj)")
    for k, (name, lab) in enumerate(variants.items()):
        _show(axes[2 + k], colorize_labels(lab), f"{name} ({int(lab.max())} obj)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def seam_zoom(image, variants, window, path=None, title="objeto na emenda entre dois tiles"):
    """Zoom numa faixa que cai em cima da emenda, que e o item 3 da Parte 4."""
    y0, y1, x0, x1 = window
    cols = 1 + len(variants)
    fig, axes = plt.subplots(1, cols, figsize=(3.6 * cols, 3.8))
    crop = image[y0:y1, x0:x1]
    _show(axes[0], crop if crop.dtype != np.uint8 else crop / 255.0, "imagem")
    for k, (name, lab) in enumerate(variants.items()):
        _show(axes[1 + k], colorize_labels(lab[y0:y1, x0:x1]), name)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
