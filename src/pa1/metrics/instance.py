"""Avaliacao no nivel de instancia, implementada por nos (o PA proibe usar AP de biblioteca).

O slide 6 define AP como precisao media sobre as classes, que e a metrica
semantica. Aqui a gente generaliza pro nivel de instancia seguindo a convencao do
Data Science Bowl 2018:

para cada limiar de IoU t em {0.50, 0.55, ..., 0.95}, casa cada instancia prevista
com no maximo uma instancia verdadeira. Par com IoU >= t e TP, previsao sem par e
FP, GT sem par e FN, e a precisao naquele limiar e TP / (TP + FP + FN). O AP da
imagem e a media sobre os limiares, e o mAP do dataset e a media dos AP das imagens.

A regra de matching e escolha nossa e tem que estar explicita na apresentacao.
"greedy" casa em ordem de IoU decrescente, "hungarian" resolve a atribuicao global
com scipy e depois corta os pares abaixo do limiar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(round(t, 2) for t in np.arange(0.50, 0.96, 0.05))


def _as_sequential(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Renumera os ids pra 1..n. Buraco na numeracao viraria instancia fantasma na matriz de IoU."""
    labels = np.asarray(labels).astype(np.int64)
    present = np.unique(labels)
    present = present[present > 0]
    if present.size == 0:
        return np.zeros_like(labels), 0
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    lut[present] = np.arange(1, present.size + 1)
    return lut[labels], int(present.size)


def instance_iou_matrix(pred_labels: np.ndarray, gt_labels: np.ndarray) -> np.ndarray:
    """Matriz (n_pred, n_gt) de IoU entre duas imagens de label."""
    pred, n_pred = _as_sequential(pred_labels)
    gt, n_gt = _as_sequential(gt_labels)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float64)

    pred_area = np.bincount(pred.ravel(), minlength=n_pred + 1).astype(np.int64)
    gt_area = np.bincount(gt.ravel(), minlength=n_gt + 1).astype(np.int64)

    # intersecao de todos os pares de uma vez, contando os pixels onde os dois tem label
    both = (pred > 0) & (gt > 0)
    pair_ids = pred[both].astype(np.int64) * (n_gt + 1) + gt[both].astype(np.int64)
    inter = np.bincount(pair_ids, minlength=(n_pred + 1) * (n_gt + 1))
    inter = inter.reshape(n_pred + 1, n_gt + 1)[1:, 1:].astype(np.float64)

    union = pred_area[1:, None] + gt_area[None, 1:] - inter
    return inter / np.maximum(union, 1.0)


def match_instances(iou: np.ndarray, threshold: float, method: str = "greedy") -> list[tuple[int, int]]:
    """Devolve os pares (pred, gt) casados com IoU >= threshold."""
    if iou.size == 0:
        return []

    if method == "greedy":
        cand = np.argwhere(iou >= threshold)
        if cand.size == 0:
            return []
        order = np.argsort(-iou[cand[:, 0], cand[:, 1]])
        used_p: set[int] = set()
        used_g: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for k in order:
            p, g = int(cand[k, 0]), int(cand[k, 1])
            if p in used_p or g in used_g:
                continue
            used_p.add(p)
            used_g.add(g)
            pairs.append((p, g))
        return pairs

    if method == "hungarian":
        row, col = linear_sum_assignment(-iou)
        return [(int(r), int(c)) for r, c in zip(row, col) if iou[r, c] >= threshold]

    raise ValueError(f"matching '{method}' nao existe, usa 'greedy' ou 'hungarian'")


def precision_at_threshold(
    iou: np.ndarray, threshold: float, n_pred: int, n_gt: int, method: str = "greedy"
) -> tuple[float, int, int, int]:
    """Precisao do DSB2018 num limiar so. Devolve (precisao, tp, fp, fn)."""
    pairs = match_instances(iou, threshold, method)
    tp = len(pairs)
    fp = n_pred - tp
    fn = n_gt - tp
    denom = tp + fp + fn
    # nada previsto e nada pra achar conta como acerto
    precision = 1.0 if denom == 0 else tp / denom
    return precision, tp, fp, fn


@dataclass
class ImageInstanceResult:
    ap: float
    per_threshold: dict[float, float]
    n_pred: int
    n_gt: int
    count_error: int

    @property
    def density(self) -> int:
        return self.n_gt


def evaluate_image(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    method: str = "greedy",
) -> ImageInstanceResult:
    iou = instance_iou_matrix(pred_labels, gt_labels)
    n_pred, n_gt = iou.shape
    per_t = {}
    for t in thresholds:
        precision, *_ = precision_at_threshold(iou, t, n_pred, n_gt, method)
        per_t[float(t)] = precision
    ap = float(np.mean(list(per_t.values()))) if per_t else 0.0
    return ImageInstanceResult(
        ap=ap,
        per_threshold=per_t,
        n_pred=n_pred,
        n_gt=n_gt,
        count_error=abs(n_pred - n_gt),
    )


@dataclass
class InstanceMeter:
    """Acumula as metricas de instancia do dataset inteiro."""

    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    method: str = "greedy"
    results: list[ImageInstanceResult] = field(default_factory=list)

    def update(self, pred_labels: np.ndarray, gt_labels: np.ndarray) -> ImageInstanceResult:
        res = evaluate_image(pred_labels, gt_labels, self.thresholds, self.method)
        self.results.append(res)
        return res

    def compute(self) -> dict:
        if not self.results:
            return {"mAP": 0.0, "mean_count_error": 0.0, "n": 0}
        aps = np.array([r.ap for r in self.results])
        cerr = np.array([r.count_error for r in self.results])
        per_t = {
            float(t): float(np.mean([r.per_threshold[float(t)] for r in self.results]))
            for t in self.thresholds
        }
        return {
            "mAP": float(aps.mean()),
            "AP50": per_t[float(self.thresholds[0])],
            "per_threshold": per_t,
            "mean_count_error": float(cerr.mean()),
            "n": len(self.results),
        }

    def density_table(self) -> np.ndarray:
        """(n_imagens, 3) com [densidade, ap, erro de contagem], pro grafico da Parte 1."""
        return np.array([[r.n_gt, r.ap, r.count_error] for r in self.results], dtype=np.float64)


def instance_areas(labels: np.ndarray) -> np.ndarray:
    """Area de cada instancia presente, na mesma ordem que instance_iou_matrix usa.

    bincount()[1:] daria errado quando a numeracao tem buraco: sairia com tamanho
    max(label) em vez do numero de instancias, e nao alinharia com a matriz de IoU,
    que renumera pra 1..n. Foi o bug que derrubou o primeiro run da Parte 5.
    """
    counts = np.bincount(np.asarray(labels).ravel())
    present = np.nonzero(counts)[0]
    return counts[present[present > 0]]


def error_breakdown(pred_labels, gt_labels, overlap: float = 0.5) -> dict:
    """Separa o erro em fusao, fragmentacao, faltou e sobrou (diagnostico da Parte 5).

    Fusao: uma instancia prevista que cobre boa parte de duas ou mais verdadeiras. E o
    modo de falha classico da Parte 1, e o que a trilha A deveria ter matado.

    Fragmentacao: uma instancia verdadeira coberta por duas ou mais previsoes. E o modo
    de falha novo que a trilha A introduz, quando o interior previsto racha em dois
    marcadores e o watershed corta um nucleo no meio.

    Saber qual dos dois domina e o que diz se o proximo ajuste deve ser no
    interior_threshold (que troca um pelo outro) ou em outro lugar.
    """
    iou = instance_iou_matrix(pred_labels, gt_labels)
    n_pred, n_gt = iou.shape
    if n_pred == 0 or n_gt == 0:
        return {"merged": 0, "fragmented": 0, "missed": n_gt, "spurious": n_pred,
                "n_pred": n_pred, "n_gt": n_gt}

    gt_area = instance_areas(gt_labels)
    pred_area = instance_areas(pred_labels)
    # de iou = i / (a + b - i) sai i = iou * (a + b) / (1 + iou)
    inter = iou * (pred_area[:, None] + gt_area[None, :]) / (1 + iou)

    cover_gt = inter / np.maximum(gt_area[None, :], 1)      # quanto de cada gt cada pred cobre
    cover_pred = inter / np.maximum(pred_area[:, None], 1)  # quanto de cada pred cai em cada gt

    return {
        "merged": int(((cover_gt >= overlap).sum(axis=1) >= 2).sum()),
        "fragmented": int(((cover_pred >= overlap).sum(axis=0) >= 2).sum()),
        "missed": int((iou.max(axis=0) < 0.5).sum()),
        "spurious": int((iou.max(axis=1) < 0.5).sum()),
        "n_pred": n_pred,
        "n_gt": n_gt,
    }
