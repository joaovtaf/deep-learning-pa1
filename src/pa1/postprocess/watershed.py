"""Watershed da Parte 2, trilha A. Ainda nao implementado.

Deixei a assinatura ja definida pra nao ter que mexer no harness de avaliacao nem
no notebook de inferencia depois. A ideia e usar os interiores previstos erodidos
como marcador, o mapa de distancia previsto (negado) como relevo, e interior +
fronteira acima de um corte como mascara.
"""

from __future__ import annotations

import numpy as np


def labels_from_boundary_head(*args, **kwargs) -> np.ndarray:
    raise NotImplementedError("watershed da trilha A entra na iteracao da Parte 2")
