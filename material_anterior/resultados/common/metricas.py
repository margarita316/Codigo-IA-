"""Las 4 metricas de regresion pedidas: CI, RMSE, MSE, R2.

CI (concordance index, definicion de Pahikkala et al. 2014/2015, la misma
que usan DeepDTA/GraphDTA/SubMDTA/DeepDTAGen) se calcula con un algoritmo
O(n log n) (arbol de Fenwick sobre los rangos de la prediccion) en vez del
O(n^2) ingenuo, porque KIBA tiene folds de test de mas de 20000 pares y el
calculo cuadratico seria demasiado lento repetido en 50 folds x 3 metodos.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class _Fenwick:
    def __init__(self, n: int):
        self.n = n
        self.tree = [0.0] * (n + 1)

    def add(self, i: int, delta: float = 1.0) -> None:
        i += 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def prefix(self, i: int) -> float:
        i += 1
        total = 0.0
        while i > 0:
            total += self.tree[i]
            i -= i & (-i)
        return total


def concordance_index(y_true, y_pred) -> float:
    """CI = (1/Z) * suma_{y_i>y_j} [1 si pred_i>pred_j, 0.5 si empatan, 0 si no].
    Los pares con y_true empatado se excluyen (no cuentan ni en Z ni en la suma).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)
    if n < 2:
        return float("nan")

    order = np.argsort(y_true, kind="mergesort")
    yt = y_true[order]
    yp = y_pred[order]

    # Compresion de rangos de la prediccion para indexar el Fenwick tree.
    uniq_pred = np.unique(yp)
    rank_of = {v: i for i, v in enumerate(uniq_pred)}
    ranks = np.array([rank_of[v] for v in yp], dtype=np.int64)
    fenwick = _Fenwick(len(uniq_pred))

    numerator = 0.0
    denominator = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and yt[j] == yt[i]:
            j += 1
        # grupo [i, j) comparte el mismo y_true -> se compara solo contra lo
        # ya insertado (estrictamente menor y_true), nunca entre si mismos.
        for k in range(i, j):
            r = ranks[k]
            less = fenwick.prefix(r - 1) if r > 0 else 0.0
            less_equal = fenwick.prefix(r)
            equal = less_equal - less
            total_inserted = fenwick.prefix(len(uniq_pred) - 1)
            numerator += less + 0.5 * equal
            denominator += total_inserted
        for k in range(i, j):
            fenwick.add(int(ranks[k]), 1.0)
        i = j

    if denominator == 0:
        return float("nan")
    return numerator / denominator


def calcular_metricas(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "CI": concordance_index(y_true, y_pred),
        "RMSE": float(np.sqrt(mse)),
        "MSE": float(mse),
        "R2": float(r2_score(y_true, y_pred)),
    }
