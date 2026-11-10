"""DBSCAN clássico: implementação didática do zero + validação com scikit-learn."""
import time
from collections import deque

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

RUIDO = -1
NAO_VISITADO = -2


def dbscan_do_zero(D: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """DBSCAN sobre uma matriz de distâncias D (n x n).

    Passos do algoritmo (Ester et al., 1996):
      1. Para cada ponto, encontra os vizinhos dentro do raio eps.
      2. Se tiver >= min_samples vizinhos (incluindo ele), é ponto NÚCLEO.
      3. Expande o cluster a partir dos núcleos (busca em largura).
      4. Pontos alcançáveis, mas não núcleo, são de BORDA; o resto é RUÍDO.
    """
    n = D.shape[0]
    vizinhos = [np.flatnonzero(D[i] <= eps) for i in range(n)]
    nucleo = np.array([len(v) >= min_samples for v in vizinhos])
    rotulos = np.full(n, NAO_VISITADO)
    cluster = 0
    for i in range(n):
        if rotulos[i] != NAO_VISITADO:
            continue
        if not nucleo[i]:
            rotulos[i] = RUIDO
            continue
        rotulos[i] = cluster
        fila = deque(vizinhos[i])
        while fila:
            j = fila.popleft()
            if rotulos[j] == RUIDO:          # ruído vira borda
                rotulos[j] = cluster
            if rotulos[j] != NAO_VISITADO:
                continue
            rotulos[j] = cluster
            if nucleo[j]:
                fila.extend(vizinhos[j])
        cluster += 1
    return rotulos, nucleo


def distancias_euclidianas(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def k_distancia(D: np.ndarray, k: int) -> list:
    """Curva k-distância ordenada (ajuda a escolher eps — 'cotovelo')."""
    kd = np.sort(D, axis=1)[:, min(k, D.shape[0] - 1)]
    return np.sort(kd).tolist()


def executar_classico(X: np.ndarray, eps: float, min_samples: int):
    t0 = time.perf_counter()
    D = distancias_euclidianas(X)
    rotulos, nucleo = dbscan_do_zero(D, eps, min_samples)
    t_proprio = time.perf_counter() - t0

    t0 = time.perf_counter()
    sk = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    t_sk = time.perf_counter() - t0

    concorda = float(np.mean((rotulos == RUIDO) == (sk.labels_ == RUIDO)))
    return {
        "rotulos": rotulos,
        "nucleo": nucleo,
        "D": D,
        "tempo_ms": t_proprio * 1000,
        "tempo_sklearn_ms": t_sk * 1000,
        "concordancia_ruido_sklearn": concorda,
        "k_distancia": k_distancia(D, min_samples - 1),
    }
