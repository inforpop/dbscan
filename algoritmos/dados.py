"""Geração dos conjuntos de dados sintéticos usados no dashboard."""
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler

DATASETS = {
    "luas": "Duas Luas (make_moons)",
    "circulos": "Círculos Concêntricos (make_circles)",
    "blobs": "Blobs Gaussianos (make_blobs)",
    "anisotropico": "Blobs Anisotrópicos",
    "ruido": "Blobs + Ruído Uniforme",
}


def gerar_dados(nome: str = "luas", n: int = 300, ruido: float = 0.06, seed: int = 42):
    """Retorna (X padronizado, rótulos verdadeiros)."""
    rng = np.random.RandomState(seed)
    if nome == "luas":
        X, y = make_moons(n_samples=n, noise=ruido, random_state=seed)
    elif nome == "circulos":
        X, y = make_circles(n_samples=n, factor=0.45, noise=ruido, random_state=seed)
    elif nome == "blobs":
        X, y = make_blobs(n_samples=n, centers=3, cluster_std=0.6 + ruido * 3, random_state=seed)
    elif nome == "anisotropico":
        X, y = make_blobs(n_samples=n, centers=3, random_state=170)
        X = X @ np.array([[0.6, -0.64], [-0.41, 0.85]])
        X += rng.normal(scale=ruido, size=X.shape)
    elif nome == "ruido":
        n_ruido = max(1, int(n * 0.15))
        Xb, yb = make_blobs(n_samples=n - n_ruido, centers=3, cluster_std=0.5, random_state=seed)
        lo, hi = Xb.min(axis=0) - 1, Xb.max(axis=0) + 1
        Xr = rng.uniform(lo, hi, size=(n_ruido, 2))
        X = np.vstack([Xb, Xr])
        y = np.concatenate([yb, -np.ones(n_ruido, dtype=int)])
    else:
        raise ValueError(f"Dataset desconhecido: {nome}")
    X = StandardScaler().fit_transform(X)
    return X, y
