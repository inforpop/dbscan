"""Testes básicos: python -m pytest -q"""
import numpy as np
from sklearn.cluster import DBSCAN

from app import app
from algoritmos.classico import distancias_euclidianas, dbscan_do_zero
from algoritmos.dados import DATASETS, gerar_dados
from algoritmos.quantico import estado_ponto, swap_test_exato


def test_pagina_e_health():
    c = app.test_client()
    assert c.get("/").status_code == 200
    assert c.get("/health").json == {"status": "ok"}


def test_dbscan_igual_sklearn_no_ruido():
    for nome in DATASETS:
        X, _ = gerar_dados(nome)
        rot, _ = dbscan_do_zero(distancias_euclidianas(X), 0.3, 5)
        sk = DBSCAN(eps=0.3, min_samples=5).fit(X).labels_
        assert np.array_equal(rot == -1, sk == -1)


def test_swap_test_confere_formula():
    a, b = estado_ponto([0.4, 1.1]), estado_ponto([0.9, 2.0])
    assert abs(swap_test_exato(a, b) - (1 + (a @ b) ** 2) / 2) < 1e-9


def test_apis():
    c = app.test_client()
    for rota in ("/api/classico", "/api/quantico"):
        d = c.post(rota, json={"dataset": "luas"}).json
        assert d["metricas"]["clusters"] == 2
