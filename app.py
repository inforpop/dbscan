"""Dashboard Flask — DBSCAN Clássico vs Quântico.

Autor: Pedro Paulo Ribeiro — www.inforpop.com
"""
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.metrics import adjusted_rand_score, silhouette_score

from algoritmos.classico import executar_classico
from algoritmos.dados import DATASETS, gerar_dados
from algoritmos.quantico import CIRCUITO_ASCII, executar_quantico

app = Flask(__name__)

LIMITES = {"n": (50, 600), "min_samples": (2, 30), "shots": (64, 8192)}


def _num(valor, tipo, padrao, faixa=None):
    try:
        v = tipo(valor)
    except (TypeError, ValueError):
        return padrao
    if faixa:
        v = max(faixa[0], min(faixa[1], v))
    return v


def metricas(X, D, rotulos, y_verdadeiro, nucleo):
    mask = rotulos != -1
    n_clusters = len(set(rotulos[mask].tolist()))
    sil = None
    if n_clusters >= 2 and mask.sum() > n_clusters:
        try:
            sil = float(silhouette_score(D[np.ix_(mask, mask)], rotulos[mask],
                                         metric="precomputed"))
        except ValueError:
            sil = None
    return {
        "clusters": n_clusters,
        "ruido": int((~mask).sum()),
        "nucleo": int(nucleo.sum()),
        "borda": int((mask & ~nucleo).sum()),
        "ari": float(adjusted_rand_score(y_verdadeiro, rotulos)),
        "silhueta": sil,
    }


@app.route("/")
def index():
    return render_template("index.html", datasets=DATASETS, circuito=CIRCUITO_ASCII)


@app.route("/api/classico", methods=["POST"])
def api_classico():
    p = request.get_json(silent=True) or {}
    X, y = gerar_dados(p.get("dataset", "luas"),
                       _num(p.get("n"), int, 300, LIMITES["n"]),
                       _num(p.get("ruido"), float, 0.06, (0, 0.5)))
    eps = _num(p.get("eps"), float, 0.3, (0.01, 5))
    ms = _num(p.get("min_samples"), int, 5, LIMITES["min_samples"])
    r = executar_classico(X, eps, ms)
    return jsonify({
        "X": X.round(4).tolist(), "rotulos": r["rotulos"].tolist(),
        "nucleo": r["nucleo"].tolist(), "k_distancia": r["k_distancia"],
        "metricas": metricas(X, r["D"], r["rotulos"], y, r["nucleo"]),
        "tempo_ms": round(r["tempo_ms"], 2),
        "tempo_sklearn_ms": round(r["tempo_sklearn_ms"], 2),
        "concordancia_sklearn": round(r["concordancia_ruido_sklearn"] * 100, 1),
    })


@app.route("/api/quantico", methods=["POST"])
def api_quantico():
    p = request.get_json(silent=True) or {}
    X, y = gerar_dados(p.get("dataset", "luas"),
                       _num(p.get("n"), int, 300, LIMITES["n"]),
                       _num(p.get("ruido"), float, 0.06, (0, 0.5)))
    eps = _num(p.get("eps"), float, 0.2, (0.01, 1.5))
    ms = _num(p.get("min_samples"), int, 5, LIMITES["min_samples"])
    shots = _num(p.get("shots"), int, 1024, LIMITES["shots"])
    r = executar_quantico(X, eps, ms, shots)
    return jsonify({
        "X": X.round(4).tolist(), "rotulos": r["rotulos"].tolist(),
        "nucleo": r["nucleo"].tolist(), "k_distancia": r["k_distancia"],
        "metricas": metricas(X, r["D"], r["rotulos"], y, r["nucleo"]),
        "tempo_ms": round(r["tempo_ms"], 2),
        "qubits": r["qubits"], "n_circuitos": r["n_circuitos"],
        "shots_total": r["shots_total"],
        "erro_fidelidade": round(r["erro_medio_fidelidade"], 4),
        "verificacao": r["verificacao"],
        "fidelidades": r["fidelidades_amostra"],
    })


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
