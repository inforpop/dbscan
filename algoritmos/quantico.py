"""DBSCAN Quântico (simulado): distância por Swap Test com codificação por ângulo.

Ideia
-----
1. Cada ponto x = (x1, x2) é mapeado para ângulos θ ∈ [0, π] e codificado em
   2 qubits:  |ψ(x)⟩ = RY(θ1)|0⟩ ⊗ RY(θ2)|0⟩.
2. O *Swap Test* entre |ψ(x)⟩ e |ψ(y)⟩ usa 1 qubit ancilla + 4 qubits de dados:
       ancilla: H ── ● ── H ── medida
       dados:      SWAP(x, y)
   P(ancilla = 0) = (1 + F) / 2, com fidelidade F = |⟨ψ(x)|ψ(y)⟩|².
3. Com `shots` medições, estimamos F (com ruído de amostragem, como num
   hardware real) e definimos a distância quântica  d_Q = sqrt(2·(1 − F)).
4. O DBSCAN roda sobre a matriz de distâncias quânticas.

A simulação usa NumPy (vetor de estados exato + amostragem binomial dos shots),
o que mantém o app leve para o Heroku. O código do circuito em Qiskit está em
`circuito_qiskit_exemplo()` para quem quiser rodar em simulador/hardware IBM.
"""
import time

import numpy as np

from .classico import dbscan_do_zero, k_distancia

# ---- Portas e estados (simulação de vetor de estados) -----------------------
KET0 = np.array([1.0, 0.0])


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])


def estado_ponto(theta: np.ndarray) -> np.ndarray:
    """|ψ⟩ = ⊗_i RY(θ_i)|0⟩  (dimensão 2^d)."""
    psi = np.array([1.0])
    for t in theta:
        psi = np.kron(psi, ry(t) @ KET0)
    return psi


def para_angulos(X: np.ndarray, escala: float = np.pi) -> np.ndarray:
    lo, hi = X.min(axis=0), X.max(axis=0)
    return (X - lo) / np.where(hi - lo == 0, 1, hi - lo) * escala


def swap_test_exato(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    """Simula o circuito do swap test no espaço completo e devolve P(ancilla=0)."""
    dim = psi_a.size
    estado = np.kron(KET0, np.kron(psi_a, psi_b))          # |0⟩|a⟩|b⟩
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    I = np.eye(dim * dim)
    H_anc = np.kron(H, I)
    # SWAP controlado: |0⟩⟨0|⊗I + |1⟩⟨1|⊗SWAP
    SWAP = np.zeros((dim * dim, dim * dim))
    for i in range(dim):
        for j in range(dim):
            SWAP[j * dim + i, i * dim + j] = 1
    CSWAP = np.kron(np.diag([1, 0]), I) + np.kron(np.diag([0, 1]), SWAP)
    final = H_anc @ CSWAP @ H_anc @ estado
    return float(np.sum(np.abs(final[: dim * dim]) ** 2))


def matriz_fidelidade(theta: np.ndarray) -> np.ndarray:
    """F_ij = |⟨ψ_i|ψ_j⟩|² vetorizado (equivalente ao swap test exato)."""
    estados = np.array([estado_ponto(t) for t in theta])
    return (estados @ estados.T) ** 2


def executar_quantico(X: np.ndarray, eps: float, min_samples: int,
                      shots: int = 1024, seed: int = 7):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    theta = para_angulos(X)
    F = matriz_fidelidade(theta)

    # Amostragem dos shots: conta quantas vezes a ancilla deu 0
    p0 = np.clip((1 + F) / 2, 0, 1)
    iu = np.triu_indices(len(X), k=1)
    contagens = rng.binomial(shots, p0[iu])
    F_est = np.ones_like(F)
    F_est[iu] = np.clip(2 * contagens / shots - 1, 0, 1)
    F_est[(iu[1], iu[0])] = F_est[iu]

    D = np.sqrt(2 * (1 - F_est))
    rotulos, nucleo = dbscan_do_zero(D, eps, min_samples)
    tempo = (time.perf_counter() - t0) * 1000

    # Verificação: swap test pelo circuito completo num par de pontos
    a, b = 0, min(1, len(X) - 1)
    p0_circ = swap_test_exato(estado_ponto(theta[a]), estado_ponto(theta[b]))

    n_pares = len(iu[0])
    return {
        "rotulos": rotulos,
        "nucleo": nucleo,
        "D": D,
        "tempo_ms": tempo,
        "qubits": 1 + 2 * X.shape[1],
        "n_circuitos": int(n_pares),
        "shots_total": int(n_pares * shots),
        "erro_medio_fidelidade": float(np.mean(np.abs(F_est[iu] - F[iu]))),
        "verificacao": {
            "p0_circuito": p0_circ,
            "p0_formula": float((1 + F[a, b]) / 2),
        },
        "k_distancia": k_distancia(D, min_samples - 1),
        "fidelidades_amostra": F_est[iu][:2000].tolist(),
    }


CIRCUITO_ASCII = """\
 q0 ancilla : ─[H]──────●───────●──────[H]──[M]
                        │       │             ║
 q1 x₁      : ─[RY θ₁]──✕───────┼─────────────╫─
                        │       │             ║
 q2 x₂      : ─[RY θ₂]──┼───────✕─────────────╫─
                        │       │             ║
 q3 y₁      : ─[RY φ₁]──✕───────┼─────────────╫─
                                │             ║
 q4 y₂      : ─[RY φ₂]──────────✕─────────────╫─
 c0         : ════════════════════════════════╩═

 P(ancilla = 0) = (1 + |⟨ψ(x)|ψ(y)⟩|²) / 2
 d_Q(x, y)      = √(2·(1 − F))
"""


def circuito_qiskit_exemplo():  # pragma: no cover - opcional
    """Mesmo swap test em Qiskit (pip install qiskit qiskit-aer)."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(5, 1)
    qc.ry(0.4, 1); qc.ry(1.1, 2); qc.ry(0.9, 3); qc.ry(2.0, 4)
    qc.h(0)
    qc.cswap(0, 1, 3)
    qc.cswap(0, 2, 4)
    qc.h(0)
    qc.measure(0, 0)
    return qc
