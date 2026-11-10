# DBSCAN · Clássico vs Quântico — Dashboard Flask

📞 **Meus Contatos:** [www.inforpop.com](https://www.inforpop.com)
📱 **WhatsApp:** [55 21 99961-8643](https://wa.me/5521999618643)
✉️ **Email:** [inforpop@hotmail.com](mailto:inforpop@hotmail.com)
🔗 **LinkedIn:** [Pedro Paulo Ribeiro](https://www.linkedin.com/in/pedro-paulo-ribeiro-pcd-1451498/)

---

Projeto prático de **clustering (agrupamento) por densidade** com o algoritmo **DBSCAN**,
em Python, com um dashboard Flask de duas abas:

| Aba | O que faz |
|---|---|
| 🖥️ **Computação Clássica** | DBSCAN implementado do zero (busca em largura sobre a matriz de distâncias euclidianas), validado contra `sklearn.cluster.DBSCAN`. |
| ⚛️ **Computação Quântica** | DBSCAN sobre uma **distância quântica**: cada par de pontos é comparado por um circuito **Swap Test** de 5 qubits, com codificação por ângulo (RY) e amostragem de *shots*. |
| 🎓 **Sobre o Autor** | Formação em Ciência de Dados (UNINASSAU) e diploma. |

## Como o DBSCAN funciona

- **ε (eps)**: raio da vizinhança.
- **min_samples**: mínimo de pontos no raio para um ponto ser **núcleo**.
- **Borda**: ponto alcançável a partir de um núcleo, mas sem vizinhos suficientes.
- **Ruído**: pontos que não pertencem a nenhum cluster.
- A **curva k-distância** ajuda a escolher ε (procure o "cotovelo").

## A parte quântica

1. Os pontos são normalizados para ângulos θ ∈ [0, π] e codificados em
   `|ψ(x)⟩ = RY(θ₁)|0⟩ ⊗ RY(θ₂)|0⟩`.
2. O **Swap Test** (ancilla + Hadamard + SWAPs controlados + Hadamard + medida) dá
   `P(0) = (1 + F) / 2`, com `F = |⟨ψ(x)|ψ(y)⟩|²`.
3. Com N *shots*, estima-se F (com ruído de amostragem, como num hardware real) e
   calcula-se `d_Q = √(2·(1 − F))`.
4. O DBSCAN roda sobre a matriz de distâncias quânticas.

A simulação é feita em NumPy (vetor de estados + amostragem binomial), deixando o
app leve para o Heroku. O dashboard confere o simulador rodando o circuito completo
em um par de pontos e comparando com a fórmula. O mesmo circuito em **Qiskit** está
em `algoritmos/quantico.py → circuito_qiskit_exemplo()` (requer `pip install qiskit qiskit-aer`).

## Métricas exibidas

Clusters, ruído, núcleo, borda, **ARI** (Adjusted Rand Index contra o rótulo real),
**Silhueta**, tempo de execução, nº de circuitos, total de shots e erro médio da fidelidade.

## Estrutura

```
dbscan-quantico/
├── app.py                  # Flask: rotas e API JSON
├── algoritmos/
│   ├── dados.py            # 5 conjuntos sintéticos
│   ├── classico.py         # DBSCAN do zero + sklearn
│   └── quantico.py         # Swap Test simulado + DBSCAN quântico
├── templates/index.html    # Dashboard (Plotly.js)
├── static/img/diploma.png
├── tests/test_app.py       # testes (pytest)
├── .github/workflows/ci.yml
├── LICENSE                 # MIT
├── requirements.txt
├── Procfile
└── .python-version
```

## Rodar localmente

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                    # http://localhost:5000
```

## Testes

```bash
pip install pytest
python -m pytest -q
```

A cada `git push` no GitHub, o workflow `.github/workflows/ci.yml` roda os testes automaticamente (aba **Actions**).

## Publicar no GitHub

```bash
# crie um repositório vazio em https://github.com/new (ex.: dbscan-quantico), sem README
git remote add origin https://github.com/SEU_USUARIO/dbscan-quantico.git
git push -u origin main
```

## Implantação no Heroku

```bash
heroku login
heroku create dbscan-quantico-inforpop     # escolha um nome livre
git push heroku main
heroku ps:scale web=1
heroku open
```

Logs: `heroku logs --tail` · Health check: `/health`

---
© 2026 Pedro Paulo Ribeiro — [www.inforpop.com](https://www.inforpop.com)
