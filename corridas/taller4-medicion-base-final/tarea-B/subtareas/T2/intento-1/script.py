# -*- coding: utf-8 -*-
"""
T2 — Muestreo de núcleo (top-p) con p = 0.9 sobre la distribución de T = 1 (Parte 1).

Entrada : entrada/T1/resultados.json  -> distribución softmax con T = 1 de la Parte 1
Salidas : resultados.json             -> contrato de la subtarea (cifras sin redondear)
          top_p_nucleo.png            -> figura comparativa (dpi = 120)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = 0.9  # umbral top-p exigido por el enunciado

# ----------------------------------------------------------------------
# 1) Distribución de T = 1 obtenida en la Parte 1 (subtarea T1)
# ----------------------------------------------------------------------
ruta_T1 = Path("entrada") / "T1" / "resultados.json"
dist_T1 = None
if ruta_T1.exists():
    with ruta_T1.open(encoding="utf-8") as f:
        res_T1 = json.load(f)
    dists = res_T1.get("distribuciones", {})
    dist_T1 = dists.get("T=1.0", dists.get("T=1"))

if dist_T1 is not None:
    tokens = list(dist_T1.keys())
    probs = np.array([dist_T1[t] for t in tokens], dtype=float)
    fuente = "entrada/T1/resultados.json (distribucion de la Parte 1 con T=1)"
else:
    # Respaldo: softmax con temperatura, numéricamente estable (misma definición de T1)
    tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
    logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
    z = logits / 1.0
    z = z - z.max()
    ez = np.exp(z)
    probs = ez / ez.sum()
    fuente = "respaldo: softmax(T=1) recalculada de forma numericamente estable"

assert abs(probs.sum() - 1.0) < 1e-9, "la distribución de entrada debe sumar 1"

# ----------------------------------------------------------------------
# 2) Top-p: ordenar por probabilidad descendente y hallar el núcleo mínimo
# ----------------------------------------------------------------------
orden = np.argsort(-probs, kind="stable")      # índices de mayor a menor probabilidad
probs_ord = probs[orden]
acum = np.cumsum(probs_ord)

# Número mínimo de tokens cuya acumulada alcanza p (se incluye el token que cruza el umbral)
k = int(np.searchsorted(acum, P, side="left")) + 1
k = min(k, len(tokens))
idx_nucleo = orden[:k]
idx_fuera = orden[k:]

acum_nucleo = float(acum[k - 1])
acum_sin_ultimo = float(acum[k - 2]) if k >= 2 else 0.0

tokens_nucleo = [tokens[i] for i in idx_nucleo]
tokens_fuera = [tokens[i] for i in idx_fuera]

# ----------------------------------------------------------------------
# 3) Renormalización (los tokens fuera del núcleo quedan con probabilidad 0)
# ----------------------------------------------------------------------
probs_renorm = np.zeros_like(probs)
masa_nucleo = float(probs[idx_nucleo].sum())
probs_renorm[idx_nucleo] = probs[idx_nucleo] / masa_nucleo

# ----------------------------------------------------------------------
# 4) Tabla del proceso (4 decimales) en texto/markdown
# ----------------------------------------------------------------------
nucleo_set = set(idx_nucleo.tolist())
header = ["rank", "token", "p(T=1)", "acumulada", "en_nucleo", "p_renormalizada"]
filas = []
for rank, i in enumerate(orden, start=1):
    dentro = i in nucleo_set
    filas.append([
        str(rank),
        tokens[i],
        f"{probs[i]:.4f}",
        f"{acum[rank - 1]:.4f}",
        "si" if dentro else "no",
        f"{probs_renorm[i]:.4f}" if dentro else "0.0000",
    ])
markdown = (
    "| " + " | ".join(header) + " |\n"
    + "|" + "---|" * len(header) + "\n"
    + "\n".join("| " + " | ".join(f) + " |" for f in filas)
    + "\n"
)

# ----------------------------------------------------------------------
# 5) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T2: top-p (muestreo de nucleo) con p=0.9 sobre la distribucion de T=1",
    "entrada": {
        "fuente": fuente,
        "temperatura": 1.0,
        "p": P,
        "distribucion_T1": {t: float(v) for t, v in zip(tokens, probs)},
    },
    "orden_descendente": {
        "tokens": [tokens[i] for i in orden],
        "probabilidades": [float(v) for v in probs_ord],
        "probabilidad_acumulada": [float(v) for v in acum],
    },
    "nucleo_top_p": {
        "tokens_supervivientes": tokens_nucleo,
        "num_tokens": k,
        "probabilidad_acumulada_nucleo": acum_nucleo,
        "acumulada_hasta_token_anterior": acum_sin_ultimo,
        "es_minimo_con_acumulada_ge_p": bool(acum_nucleo >= P and acum_sin_ultimo < P),
        "tokens_eliminados": tokens_fuera,
    },
    "distribucion_renormalizada": {
        t: float(probs_renorm[i]) for i, t in enumerate(tokens)
    },
    "distribucion_renormalizada_orden_descendente": {
        tokens[i]: float(probs_renorm[i]) for i in orden
    },
    "tabla_4_decimales": {"columnas": header, "filas": filas, "markdown": markdown},
    "verificacion": {
        "suma_distribucion_renormalizada": float(probs_renorm.sum()),
        "suma_igual_1": bool(abs(probs_renorm.sum() - 1.0) < 1e-12),
        "tokens_fuera_del_nucleo_con_probabilidad_0": bool(
            all(probs_renorm[i] == 0.0 for i in idx_fuera)
        ),
        "proporciones_del_nucleo_conservadas": bool(
            np.allclose(probs_renorm[idx_nucleo] * masa_nucleo, probs[idx_nucleo],
                        rtol=0.0, atol=1e-12)
        ),
        "acumulada_nucleo_ge_p": bool(acum_nucleo >= P),
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 6) Figura comparativa
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(len(tokens))
colores = ["#2a9d8f" if i in nucleo_set else "#e76f51" for i in range(len(tokens))]

ax = axes[0]
ax.bar(x, probs, color=colores, edgecolor="black", linewidth=0.5, label="p(T=1)")
ax.plot(x, acum, "o--", color="#264653", markersize=4, label="acumulada (orden desc.)")
ax.axhline(P, color="gray", linestyle=":", linewidth=1.5, label="p = 0.9")
ax.set_title("Distribución de T=1 y probabilidad acumulada")
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_ylabel("probabilidad")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8)

ax = axes[1]
ax.bar(x, probs_renorm, color=colores, edgecolor="black", linewidth=0.5)
ax.set_title("Distribución renormalizada (núcleo: " + ", ".join(tokens_nucleo) + ")")
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_ylabel("probabilidad renormalizada")
ax.set_ylim(0, 1.05)

fig.suptitle(
    "Muestreo de núcleo (top-p) con p = 0.9 — verde: dentro del núcleo, rojo: fuera",
    fontsize=10,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("top_p_nucleo.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 7) Resumen
# ----------------------------------------------------------------------
print("T2 — Muestreo de núcleo (top-p), p = 0.9, sobre la distribución de T = 1")
print(f"Fuente: {fuente}")
print("Orden descendente de probabilidad:")
for rank, i in enumerate(orden, start=1):
    print(f"  {rank}. {tokens[i]}  p = {probs[i]:.4f}  acumulada = {acum[rank-1]:.4f}")
print(f"Tokens supervivientes (núcleo mínimo con acumulada >= 0.9): {tokens_nucleo}")
print(f"Probabilidad acumulada del núcleo: {acum_nucleo:.4f}")
print(f"Tokens eliminados (probabilidad 0): {tokens_fuera}")
print("Distribución renormalizada:")
for t, v in zip(tokens, probs_renorm):
    print(f"  {t}: {v:.6f}")
print(f"Suma de la distribución renormalizada: {probs_renorm.sum():.12f}")
print("Archivos escritos: resultados.json, top_p_nucleo.png")
