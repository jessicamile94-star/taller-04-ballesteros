# =====================================================================
# T3 — Parte 3: Comprobación empírica del muestreo top-p (Tarea B)
# Toma 10,000 muestras de la distribución renormalizada de la Parte 2
# con numpy.random.default_rng(0), compara frecuencias observadas vs.
# probabilidades teóricas (gráfico de barras) y calcula la divergencia
# KL (en bits) de la empírica respecto de la teórica.
# =====================================================================
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Parámetros fijados por el enunciado
# ---------------------------------------------------------------------
TOKENS = ["t0", "t1", "t2", "t3", "t4", "t5"]
N_MUESTRAS = 10_000
SEMILLA = 0

# ---------------------------------------------------------------------
# 1) Distribución renormalizada de la Parte 2 (resultado de T2)
# ---------------------------------------------------------------------
ruta_t2 = Path("entrada/T2/resultados.json")
if ruta_t2.exists():
    with open(ruta_t2, "r", encoding="utf-8") as f:
        t2 = json.load(f)
    p_teo = np.array(
        [t2["distribucion_renormalizada"][t] for t in TOKENS], dtype=float
    )
    fuente = "entrada/T2/resultados.json (distribucion renormalizada top-p p=0.9 sobre T=1)"
else:
    # Respaldo determinista: softmax(T=1) + top-p(0.9) + renormalización
    logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0])
    z = logits - logits.max()                      # estabilidad numérica
    p1 = np.exp(z) / np.exp(z).sum()
    orden = np.argsort(-p1)                        # descendente
    acum = np.cumsum(p1[orden])
    k = int(np.searchsorted(acum, 0.9, side="left") + 1)  # mínimo con acum >= 0.9
    nucleo = orden[:k]
    p_teo = np.zeros_like(p1)
    p_teo[nucleo] = p1[nucleo] / p1[nucleo].sum()
    fuente = "recalculada (softmax T=1 + top-p p=0.9 + renormalizacion)"

# Salvaguarda numérica para el muestreador (no altera los valores salvo ~1e-16)
if abs(p_teo.sum() - 1.0) > 1e-12:
    p_teo = p_teo / p_teo.sum()

# ---------------------------------------------------------------------
# 2) Muestreo: 10,000 muestras con numpy.random.default_rng(0)
# ---------------------------------------------------------------------
rng = np.random.default_rng(SEMILLA)
muestras = rng.choice(len(TOKENS), size=N_MUESTRAS, p=p_teo)

conteos = np.bincount(muestras, minlength=len(TOKENS))
f_obs = conteos / N_MUESTRAS

# ---------------------------------------------------------------------
# 3) Divergencia KL en bits: KL(empírica || teórica)
#    KL(P||Q) = sum_i P(i) * log2(P(i)/Q(i)); términos con P(i)=0 valen 0
# ---------------------------------------------------------------------
mask = (p_teo > 0) & (f_obs > 0)
contrib_kl = np.zeros(len(TOKENS))
contrib_kl[mask] = f_obs[mask] * np.log2(f_obs[mask] / p_teo[mask])
kl_bits = float(contrib_kl.sum())

# ---------------------------------------------------------------------
# 4) Gráfico de barras: frecuencias observadas vs. probabilidades teóricas
# ---------------------------------------------------------------------
x = np.arange(len(TOKENS))
w = 0.38
fig, ax = plt.subplots(figsize=(8.5, 5.2))
b1 = ax.bar(x - w / 2, f_obs, w, label="Frecuencia observada (10,000 muestras)",
            color="#4C72B0")
b2 = ax.bar(x + w / 2, p_teo, w, label="Probabilidad teórica (renormalizada)",
            color="#DD8452")
for rect, v in zip(b1, f_obs):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.006, f"{v:.4f}",
            ha="center", fontsize=8)
for rect, v in zip(b2, p_teo):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.006, f"{v:.4f}",
            ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(TOKENS)
ax.set_ylabel("Probabilidad / frecuencia")
ax.set_ylim(0, max(f_obs.max(), p_teo.max()) * 1.18)
ax.set_title(
    "Parte 3 — Frecuencias observadas vs. probabilidades teóricas\n"
    f"top-p p=0.9 sobre T=1 · {N_MUESTRAS:,} muestras · "
    f"default_rng({SEMILLA}) · KL = {kl_bits:.6f} bits"
)
ax.legend()
fig.tight_layout()
fig.savefig("T3_frecuencias_vs_teoricas.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------
# 5) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------
resultados = {
    "subtarea": (
        "T3: comprobacion empirica — 10,000 muestras de la distribucion "
        "renormalizada de la Parte 2 con numpy.random.default_rng(0), "
        "comparacion de frecuencias observadas vs. probabilidades teoricas "
        "y divergencia KL (bits) de la empirica respecto de la teorica"
    ),
    "entrada": {
        "fuente_distribucion": fuente,
        "probabilidades_teoricas": {t: float(p) for t, p in zip(TOKENS, p_teo)},
        "num_muestras": N_MUESTRAS,
        "semilla": SEMILLA,
        "generador": f"numpy.random.default_rng({SEMILLA})",
    },
    "muestras": {
        "n": N_MUESTRAS,
        "conteos_observados": {t: int(c) for t, c in zip(TOKENS, conteos)},
        "frecuencias_observadas": {t: float(fr) for t, fr in zip(TOKENS, f_obs)},
    },
    "tabla_comparativa": {
        "columnas": ["token", "frecuencia_observada", "probabilidad_teorica",
                     "diferencia_obs_menos_teo", "aporte_KL_bits"],
        "filas": [
            [t, float(fo), float(pt), float(fo - pt), float(ck)]
            for t, fo, pt, ck in zip(TOKENS, f_obs, p_teo, contrib_kl)
        ],
    },
    "divergencia_kl_bits": kl_bits,
    "figura": "T3_frecuencias_vs_teoricas.png",
    "verificacion": {
        "suma_frecuencias_observadas": float(f_obs.sum()),
        "suma_probabilidades_teoricas": float(p_teo.sum()),
        "tokens_con_probabilidad_teorica_cero_sin_muestras": bool(
            np.all(conteos[p_teo == 0] == 0)
        ),
        "kl_cercana_a_cero": bool(kl_bits < 0.01),
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------
print("=== T3: Comprobación empírica (Parte 3) ===")
print(f"Fuente de la distribución teórica: {fuente}")
print(f"Muestras: {N_MUESTRAS:,} · generador: numpy.random.default_rng({SEMILLA})")
print("token | frec. observada | prob. teórica")
for t, fo, pt in zip(TOKENS, f_obs, p_teo):
    print(f"  {t}  |     {fo:.6f}    |   {pt:.6f}")
print(f"Divergencia KL (empírica || teórica) = {kl_bits:.6f} bits")
print("Figura guardada: T3_frecuencias_vs_teoricas.png")
print("Resultados escritos en resultados.json")
