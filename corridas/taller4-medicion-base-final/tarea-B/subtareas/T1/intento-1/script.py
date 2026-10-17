import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Datos del enunciado (Parte 1 — Softmax con temperatura)
# ------------------------------------------------------------------
TOKENS = ["t0", "t1", "t2", "t3", "t4", "t5"]
LOGITS = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
TEMPERATURAS = [0.5, 1.0, 2.0]


# ------------------------------------------------------------------
# Softmax con temperatura, numéricamente estable:
# p_i = exp(z_i / T) / sum_j exp(z_j / T)
# Se resta el máximo de los logits escalados antes de exponenciar.
# ------------------------------------------------------------------
def softmax_temperatura(z, T):
    z = np.asarray(z, dtype=float)
    if T <= 0:
        raise ValueError("La temperatura debe ser positiva.")
    escalados = z / T
    escalados = escalados - np.max(escalados)  # truco de estabilidad numérica
    exps = np.exp(escalados)
    return exps / np.sum(exps)


def entropia_bits(p):
    """Entropía de Shannon H(p) = -sum_i p_i * log2(p_i), en bits."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]  # convención 0*log2(0) = 0
    return float(-np.sum(p * np.log2(p)))


# ------------------------------------------------------------------
# Cálculo de distribuciones, entropías y verificación de suma
# ------------------------------------------------------------------
distribuciones = {}
entropias = {}
sumas = {}
for T in TEMPERATURAS:
    p = softmax_temperatura(LOGITS, T)
    distribuciones[T] = p
    entropias[T] = entropia_bits(p)
    sumas[T] = float(np.sum(p))

# ------------------------------------------------------------------
# Tabla con cuatro decimales: 3 distribuciones (6 probs c/u) + 3 entropías
# ------------------------------------------------------------------
columnas = ["T"] + [f"p({t})" for t in TOKENS] + ["H (bits)", "suma_p"]
filas = []
for T in TEMPERATURAS:
    filas.append(
        [f"{T:.1f}"]
        + [f"{v:.4f}" for v in distribuciones[T]]
        + [f"{entropias[T]:.4f}", f"{sumas[T]:.4f}"]
    )
tabla = pd.DataFrame(filas, columns=columnas)
print("Tabla de resultados (4 decimales):")
print(tabla.to_string(index=False))

# Tabla en formato markdown (sin dependencias externas)
md = "| " + " | ".join(columnas) + " |\n"
md += "|" + "|".join(["---"] * len(columnas)) + "|\n"
for fila in filas:
    md += "| " + " | ".join(fila) + " |\n"

# ------------------------------------------------------------------
# Verificaciones del criterio de éxito
# ------------------------------------------------------------------
suma_ok = all(abs(sumas[T] - 1.0) < 1e-12 for T in TEMPERATURAS)
entropia_monotona = bool(
    entropias[0.5] < entropias[1.0] < entropias[2.0]
)
assert suma_ok, "Cada distribución debe sumar 1."
assert entropia_monotona, "La entropía debe crecer con T: H(0.5) < H(1) < H(2)."

# ------------------------------------------------------------------
# Figura: distribuciones para las tres temperaturas
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(TOKENS))
ancho = 0.25
for i, T in enumerate(TEMPERATURAS):
    ax.bar(x + (i - 1) * ancho, distribuciones[T], ancho, label=f"T = {T}")
ax.set_xticks(x)
ax.set_xticklabels(TOKENS)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad")
ax.set_title("Softmax con temperatura (logits: 2.0, 1.0, 0.5, 0.2, -1.0, -3.0)")
ax.legend()
fig.tight_layout()
fig.savefig("softmax_temperatura.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "logits": {t: float(z) for t, z in zip(TOKENS, LOGITS)},
    "definicion": (
        "p_i = exp(z_i/T) / sum_j exp(z_j/T), restando el maximo de los "
        "logits escalados antes de exponenciar (estabilidad numerica)"
    ),
    "distribuciones": {
        f"T={T}": {t: float(v) for t, v in zip(TOKENS, distribuciones[T])}
        for T in TEMPERATURAS
    },
    "entropias_bits": {f"T={T}": entropias[T] for T in TEMPERATURAS},
    "sumas_distribuciones": {f"T={T}": sumas[T] for T in TEMPERATURAS},
    "tabla_4_decimales": {
        "columnas": columnas,
        "filas": filas,
        "markdown": md,
    },
    "verificacion": {
        "cada_distribucion_suma_1": bool(suma_ok),
        "entropia_crece_con_T": entropia_monotona,
    },
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# Resumen
# ------------------------------------------------------------------
print("\nResumen:")
for T in TEMPERATURAS:
    print(
        f"T={T}: H = {entropias[T]:.4f} bits | "
        + ", ".join(f"p({t})={v:.4f}" for t, v in zip(TOKENS, distribuciones[T]))
        + f" | suma = {sumas[T]:.6f}"
    )
print(f"Orden de entropías: H(0.5)={entropias[0.5]:.4f} < "
      f"H(1)={entropias[1.0]:.4f} < H(2)={entropias[2.0]:.4f}")
print("Archivos generados: resultados.json, softmax_temperatura.png")
