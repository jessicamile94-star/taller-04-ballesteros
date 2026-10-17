import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def softmax_con_temperatura(z, T):
    """Softmax con temperatura, numéricamente estable: resta el máximo antes de exponenciar."""
    z = np.asarray(z, dtype=float)
    x = z / T
    x = x - np.max(x)  # estabilidad numérica
    e = np.exp(x)
    return e / e.sum()


tokens = [f"t{i}" for i in range(6)]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0])
temperaturas = [0.5, 1.0, 2.0]

distribuciones = {f"T={T}": softmax_con_temperatura(logits, T) for T in temperaturas}
entropias_bits = {
    f"T={T}": float(-(p * np.log2(p)).sum()) for T, p in distribuciones.items()
}
sumas = {f"T={T}": float(p.sum()) for T, p in distribuciones.items()}

# Validaciones del criterio de éxito
assert all(abs(s - 1.0) < 1e-12 for s in sumas.values()), "Cada distribución debe sumar 1"
h = [entropias_bits[f"T={T}"] for T in temperaturas]
assert h[0] < h[1] < h[2], "La entropía debe crecer monotónicamente con T"

# Tabla con cuatro decimales
filas = []
for T in temperaturas:
    p = distribuciones[f"T={T}"]
    filas.append([f"T={T}"] + [f"{v:.4f}" for v in p] + [f"{entropias_bits[f'T={T}']:.4f}"])

columnas = ["Temperatura"] + tokens + ["Entropía (bits)"]

resultados = {
    "tokens": tokens,
    "logits": logits.tolist(),
    "temperaturas": temperaturas,
    "distribuciones": {f"T={T}": distribuciones[f"T={T}"].tolist() for T in temperaturas},
    "entropias_bits": entropias_bits,
    "sumas_distribuciones": sumas,
    "tabla_4_decimales": {"columnas": columnas, "filas": filas},
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# Figura: tabla en PNG
fig, ax = plt.subplots(figsize=(9.5, 2.8))
ax.axis("off")
tbl = ax.table(cellText=filas, colLabels=columnas, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.7)
ax.set_title("Softmax con temperatura: distribuciones y entropía (bits)", fontsize=11)
plt.savefig("tabla_softmax_temperatura.png", dpi=120, bbox_inches="tight")
plt.close(fig)

print("Distribuciones (sin redondear):")
for T in temperaturas:
    print(f"  T={T}: {np.round(distribuciones[f'T={T}'], 6).tolist()}")
print("Entropías (bits):", {k: round(v, 4) for k, v in entropias_bits.items()})
print("Sumas:", sumas)
print("Monotonía H(0.5) < H(1) < H(2):", h[0] < h[1] < h[2])
print("Archivos: resultados.json, tabla_softmax_temperatura.png")
