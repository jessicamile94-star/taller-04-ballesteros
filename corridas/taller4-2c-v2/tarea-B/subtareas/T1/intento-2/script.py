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

distribuciones = [softmax_con_temperatura(logits, T) for T in temperaturas]
entropias_bits = [float(-(p * np.log2(p)).sum()) for p in distribuciones]
sumas = [float(p.sum()) for p in distribuciones]

# Validaciones del criterio de éxito
assert all(abs(s - 1.0) < 1e-12 for s in sumas), "Cada distribución debe sumar 1"
assert entropias_bits[0] < entropias_bits[1] < entropias_bits[2], \
    "La entropía debe crecer monotónicamente con T"

# Tabla con cuatro decimales
filas = []
for T, p, h in zip(temperaturas, distribuciones, entropias_bits):
    filas.append([f"T={T}"] + [f"{v:.4f}" for v in p] + [f"{h:.4f}"])
columnas = ["Temperatura"] + tokens + ["Entropía (bits)"]

resultados = {
    "tokens": tokens,
    "logits": logits.tolist(),
    "temperaturas": temperaturas,
    "distribuciones": {f"T={T}": p.tolist() for T, p in zip(temperaturas, distribuciones)},
    "entropias_bits": {f"T={T}": h for T, h in zip(temperaturas, entropias_bits)},
    "sumas_distribuciones": {f"T={T}": s for T, s in zip(temperaturas, sumas)},
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
for T, p in zip(temperaturas, distribuciones):
    print(f"  T={T}: {np.round(p, 6).tolist()}")
print("Entropías (bits):", {f"T={T}": round(h, 4) for T, h in zip(temperaturas, entropias_bits)})
print("Sumas:", sumas)
print("Monotonía H(0.5) < H(1) < H(2):",
      entropias_bits[0] < entropias_bits[1] < entropias_bits[2])
print("Archivos: resultados.json, tabla_softmax_temperatura.png")
