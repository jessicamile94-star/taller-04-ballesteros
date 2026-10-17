import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Distribución renormalizada de la Parte 2 (resultado previo T2) ---
with open("entrada/T2/resultados.json", encoding="utf-8") as f:
    t2 = json.load(f)

tokens = t2["tokens"]
p_teo = np.array([t2["distribucion_renormalizada"][t] for t in tokens], dtype=float)

# Validación: recalcular top-p (T=1, p=0.9) desde los logits y comparar
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0])
z = logits - logits.max()
p1 = np.exp(z) / np.exp(z).sum()
orden = np.argsort(-p1)
k = int(np.searchsorted(np.cumsum(p1[orden]), 0.9, side="left") + 1)
p_check = np.zeros_like(p1)
nuc = orden[:k]
p_check[nuc] = p1[nuc] / p1[nuc].sum()
assert np.allclose(p_check, p_teo, atol=1e-9), "Inconsistencia con la distribución renormalizada de T2"

# --- Muestreo: 10,000 muestras con numpy.random.default_rng(0) ---
N = 10000
rng = np.random.default_rng(0)
muestras = rng.choice(len(tokens), size=N, p=p_teo)
conteos = np.bincount(muestras, minlength=len(tokens))
freq_obs = conteos / N

# --- Divergencia KL (bits) de la empírica respecto de la teórica ---
# Términos con frecuencia observada 0 aportan 0; no hay p_teo=0 con freq>0 (se muestreó de p_teo)
mask = freq_obs > 0
kl_bits = float(np.sum(freq_obs[mask] * np.log2(freq_obs[mask] / p_teo[mask])))
kl_nats = float(np.sum(freq_obs[mask] * np.log(freq_obs[mask] / p_teo[mask])))

# --- Gráfico de barras: frecuencias observadas vs probabilidades teóricas ---
x = np.arange(len(tokens))
w = 0.38
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w / 2, freq_obs, w, label="Frecuencia observada", color="#4C72B0")
ax.bar(x + w / 2, p_teo, w, label="Probabilidad teórica", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_xlabel("Token")
ax.set_ylabel("Frecuencia / probabilidad")
ax.set_title(f"Parte 3: muestreo top-p (T=1, p=0.9), N={N}, semilla=0\n"
             f"KL(empírica || teórica) = {kl_bits:.6f} bits")
ax.legend()
for xi, (fo, pt) in enumerate(zip(freq_obs, p_teo)):
    ax.text(xi - w / 2, fo, f"{fo:.4f}", ha="center", va="bottom", fontsize=8)
    if pt > 0:
        ax.text(xi + w / 2, pt, f"{pt:.4f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig("parte3_frecuencias_vs_teoricas.png", dpi=120)
plt.close(fig)

# --- resultados.json (contrato de la subtarea) ---
resultados = {
    "subtarea": "T3",
    "num_muestras": N,
    "semilla": 0,
    "tokens": tokens,
    "probabilidades_teoricas": p_teo.tolist(),
    "conteos_observados": conteos.tolist(),
    "frecuencias_observadas": freq_obs.tolist(),
    "divergencia_KL_bits": kl_bits,
    "divergencia_KL_nats": kl_nats,
    "desviacion_maxima_abs": float(np.max(np.abs(freq_obs - p_teo))),
    "figura": "parte3_frecuencias_vs_teoricas.png",
    "tabla": {
        "columnas": ["Token", "Prob. teórica", "Conteo observado", "Frecuencia observada", "Diferencia"],
        "filas": [[t, float(pt), int(c), float(fo), float(fo - pt)]
                  for t, pt, c, fo in zip(tokens, p_teo, conteos, freq_obs)],
    },
}
Path("resultados.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")

# --- Resumen ---
print("T3 — Comprobación empírica (N=10000, semilla=0)")
for t, pt, c, fo in zip(tokens, p_teo, conteos, freq_obs):
    print(f"  {t}: teórica={pt:.6f}  observada={fo:.6f}  (conteo={c})")
print(f"Divergencia KL (bits): {kl_bits:.8f}")
print(f"Divergencia KL (nats): {kl_nats:.8f}")
print("Figura guardada: parte3_frecuencias_vs_teoricas.png")
