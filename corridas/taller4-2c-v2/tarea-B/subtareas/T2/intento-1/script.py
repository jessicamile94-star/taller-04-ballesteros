import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = 0.9
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0])

# Distribución T=1 obtenida en la Parte 1 (entrada/T1/resultados.json)
try:
    with open("entrada/T1/resultados.json", encoding="utf-8") as f:
        prev = json.load(f)
    tokens = list(prev["tokens"])
    probs = np.array(prev["distribuciones"]["T=1.0"], dtype=float)
except FileNotFoundError:
    z = logits - logits.max()
    probs = np.exp(z) / np.exp(z).sum()

assert abs(probs.sum() - 1.0) < 1e-9 and probs.min() >= 0

# Orden descendente por probabilidad y acumuladas
orden = np.argsort(-probs, kind="stable")
probs_ord = probs[orden]
tokens_ord = [tokens[i] for i in orden]
acum = np.cumsum(probs_ord)

# Conjunto mínimo cuya probabilidad acumulada alcanza p = 0.9
k = min(int(np.searchsorted(acum, p, side="left")) + 1, len(tokens))
nucleo_idx = orden[:k]
nucleo_tokens = tokens_ord[:k]
acum_nucleo = float(acum[k - 1])
fuera = [t for t in tokens if t not in nucleo_tokens]

# Renormalización del núcleo; tokens fuera del núcleo quedan en 0
renorm_nucleo = probs_ord[:k] / acum_nucleo
renorm = np.zeros(len(tokens))
renorm[nucleo_idx] = renorm_nucleo

# Validaciones
assert acum_nucleo >= p, "la acumulada del núcleo debe alcanzar p"
assert k == 1 or acum[k - 2] < p, "el núcleo debe ser el conjunto mínimo"
assert abs(renorm.sum() - 1.0) < 1e-12
assert np.all(renorm >= 0.0)

# Figura: distribución T=1 vs renormalizada tras top-p
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(tokens))
w = 0.38
ax.bar(x - w / 2, probs, w, label="T=1 original", color="#4C72B0")
ax.bar(x + w / 2, renorm, w, label="Top-p (p=0.9) renormalizada", color="#DD8452")
for i in range(len(tokens)):
    ax.text(x[i] - w / 2, probs[i] + 0.008, f"{probs[i]:.4f}", ha="center", fontsize=8)
    v = renorm[i]
    ax.text(x[i] + w / 2, v + 0.008, f"{v:.4f}" if v > 0 else "0", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_ylim(0, 0.75)
ax.set_ylabel("Probabilidad")
ax.set_title(f"Top-p (núcleo) con p={p} sobre la distribución T=1\n"
             f"Núcleo: {', '.join(nucleo_tokens)} (acumulada {acum_nucleo:.4f})")
ax.legend()
fig.tight_layout()
fig.savefig("top_p_renormalizada.png", dpi=120)
plt.close(fig)

tabla_filas = [
    [tokens_ord[i], float(probs_ord[i]), float(acum[i]), bool(i < k),
     float(renorm_nucleo[i]) if i < k else 0.0]
    for i in range(len(tokens))
]

resultados = {
    "p": p,
    "temperatura": 1.0,
    "tokens": tokens,
    "distribucion_T1": probs.tolist(),
    "orden_descendente": tokens_ord,
    "probabilidades_ordenadas": probs_ord.tolist(),
    "probabilidades_acumuladas": acum.tolist(),
    "tokens_nucleo": nucleo_tokens,
    "num_tokens_nucleo": k,
    "probabilidad_acumulada_nucleo": acum_nucleo,
    "tokens_fuera_nucleo": fuera,
    "distribucion_renormalizada": {t: float(v) for t, v in zip(tokens, renorm)},
    "distribucion_renormalizada_nucleo": {t: float(v) for t, v in zip(nucleo_tokens, renorm_nucleo)},
    "suma_distribucion_renormalizada": float(renorm.sum()),
    "tabla": {
        "columnas": ["Token", "Prob. T=1", "Acumulada", "En_nucleo", "Prob. renormalizada"],
        "filas": tabla_filas,
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print("Distribución T=1:", dict(zip(tokens, np.round(probs, 6))))
print("Orden descendente:", tokens_ord)
print("Acumuladas:", np.round(acum, 6).tolist())
print(f"Núcleo top-p (p={p}): {nucleo_tokens} (mínimo, acumulada={acum_nucleo:.6f} >= 0.9)")
print("Fuera del núcleo (probabilidad 0):", fuera)
print("Distribución renormalizada:", {t: round(float(v), 6) for t, v in zip(tokens, renorm)})
print("Suma renormalizada:", float(renorm.sum()))
print("Figura guardada: top_p_renormalizada.png")
