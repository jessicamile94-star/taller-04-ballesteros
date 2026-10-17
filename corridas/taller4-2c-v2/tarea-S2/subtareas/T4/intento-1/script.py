import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ---------- Carga y validación de datos ----------
csv_path = Path("data/model_errors.csv")
df = pd.read_csv(csv_path)
req = ["store_id", "error_a", "error_b"]
missing = [c for c in req if c not in df.columns]
if missing:
    raise ValueError(f"Faltan columnas {missing} en {csv_path}")
if df["store_id"].isna().any() or df["store_id"].duplicated().any():
    raise ValueError("store_id contiene valores nulos o duplicados")
for c in ["error_a", "error_b"]:
    v = pd.to_numeric(df[c], errors="coerce")
    if v.isna().any() or not np.isfinite(v.to_numpy(dtype=float)).all() or (v < 0).any():
        raise ValueError(f"Valores no numéricos, no finitos o negativos en columna {c}")
    df[c] = v.astype(float)

# ---------- Vector de diferencias (pareada: error_a − error_b) ----------
df["difference"] = df["error_a"] - df["error_b"]
df["favors"] = np.where(df["difference"] > 0, "B", np.where(df["difference"] < 0, "A", "tie"))

d = df["difference"].to_numpy(dtype=float)
n = int(d.size)
if n < 2:
    raise ValueError("Se requieren al menos 2 diferencias para la prueba t")

# ---------- Estadísticos exactos ----------
mean = float(np.mean(d))
sample_sd = float(np.std(d, ddof=1))          # desviación muestral (ddof=1)
standard_error = float(sample_sd / np.sqrt(n))  # SE = sd / sqrt(n)
dfree = n - 1

assert abs(standard_error - sample_sd / np.sqrt(n)) < 1e-12

# ---------- Prueba t de una muestra bilateral sobre las diferencias ----------
res = stats.ttest_1samp(d, 0.0, alternative="two-sided")
t_stat = float(res.statistic)
p_val = float(res.pvalue)

# ---------- IC t bilateral del 95 % para la media pareada ----------
alpha = 0.05
tcrit = float(stats.t.ppf(1 - alpha / 2, dfree))
ci_low = float(mean - tcrit * standard_error)
ci_high = float(mean + tcrit * standard_error)
ci_check = stats.t.interval(0.95, dfree, loc=mean, scale=standard_error)
assert abs(ci_check[0] - ci_low) < 1e-9 and abs(ci_check[1] - ci_high) < 1e-9

# ---------- Archivos exigidos ----------
out_dir = Path("output")
out_dir.mkdir(parents=True, exist_ok=True)

df.to_csv("output/store_differences.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 5))
colors = np.where(df["difference"].to_numpy() >= 0, "#2a9d8f", "#e76f51")
ax.bar(df["store_id"], df["difference"], color=colors, edgecolor="black", linewidth=0.4)
ax.axhline(0, color="black", lw=0.8)
ax.axhline(mean, color="#264653", lw=1.6, label=f"media = {mean:.4f}")
ax.axhspan(mean - standard_error, mean + standard_error, color="#264653", alpha=0.20,
           label=f"media ± EE = ±{standard_error:.4f}")
ax.axhspan(mean - sample_sd, mean + sample_sd, color="#e9c46a", alpha=0.18,
           label=f"media ± sd = ±{sample_sd:.4f}")
ax.set_xlabel("Tienda")
ax.set_ylabel("Diferencia (error_a − error_b)")
ax.set_title("Diferencias pareadas por tienda: media, error estándar y dispersión")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig("output/store_differences.png", dpi=120)
plt.close(fig)

table = [
    {"store_id": str(s), "error_a": float(a), "error_b": float(b),
     "difference": float(dd), "favors": str(fv)}
    for s, a, b, dd, fv in zip(df["store_id"], df["error_a"], df["error_b"],
                               df["difference"], df["favors"])
]

results = {
    "n": n,
    "mean": mean,
    "sample_sd": sample_sd,
    "standard_error": standard_error,
    "se_identity_check": bool(abs(standard_error - sample_sd / np.sqrt(n)) < 1e-12),
    "degrees_of_freedom": dfree,
    "t_statistic": t_stat,
    "p_value_two_sided": p_val,
    "null_mean": 0.0,
    "alpha": alpha,
    "confidence_level": 0.95,
    "ci95_low": ci_low,
    "ci95_high": ci_high,
    "ci95": [ci_low, ci_high],
    "test": "prueba t de una muestra bilateral sobre las diferencias (equivalente pareada)",
    "differences": [float(x) for x in d],
    "table": table,
    "interpretacion": (
        f"La diferencia media es {mean:.4f} unidades (error_a − error_b). La dispersión entre tiendas "
        f"(sd = {sample_sd:.4f}) es mucho mayor que el error estándar de la media "
        f"(EE = sd/√n = {standard_error:.4f}): la media se estima con precisión aunque las tiendas varíen. "
        f"IC t 95% = [{ci_low:.6f}, {ci_high:.6f}]; p bilateral = {p_val:.6f}."
    ),
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"n={n}  mean={mean:.6f}  sample_sd={sample_sd:.6f}  SE={standard_error:.6f}")
print(f"df={dfree}  t={t_stat:.6f}  p(bilateral)={p_val:.6f}")
print(f"IC t 95%: ({ci_low:.6f}, {ci_high:.6f})")
print("Escritos: resultados.json, output/store_differences.csv, output/store_differences.png")
