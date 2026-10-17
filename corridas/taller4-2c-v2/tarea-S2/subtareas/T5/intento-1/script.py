import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- datos ----------
df = pd.read_csv("data/model_errors.csv")
req = ["store_id", "error_a", "error_b"]
faltan = [c for c in req if c not in df.columns]
if faltan:
    raise ValueError(f"Faltan columnas requeridas: {faltan}")
if df["store_id"].isna().any() or df["store_id"].duplicated().any():
    raise ValueError("store_id debe ser único y no nulo")
d = (df["error_a"].astype(float) - df["error_b"].astype(float)).to_numpy()
if not np.all(np.isfinite(d)):
    raise ValueError("Las diferencias deben ser finitas")
n = int(d.size)
if n != 16:
    raise ValueError(f"Se esperaban 16 tiendas, hay {n}")

df["difference"] = d
df["favors"] = np.where(d > 0, "B", np.where(d < 0, "A", "tie"))
obs_mean = float(d.mean())

# ---------- prueba t (procedimiento principal): p-value de T4 o recálculo ----------
t_info = {"alternative": "two-sided", "null_mean": 0.0}
fuente_t = "recalculada con scipy.stats"
ruta_t4 = Path("entrada/T4/resultados.json")
if ruta_t4.exists():
    with open(ruta_t4) as fh:
        t4 = json.load(fh)
    if not np.allclose(np.asarray(t4["differences"], dtype=float), d, atol=1e-9):
        raise ValueError("Las diferencias de T4 no coinciden con las recalculadas")
    t_info.update({
        "mean": float(t4["mean"]),
        "sample_sd": float(t4["sample_sd"]),
        "t_statistic": float(t4["t_statistic"]),
        "degrees_of_freedom": int(t4["degrees_of_freedom"]),
        "p_value_two_sided": float(t4["p_value_two_sided"]),
    })
    fuente_t = "entrada/T4/resultados.json"
else:
    tt = stats.ttest_1samp(d, 0.0, alternative="two-sided")
    t_info.update({
        "mean": obs_mean,
        "sample_sd": float(d.std(ddof=1)),
        "t_statistic": float(tt.statistic),
        "degrees_of_freedom": int(n - 1),
        "p_value_two_sided": float(tt.pvalue),
    })
p_t = float(t_info["p_value_two_sided"])

# ---------- enumeración exacta de las 2^16 configuraciones de signos ----------
total = 2 ** n
if total != 65536:
    raise ValueError("El total de configuraciones debe ser 65536")
bits = np.arange(n, dtype=np.int64)
signos = (((np.arange(total, dtype=np.int64)[:, None] >> bits[None, :]) & 1)
          .astype(np.float64) * 2.0 - 1.0)
medias_nulas = (signos @ d) / n
del signos

tol = 1e-9  # las diferencias son múltiplos de 0.1; el ruido float es ~1e-14
abs_nulas = np.abs(medias_nulas)
abs_obs = abs(obs_mean)
extremas = abs_nulas >= abs_obs - tol
conteo_extremas = int(extremas.sum())
conteo_iguales = int(np.sum(np.abs(abs_nulas - abs_obs) <= tol))
conteo_mas_extremas = conteo_extremas - conteo_iguales
p_signflip = conteo_extremas / total

if not extremas[0]:
    raise ValueError("La configuración observada (todos los signos +1) debe contarse como extrema")
if conteo_iguales < 2:
    raise ValueError("Deben contarse al menos la configuración observada y su negación")
if not np.isclose(abs_nulas.max(), np.abs(d).sum() / n, atol=1e-9):
    raise ValueError("El máximo |media nula| debe ser sum|d|/n")

# ---------- archivos exigidos ----------
Path("output").mkdir(parents=True, exist_ok=True)
df[["store_id", "error_a", "error_b", "difference", "favors"]].to_csv(
    "output/store_differences.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
ax = axes[0]
ax.bar(df["store_id"], d, color=np.where(d > 0, "#1f77b4", "#d62728"))
ax.axhline(0.0, color="black", lw=0.8)
ax.axhline(obs_mean, color="green", ls="--", lw=1.3,
           label=f"media observada = {obs_mean:.4f}")
ax.set_ylabel("difference = error_a - error_b")
ax.set_title("Diferencias pareadas por tienda")
ax.tick_params(axis="x", rotation=90)
ax.legend(fontsize=9)

ax = axes[1]
bins = np.linspace(medias_nulas.min(), medias_nulas.max(), 121)
ax.hist(medias_nulas, bins=bins, color="#9ecae1", label="65536 medias nulas")
ax.hist(medias_nulas[extremas], bins=bins, color="#d62728",
        label=f"configuraciones extremas: {conteo_extremas}")
ax.axvline(obs_mean, color="green", lw=1.6)
ax.axvline(-obs_mean, color="green", lw=1.6, ls="--")
ax.set_xlabel("media nula bajo cambios de signo")
ax.set_title("Referencia nula exacta por cambios de signo\n"
             f"p = {conteo_extremas}/{total} = {p_signflip:.6f}  |  p_t = {p_t:.6f}")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("output/store_differences.png", dpi=120)
plt.close(fig)

# ---------- resultados.json ----------
resultados = {
    "subtask": "T5",
    "test": "prueba exacta por cambios de signo (enumeración completa de las 2^16 configuraciones de signos de las diferencias)",
    "n": n,
    "differences": [float(x) for x in d],
    "observed_mean_difference": obs_mean,
    "total_configurations": int(total),
    "count_extreme_configurations": conteo_extremas,
    "n_configurations_equal_to_observed": conteo_iguales,
    "n_configurations_more_extreme": conteo_mas_extremas,
    "p_value_signflip_exact": p_signflip,
    "p_value_fraction": [conteo_extremas, int(total)],
    "null_mean_min": float(medias_nulas.min()),
    "null_mean_max": float(medias_nulas.max()),
    "max_abs_null_mean": float(abs_nulas.max()),
    "t_test_main": {**t_info, "source": fuente_t},
    "p_value_t": p_t,
    "abs_difference_pvalues": abs(p_t - p_signflip),
    "relative_difference_vs_t": abs(p_t - p_signflip) / p_t,
    "not_interchangeable": True,
    "comparison_note": (
        "P-value exacto condicional a H0 (diferencias simétricas alrededor de 0): fracción de las "
        "65536 configuraciones de signos cuya media nula tiene magnitud >= |media observada|. "
        "La prueba t (procedimiento principal) asume diferencias normales y usa la distribución t "
        "con varianza estimada (gl=15). Los p-values son numéricamente cercanos pero los "
        "procedimientos NO son intercambiables: supuesto distribucional distinto (normalidad vs "
        "solo simetría) y referencia nula distinta (distribución t teórica vs enumeración exacta)."
    ),
}
with open("resultados.json", "w") as fh:
    json.dump(resultados, fh, indent=2)

print(f"n={n} | media observada de las diferencias = {obs_mean:.6f}")
print(f"Configuraciones totales = {total} | extremas = {conteo_extremas} "
      f"({conteo_iguales} con |media| igual a la observada + {conteo_mas_extremas} más extremas)")
print(f"p-value exacto por cambios de signo = {conteo_extremas}/{total} = {p_signflip:.8f}")
print(f"p-value t bilateral (principal) = {p_t:.8f} "
      f"(t={t_info['t_statistic']:.4f}, gl={t_info['degrees_of_freedom']}, fuente={fuente_t})")
print(f"|p_t - p_cambios_de_signo| = {abs(p_t - p_signflip):.8f} "
      f"(procedimientos no intercambiables)")
print("Escritos: resultados.json, output/store_differences.png, output/store_differences.csv")
