import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20260829
BOOTSTRAP_RESAMPLES = 10_000

# --- Cargar datos y construir vector de diferencias pareadas (una diferencia por tienda) ---
df = pd.read_csv("data/model_errors.csv")
req = ["store_id", "error_a", "error_b"]
if list(df.columns[:3]) != req:
    raise ValueError(f"Columnas esperadas {req}, encontradas {list(df.columns)}")
if df["store_id"].duplicated().any() or df["store_id"].isna().any():
    raise ValueError("store_id debe ser único y no nulo")
diffs = (df["error_a"].astype(float) - df["error_b"].astype(float)).to_numpy()
n = len(diffs)

# --- Bootstrap pareado: se remuestrean TIENDAS COMPLETAS (índices) sobre el vector de diferencias ---
def bootstrap_pareado(seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))  # tiendas con reemplazo
    return diffs[idx].mean(axis=1)                            # media de diferencias por remuestra

boot_means_1 = bootstrap_pareado(SEED)
ci1 = np.percentile(boot_means_1, [2.5, 97.5])

# --- Verificación de reproducibilidad: segunda ejecución con la misma semilla ---
boot_means_2 = bootstrap_pareado(SEED)
ci2 = np.percentile(boot_means_2, [2.5, 97.5])
means_identical = bool(np.array_equal(boot_means_1, boot_means_2))
ci_identical = bool(np.array_equal(ci1, ci2))
reproducible = means_identical and ci_identical
if not reproducible:
    raise RuntimeError("El bootstrap no fue reproducible con la misma semilla")

obs_mean = float(diffs.mean())

# --- Tabla pareada -> output/store_differences.csv ---
tabla = df[["store_id", "error_a", "error_b"]].copy()
tabla["difference"] = diffs
tabla["favors"] = np.where(diffs > 0, "B", np.where(diffs < 0, "A", "empate"))
Path("output").mkdir(parents=True, exist_ok=True)
tabla.to_csv("output/store_differences.csv", index=False)

# --- Figura -> output/store_differences.png ---
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#1f77b4" if d > 0 else "#d62728" for d in diffs]
ax.bar(tabla["store_id"], diffs, color=colors)
ax.axhline(0, color="black", lw=0.8)
ax.axhline(obs_mean, color="green", lw=1.5, label=f"Media observada = {obs_mean:.4f}")
ax.axhline(ci1[0], color="orange", ls="--", lw=1.5, label=f"IC 95% percentil = [{ci1[0]:.4f}, {ci1[1]:.4f}]")
ax.axhline(ci1[1], color="orange", ls="--", lw=1.5)
ax.set_xlabel("Tienda")
ax.set_ylabel("Diferencia (error_a − error_b)")
ax.set_title(f"Bootstrap pareado por tienda completa\nSEED={SEED}, {BOOTSTRAP_RESAMPLES} remuestras (análisis de sensibilidad)")
ax.legend(fontsize=9)
fig.tight_layout()
plt.savefig("output/store_differences.png", dpi=120)
plt.close(fig)

# --- resultados.json (contrato) ---
resultados = {
    "seed": SEED,
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "n_stores": n,
    "resampling_unit": "tienda completa (índices remuestreados sobre el vector de diferencias; error_a y error_b nunca por separado)",
    "observed_mean_difference": obs_mean,
    "bootstrap_mean": float(boot_means_1.mean()),
    "bootstrap_std": float(boot_means_1.std(ddof=1)),
    "ci95_percentile_low": float(ci1[0]),
    "ci95_percentile_high": float(ci1[1]),
    "ci95_percentile": [float(ci1[0]), float(ci1[1])],
    "reproducibility": {
        "second_run_ci95_percentile": [float(ci2[0]), float(ci2[1])],
        "intervals_identical": ci_identical,
        "bootstrap_means_identical": means_identical,
        "reproducible": reproducible,
    },
    "table": tabla.to_dict(orient="records"),
}
with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

print("Bootstrap pareado (remuestreando tiendas completas vía vector de diferencias)")
print(f"SEED={SEED}, remuestras={BOOTSTRAP_RESAMPLES}, tiendas={n}")
print(f"Media observada de diferencias: {obs_mean:.6f}")
print(f"IC 95% percentil: [{ci1[0]:.6f}, {ci1[1]:.6f}]")
print(f"Reproducibilidad (2 ejecuciones, misma semilla): intervalos idénticos={ci_identical}, medias bootstrap idénticas={means_identical}")
print("Archivos: output/store_differences.png, output/store_differences.csv, resultados.json")
