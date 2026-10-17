"""
T2: Normalizar la matriz de características de T1 mediante Z-score (vectorizado)
y calcular load_score = zscores @ [0.50, 0.30, 0.20].
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1. Recuperar la matriz de características definida en T1
# ----------------------------------------------------------------------------
RUTA_CSV = "data/server_measurements.csv"
RUTA_T1 = Path("entrada/T1/resultados.json")
RUTA_PARQUET = Path("output/server_analysis.parquet")
PESOS = np.array([0.50, 0.30, 0.20])
FEATURES_POR_DEFECTO = ["gpu_utilization", "cpu_utilization", "memory_gb"]

feature_names = list(FEATURES_POR_DEFECTO)
if RUTA_T1.exists():
    with open(RUTA_T1, "r", encoding="utf-8") as f:
        t1 = json.load(f)
    nombres_t1 = t1.get("feature_names")
    if isinstance(nombres_t1, list) and len(nombres_t1) == 3:
        feature_names = [str(n) for n in nombres_t1]

df = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"])

try:
    values = df[feature_names].to_numpy(dtype=np.float64)
except (KeyError, TypeError, ValueError) as exc:
    raise ValueError(f"No se pudo construir la matriz numérica de características: {exc}") from exc

# Validaciones estilo T1
if values.ndim != 2:
    raise ValueError("La matriz de características debe ser bidimensional")
if values.shape[1] != len(feature_names):
    raise ValueError("El número de columnas no coincide con feature_names")
if not np.issubdtype(values.dtype, np.number):
    raise ValueError("Los datos deben ser numéricos")

# ----------------------------------------------------------------------------
# 2. Media y desviación estándar por columna (operaciones vectorizadas)
# ----------------------------------------------------------------------------
means = values.mean(axis=0)
stds = values.std(axis=0, ddof=0)  # desviación estándar poblacional

# ----------------------------------------------------------------------------
# 3. Z-score vectorizado (sin for sobre observaciones), evitando división por 0
# ----------------------------------------------------------------------------
stds_seguras = np.where(stds == 0.0, 1.0, stds)
zscores = (values - means) / stds_seguras

# ----------------------------------------------------------------------------
# 4. load_score = zscores @ [0.50, 0.30, 0.20] (vectorizado)
# ----------------------------------------------------------------------------
load_score = zscores @ PESOS

# ----------------------------------------------------------------------------
# 5. Verificaciones del criterio de éxito
# ----------------------------------------------------------------------------
medias_z = zscores.mean(axis=0)
stds_z = zscores.std(axis=0, ddof=0)
max_error_media = float(np.max(np.abs(medias_z)))
max_error_std = float(np.max(np.abs(stds_z - 1.0)))
max_error_load = float(np.max(np.abs(load_score - zscores @ PESOS)))

assert zscores.shape == (values.shape[0], 3), f"zscores.shape incorrecto: {zscores.shape}"
assert load_score.shape == (values.shape[0],), f"load_score.shape incorrecto: {load_score.shape}"
assert max_error_media <= 1e-10, f"Media de zscores distinta de 0: {max_error_media}"
assert max_error_std <= 1e-10, f"Desviación de zscores distinta de 1: {max_error_std}"
assert max_error_load <= 1e-12, f"load_score no coincide con zscores @ pesos: {max_error_load}"

# ----------------------------------------------------------------------------
# 6. Salidas exigidas por el enunciado
# ----------------------------------------------------------------------------
RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df_salida = df.copy()
df_salida["load_score"] = load_score
df_salida.to_parquet(str(RUTA_PARQUET), index=False)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(load_score, bins=30, color="#4C72B0", edgecolor="white")
ax.set_xlabel("load_score")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución de load_score (T2)")
fig.tight_layout()
fig.savefig("load_score_histograma.png", dpi=120)
plt.close(fig)

resultados = {
    "subtarea": "T2_zscore_load_score",
    "archivo_datos": RUTA_CSV,
    "feature_names": feature_names,
    "n_observaciones": int(values.shape[0]),
    "means": means.tolist(),
    "stds": stds.tolist(),
    "stds_cero_detectados": [bool(s == 0.0) for s in stds],
    "pesos": PESOS.tolist(),
    "zscores_shape": list(zscores.shape),
    "load_score_shape": list(load_score.shape),
    "zscores_medias_por_columna": medias_z.tolist(),
    "zscores_desviaciones_por_columna": stds_z.tolist(),
    "max_error_media_zscore": max_error_media,
    "max_error_std_zscore": max_error_std,
    "max_error_load_score": max_error_load,
    "load_score": load_score.tolist(),
    "load_score_min": float(load_score.min()),
    "load_score_max": float(load_score.max()),
    "load_score_media": float(load_score.mean()),
    "archivo_parquet": str(RUTA_PARQUET),
    "figura": "load_score_histograma.png",
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2)

# ----------------------------------------------------------------------------
# 7. Resumen
# ----------------------------------------------------------------------------
print("means:", means)
print("zscores.shape    ->", zscores.shape)
print("load_score.shape ->", load_score.shape)
print("Medias de zscores por columna:", medias_z)
print("Desviaciones de zscores por columna:", stds_z)
print(f"max|media zscore - 0| = {max_error_media:.3e} (tol 1e-10)")
print(f"max|std zscore - 1|   = {max_error_std:.3e} (tol 1e-10)")
print(f"max|load_score - zscores@pesos| = {max_error_load:.3e}")
print(f"load_score: min={load_score.min():.6f}, max={load_score.max():.6f}, media={load_score.mean():.6f}")
print(f"Parquet escrito en {RUTA_PARQUET} ({len(df_salida)} filas).")
