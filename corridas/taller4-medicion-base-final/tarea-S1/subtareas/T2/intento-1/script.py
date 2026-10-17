#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2 — Normalizar la matriz de características de T1 (Z-score) y calcular load_score.

- Recupera la matriz de características de T1 (300x3: gpu_utilization, cpu_utilization,
  memory_gb) desde entrada/T1/ si está disponible; si no, la reconstruye exactamente
  como build_measurements de T1 a partir de data/server_measurements.csv.
- Calcula media y desviación estándar por columna (vectorizado, ddof=0).
- Normaliza con Z-score de forma vectorizada protegiendo la división por cero
  (si std == 0 la columna es constante, luego (x - media) == 0 y se divide entre 1.0).
- Calcula load_score = zscores @ [0.50, 0.30, 0.20] con operaciones vectorizadas.
- Sin bucles for sobre observaciones.
- Escribe resultados.json y output/server_analysis.parquet (df + load_score).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

np.set_printoptions(precision=8)

TOL = 1e-10
WEIGHTS = np.array([0.50, 0.30, 0.20], dtype=float)
T1_DIR = Path("entrada/T1")
CSV_PATH = Path("data/server_measurements.csv")
PARQUET_PATH = Path("output/server_analysis.parquet")
RESULTS_PATH = Path("resultados.json")

# ---------------------------------------------------------------------------
# 1) Matriz de características producida por T1
# ---------------------------------------------------------------------------
t1 = {}
if (T1_DIR / "resultados.json").exists():
    with open(T1_DIR / "resultados.json", "r", encoding="utf-8") as fh:
        t1 = json.load(fh)

feature_names = tuple(t1.get("feature_names",
                             ["gpu_utilization", "cpu_utilization", "memory_gb"]))
expected_shape = tuple(t1.get("shape", (300, 3)))

t1_means_ref = None
if "column_means" in t1:
    try:
        t1_means_ref = np.array([float(t1["column_means"][c]) for c in feature_names])
    except (KeyError, TypeError, ValueError):
        t1_means_ref = None

df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])

values = None
values_source = None
for fname in ("values.npy", "features.npy", "measurements_values.npy",
              "values.parquet", "features.parquet", "measurements.parquet"):
    p = T1_DIR / fname
    if not p.exists():
        continue
    try:
        raw = np.load(p) if p.suffix == ".npy" else pd.read_parquet(p)
        arr = np.asarray(raw, dtype=float)
    except Exception:
        continue
    if arr.ndim == 2 and arr.shape[1] == len(feature_names):
        if t1_means_ref is None or np.allclose(arr.mean(axis=0), t1_means_ref,
                                               rtol=0.0, atol=1e-6):
            values = arr
            values_source = str(p)
            break

if values is None:
    # Reconstrucción idéntica a build_measurements de T1:
    # mismas columnas, mismo orden, convertidas a float.
    values = df.loc[:, list(feature_names)].to_numpy(dtype=float)
    values_source = (f"{CSV_PATH} (build_measurements de T1: "
                     f"{', '.join(feature_names)})")

# Comprobaciones cruzadas con T1
means_match_t1 = bool(t1_means_ref is not None
                      and np.all(np.abs(values.mean(axis=0) - t1_means_ref) <= 1e-9))
first_row_match_t1 = False
if "first_row_values" in t1:
    first_row_match_t1 = bool(np.allclose(
        values[0], np.asarray(t1["first_row_values"], dtype=float),
        rtol=0.0, atol=1e-9))

# ---------------------------------------------------------------------------
# 2) Media y desviación estándar por columna (vectorizado)
# ---------------------------------------------------------------------------
means = values.mean(axis=0)
stds = values.std(axis=0, ddof=0)   # desviación estándar poblacional

# ---------------------------------------------------------------------------
# 3) Z-score vectorizado con protección de división por cero
# ---------------------------------------------------------------------------
zero_std_mask = ~(stds > 0.0)                  # True donde std == 0 (columna constante)
stds_safe = np.where(zero_std_mask, 1.0, stds)
zscores = (values - means) / stds_safe         # broadcasting NumPy; sin bucles

# ---------------------------------------------------------------------------
# 4) load_score con operaciones vectorizadas
# ---------------------------------------------------------------------------
load_score = zscores @ WEIGHTS

# Control independiente (también vectorizado): suma ponderada por columnas
load_score_alt = (WEIGHTS[0] * zscores[:, 0]
                  + WEIGHTS[1] * zscores[:, 1]
                  + WEIGHTS[2] * zscores[:, 2])
load_score_max_diff = float(np.max(np.abs(load_score - load_score_alt)))

# ---------------------------------------------------------------------------
# 5) Verificaciones
# ---------------------------------------------------------------------------
z_col_means = zscores.mean(axis=0)
z_col_stds = zscores.std(axis=0, ddof=0)

expected_means = (t1_means_ref if t1_means_ref is not None
                  else np.array([57.993133333333326, 50.54163333333333,
                                 29.455333333333336]))
means_printed = str(means)
means_printed_ok = means_printed == "[57.99313333 50.54163333 29.45533333]"

checks = {
    "means_impresas_iguales_a_esperado": means_printed_ok,
    "means_coinciden_con_esperado": bool(np.allclose(means, expected_means,
                                                     rtol=0.0, atol=1e-9)),
    "zscores_shape_300x3": zscores.shape == (300, 3),
    "zscores_shape_coincide_con_T1": tuple(zscores.shape) == expected_shape,
    "load_score_shape_300": load_score.shape == (300,),
    "zscores_media_0_tol_1e-10": bool(np.all(np.abs(z_col_means) <= TOL)),
    "zscores_std_1_tol_1e-10": bool(np.all(np.abs(z_col_stds - 1.0) <= TOL)),
    "load_score_igual_a_suma_ponderada": bool(np.allclose(load_score, load_score_alt,
                                                          rtol=0.0, atol=1e-12)),
}

# ---------------------------------------------------------------------------
# Impresiones exigidas: medias, formas y verificaciones
# ---------------------------------------------------------------------------
print("=== T2: Z-score y load_score ===")
print("means:", means)
print("stds :", stds)
print("zscores.shape    ->", zscores.shape)
print("load_score.shape ->", load_score.shape)
print("pesos:", WEIGHTS)
print("media por columna de zscores (tol 1e-10):", z_col_means)
print("std por columna de zscores   (tol 1e-10):", z_col_stds)
print("max |media_col(zscores)|          =", float(np.max(np.abs(z_col_means))))
print("max |std_col(zscores) - 1|        =", float(np.max(np.abs(z_col_stds - 1.0))))
print("max |load_score - suma_ponderada| =", load_score_max_diff)
print("columnas con std == 0:", int(zero_std_mask.sum()))
print("fuente de values:", values_source)
print("cruce con T1 -> means_match:", means_match_t1,
      "| first_row_match:", first_row_match_t1)
print("\nVerificaciones:")
for nombre, ok in checks.items():
    print(f"  {nombre}: {'OK' if ok else 'FALLO'}")

# ---------------------------------------------------------------------------
# 6) Parquet exigido por el enunciado: df + load_score
# ---------------------------------------------------------------------------
PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
df_out = df.copy()
df_out["load_score"] = load_score
df_out.to_parquet(PARQUET_PATH, index=False)

# ---------------------------------------------------------------------------
# 7) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
resultados = {
    "subtask": "T2",
    "feature_names": list(feature_names),
    "values_shape": list(values.shape),
    "values_dtype": str(values.dtype),
    "values_source": values_source,
    "means": means.tolist(),
    "stds": stds.tolist(),
    "std_ddof": 0,
    "means_printed": means_printed,
    "means_printed_matches_expected": means_printed_ok,
    "means_match_expected": checks["means_coinciden_con_esperado"],
    "weights": WEIGHTS.tolist(),
    "zscores_shape": list(zscores.shape),
    "load_score_shape": list(load_score.shape),
    "zscores": zscores.tolist(),
    "zscores_column_means": z_col_means.tolist(),
    "zscores_column_stds": z_col_stds.tolist(),
    "zscores_mean_zero_within_1e-10": checks["zscores_media_0_tol_1e-10"],
    "zscores_std_one_within_1e-10": checks["zscores_std_1_tol_1e-10"],
    "max_abs_zscores_column_mean": float(np.max(np.abs(z_col_means))),
    "max_abs_zscores_std_minus_one": float(np.max(np.abs(z_col_stds - 1.0))),
    "n_zero_std_columns": int(zero_std_mask.sum()),
    "zero_std_columns": [feature_names[i] for i in np.flatnonzero(zero_std_mask)],
    "load_score": load_score.tolist(),
    "load_score_matches_weighted_sum": checks["load_score_igual_a_suma_ponderada"],
    "load_score_max_abs_diff_vs_weighted_sum": load_score_max_diff,
    "load_score_min": float(load_score.min()),
    "load_score_max": float(load_score.max()),
    "load_score_mean": float(load_score.mean()),
    "load_score_std": float(load_score.std()),
    "t1_cross_check": {
        "means_match": means_match_t1,
        "first_row_match": first_row_match_t1,
    },
    "parquet_path": str(PARQUET_PATH),
    "parquet_rows": int(len(df_out)),
    "parquet_columns": list(df_out.columns),
    "no_loops_over_observations": True,
    "all_checks_passed": bool(all(checks.values())),
}
with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
    json.dump(resultados, fh, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Resumen breve
# ---------------------------------------------------------------------------
print("\nResumen:")
print("  means =", means)
print("  stds  =", stds)
print("  zscores", zscores.shape, "| load_score", load_score.shape,
      "| pesos", WEIGHTS.tolist())
print("  verificaciones OK:", all(checks.values()),
      f"({sum(checks.values())}/{len(checks)})")
print("  load_score: min=%.6f max=%.6f mean=%.6f"
      % (load_score.min(), load_score.max(), load_score.mean()))
print("  Parquet escrito en:", PARQUET_PATH)
print("  resultados.json escrito con", len(resultados), "claves.")
