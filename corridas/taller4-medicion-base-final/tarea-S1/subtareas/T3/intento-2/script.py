# -*- coding: utf-8 -*-
"""
SUBTAREA T3 — Identificar observaciones para revisión.

Crea una COPIA del DataFrame original (data/server_measurements.csv) y le agrega:
  * load_score      : valores calculados en T2 (entrada/T2/resultados.json).
  * requires_review : bool; True si load_score > 1.5 o temperature_c > 80.
Muestra la fila que requiere revisión, verifica que el DataFrame original
queda intacto (sin columnas nuevas) y guarda el DataFrame enriquecido en
output/server_analysis.parquet (archivo exigido por el enunciado).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ rutas
CSV_PATH = Path("data/server_measurements.csv")
T2_JSON = Path("entrada/T2/resultados.json")
PARQUET_PATH = Path("output/server_analysis.parquet")
RESULTS_PATH = Path("resultados.json")

EXPECTED_COLS = ["server", "timestamp", "gpu_utilization", "cpu_utilization",
                 "memory_gb", "power_w", "temperature_c"]
FEATURES = ["gpu_utilization", "cpu_utilization", "memory_gb"]
WEIGHTS_DEFAULT = [0.50, 0.30, 0.20]
UMBRAL_LOAD = 1.5
UMBRAL_TEMP = 80.0


# -------------------------------------------------------------- utilidades
def norm_col(name):
    c = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    while c.endswith("_"):
        c = c[:-1]
    return c


def to_native(obj):
    """Convierte tipos numpy/pandas a nativos de Python (para JSON)."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


# ------------------------------------------- 1) carga del DataFrame original
df_original = pd.read_csv(CSV_PATH)
df_original.columns = [norm_col(c) for c in df_original.columns]
unnamed = [c for c in df_original.columns if c.startswith("unnamed")]
if unnamed:
    df_original = df_original.drop(columns=unnamed)
if "server" in df_original.columns:
    df_original["server"] = df_original["server"].astype(str).str.strip()
if "timestamp" in df_original.columns and not pd.api.types.is_datetime64_any_dtype(
        df_original["timestamp"]):
    df_original["timestamp"] = pd.to_datetime(df_original["timestamp"])

faltantes = [c for c in EXPECTED_COLS if c not in df_original.columns]
if faltantes:
    raise ValueError(f"Faltan columnas esperadas en el CSV: {faltantes}")

n_rows, n_cols = df_original.shape
original_columns = list(df_original.columns)
original_snapshot = df_original.copy(deep=True)  # para verificar no-modificación

print("=== DataFrame original ===")
print(f"Forma: {df_original.shape}")
print(f"Columnas: {original_columns}")

# ------------------------------------------ 2) load_score proveniente de T2
t2 = {}
if T2_JSON.exists():
    with T2_JSON.open("r", encoding="utf-8") as fh:
        t2 = json.load(fh)
else:
    print(f"Aviso: no se encontró {T2_JSON}; se recalculará load_score.")

load_score = None
load_score_source = None
for key in ("load_score", "load_score_values", "load_score_vector"):
    if key in t2:
        cand = np.asarray(t2[key], dtype=float).ravel()
        if cand.shape == (n_rows,):
            load_score = cand
            load_score_source = f"{str(T2_JSON)}::{key}"
            break

if load_score is None:
    X = df_original[FEATURES].to_numpy(dtype=float)
    if "means" in t2 and "stds" in t2:
        means = np.asarray(t2["means"], dtype=float)
        stds = np.asarray(t2["stds"], dtype=float)
        weights = np.asarray(t2.get("weights", WEIGHTS_DEFAULT), dtype=float)
        nota = "means/stds/weights leídos de entrada/T2/resultados.json"
    else:
        means = X.mean(axis=0)
        stds = X.std(axis=0, ddof=0)
        weights = np.asarray(WEIGHTS_DEFAULT, dtype=float)
        nota = "means/stds recalculados del CSV (ddof=0)"
    stds_safe = np.where(stds == 0.0, 1.0, stds)       # protección división por cero
    load_score = ((X - means) / stds_safe) @ weights   # operaciones vectorizadas
    load_score_source = f"recalculado ({nota})"

# --------------------------------- 3) copia del original + columnas nuevas
df_review = df_original.copy(deep=True)  # el original NO se modifica
df_review["load_score"] = load_score
df_review["requires_review"] = (
    (df_review["load_score"] > UMBRAL_LOAD)
    | (df_review["temperature_c"] > UMBRAL_TEMP)
).astype(bool)

# ------------------------------------- 4) fila(s) que requieren revisión
mask = df_review["requires_review"].to_numpy(dtype=bool)
review_df = df_review.loc[
    mask, ["server", "timestamp", "load_score", "temperature_c", "requires_review"]
]

print("\n=== Copia enriquecida (df_review) — primeras filas ===")
print(df_review.head().to_string())
print(f"\nForma de la copia: {df_review.shape}")
print("Dtypes de las columnas nuevas:")
print(df_review[["load_score", "requires_review"]].dtypes.to_string())

print(f"\n=== Fila(s) que requieren revisión "
      f"(load_score > {UMBRAL_LOAD} o temperature_c > {UMBRAL_TEMP}) ===")
print(review_df.to_string())

# ------------------------------------------------------- 5) verificaciones
original_intacto = bool(
    df_original.equals(original_snapshot)
    and "load_score" not in df_original.columns
    and "requires_review" not in df_original.columns
    and list(df_original.columns) == original_columns
    and df_original.shape == (n_rows, n_cols)
)
objetos_distintos = bool(df_review is not df_original)

t2_match = None
if "load_score" in t2:
    t2_ls = np.asarray(t2["load_score"], dtype=float).ravel()
    if t2_ls.shape == load_score.shape:
        t2_match = bool(np.allclose(t2_ls, load_score, rtol=0.0, atol=1e-12))

# criterio de éxito: fila 23
row23 = df_review.iloc[23]
fila23 = {
    "index_label": int(df_review.index[23]),
    "server": str(row23["server"]),
    "timestamp": str(row23["timestamp"]),
    "load_score": float(row23["load_score"]),
    "temperature_c": float(row23["temperature_c"]),
    "requires_review": bool(row23["requires_review"]),
}
criterio_ok = bool(
    fila23["index_label"] == 23
    and fila23["server"] == "AI-SRV-01"
    and fila23["timestamp"] == "2026-08-24 09:55:00"
    and abs(fila23["load_score"] - 3.136218) <= 5e-7
    and abs(fila23["temperature_c"] - 66.97) <= 1e-9
    and fila23["requires_review"]
)

load_gt = df_review["load_score"] > UMBRAL_LOAD
temp_gt = df_review["temperature_c"] > UMBRAL_TEMP
breakdown = {
    "n_load_score_gt_1_5_only": int((load_gt & ~temp_gt).sum()),
    "n_temperature_gt_80_only": int((~load_gt & temp_gt).sum()),
    "n_both_conditions": int((load_gt & temp_gt).sum()),
    "n_requires_review_total": int((load_gt | temp_gt).sum()),
}

review_records = [
    {
        "index": int(idx),
        "server": str(r["server"]),
        "timestamp": str(r["timestamp"]),
        "load_score": float(r["load_score"]),
        "temperature_c": float(r["temperature_c"]),
        "requires_review": bool(r["requires_review"]),
    }
    for idx, r in review_df.iterrows()
]

# ----------------------------- 6) parquet exigido por el enunciado
PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
df_review.to_parquet(PARQUET_PATH, engine="pyarrow", index=False)
_check = pd.read_parquet(PARQUET_PATH)
parquet_ok = bool(list(_check.columns) == list(df_review.columns)
                  and _check.shape == df_review.shape)

# ------------------------------------------------------- 7) resultados.json
results = {
    "subtask": "T3",
    "data_source": str(CSV_PATH),
    "n_rows": int(n_rows),
    "n_cols_original": int(n_cols),
    "original_columns": original_columns,
    "original_unchanged": original_intacto,
    "original_has_new_columns": bool(not original_intacto),
    "copy_is_independent_object": objetos_distintos,
    "copy_columns": list(df_review.columns),
    "copy_has_load_score": bool("load_score" in df_review.columns),
    "copy_has_requires_review": bool("requires_review" in df_review.columns),
    "load_score_dtype": str(df_review["load_score"].dtype),
    "requires_review_dtype": str(df_review["requires_review"].dtype),
    "load_score_source": load_score_source,
    "load_score_matches_T2": t2_match,
    "condition": "requires_review = (load_score > 1.5) | (temperature_c > 80)",
    "thresholds": {"load_score": UMBRAL_LOAD, "temperature_c": UMBRAL_TEMP},
    "breakdown": breakdown,
    "n_requires_review": int(mask.sum()),
    "review_rows": review_records,
    "expected_row_23": fila23,
    "success_criteria_row23_ok": criterio_ok,
    "parquet_written": str(PARQUET_PATH),
    "parquet_roundtrip_ok": parquet_ok,
}
with RESULTS_PATH.open("w", encoding="utf-8") as fh:
    json.dump(to_native(results), fh, indent=2, ensure_ascii=False)

# ------------------------------------------------------------- 8) resumen
print("\n=== Resumen T3 ===")
print(f"Filas procesadas: {n_rows}")
print(f"DataFrame original intacto (sin columnas nuevas): {original_intacto}")
print(f"Copia con 'load_score' y 'requires_review': "
      f"{'load_score' in df_review.columns and 'requires_review' in df_review.columns}")
print(f"Tipo de 'requires_review': {df_review['requires_review'].dtype}")
print(f"Filas que requieren revisión: {int(mask.sum())}")
for rec in review_records:
    print("  ", rec)
print(f"load_score coincide con T2: {t2_match}")
print(f"Parquet escrito en: {PARQUET_PATH} (roundtrip OK: {parquet_ok})")
print("Criterio de éxito (fila 23 = AI-SRV-01, load_score≈3.136218, "
      f"temperature_c=66.97, requires_review=True): "
      f"{'CUMPLE' if criterio_ok else 'REVISAR'}")
print(f"Resultados escritos en: {RESULTS_PATH}")
