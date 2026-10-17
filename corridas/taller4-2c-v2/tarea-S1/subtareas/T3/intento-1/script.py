import json
from pathlib import Path

import numpy as np
import pandas as pd

# --- Datos de la tarea ---
df = pd.read_csv("data/server_measurements.csv", parse_dates=["timestamp"])

# --- Resultados de T2: medias y desviaciones para reproducir load_score ---
with open("entrada/T2/resultados.json", encoding="utf-8") as f:
    t2 = json.load(f)

feats = t2["feature_names"]
means = np.asarray(t2["means"], dtype=float)
stds = np.asarray(t2["stds"], dtype=float)
weights = np.array([0.50, 0.30, 0.20])

X = df[feats].to_numpy(dtype=float)
stds_safe = np.where(stds == 0.0, 1.0, stds)  # evitar división por cero
load_score = ((X - means) / stds_safe) @ weights
assert load_score.shape == (len(df),)

# --- Copia del original (el original no se modifica) ---
df_snapshot = df.copy(deep=True)
df_new = df.copy()
df_new["load_score"] = load_score
df_new["requires_review"] = (
    (df_new["load_score"] > 1.5) | (df_new["temperature_c"] > 80)
).astype(bool)

# --- Verificaciones ---
original_sin_cambios = (
    df.equals(df_snapshot)
    and "load_score" not in df.columns
    and "requires_review" not in df.columns
)
copia_fiel = df_new[list(df.columns)].equals(df)
dtype_bool = str(df_new["requires_review"].dtype) == "bool"


def jsonable(v):
    if isinstance(v, pd.Timestamp):
        return str(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


fila = df_new.loc[23]
fila23 = {c: jsonable(fila[c]) for c in df_new.columns}

esperado = {
    "server": "AI-SRV-01",
    "timestamp": "2026-08-24 09:55:00",
    "load_score": 3.136218,
    "temperature_c": 66.97,
    "requires_review": True,
}
fila23_ok = (
    fila23["server"] == esperado["server"]
    and fila23["timestamp"] == esperado["timestamp"]
    and abs(fila23["load_score"] - esperado["load_score"]) <= 5e-7
    and abs(fila23["temperature_c"] - esperado["temperature_c"]) <= 1e-9
    and fila23["requires_review"] is True
)

por_servidor = (
    df_new.loc[df_new["requires_review"], "server"].value_counts().sort_index().to_dict()
)

resultados = {
    "subtarea": "T3",
    "n_observaciones": int(df_new.shape[0]),
    "columnas_original": list(df.columns),
    "columnas_df_nuevo": list(df_new.columns),
    "requires_review_dtype": str(df_new["requires_review"].dtype),
    "requires_review_es_bool": bool(dtype_bool),
    "requires_review_true": int(df_new["requires_review"].sum()),
    "requires_review_true_por_servidor": {k: int(v) for k, v in por_servidor.items()},
    "umbral_load_score": 1.5,
    "umbral_temperature_c": 80.0,
    "original_sin_cambios": bool(original_sin_cambios),
    "copia_fiel_del_original": bool(copia_fiel),
    "fila_23": fila23,
    "fila_23_esperado": esperado,
    "fila_23_coincide": bool(fila23_ok),
    "load_score_resumen": {
        "min": float(load_score.min()),
        "max": float(load_score.max()),
        "mean": float(load_score.mean()),
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# --- Archivo exigido por el enunciado ---
Path("output").mkdir(parents=True, exist_ok=True)
df_new.to_parquet("output/server_analysis.parquet", index=False)

print("T3 completada.")
print("Original sin cambios:", original_sin_cambios, "| copia fiel:", copia_fiel)
print(
    "requires_review=True:",
    int(df_new["requires_review"].sum()),
    "filas | dtype:",
    df_new["requires_review"].dtype,
)
print("Fila 23:", fila23)
print("Fila 23 coincide con lo esperado:", fila23_ok)
print("Parquet escrito en output/server_analysis.parquet")
