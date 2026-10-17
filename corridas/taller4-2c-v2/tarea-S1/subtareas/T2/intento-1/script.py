import json
from pathlib import Path
import numpy as np
import pandas as pd


class ServerMeasurements:
    def __init__(self, values, feature_names):
        arr = np.asarray(values)
        if arr.ndim != 2:
            raise ValueError("values debe ser una matriz NumPy 2D")
        names = tuple(feature_names)
        if arr.shape[1] != len(names):
            raise ValueError("el número de columnas no coincide con feature_names")
        try:
            arr = arr.astype(float)
        except (TypeError, ValueError):
            raise ValueError("los datos no son numéricos")
        if not np.isfinite(arr).all():
            raise ValueError("los datos no son numéricos")
        self.values = arr
        self.feature_names = names

    def __repr__(self):
        return f"ServerMeasurements(shape={self.values.shape}, features={self.feature_names})"


def build_measurements(df):
    cols = ["gpu_utilization", "cpu_utilization", "memory_gb"]
    return ServerMeasurements(df[cols].to_numpy(dtype=float), cols)


# --- Datos y matriz de T1 ---
df = pd.read_csv("data/server_measurements.csv", parse_dates=["timestamp"])
sm = build_measurements(df)
X = sm.values

# --- Estadísticos por columna (vectorizado, sin for sobre observaciones) ---
means = X.mean(axis=0)
stds = X.std(axis=0)  # ddof=0

# --- Z-score evitando división por cero ---
safe_stds = np.where(stds == 0, 1.0, stds)
zscores = (X - means) / safe_stds

# --- load_score vectorizado ---
weights = np.array([0.50, 0.30, 0.20])
load_score = zscores @ weights

# --- Verificaciones ---
expected_means = np.array([57.99313333, 50.54163333, 29.45533333])
col_means = zscores.mean(axis=0)
col_stds = zscores.std(axis=0)
load_check = zscores[:, 0] * 0.50 + zscores[:, 1] * 0.30 + zscores[:, 2] * 0.20

coincide_t1 = None
p_t1 = Path("entrada/T1/resultados.json")
if p_t1.exists():
    with open(p_t1) as f:
        t1 = json.load(f)
    t1_means = np.array([t1["medias_por_feature"][c] for c in sm.feature_names])
    coincide_t1 = bool(np.allclose(means, t1_means, rtol=0, atol=1e-9))

results = {
    "feature_names": list(sm.feature_names),
    "means": means.tolist(),
    "stds": stds.tolist(),
    "stds_cero": int((stds == 0).sum()),
    "zscores_shape": list(zscores.shape),
    "load_score_shape": list(load_score.shape),
    "zscores_column_means": col_means.tolist(),
    "zscores_column_stds": col_stds.tolist(),
    "verificaciones": {
        "means_coinciden_esperado": bool(np.allclose(means, expected_means, rtol=0, atol=1e-6)),
        "max_diff_medias_esperadas": float(np.max(np.abs(means - expected_means))),
        "zscores_shape_ok": zscores.shape == (300, 3),
        "load_score_shape_ok": load_score.shape == (300,),
        "media_0_tol_1e-10": bool(np.all(np.abs(col_means) < 1e-10)),
        "std_1_tol_1e-10": bool(np.all(np.abs(col_stds - 1.0) < 1e-10)),
        "load_score_igual_suma_ponderada": bool(np.allclose(load_score, load_check, rtol=0, atol=1e-12)),
        "max_diff_load_score": float(np.max(np.abs(load_score - load_check))),
        "coincide_con_T1": coincide_t1,
    },
    "load_score_resumen": {
        "min": float(load_score.min()),
        "max": float(load_score.max()),
        "mean": float(load_score.mean()),
    },
}

# --- Archivos exigidos y artefactos para subtareas posteriores ---
Path("output").mkdir(parents=True, exist_ok=True)
df_out = df.copy()
df_out["load_score"] = load_score
df_out.to_parquet("output/server_analysis.parquet", index=False)

np.save("zscores.npy", zscores)
np.save("load_score.npy", load_score)

with open("resultados.json", "w") as f:
    json.dump(results, f, indent=2)

print("repr:", repr(sm))
print("means:", means)
print("stds:", stds)
print("zscores.shape:", zscores.shape, "| load_score.shape:", load_score.shape)
print("zscores col means:", col_means, "| col stds:", col_stds)
print("load_score min/mean/max:", load_score.min(), load_score.mean(), load_score.max())
print("verificaciones:", json.dumps(results["verificaciones"], indent=2))
