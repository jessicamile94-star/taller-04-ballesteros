#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4 — Construir un resumen por servidor.

Agrupa el DataFrame enriquecido de T3 por `server` y construye la tabla resumen con
exactamente las columnas: server, observations, mean_power_w, max_temperature_c,
mean_load, review_count (review_count = nº de filas con requires_review=True).

Entradas:
  - entrada/T3/  -> resultados.json y archivos del DataFrame enriquecido (fuente preferida)
  - entrada/T2/  -> resultados.json / archivos (fuente alternativa de load_score)
  - data/server_measurements.csv -> respaldo para reconstruir load_score
Salidas:
  - resultados.json                 (contrato de la subtarea; cifras sin redondear)
  - output/server_analysis.parquet  (tabla resumen; ruta exigida por el enunciado)
  - t4_resumen_por_servidor.png     (figura resumen)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPECTED_COLS = ["server", "observations", "mean_power_w",
                 "max_temperature_c", "mean_load", "review_count"]
REQUIRED_COLS = ["server", "power_w", "temperature_c", "load_score", "requires_review"]
EXPECTED_ROW = {"server": "AI-SRV-01", "observations": 50, "mean_power_w": 363.4598,
                "max_temperature_c": 86.50, "mean_load": -0.614527, "review_count": 2}


# --------------------------------------------------------------------------- utilidades
def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def to_py(obj):
    """Convierte tipos numpy/pandas a tipos nativos para JSON."""
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def as_bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "t", "yes", "si", "sí"])


def list_data_files(folder):
    d = Path(folder)
    if not d.is_dir():
        return []
    files = sorted(d.rglob("*.parquet")) + sorted(d.rglob("*.csv"))
    files += sorted(d.rglob("*.pkl")) + sorted(d.rglob("*.pickle"))
    files += [f for f in sorted(d.rglob("*.json")) if f.name != "resultados.json"]
    return files


def try_read_json_records(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        for key in ("data", "rows", "records", "dataframe", "copy", "df"):
            v = data.get(key)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return pd.DataFrame(v)
    return None


def try_read(path):
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix in (".pkl", ".pickle"):
            return pd.read_pickle(path)
        if path.suffix == ".json":
            return try_read_json_records(path)
        return pd.read_csv(path)
    except Exception:
        return None


# ---------------------------------------------------------------- contexto de T3 / T2
t3_res = read_json(Path("entrada/T3/resultados.json")) or {}
t2_res = read_json(Path("entrada/T2/resultados.json")) or {}
EXPECTED_N = int(t3_res.get("n_rows", 300))
EXPECTED_N_REVIEW = int(t3_res.get("n_requires_review", 49))
notes = []


def known_load_scores():
    """Pares (índice original, load_score) conocidos a partir de T3."""
    out = []
    rows = t3_res.get("review_rows") if isinstance(t3_res, dict) else None
    if isinstance(rows, list):
        for r in rows:
            try:
                out.append((int(r["index"]), float(r["load_score"])))
            except (KeyError, TypeError, ValueError):
                continue
    return out


def find_full_df():
    """DataFrame enriquecido de T3: columnas requeridas y nº de filas esperado."""
    best, best_score, best_path = None, -1, None
    for folder in ("entrada/T3", "entrada/T2"):
        for f in list_data_files(folder):
            cand = try_read(f)
            if cand is None or len(cand) != EXPECTED_N:
                continue
            if not all(c in cand.columns for c in REQUIRED_COLS):
                continue
            score = 1
            if f.suffix == ".parquet":
                score += 1
            if "T3" in f.parts:
                score += 1
            try:
                if int(as_bool_series(cand["requires_review"]).sum()) == EXPECTED_N_REVIEW:
                    score += 2
            except Exception:
                pass
            if score > best_score:
                best, best_score, best_path = cand.copy(), score, f
    return best, best_path


def find_base_df():
    """DataFrame base (server, power_w, temperature_c, ...) para reconstruir load_score."""
    for folder in ("entrada/T3", "entrada/T2"):
        for f in list_data_files(folder):
            cand = try_read(f)
            if cand is None or len(cand) != EXPECTED_N:
                continue
            if all(c in cand.columns for c in ("server", "power_w", "temperature_c")):
                return cand.copy(), f
    f = Path("data/server_measurements.csv")
    if f.exists():
        cand = pd.read_csv(f)
        cand = cand.loc[:, ~cand.columns.astype(str).str.startswith("Unnamed")]
        if all(c in cand.columns for c in ("server", "power_w", "temperature_c")):
            return cand, f
    return None, None


def load_score_from_json():
    """Lista de valores de load_score guardada por T2 en su resultados.json."""
    for res in (t2_res, t3_res):
        if not isinstance(res, dict):
            continue
        ls = res.get("load_score")
        if isinstance(ls, dict):
            ls = ls.get("values", ls.get("scores"))
        if isinstance(ls, list) and len(ls) == EXPECTED_N:
            try:
                arr = np.asarray(ls, dtype=float)
                if np.isfinite(arr).all():
                    return arr
            except (TypeError, ValueError):
                continue
    return None


def normalize_by_index(cand):
    if "index" in cand.columns and cand["index"].nunique() == len(cand):
        cand = cand.sort_values("index").reset_index(drop=True)
    return cand


def reconstruct_load_score(dframe):
    """
    Reconstruye load_score validando contra los valores conocidos de T3 (review_rows).
    Cubre cualquier load_score afín en (gpu, cpu, memory, power, temperature) — p. ej.
    z-scores de combinaciones ponderadas — más algunas formas robustas habituales.
    """
    known = known_load_scores()
    if len(known) < 6:
        return None, "referencias insuficientes"
    idx = np.array([k[0] for k in known], dtype=int)
    vals = np.array([k[1] for k in known], dtype=float)
    n = len(dframe)
    if idx.min() < 0 or idx.max() >= n:
        return None, "índices de referencia fuera de rango"

    cols = ["gpu_utilization", "cpu_utilization", "memory_gb", "power_w", "temperature_c"]
    X = np.column_stack([np.ones(n)] + [dframe[c].to_numpy(dtype=float) for c in cols])
    theta, *_ = np.linalg.lstsq(X[idx], vals, rcond=None)
    resid = float(np.max(np.abs(X[idx] @ theta - vals)))
    if resid < 1e-8:
        return X @ theta, f"afin en ({', '.join(cols)}); residuo max={resid:.2e}"

    def z(a, ddof=1):
        sd = a.std(ddof=ddof)
        return (a - a.mean()) / sd if sd > 0 else np.zeros_like(a)

    def robust_z(a):
        med = np.median(a)
        mad = np.median(np.abs(a - med)) * 1.4826
        return (a - med) / mad if mad > 0 else np.zeros_like(a)

    g = dframe["gpu_utilization"].to_numpy(float)
    c = dframe["cpu_utilization"].to_numpy(float)
    m = dframe["memory_gb"].to_numpy(float)
    p = dframe["power_w"].to_numpy(float)
    bases = {"power_w": p, "gpu_utilization": g, "cpu_utilization": c, "memory_gb": m,
             "gpu+cpu+memory": g + c + m, "gpu+cpu": g + c}
    cands = {}
    for name, b in bases.items():
        for ddof in (0, 1):
            cands[f"z({name})[ddof={ddof}]"] = z(b, ddof)
        cands[f"robust_z({name})"] = robust_z(b)
    for ddof in (0, 1):
        zg, zc, zm = z(g, ddof), z(c, ddof), z(m, ddof)
        cands[f"z(g)+z(c)+z(m)[ddof={ddof}]"] = zg + zc + zm
        cands[f"0.5z(g)+0.3z(c)+0.2z(m)[ddof={ddof}]"] = 0.5 * zg + 0.3 * zc + 0.2 * zm
    for name, score in cands.items():
        if np.allclose(score[idx], vals, rtol=0.0, atol=1e-8):
            return score, name
    return None, "sin coincidencia"


def check_known_load_scores(dframe):
    """Comprueba que el load_score cargado coincide con los valores conocidos de T3."""
    try:
        known = known_load_scores()
        if not known:
            return None
        ls = dframe["load_score"].to_numpy(float)
        if "index" in dframe.columns and dframe["index"].nunique() == len(dframe):
            lookup = pd.Series(ls, index=dframe["index"].to_numpy(int))
            aligned = np.array([lookup.get(i, np.nan) for i, _ in known], dtype=float)
        else:
            pos = np.array([i for i, _ in known], dtype=int)
            aligned = ls[pos] if pos.max() < len(ls) else np.full(len(known), np.nan)
        vals = np.array([v for _, v in known], dtype=float)
        if not np.isfinite(aligned).all():
            return None
        return bool(np.allclose(aligned, vals, rtol=0.0, atol=1e-8))
    except Exception:
        return None


# ------------------------------------------------------- 1) Cargar el DataFrame de T3
df, source_path = find_full_df()
load_score_source = None
if df is not None:
    load_score_source = f"{source_path}::load_score"
    df["requires_review"] = as_bool_series(df["requires_review"])
    df["load_score"] = pd.to_numeric(df["load_score"], errors="coerce").astype(float)
else:
    base, bpath = find_base_df()
    if base is None:
        raise SystemExit("ERROR: sin datos en entrada/T3, entrada/T2 ni data/server_measurements.csv")
    base = normalize_by_index(base)
    df = base.copy()
    source_path = bpath
    arr = load_score_from_json()
    if arr is not None:
        df["load_score"] = arr
        load_score_source = "entrada/T2/resultados.json::load_score"
        notes.append("load_score leido de la lista guardada en entrada/T2/resultados.json")
    else:
        try:
            arr, how = reconstruct_load_score(df)
        except Exception as exc:
            arr, how = None, f"error en reconstruccion ({exc})"
        if arr is not None:
            df["load_score"] = arr
            load_score_source = f"reconstruido y validado contra T3 [{how}]"
            notes.append(f"load_score reconstruido: {how}")
        else:
            g = df["gpu_utilization"].to_numpy(float)
            c = df["cpu_utilization"].to_numpy(float)
            m = df["memory_gb"].to_numpy(float)
            combo = g + c + m
            df["load_score"] = (combo - combo.mean()) / combo.std(ddof=1)
            load_score_source = "FALLBACK sin verificar: z(gpu+cpu+memory, ddof=1)"
            notes.append("AVISO: load_score no pudo validarse; se uso z(gpu+cpu+memory)")
    if "requires_review" in df.columns:
        df["requires_review"] = as_bool_series(df["requires_review"])
    else:
        df["requires_review"] = ((df["load_score"].to_numpy(float) > 1.5)
                                 | (df["temperature_c"].to_numpy(float) > 80.0))
        notes.append("requires_review recalculada: (load_score > 1.5) | (temperature_c > 80)")

# limpieza básica
df["server"] = df["server"].astype(str).str.strip()
for col in ("power_w", "temperature_c", "load_score"):
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["requires_review"] = df["requires_review"].astype(bool)

# coherencia con T3
total_review = int(df["requires_review"].sum())
if total_review != EXPECTED_N_REVIEW:
    recomputed = ((df["load_score"].to_numpy(float) > 1.5)
                  | (df["temperature_c"].to_numpy(float) > 80.0))
    if int(recomputed.sum()) == EXPECTED_N_REVIEW:
        df["requires_review"] = recomputed
        total_review = int(recomputed.sum())
        notes.append("requires_review ajustada a la condicion de T3 (49 filas)")

ls_match = check_known_load_scores(df)

# ------------------------------------------------------------- 2) Tabla resumen (T4)
summary = (
    df.groupby("server", sort=True)
      .agg(observations=("server", "size"),
           mean_power_w=("power_w", "mean"),
           max_temperature_c=("temperature_c", "max"),
           mean_load=("load_score", "mean"),
           review_count=("requires_review", "sum"))
      .reset_index()
      .loc[:, EXPECTED_COLS]
)
summary["observations"] = summary["observations"].astype(int)
summary["review_count"] = summary["review_count"].astype(int)

# ------------------------------------------------------------- 3) Imprimir la tabla
print("=" * 78)
print("T4 — Tabla resumen por servidor")
print("=" * 78)
disp = summary.copy()
disp["mean_power_w"] = disp["mean_power_w"].map("{:.4f}".format)
disp["max_temperature_c"] = disp["max_temperature_c"].map("{:.2f}".format)
disp["mean_load"] = disp["mean_load"].map("{:.6f}".format)
print(disp.to_string(index=False))
print("=" * 78)

# ------------------------------------------------- 4) Parquet exigido por el enunciado
out_dir = Path("output")
out_dir.mkdir(parents=True, exist_ok=True)
parquet_path = out_dir / "server_analysis.parquet"
summary.to_parquet(parquet_path, index=False)
parquet_ok = False
try:
    back = pd.read_parquet(parquet_path)
    parquet_ok = (list(back.columns) == EXPECTED_COLS
                  and len(back) == len(summary)
                  and np.allclose(back["mean_load"].to_numpy(float), summary["mean_load"].to_numpy(float))
                  and np.allclose(back["mean_power_w"].to_numpy(float), summary["mean_power_w"].to_numpy(float)))
except Exception:
    parquet_ok = False

# ------------------------------------------------------------------ 5) Verificación
row1 = summary.loc[summary["server"] == EXPECTED_ROW["server"]]
actual_row = None
if len(row1):
    r1 = row1.iloc[0]
    actual_row = {"server": EXPECTED_ROW["server"],
                  "observations": int(r1["observations"]),
                  "mean_power_w": float(r1["mean_power_w"]),
                  "max_temperature_c": float(r1["max_temperature_c"]),
                  "mean_load": float(r1["mean_load"]),
                  "review_count": int(r1["review_count"])}

checks = {
    "una_fila_por_servidor_6_servidores": len(summary) == 6,
    "orden_exacto_de_columnas": list(summary.columns) == EXPECTED_COLS,
    "AI-SRV-01_observations_50": actual_row is not None and actual_row["observations"] == EXPECTED_ROW["observations"],
    "AI-SRV-01_mean_power_w_363.4598": actual_row is not None and round(actual_row["mean_power_w"], 4) == EXPECTED_ROW["mean_power_w"],
    "AI-SRV-01_max_temperature_c_86.50": actual_row is not None and round(actual_row["max_temperature_c"], 2) == EXPECTED_ROW["max_temperature_c"],
    "AI-SRV-01_mean_load_-0.614527": actual_row is not None and round(actual_row["mean_load"], 6) == EXPECTED_ROW["mean_load"],
    "AI-SRV-01_review_count_2": actual_row is not None and actual_row["review_count"] == EXPECTED_ROW["review_count"],
    "total_observations_300": int(summary["observations"].sum()) == EXPECTED_N,
    "total_review_count_49": int(summary["review_count"].sum()) == EXPECTED_N_REVIEW,
    "parquet_escrito_y_verificado": bool(parquet_ok),
}
all_ok = all(bool(v) for v in checks.values())
checks["ALL_OK"] = all_ok

# ---------------------------------------------------------------------- 6) Figura
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
x = np.arange(len(summary))
panels = [("mean_power_w", "Potencia media (W)", "#4C72B0"),
          ("max_temperature_c", "Temperatura máxima (°C)", "#DD8452"),
          ("mean_load", "Carga media (load_score)", "#55A868"),
          ("review_count", "Observaciones a revisar", "#C44E52")]
for ax, (col, title, color) in zip(axes.flat, panels):
    ax.bar(x, summary[col].to_numpy(float), color=color)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["server"], rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle(f"T4 — Resumen por servidor ({int(summary['observations'].sum())} observaciones, "
             f"{int(summary['review_count'].sum())} a revisar)")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("t4_resumen_por_servidor.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------ 7) resultados.json
records = []
for _, r in summary.iterrows():
    records.append({"server": str(r["server"]),
                    "observations": int(r["observations"]),
                    "mean_power_w": float(r["mean_power_w"]),
                    "max_temperature_c": float(r["max_temperature_c"]),
                    "mean_load": float(r["mean_load"]),
                    "review_count": int(r["review_count"])})

results = {
    "subtask": "T4",
    "task": "Resumen por servidor: groupby(server) con columnas server, observations, "
            "mean_power_w, max_temperature_c, mean_load, review_count",
    "data_source": str(source_path),
    "load_score_source": load_score_source,
    "n_rows_input": int(len(df)),
    "n_requires_review_input": int(total_review),
    "n_servers": int(summary.shape[0]),
    "servers": [str(s) for s in summary["server"]],
    "column_order": EXPECTED_COLS,
    "table": records,
    "table_by_server": {rec["server"]: rec for rec in records},
    "table_display": disp.to_string(index=False),
    "totals": {"observations": int(summary["observations"].sum()),
               "review_count": int(summary["review_count"].sum())},
    "load_score_matches_T3_review_rows": ls_match,
    "verification": {"expected_row_AI-SRV-01": EXPECTED_ROW,
                     "actual_row_AI-SRV-01": actual_row,
                     "checks": checks},
    "notes": notes,
    "outputs": {"parquet": str(parquet_path), "figure": "t4_resumen_por_servidor.png"},
}
with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(to_py(results), fh, indent=2, ensure_ascii=False)

# ------------------------------------------------------------------ 8) Resumen final
print("\nResumen T4:")
print(f"  Origen de datos      : {source_path}")
print(f"  Origen de load_score : {load_score_source}")
print(f"  Filas de entrada     : {len(df)} | Servidores: {summary.shape[0]} | A revisar: {total_review}")
print(f"  Parquet              : {parquet_path} (escrito y verificado: {parquet_ok})")
print(f"  Figura               : t4_resumen_por_servidor.png")
print(f"  Verificacion AI-SRV-01 y estructura: {'OK' if all_ok else 'REVISAR'}")
print("  Notas:", "; ".join(notes) if notes else "(ninguna)")
