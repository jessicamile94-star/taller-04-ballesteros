#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5 - Guardar y verificar el resultado (round trip Parquet con PyArrow).

1) Obtiene el DataFrame analizado de la Tarea 3:
   a) si una subtarea previa dejo el DataFrame completo en entrada/<id>/, se usa
      ese archivo (validandolo contra la referencia de T3);
   b) si no, se reconstruye EXACTAMENTE con la definicion de Tarea 2/Tarea 3:
          load_score = zscores @ [0.50, 0.30, 0.20]
          zscores    = (x - media) / desviacion  por columna de gpu_utilization,
                       cpu_utilization y memory_gb (se usan la media y la
                       desviacion almacenadas por T2 si estan disponibles)
          requires_review = (load_score > 1.5) | (temperature_c > 80)
   El DataFrame se valida ANTES de guardar contra la referencia de T3
   (entrada/T3/resultados.json): n filas, columnas, n_requires_review, desglose
   y los load_score exactos de review_rows. Si no coincide, se aborta sin
   escribir el Parquet (nunca se guarda contenido que no sea el de T3).
2) Guarda output/server_analysis.parquet con PyArrow y sin el indice.
3) Lo vuelve a leer y verifica: filas, columnas, esquema y valores.
4) Imprime el mensaje de round trip y escribe resultados.json.
"""

import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_PATH = Path("output") / "server_analysis.parquet"
ENTRADA = Path("entrada")
DATA_CSV = Path("data") / "server_measurements.csv"

FEATURE_COLS = ["gpu_utilization", "cpu_utilization", "memory_gb"]
WEIGHTS = np.array([0.50, 0.30, 0.20], dtype="float64")
BASE_COLS = ["server", "timestamp", "gpu_utilization", "cpu_utilization",
             "memory_gb", "power_w", "temperature_c"]
CANONICAL_COLS = BASE_COLS + ["load_score", "requires_review"]
REQUIRED_COLS = set(CANONICAL_COLS)

READERS = {
    ".parquet": pd.read_parquet,
    ".pq": pd.read_parquet,
    ".feather": pd.read_feather,
    ".pkl": pd.read_pickle,
    ".pickle": pd.read_pickle,
    ".csv": pd.read_csv,
}


# --------------------------------------------------------------------------- #
# Utilidades genericas
# --------------------------------------------------------------------------- #
def load_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def iter_nodes(obj, path=""):
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            child = "%s.%s" % (path, k) if path else str(k)
            yield from iter_nodes(v, child)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_nodes(v, "%s[%d]" % (path, i))


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def flatten_numbers(v):
    out = []

    def rec(x):
        if isinstance(x, list):
            for e in x:
                rec(e)
        elif _is_num(x):
            out.append(float(x))
        else:
            raise ValueError("no numerico")

    try:
        rec(v)
    except Exception:
        return None
    return out or None


# --------------------------------------------------------------------------- #
# Lectura del CSV original (mismas columnas que T3)
# --------------------------------------------------------------------------- #
def load_csv():
    if not DATA_CSV.exists():
        return None
    try:
        df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    except Exception:
        try:
            df = pd.read_csv(DATA_CSV, sep=";")
        except Exception:
            return None
    df.columns = [str(c).strip() for c in df.columns]
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Extraccion de la salida de T2 (serie load_score o parametros mean/std/weights)
# --------------------------------------------------------------------------- #
def _concat_by_keys(node, server_order):
    keys = [str(k) for k in node.keys()]
    flats = []
    for k in keys:
        f = flatten_numbers(node[k])
        if f is None:
            return None
        flats.append(f)
    if server_order is not None and len(keys) == len(server_order):
        norm = lambda s: s.strip().upper()
        if sorted(map(norm, keys)) == sorted(map(norm, server_order)):
            kmap = {norm(k): i for i, k in enumerate(keys)}
            flats = [flats[kmap[norm(s)]] for s in server_order]
    try:
        return np.concatenate([np.asarray(f, dtype="float64") for f in flats])
    except Exception:
        return None


def extract_t2_series(t2, n, server_order=None):
    """Busca en el JSON de T2 una serie load_score de longitud n (si se guardo)."""
    if not isinstance(t2, dict) or n <= 0:
        return None
    best = None
    for path, node in iter_nodes(t2):
        if "load" not in path.lower():
            continue
        arr = None
        if isinstance(node, list):
            flat = flatten_numbers(node)
            if flat is not None and len(flat) == n:
                arr = np.asarray(flat, dtype="float64")
        elif isinstance(node, dict) and node:
            keys = [str(k) for k in node.keys()]
            if all(k.lstrip("-").isdigit() for k in keys):
                try:
                    okeys = sorted(keys, key=int)
                    if okeys[0] == "0" and int(okeys[-1]) == len(okeys) - 1:
                        flat = [float(node[k]) for k in okeys]
                        if len(flat) == n:
                            arr = np.asarray(flat, dtype="float64")
                except Exception:
                    arr = None
            else:
                arr = _concat_by_keys(node, server_order)
                if arr is not None and len(arr) != n:
                    arr = None
        if arr is not None:
            best = arr
    return best


def _mentions_features(path, feat_l):
    return any(f in path for f in feat_l)


def extract_t2_params(t2, feat_cols):
    """Extrae weights/means/stds de load_score almacenados por T2 (si existen)."""
    weights, means, stds = None, None, None
    if not isinstance(t2, dict):
        return weights, means, stds
    feat_l = [f.lower() for f in feat_cols]

    def dict_to_vec(node):
        low = {str(k).strip().lower(): v for k, v in node.items()}
        vals = []
        for f in feat_l:
            v = low.get(f)
            if not _is_num(v):
                return None
            vals.append(float(v))
        return np.asarray(vals, dtype="float64")

    for path, node in iter_nodes(t2):
        pl = path.lower()
        if isinstance(node, list):
            flat = flatten_numbers(node)
            if flat is None or len(flat) != len(feat_cols):
                continue
            if weights is None and any(t in pl for t in ("weight", "peso", "ponder", "coef")):
                weights = np.asarray(flat, dtype="float64")
            elif means is None and any(t in pl for t in ("mean", "media")) and _mentions_features(pl, feat_l):
                means = np.asarray(flat, dtype="float64")
            elif stds is None and any(t in pl for t in ("std", "sd", "desv", "sigma")) and _mentions_features(pl, feat_l):
                stds = np.asarray(flat, dtype="float64")
        elif isinstance(node, dict):
            vec = dict_to_vec(node)
            if vec is None:
                continue
            if means is None and any(t in pl for t in ("mean", "media")):
                means = vec
            elif stds is None and any(t in pl for t in ("std", "sd", "desv", "sigma")):
                stds = vec

    # medias/desviaciones como claves planas, p. ej. "gpu_utilization_mean"
    m, s = {}, {}
    for path, node in iter_nodes(t2):
        if not _is_num(node):
            continue
        pl = path.lower()
        for f, fl in zip(feat_cols, feat_l):
            if fl not in pl:
                continue
            if any(t in pl for t in ("mean", "media")) and f not in m:
                m[f] = float(node)
            elif any(t in pl for t in ("std", "sd", "desv", "sigma")) and f not in s:
                s[f] = float(node)
    if means is None and len(m) == len(feat_cols):
        means = np.asarray([m[f] for f in feat_cols], dtype="float64")
    if stds is None and len(s) == len(feat_cols):
        stds = np.asarray([s[f] for f in feat_cols], dtype="float64")

    if weights is None:
        for path, node in iter_nodes(t2):
            if isinstance(node, list):
                flat = flatten_numbers(node)
                if (flat is not None and len(flat) == len(feat_cols)
                        and all(0.0 <= x <= 1.0 for x in flat)
                        and abs(sum(flat) - 1.0) < 1e-6):
                    weights = np.asarray(flat, dtype="float64")
                    break
    return weights, means, stds


# --------------------------------------------------------------------------- #
# Calculo de load_score segun Tarea 2
# --------------------------------------------------------------------------- #
def weighted_zscore(X, weights, mean=None, std=None, ddof=1):
    """load_score = ((X - media) / desviacion) @ weights, por columna."""
    X = np.asarray(X, dtype="float64")
    mu = X.mean(axis=0) if mean is None else np.asarray(mean, dtype="float64")
    sd = X.std(axis=0, ddof=ddof) if std is None else np.asarray(std, dtype="float64")
    if np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        return None
    Z = (X - mu) / sd
    return Z @ np.asarray(weights, dtype="float64")


def weighted_zscore_per_server(df, weights, ddof=1):
    """Reserva: z-scores calculados dentro de cada servidor."""
    X = df[FEATURE_COLS].astype("float64")
    g = df["server"].astype(str)
    mu = X.groupby(g).transform("mean")
    sd = X.groupby(g).transform(lambda s: s.std(ddof=ddof))
    if bool((sd <= 0).any()) or bool(sd.isna().any()):
        return None
    Z = (X - mu) / sd
    return Z.to_numpy(dtype="float64") @ np.asarray(weights, dtype="float64")


def assemble(df_base, load_score):
    """DataFrame analizado de T3: columnas originales + load_score + requires_review."""
    df = df_base.copy().reset_index(drop=True)
    ls = np.asarray(load_score, dtype="float64")
    if len(ls) != len(df):
        raise ValueError("longitud de load_score (%d) != filas (%d)" % (len(ls), len(df)))
    df["load_score"] = ls
    tp = pd.to_numeric(df["temperature_c"], errors="coerce")
    df["requires_review"] = ((df["load_score"] > 1.5) | (tp > 80.0)).astype(bool)
    df = df[[c for c in CANONICAL_COLS if c in df.columns]]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Salidas de subtareas previas en entrada/<id>/
# --------------------------------------------------------------------------- #
def normalize_df(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    drop = [c for c in df.columns
            if c.lower() in ("index", "unnamed: 0", "level_0", "__index_level_0__")]
    if drop and set(df.columns) - set(drop) == set(CANONICAL_COLS):
        df = df.drop(columns=drop)
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception:
            pass
    if "load_score" in df.columns:
        df["load_score"] = pd.to_numeric(df["load_score"], errors="coerce").astype("float64")
    if "requires_review" in df.columns:
        col = df["requires_review"]
        if col.dtype == object:
            df["requires_review"] = (col.astype(str).str.strip().str.lower()
                                     .isin(["true", "1", "1.0", "yes", "si"]).astype(bool))
        else:
            df["requires_review"] = col.fillna(False).astype(bool)
    cols = [c for c in CANONICAL_COLS if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    return df[cols + extra].reset_index(drop=True)


def scan_prev_full_dfs():
    """DataFrames completos (con las 9 columnas) dejados por subtareas previas."""
    out = []
    if not ENTRADA.is_dir():
        return out
    dirs = sorted([p for p in ENTRADA.iterdir() if p.is_dir()],
                  key=lambda p: (0 if p.name.upper() == "T3" else 1, p.name))
    for d in dirs:
        for f in sorted(d.rglob("*")):
            if not f.is_file():
                continue
            reader = READERS.get(f.suffix.lower())
            if reader is None:
                continue
            try:
                dfx = reader(f)
            except Exception:
                continue
            if isinstance(dfx, pd.DataFrame) and REQUIRED_COLS.issubset(set(map(str, dfx.columns))):
                out.append((f.as_posix(), normalize_df(dfx)))
    return out


def scan_prev_series(n_expected):
    """Archivos previos con una columna load_score (sin requires_review)."""
    out = []
    if not ENTRADA.is_dir():
        return out
    for d in sorted([p for p in ENTRADA.iterdir() if p.is_dir()]):
        for f in sorted(d.rglob("*")):
            reader = READERS.get(f.suffix.lower())
            if reader is None:
                continue
            try:
                dfx = reader(f)
            except Exception:
                continue
            if not isinstance(dfx, pd.DataFrame):
                continue
            cols = list(map(str, dfx.columns))
            if ("load_score" in cols and "requires_review" not in cols
                    and len(dfx) == n_expected):
                arr = pd.to_numeric(dfx["load_score"], errors="coerce").to_numpy(dtype="float64")
                out.append((arr, f.as_posix()))
    return out


# --------------------------------------------------------------------------- #
# Validacion estricta contra la referencia de T3
# --------------------------------------------------------------------------- #
def strict_validate(df, t3):
    """Comprueba que el DataFrame ES el analizado en T3. Devuelve (ok, checks)."""
    checks = {}
    if not isinstance(t3, dict) or not t3:
        return False, {"referencia_T3_disponible": False}
    if "n_rows" in t3:
        checks["n_rows"] = len(df) == int(t3["n_rows"])
    cols_ref = t3.get("copy_columns")
    if cols_ref:
        checks["columnas"] = set(map(str, df.columns)) == set(map(str, cols_ref))
        checks["orden_columnas"] = list(map(str, df.columns)) == list(map(str, cols_ref))
    if "requires_review" in df.columns and "n_requires_review" in t3:
        checks["n_requires_review"] = (int(df["requires_review"].astype(bool).sum())
                                       == int(t3["n_requires_review"]))
    br = t3.get("breakdown") or {}
    if br and {"load_score", "temperature_c"}.issubset(df.columns):
        ls = df["load_score"].to_numpy(dtype="float64")
        tp = pd.to_numeric(df["temperature_c"], errors="coerce").to_numpy(dtype="float64")
        c_ls, c_tp = ls > 1.5, tp > 80.0
        checks["breakdown"] = (
            int((c_ls & ~c_tp).sum()) == int(br.get("n_load_score_gt_1_5_only", -1))
            and int((~c_ls & c_tp).sum()) == int(br.get("n_temperature_gt_80_only", -1))
            and int((c_ls & c_tp).sum()) == int(br.get("n_both_conditions", -1))
        )
    rows = t3.get("review_rows") or []
    if rows and "load_score" in df.columns:
        n_ok_ls = n_ok_full = n_tot = 0
        for r in rows:
            try:
                i = int(r["index"])
            except Exception:
                continue
            if not (0 <= i < len(df)):
                continue
            n_tot += 1
            row = df.iloc[i]
            try:
                ok_ls = math.isclose(float(row["load_score"]), float(r["load_score"]),
                                     rel_tol=1e-9, abs_tol=1e-12)
            except Exception:
                ok_ls = False
            if ok_ls:
                n_ok_ls += 1
            try:
                ok_full = (ok_ls
                           and str(row["server"]) == str(r["server"])
                           and pd.Timestamp(row["timestamp"]) == pd.Timestamp(r["timestamp"])
                           and float(row["temperature_c"]) == float(r["temperature_c"])
                           and bool(row["requires_review"]) == bool(r["requires_review"]))
            except Exception:
                ok_full = False
            if ok_full:
                n_ok_full += 1
        checks["review_rows_load_score"] = bool(n_tot > 0 and n_ok_ls == n_tot)
        checks["review_rows_filas_completas"] = bool(n_tot > 0 and n_ok_full == n_tot)
    obligatorios = [k for k in ("n_rows", "columnas", "n_requires_review",
                                "breakdown", "review_rows_load_score") if k in checks]
    ok = bool(obligatorios) and all(bool(checks[k]) for k in obligatorios)
    return ok, checks


# --------------------------------------------------------------------------- #
# Construccion del DataFrame analizado de T3 (con candidatos validados)
# --------------------------------------------------------------------------- #
def build_analyzed_dataframe():
    t3 = load_json_safe(ENTRADA / "T3" / "resultados.json")
    if not isinstance(t3, dict):
        t3 = {}
    t2 = load_json_safe(ENTRADA / "T2" / "resultados.json")
    df_csv = load_csv()
    n_expected = int(t3["n_rows"]) if "n_rows" in t3 else (len(df_csv) if df_csv is not None else 0)
    server_order = None
    if df_csv is not None and "server" in df_csv.columns:
        server_order = [str(s) for s in pd.unique(df_csv["server"].astype(str))]

    cands = []  # (prioridad, descripcion, df, info)

    # (0/1) DataFrame completo dejado por una subtarea previa (T3 primero)
    for path, dfn in scan_prev_full_dfs():
        prio = 0 if "/T3/" in ("/" + path) else 1
        cands.append((prio, "DataFrame completo leido de %s" % path, dfn, {"origen": path}))

    if df_csv is not None and n_expected == len(df_csv):
        # (2) serie load_score almacenada por T2 (JSON) o en archivos previos
        serie = extract_t2_series(t2, n_expected, server_order)
        if serie is not None:
            cands.append((2, "serie load_score almacenada en entrada/T2/resultados.json",
                          assemble(df_csv, serie), {"origen": "entrada/T2/resultados.json"}))
        for arr, path in scan_prev_series(n_expected):
            cands.append((2, "serie load_score leida de %s" % path,
                          assemble(df_csv, arr), {"origen": path}))

        # (3) media/desviacion almacenadas por T2 + pesos de T2 (o [0.50,0.30,0.20])
        w_t2, mu_t2, sd_t2 = extract_t2_params(t2, FEATURE_COLS)
        w = w_t2 if w_t2 is not None else WEIGHTS
        X = df_csv[FEATURE_COLS].to_numpy(dtype="float64")
        if mu_t2 is not None and sd_t2 is not None:
            ls = weighted_zscore(X, w, mean=mu_t2, std=sd_t2)
            if ls is not None:
                cands.append((3, "load_score = zscores @ %s con media/desviacion "
                                 "almacenadas por T2" % np.array2string(w),
                              assemble(df_csv, ls),
                              {"weights": w.tolist(), "means": mu_t2.tolist(),
                               "stds": sd_t2.tolist(), "fuente_stats": "entrada/T2"}))
        # (4) formula de T2/T3: z-scores por columna (global), ddof de pandas (1) y numpy (0)
        for ddof in (1, 0):
            ls = weighted_zscore(X, w, ddof=ddof)
            if ls is not None:
                cands.append((4, "load_score = zscores @ %s (z por columna: media y "
                                 "desviacion del dataset, ddof=%d)" % (np.array2string(w), ddof),
                              assemble(df_csv, ls),
                              {"weights": w.tolist(), "ddof": ddof, "ambito": "global"}))
        # (5) reserva: z-scores por servidor
        for ddof in (1, 0):
            ls = weighted_zscore_per_server(df_csv, w, ddof=ddof)
            if ls is not None:
                cands.append((5, "load_score = zscores @ %s (z por servidor, ddof=%d)"
                              % (np.array2string(w), ddof),
                              assemble(df_csv, ls),
                              {"weights": w.tolist(), "ddof": ddof, "ambito": "por_servidor"}))

    if not cands:
        raise SystemExit("ERROR: no hay candidatos para el DataFrame analizado de T3.")

    reports = []
    for prio, desc, df, info in sorted(cands, key=lambda c: c[0]):
        ok, checks = strict_validate(df, t3)
        reports.append({"prioridad": int(prio), "descripcion": desc,
                        "valido": bool(ok), "checks": {k: bool(v) for k, v in checks.items()}})
        if ok:
            return df, desc, info, reports, t3

    print("Ningun candidato coincide con la referencia de T3:")
    for r in reports:
        print("  - [%d] %s -> %s" % (r["prioridad"], r["descripcion"], r["checks"]))
    raise SystemExit("ERROR: el DataFrame analizado de T3 no pudo reproducirse/validarse; "
                     "no se escribe el Parquet para no guardar contenido incorrecto.")


# --------------------------------------------------------------------------- #
# Round trip: guardar con PyArrow (sin indice), releer y verificar
# --------------------------------------------------------------------------- #
def values_equal(df1, df2):
    """Comparacion celda a celda (NaN/NaT en la misma posicion cuenta como igual)."""
    if df1.shape != df2.shape or list(df1.columns) != list(df2.columns):
        return False
    for c in df1.columns:
        a, b = df1[c], df2[c]
        try:
            eq = (a == b)
        except Exception:
            try:
                eq = (a.astype(str) == b.astype(str))
            except Exception:
                return False
        try:
            eq = eq | (a.isna() & b.isna())
        except Exception:
            pass
        if not bool(eq.all()):
            return False
    return True


def save_and_roundtrip(df_analyzed, source_desc, info, reports, t3):
    df_before = df_analyzed.reset_index(drop=True)

    # Comprobacion final ANTES de escribir: el contenido debe ser el de T3
    ok_t3, checks_t3 = strict_validate(df_before, t3)
    if not ok_t3:
        raise SystemExit("ERROR: el DataFrame no coincide con la referencia de T3; "
                         "no se escribe el Parquet. Checks: %s" % checks_t3)

    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df_before, preserve_index=False)  # sin el indice
    pq.write_table(table, PARQUET_PATH)

    df_after = pq.read_table(PARQUET_PATH).to_pandas().reset_index(drop=True)
    schema_after = pq.read_schema(PARQUET_PATH)

    filas_ok = bool(len(df_before) == len(df_after))
    columnas_ok = bool(list(df_before.columns) == list(df_after.columns))

    dtypes_before = {c: str(t) for c, t in df_before.dtypes.items()}
    dtypes_after = {c: str(t) for c, t in df_after.dtypes.items()}
    dtypes_ok = bool(columnas_ok and all(dtypes_before[c] == dtypes_after.get(c)
                                         for c in dtypes_before))
    try:
        arrow_eq = bool(table.schema.equals(schema_after, check_metadata=False))
    except Exception:
        arrow_eq = False
    try:
        arrow_eq_meta = bool(table.schema.equals(schema_after, check_metadata=True))
    except Exception:
        arrow_eq_meta = False
    index_cols = [f.name for f in schema_after if "__index_level" in f.name]
    esquema_ok = bool(dtypes_ok and arrow_eq and not index_cols)
    valores_ok = bool(values_equal(df_before, df_after))
    try:
        df_equals = bool(df_before.equals(df_after))
    except Exception:
        df_equals = False

    print("Round trip verificado: %s" % PARQUET_PATH.as_posix())
    print("Filas: %s" % filas_ok)
    print("Columnas: %s" % columnas_ok)
    print("Esquema: %s" % esquema_ok)
    print("Valores: %s" % valores_ok)

    if not (filas_ok and columnas_ok and esquema_ok and valores_ok):
        raise SystemExit("ERROR: el round trip no paso las cuatro verificaciones.")

    n_req = int(df_before["requires_review"].astype(bool).sum())
    ls = df_before["load_score"].to_numpy(dtype="float64")
    tp = pd.to_numeric(df_before["temperature_c"], errors="coerce").to_numpy(dtype="float64")
    c_ls, c_tp = ls > 1.5, tp > 80.0

    resultados = {
        "subtask": "T5",
        "task": "Guardar el DataFrame analizado de T3 en Parquet (PyArrow, sin indice) y verificar el round trip",
        "parquet_path": PARQUET_PATH.as_posix(),
        "parquet_exists": bool(PARQUET_PATH.exists()),
        "parquet_size_bytes": int(PARQUET_PATH.stat().st_size),
        "engine": "pyarrow",
        "pyarrow_version": str(pa.__version__),
        "pandas_version": str(pd.__version__),
        "index_saved": False,
        "index_columns_in_file": index_cols,
        "df_source": source_desc,
        "load_score_params": info,
        "load_score_formula": "load_score = zscores @ [0.50, 0.30, 0.20] con z-scores por "
                              "columna de gpu_utilization, cpu_utilization y memory_gb "
                              "(media y desviacion segun Tarea 2)",
        "t3_validation_before_saving": {k: bool(v) for k, v in checks_t3.items()},
        "candidate_reports": reports,
        "rows_before": int(len(df_before)),
        "rows_after": int(len(df_after)),
        "n_cols_before": int(df_before.shape[1]),
        "n_cols_after": int(df_after.shape[1]),
        "columns_before": list(df_before.columns),
        "columns_after": list(df_after.columns),
        "dtypes_before": dtypes_before,
        "dtypes_after": dtypes_after,
        "arrow_schema_before": str(table.schema),
        "arrow_schema_after": str(schema_after),
        "arrow_schema_equal": arrow_eq,
        "arrow_schema_equal_with_metadata": arrow_eq_meta,
        "pandas_dtypes_equal": dtypes_ok,
        "dataframe_equals": df_equals,
        "n_requires_review": n_req,
        "breakdown_reproducido": {
            "n_load_score_gt_1_5_only": int((c_ls & ~c_tp).sum()),
            "n_temperature_gt_80_only": int((~c_ls & c_tp).sum()),
            "n_both_conditions": int((c_ls & c_tp).sum()),
            "n_requires_review": n_req,
        },
        "checks": {
            "filas": filas_ok,
            "columnas": columnas_ok,
            "esquema": esquema_ok,
            "valores": valores_ok,
        },
        "all_checks_passed": bool(filas_ok and columnas_ok and esquema_ok and valores_ok),
        "message": "Round trip verificado: %s" % PARQUET_PATH.as_posix(),
    }

    with open("resultados.json", "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=2)
    try:
        t5_dir = ENTRADA / "T5"
        t5_dir.mkdir(parents=True, exist_ok=True)
        with open(t5_dir / "resultados.json", "w", encoding="utf-8") as fh:
            json.dump(resultados, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print("\nResumen T5:")
    print("  Origen del DataFrame analizado: %s" % source_desc)
    print("  Dimensiones: %d filas x %d columnas; requires_review=True en %d filas"
          % (len(df_before), df_before.shape[1], n_req))
    print("  Archivo: %s (%d bytes, PyArrow %s, sin indice)"
          % (PARQUET_PATH.as_posix(), PARQUET_PATH.stat().st_size, pa.__version__))
    print("  Round trip: Filas=%s, Columnas=%s, Esquema=%s, Valores=%s"
          % (filas_ok, columnas_ok, esquema_ok, valores_ok))
    try:
        r23 = df_before.iloc[23]
        print("  Fila 23 (esperada en T3): server=%s, timestamp=%s, load_score=%.15f, "
              "temperature_c=%.2f, requires_review=%s"
              % (r23["server"], pd.Timestamp(r23["timestamp"]), float(r23["load_score"]),
                 float(r23["temperature_c"]), bool(r23["requires_review"])))
    except Exception:
        pass


def main():
    df, desc, info, reports, t3 = build_analyzed_dataframe()
    print("DataFrame analizado (T3): %s" % desc)
    save_and_roundtrip(df, desc, info, reports, t3)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit("ERROR: fallo inesperado en T5; no se guardo un Parquet no verificado.")
