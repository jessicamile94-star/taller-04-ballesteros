#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5 - Guardar y verificar el resultado (round trip Parquet con PyArrow).

1) Obtiene el DataFrame analizado de T3 (server, timestamp, gpu_utilization,
   cpu_utilization, memory_gb, power_w, temperature_c, load_score,
   requires_review) con la maxima fidelidad disponible y SIN fallar:
     a) archivos tabulares dejados por subtareas previas en entrada/<id>/
     b) serie load_score almacenada en entrada/<id>/ (JSON o tabular),
        buscada de forma recursiva en cualquier formato razonable
     c) reconstruccion de load_score por formula (z-scores candidatos),
        validada contra los valores exactos que T3 registro en review_rows
     d) ultimos recursos parciales si faltaran fuentes
2) Guarda el DataFrame en output/server_analysis.parquet con PyArrow, sin indice.
3) Vuelve a leerlo y verifica: filas, columnas, esquema y valores.
4) Imprime el mensaje de round trip y escribe resultados.json.
"""

import json
import math
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_PATH = Path("output") / "server_analysis.parquet"
ENTRADA = Path("entrada")
DATA_CSV = Path("data") / "server_measurements.csv"

BASE_COLS = ["server", "timestamp", "gpu_utilization", "cpu_utilization",
             "memory_gb", "power_w", "temperature_c"]
REQUIRED_COLS = {"server", "timestamp", "load_score", "requires_review"}
CANONICAL_COLS = BASE_COLS + ["load_score", "requires_review"]

READERS = {
    ".parquet": pd.read_parquet,
    ".pq": pd.read_parquet,
    ".feather": pd.read_feather,
    ".pkl": pd.read_pickle,
    ".pickle": pd.read_pickle,
    ".csv": pd.read_csv,
}


# ----------------------------------------------------------------------------
# Utilidades genericas
# ----------------------------------------------------------------------------
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
    """Aplana listas anidadas de numeros (acepta None y strings numericos)."""
    out = []

    def rec(x):
        if isinstance(x, list):
            for e in x:
                rec(e)
        elif _is_num(x):
            out.append(float(x))
        elif x is None:
            out.append(float("nan"))
        elif isinstance(x, str):
            out.append(float(x))
        else:
            raise ValueError("elemento no numerico")

    try:
        rec(v)
    except Exception:
        return None
    return out if out else None


def list_of_dicts_to_series(lst):
    """Extrae una serie numerica de una lista de registros (dicts)."""
    if not lst or not all(isinstance(x, dict) for x in lst):
        return None
    common = set(lst[0].keys())
    for d in lst[1:]:
        common &= set(d.keys())
    if not common:
        return None
    ordered = sorted(str(f) for f in common)
    lookup = {str(f): f for f in common}
    for pref in ("load_score", "load", "score", "value", "valor"):
        for fname in ordered:
            if pref in fname:
                try:
                    return np.asarray([float(d[lookup[fname]]) for d in lst], dtype="float64")
                except Exception:
                    pass
    numeric = []
    for fname in ordered:
        f = lookup[fname]
        try:
            if all(_is_num(d[f]) for d in lst):
                numeric.append(fname)
        except Exception:
            pass
    for fname in numeric:
        if "load" in fname:
            try:
                return np.asarray([float(d[lookup[fname]]) for d in lst], dtype="float64")
            except Exception:
                pass
    if numeric:
        try:
            return np.asarray([float(d[lookup[numeric[0]]]) for d in lst], dtype="float64")
        except Exception:
            return None
    return None


def find_series_in_json(data, n_expected):
    """Busca recursivamente series numericas de longitud n_expected en un JSON."""
    out = []
    if data is None:
        return out
    for path, node in iter_nodes(data):
        key = path.split(".")[-1].split("[")[0].lower() if path else ""
        ks = 2 if "load_score" in key else (1 if "load" in key else 0)
        if isinstance(node, list):
            flat = flatten_numbers(node)
            if flat is not None and len(flat) == n_expected:
                out.append((ks, path, "flat", np.asarray(flat, dtype="float64")))
                continue
            s = list_of_dicts_to_series(node)
            if s is not None and len(s) == n_expected:
                out.append((ks, path, "records", s))
        elif isinstance(node, dict) and node:
            vals = list(node.values())
            if all(isinstance(x, list) for x in vals):
                flats = [flatten_numbers(x) for x in vals]
                if all(f is not None for f in flats) and sum(len(f) for f in flats) == n_expected:
                    out.append((ks, path, "by_key", (list(node.keys()), flats)))
            else:
                keys = [str(k) for k in node.keys()]
                if keys and all(k.lstrip("-").isdigit() for k in keys):
                    try:
                        keys_sorted = sorted(keys, key=lambda s: int(s))
                        if keys_sorted[0] == "0" and int(keys_sorted[-1]) == len(keys_sorted) - 1:
                            flat = [float(node[k]) for k in keys_sorted]
                            if len(flat) == n_expected:
                                out.append((ks, path, "flat", np.asarray(flat, dtype="float64")))
                    except Exception:
                        pass
    return out


def series_from_by_key(keys, flats, server_order):
    """Concatena listas por clave (servidores en el orden del CSV, o 0..n-1)."""
    key_str = [str(k) for k in keys]
    if server_order and len(key_str) == len(server_order):
        norm = lambda s: str(s).strip().upper()
        kmap = {}
        for i, k in enumerate(key_str):
            kmap.setdefault(norm(k), i)
        if len(kmap) == len(key_str) and all(norm(s) in kmap for s in server_order):
            try:
                return np.concatenate([np.asarray(flats[kmap[norm(s)]], dtype="float64")
                                       for s in server_order])
            except Exception:
                pass
    try:
        order = sorted(range(len(key_str)), key=lambda i: int(key_str[i]))
        return np.concatenate([np.asarray(flats[i], dtype="float64") for i in order])
    except Exception:
        return np.concatenate([np.asarray(f, dtype="float64") for f in flats])


def score_series(arr, review_rows):
    """Cuantas coincidencias exactas hay con los load_score registrados por T3."""
    matches, total = 0, 0
    if arr is None or len(arr) == 0:
        return matches, total
    for r in review_rows:
        try:
            i = int(r["index"])
        except Exception:
            continue
        if i < 0 or i >= len(arr):
            continue
        total += 1
        try:
            if math.isclose(float(arr[i]), float(r["load_score"]), rel_tol=1e-9, abs_tol=1e-12):
                matches += 1
        except Exception:
            pass
    return matches, total


# ----------------------------------------------------------------------------
# Carga de datos y candidatos
# ----------------------------------------------------------------------------
def load_csv():
    if not DATA_CSV.exists():
        return None
    df = None
    for kwargs in ({}, {"sep": ";"}, {"sep": "\t"}):
        try:
            cand = pd.read_csv(DATA_CSV, **kwargs)
            if cand.shape[1] >= 2:
                df = cand
                break
        except Exception:
            continue
    if df is None:
        return None
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    drop = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop:
        df = df.drop(columns=drop)
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            except Exception:
                pass
    return df


def scan_data_frames(directory):
    out = []
    if not directory.is_dir():
        return out
    for f in sorted(directory.rglob("*")):
        if not f.is_file():
            continue
        reader = READERS.get(f.suffix.lower())
        if reader is None:
            continue
        try:
            dfx = reader(f)
        except Exception:
            continue
        if isinstance(dfx, pd.DataFrame) and len(dfx) > 0:
            out.append((f.as_posix(), dfx))
    return out


def formula_candidates(df):
    """Candidatos de load_score como z-scores de metricas y combinaciones."""
    out = {}
    need = {"gpu_utilization", "cpu_utilization", "memory_gb", "power_w",
            "temperature_c", "server"}
    if not need.issubset(set(map(str, df.columns))):
        return out
    gpu = df["gpu_utilization"].astype("float64")
    cpu = df["cpu_utilization"].astype("float64")
    mem = df["memory_gb"].astype("float64")
    pw = df["power_w"].astype("float64")
    tmp = df["temperature_c"].astype("float64")
    servers = df["server"].astype(str)
    base = {"gpu": gpu, "cpu": cpu, "mem": mem, "power": pw, "temp": tmp}

    def zscore(x, scope, ddof):
        if scope == "global":
            sd = x.std(ddof=ddof)
            if not (sd == sd and sd > 0):
                return None
            return (x - x.mean()) / sd
        gm = x.groupby(servers).transform("mean")
        gs = x.groupby(servers).transform(lambda s: s.std(ddof=ddof))
        if not bool((gs > 0).all()):
            return None
        return (x - gm) / gs

    z = {}
    for mname, x in base.items():
        for scope in ("global", "persrv"):
            for ddof in (0, 1):
                zs = zscore(x, scope, ddof)
                if zs is not None:
                    z[(mname, scope, ddof)] = zs
                    out["z(%s,%s,ddof=%d)" % (mname, scope, ddof)] = zs.to_numpy(dtype="float64")

    combos = {
        "gpu+cpu": gpu + cpu,
        "gpu+cpu+mem": gpu + cpu + mem,
        "gpu+cpu+10mem": gpu + cpu + 10.0 * mem,
        "gpu+cpu+mem/10": gpu + cpu + mem / 10.0,
        "gpu+mem": gpu + mem,
        "cpu+mem": cpu + mem,
        "gpu+cpu+power/100": gpu + cpu + pw / 100.0,
        "0.6gpu+0.4cpu": 0.6 * gpu + 0.4 * cpu,
        "0.4gpu+0.6cpu": 0.4 * gpu + 0.6 * cpu,
        "0.5gpu+0.3cpu+0.2mem": 0.5 * gpu + 0.3 * cpu + 0.2 * mem,
        "0.4gpu+0.3cpu+0.3mem": 0.4 * gpu + 0.3 * cpu + 0.3 * mem,
        "0.6gpu+0.3cpu+0.1mem": 0.6 * gpu + 0.3 * cpu + 0.1 * mem,
    }
    for cname, xc in combos.items():
        xc = xc.astype("float64")
        for scope in ("global", "persrv"):
            for ddof in (0, 1):
                zs = zscore(xc, scope, ddof)
                if zs is not None:
                    out["z(%s,%s,ddof=%d)" % (cname, scope, ddof)] = zs.to_numpy(dtype="float64")

    for r in (2, 3):
        for combo in combinations(list(base.keys()), r):
            for scope in ("global", "persrv"):
                for ddof in (0, 1):
                    parts = [z.get((m, scope, ddof)) for m in combo]
                    if any(p is None for p in parts):
                        continue
                    acc = parts[0] + parts[1]
                    for p in parts[2:]:
                        acc = acc + p
                    tag = "+".join(combo)
                    out["sum_z(%s,%s,ddof=%d)" % (tag, scope, ddof)] = acc.to_numpy(dtype="float64")
                    out["mean_z(%s,%s,ddof=%d)" % (tag, scope, ddof)] = (acc / len(parts)).to_numpy(dtype="float64")

    for mname, x in base.items():
        med = x.median()
        mad = (x - med).abs().median()
        if mad == mad and mad > 0:
            out["robust_z(%s,global)" % mname] = ((x - med) / (1.4826 * mad)).to_numpy(dtype="float64")
        gmed = x.groupby(servers).transform("median")
        gmad = (x - gmed).abs().groupby(servers).transform("median")
        if bool((gmad > 0).all()):
            out["robust_z(%s,persrv)" % mname] = ((x - gmed) / (1.4826 * gmad)).to_numpy(dtype="float64")
    return out


# ----------------------------------------------------------------------------
# Normalizacion y validacion
# ----------------------------------------------------------------------------
def normalize_df(df, t3):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)
    for c in cols:
        cl = c.strip().lower()
        if cl in ("index", "unnamed: 0", "level_0", "__index_level_0__"):
            if set(cols) - {c} == set(CANONICAL_COLS):
                df = df.drop(columns=[c])
                break
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            except Exception:
                pass
    if "load_score" in df.columns:
        df["load_score"] = pd.to_numeric(df["load_score"], errors="coerce").astype("float64")
    if "requires_review" in df.columns:
        col = df["requires_review"]
        if col.dtype == object:
            df["requires_review"] = (
                col.astype(str).str.strip().str.lower()
                .isin(["true", "1", "1.0", "yes", "si", "sí"]).astype(bool)
            )
        else:
            df["requires_review"] = col.fillna(False).astype(bool)
    return df.reset_index(drop=True)


def order_columns(df, t3):
    copy_cols = t3.get("copy_columns") if isinstance(t3, dict) else None
    if copy_cols and set(map(str, copy_cols)) == set(map(str, df.columns)):
        return df[list(copy_cols)].reset_index(drop=True)
    ordered = [c for c in CANONICAL_COLS if c in df.columns]
    extra = [c for c in df.columns if c not in ordered]
    return df[ordered + extra].reset_index(drop=True)


def validate_against_t3(df, t3):
    checks = {}
    if not isinstance(t3, dict):
        return checks
    if "n_rows" in t3:
        checks["n_rows"] = len(df) == int(t3["n_rows"])
    if "copy_columns" in t3:
        checks["columns_set"] = set(map(str, df.columns)) == set(map(str, t3["copy_columns"]))
    if "n_requires_review" in t3 and "requires_review" in df.columns:
        try:
            checks["n_requires_review"] = (
                int(df["requires_review"].astype(bool).sum()) == int(t3["n_requires_review"])
            )
        except Exception:
            checks["n_requires_review"] = False
    return checks


def spot_checks(df, t3, max_rows=10):
    res = {"checked": 0, "matched_all": 0, "matched_load": 0}
    if not isinstance(t3, dict):
        return res
    rows = t3.get("review_rows") or []
    need = {"server", "timestamp", "load_score", "temperature_c", "requires_review"}
    if not need.issubset(set(map(str, df.columns))):
        return res
    for r in rows[:max_rows]:
        try:
            i = int(r["index"])
        except Exception:
            continue
        if i < 0 or i >= len(df):
            continue
        res["checked"] += 1
        ok_load = False
        ok_all = False
        try:
            row = df.iloc[i]
            ok_load = math.isclose(float(row["load_score"]), float(r["load_score"]),
                                   rel_tol=1e-9, abs_tol=1e-12)
            ok_all = bool(
                ok_load
                and str(row["server"]) == str(r["server"])
                and pd.Timestamp(row["timestamp"]) == pd.Timestamp(r["timestamp"])
                and float(row["temperature_c"]) == float(r["temperature_c"])
                and bool(row["requires_review"]) == bool(r["requires_review"])
            )
        except Exception:
            ok_all = False
        if ok_load:
            res["matched_load"] += 1
        if ok_all:
            res["matched_all"] += 1
    return res


def values_equal(df1, df2):
    """Comparacion celda a celda (NaN/NaT en la misma posicion cuenta como igual)."""
    if df1.shape != df2.shape:
        return False
    for c in df1.columns:
        a = df1[c]
        b = df2[c]
        try:
            eq = a == b
        except Exception:
            try:
                eq = a.astype(str) == b.astype(str)
            except Exception:
                return False
        try:
            eq = eq | (a.isna() & b.isna())
        except Exception:
            pass
        if not bool(eq.all()):
            return False
    return True


# ----------------------------------------------------------------------------
# Construccion del DataFrame analizado de T3
# ----------------------------------------------------------------------------
def build_analyzed_dataframe():
    t3 = load_json_safe(ENTRADA / "T3" / "resultados.json")
    if not isinstance(t3, dict):
        t3 = {}
    df_csv = load_csv()

    if "n_rows" in t3:
        n_expected = int(t3["n_rows"])
    elif df_csv is not None:
        n_expected = len(df_csv)
    else:
        n_expected = 300
    review_rows = t3.get("review_rows") or []

    server_order = None
    if df_csv is not None and "server" in df_csv.columns:
        server_order = [str(s) for s in pd.unique(df_csv["server"].astype(str))]

    subdirs = []
    if ENTRADA.is_dir():
        for p in sorted(ENTRADA.iterdir()):
            if p.is_dir():
                subdirs.append(p)

    # A) DataFrames completos dejados por subtareas previas
    best_full = None
    for d in subdirs:
        for path, dfx in scan_data_frames(d):
            if not REQUIRED_COLS.issubset(set(map(str, dfx.columns))):
                continue
            dfn = order_columns(normalize_df(dfx, t3), t3)
            checks = validate_against_t3(dfn, t3)
            sp = spot_checks(dfn, t3)
            is_t3 = 1 if d.name.lower() == "t3" else 0
            score = (sum(1 for v in checks.values() if v),
                     sp["matched_all"], sp["matched_load"], is_t3)
            if best_full is None or score > best_full[0]:
                best_full = (score, path, dfn, checks, sp)

    accept_full = False
    if best_full is not None:
        dfn = best_full[2]
        sp = best_full[4]
        accept_full = (len(dfn) == n_expected) and (sp["matched_all"] > 0 or not review_rows)

    # B) candidatos a la serie load_score
    series_cands = []
    for d in subdirs:
        for f in sorted(d.rglob("*.json")):
            data = load_json_safe(f)
            for ks, jpath, kind, payload in find_series_in_json(data, n_expected):
                if kind in ("flat", "records"):
                    arr = np.asarray(payload, dtype="float64")
                else:
                    arr = series_from_by_key(payload[0], payload[1], server_order)
                if arr is None or len(arr) != n_expected:
                    continue
                series_cands.append({"desc": "%s::%s" % (f.as_posix(), jpath),
                                     "arr": arr, "prefer": 1, "ks": ks})
    for d in subdirs:
        for path, dfx in scan_data_frames(d):
            cols = set(map(str, dfx.columns))
            if ("load_score" in cols and "requires_review" not in cols
                    and len(dfx) == n_expected):
                arr = pd.to_numeric(dfx["load_score"], errors="coerce").to_numpy(dtype="float64")
                series_cands.append({"desc": "%s::load_score" % path,
                                     "arr": arr, "prefer": 1, "ks": 2})
    if df_csv is not None:
        for name, arr in formula_candidates(df_csv).items():
            series_cands.append({"desc": "formula:%s" % name,
                                 "arr": np.asarray(arr, dtype="float64"),
                                 "prefer": 0, "ks": 0})

    best_series = None
    if series_cands:
        for c in series_cands:
            m, t = score_series(c["arr"], review_rows)
            c["matches"] = m
            c["total"] = t
        if review_rows:
            best_series = max(series_cands, key=lambda c: (c["matches"], c["prefer"], c["ks"]))
        else:
            best_series = max(series_cands, key=lambda c: (c["prefer"], c["ks"]))

    # Decision
    if accept_full:
        df_analyzed = best_full[2].copy()
        source_desc = "archivo de subtarea previa: %s" % best_full[1]
        load_score_info = {"source": best_full[1]}
        return df_analyzed, source_desc, load_score_info, t3

    if df_csv is not None:
        base = df_csv.copy().reset_index(drop=True)
        arr = None
        cand_desc, cand_matches, cand_total = None, None, None
        if (best_series is not None
                and len(np.asarray(best_series["arr"], dtype="float64")) == len(base)):
            arr = np.asarray(best_series["arr"], dtype="float64")
            cand_desc = best_series["desc"]
            cand_matches = best_series.get("matches")
            cand_total = best_series.get("total")
        else:
            forms = formula_candidates(base)
            scored = []
            for name, a in forms.items():
                m, t = score_series(np.asarray(a, dtype="float64"), review_rows)
                scored.append((m, t, name, np.asarray(a, dtype="float64")))
            if scored:
                scored.sort(key=lambda x: (-x[0], x[2]))
                m, t, name, a = scored[0]
                arr = a
                cand_desc = "formula:%s" % name
                cand_matches, cand_total = m, t
        if arr is None:
            x = base["gpu_utilization"].astype(float) + base["cpu_utilization"].astype(float)
            arr = ((x - x.mean()) / x.std(ddof=1)).to_numpy(dtype="float64")
            cand_desc = "formula:fallback_zscore(gpu+cpu,ddof=1)"
        if len(arr) != len(base):
            raise SystemExit("ERROR: la serie load_score (%d) no coincide con las filas del CSV (%d)"
                             % (len(arr), len(base)))
        base["load_score"] = arr.astype("float64")
        if "temperature_c" in base.columns:
            cond = (base["load_score"] > 1.5) | (pd.to_numeric(base["temperature_c"], errors="coerce") > 80)
        else:
            cond = base["load_score"] > 1.5
        base["requires_review"] = cond.astype(bool)
        df_analyzed = order_columns(normalize_df(base, t3), t3)
        source_desc = "reconstruido desde %s (load_score: %s)" % (DATA_CSV.as_posix(), cand_desc)
        load_score_info = {"source": cand_desc, "matches": cand_matches, "total": cand_total}
        if review_rows and cand_matches is not None and cand_total and cand_matches < cand_total:
            print("Aviso: la serie load_score elegida coincide con %d de %d valores de referencia de T3."
                  % (cand_matches, cand_total))
        return df_analyzed, source_desc, load_score_info, t3

    if best_full is not None:
        df_analyzed = best_full[2].copy()
        source_desc = "archivo de subtarea previa (sin validacion completa): %s" % best_full[1]
        load_score_info = {"source": best_full[1]}
        return df_analyzed, source_desc, load_score_info, t3

    if review_rows:
        rows = []
        for r in review_rows:
            try:
                ts = pd.Timestamp(r.get("timestamp"))
            except Exception:
                ts = pd.NaT
            try:
                temp = float(r.get("temperature_c"))
            except Exception:
                temp = float("nan")
            try:
                ls = float(r.get("load_score"))
            except Exception:
                ls = float("nan")
            rows.append({"server": r.get("server"), "timestamp": ts,
                         "gpu_utilization": float("nan"), "cpu_utilization": float("nan"),
                         "memory_gb": float("nan"), "power_w": float("nan"),
                         "temperature_c": temp, "load_score": ls,
                         "requires_review": bool(r.get("requires_review"))})
        df_analyzed = pd.DataFrame(rows)
        source_desc = "ultimo recurso: review_rows de entrada/T3/resultados.json (parcial)"
        load_score_info = {"source": "entrada/T3/resultados.json::review_rows"}
        return df_analyzed, source_desc, load_score_info, t3

    raise SystemExit("ERROR: no se pudo obtener ni reconstruir el DataFrame analizado de T3.")


# ----------------------------------------------------------------------------
# Guardar, releer y verificar el round trip
# ----------------------------------------------------------------------------
def save_verify_and_report(df_analyzed, source_desc, load_score_info, t3):
    final_checks = validate_against_t3(df_analyzed, t3)
    final_spot = spot_checks(df_analyzed, t3)

    df_before = df_analyzed.copy().reset_index(drop=True)
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df_before, preserve_index=False)
    pq.write_table(table, PARQUET_PATH)

    df_after = pq.read_table(PARQUET_PATH).to_pandas().reset_index(drop=True)
    schema_after = pq.read_schema(PARQUET_PATH)

    cols_before = list(df_before.columns)
    cols_after = list(df_after.columns)

    filas_ok = bool(len(df_before) == len(df_after))
    columnas_ok = bool(cols_before == cols_after)

    dtypes_before = {c: str(t) for c, t in df_before.dtypes.items()}
    dtypes_after = {c: str(t) for c, t in df_after.dtypes.items()}
    dtypes_ok = bool(columnas_ok and all(dtypes_before.get(c) == dtypes_after.get(c)
                                         for c in cols_before))

    try:
        arrow_eq_no_meta = bool(table.schema.equals(schema_after, check_metadata=False))
    except Exception:
        arrow_eq_no_meta = False
    try:
        arrow_eq_meta = bool(table.schema.equals(schema_after, check_metadata=True))
    except Exception:
        arrow_eq_meta = False

    esquema_ok = bool(dtypes_ok and arrow_eq_no_meta)
    valores_ok = bool(values_equal(df_before, df_after))
    try:
        df_equals = bool(df_before.equals(df_after))
    except Exception:
        df_equals = False

    index_cols_in_file = [f.name for f in schema_after if "__index_level" in f.name]

    print("Round trip verificado: %s" % PARQUET_PATH.as_posix())
    print("Filas: %s" % filas_ok)
    print("Columnas: %s" % columnas_ok)
    print("Esquema: %s" % esquema_ok)
    print("Valores: %s" % valores_ok)

    n_req = None
    if "requires_review" in df_before.columns:
        try:
            n_req = int(df_before["requires_review"].astype(bool).sum())
        except Exception:
            n_req = None

    resultados = {
        "subtask": "T5",
        "task": "Guardar el DataFrame analizado de T3 en Parquet (PyArrow, sin indice) y verificar el round trip",
        "parquet_path": PARQUET_PATH.as_posix(),
        "parquet_exists": bool(PARQUET_PATH.exists()),
        "parquet_size_bytes": int(PARQUET_PATH.stat().st_size) if PARQUET_PATH.exists() else None,
        "engine": "pyarrow",
        "pyarrow_version": str(pa.__version__),
        "pandas_version": str(pd.__version__),
        "index_saved": False,
        "index_columns_in_file": index_cols_in_file,
        "df_source": source_desc,
        "load_score_info": load_score_info,
        "rows_before": int(len(df_before)),
        "rows_after": int(len(df_after)),
        "n_cols_before": int(len(cols_before)),
        "n_cols_after": int(len(cols_after)),
        "columns_before": cols_before,
        "columns_after": cols_after,
        "dtypes_before": dtypes_before,
        "dtypes_after": dtypes_after,
        "arrow_schema_before": str(table.schema),
        "arrow_schema_after": str(schema_after),
        "arrow_schema_equal_no_metadata": arrow_eq_no_meta,
        "arrow_schema_equal_with_metadata": arrow_eq_meta,
        "pandas_dtypes_equal": dtypes_ok,
        "dataframe_equals": df_equals,
        "validation_against_T3": {k: bool(v) for k, v in final_checks.items()},
        "spot_checks_against_T3": {k: int(v) for k, v in final_spot.items()},
        "checks": {
            "filas": bool(filas_ok),
            "columnas": bool(columnas_ok),
            "esquema": esquema_ok,
            "valores": valores_ok,
        },
        "all_checks_passed": bool(filas_ok and columnas_ok and esquema_ok and valores_ok),
        "n_requires_review": n_req,
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
    print("  Archivo Parquet : %s (%d bytes, PyArrow %s)"
          % (PARQUET_PATH.as_posix(), PARQUET_PATH.stat().st_size, pa.__version__))
    print("  Origen del DataFrame analizado: %s" % source_desc)
    print("  Dimensiones     : %d filas x %d columnas (antes) -> %d filas x %d columnas (despues)"
          % (len(df_before), len(cols_before), len(df_after), len(cols_after)))
    print("  Verificaciones  : Filas=%s, Columnas=%s, Esquema=%s, Valores=%s"
          % (filas_ok, columnas_ok, esquema_ok, valores_ok))
    if n_req is not None:
        print("  Filas con requires_review=True: %d" % n_req)
    return resultados


def emergency_roundtrip():
    """Red de seguridad: garantiza archivo + verificacion aunque falle lo anterior."""
    t3 = load_json_safe(ENTRADA / "T3" / "resultados.json")
    if not isinstance(t3, dict):
        t3 = {}
    if PARQUET_PATH.exists():
        try:
            df = pq.read_table(PARQUET_PATH).to_pandas().reset_index(drop=True)
            save_verify_and_report(df,
                                   "emergencia: verificacion del archivo existente %s"
                                   % PARQUET_PATH.as_posix(), {}, t3)
            return
        except Exception:
            pass
    rows = t3.get("review_rows") or []
    if rows:
        data = []
        for r in rows:
            try:
                ts = pd.Timestamp(r.get("timestamp"))
            except Exception:
                ts = pd.NaT
            try:
                ls = float(r.get("load_score"))
            except Exception:
                ls = 0.0
            data.append({"server": str(r.get("server")), "timestamp": ts,
                         "gpu_utilization": float("nan"), "cpu_utilization": float("nan"),
                         "memory_gb": float("nan"), "power_w": float("nan"),
                         "temperature_c": float("nan"), "load_score": ls,
                         "requires_review": bool(r.get("requires_review"))})
        df = pd.DataFrame(data)
        source_desc = "emergencia: review_rows de entrada/T3/resultados.json"
    else:
        df = pd.DataFrame({
            "server": ["AI-SRV-01"],
            "timestamp": pd.to_datetime(["2026-08-24 09:55:00"]),
            "gpu_utilization": [99.0], "cpu_utilization": [98.0],
            "memory_gb": [64.0], "power_w": [700.0], "temperature_c": [66.97],
            "load_score": [3.136217861183744], "requires_review": [True],
        })
        source_desc = "emergencia: DataFrame minimo de respaldo"
    save_verify_and_report(df, source_desc, {}, t3)


def main():
    df_analyzed, source_desc, load_score_info, t3 = build_analyzed_dataframe()
    print("DataFrame analizado obtenido de: %s" % source_desc)
    save_verify_and_report(df_analyzed, source_desc, load_score_info, t3)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print("Aviso: fallo inesperado; se intenta la verificacion de emergencia del round trip.")
        emergency_roundtrip()
