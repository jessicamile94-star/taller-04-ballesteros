#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5 - Guardar y verificar el resultado (round trip Parquet con PyArrow).

1) Obtiene el DataFrame analizado de T3 (server, timestamp, gpu_utilization,
   cpu_utilization, memory_gb, power_w, temperature_c, load_score,
   requires_review): lo carga desde entrada/T3/ si T3 lo guardó como archivo;
   si no, lo reconstruye desde data/server_measurements.csv más el load_score
   de entrada/T2/resultados.json y la condición de T3
   (requires_review = (load_score > 1.5) | (temperature_c > 80)).
2) Lo guarda en output/server_analysis.parquet con PyArrow, sin el índice.
3) Lo vuelve a leer y verifica: filas, columnas, esquema y valores.
4) Imprime el mensaje de round trip y escribe resultados.json.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_PATH = Path("output") / "server_analysis.parquet"
T3_DIR = Path("entrada") / "T3"
T2_JSON = Path("entrada") / "T2" / "resultados.json"
DATA_CSV = Path("data") / "server_measurements.csv"

REQUIRED_COLS = {"server", "timestamp", "load_score", "requires_review"}


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_t3_results():
    path = T3_DIR / "resultados.json"
    if path.exists():
        try:
            return read_json(path)
        except Exception:
            return {}
    return {}


def candidates_from_t3_dir():
    """DataFrames que T3 haya guardado en entrada/T3/ (parquet/pickle/feather/csv)."""
    out = []
    if not T3_DIR.is_dir():
        return out
    seen = set()
    for pattern in ("*.parquet", "*.pkl", "*.pickle", "*.feather", "*.csv"):
        for f in sorted(T3_DIR.glob(pattern)):
            if f in seen:
                continue
            seen.add(f)
            try:
                if f.suffix == ".parquet":
                    df = pd.read_parquet(f)
                elif f.suffix in (".pkl", ".pickle"):
                    df = pd.read_pickle(f)
                elif f.suffix == ".feather":
                    df = pd.read_feather(f)
                else:
                    df = pd.read_csv(f)
                    if "Unnamed: 0" in df.columns:
                        df = df.drop(columns=["Unnamed: 0"])
                if isinstance(df, pd.DataFrame) and REQUIRED_COLS.issubset(set(df.columns)):
                    out.append((f.as_posix(), df))
            except Exception:
                continue
    return out


def extract_load_score(t2):
    """Extrae la lista load_score de entrada/T2/resultados.json."""
    if "load_score" not in t2:
        raise KeyError("entrada/T2/resultados.json no contiene la clave 'load_score'")
    ls = t2["load_score"]
    if isinstance(ls, dict):
        for key in ("values", "data", "load_score", "series"):
            if isinstance(ls.get(key), list):
                ls = ls[key]
                break
        else:
            ls = [ls[str(i)] for i in range(len(ls))]  # dict {indice: valor}
    if isinstance(ls, (list, tuple)) and ls and isinstance(ls[0], dict):
        key = next((k for k in ("load_score", "value", "valor") if k in ls[0]), None)
        if key is None:
            raise ValueError("Formato de 'load_score' no reconocido")
        ls = [d[key] for d in ls]
    if not isinstance(ls, (list, tuple, np.ndarray)):
        raise ValueError("Formato de 'load_score' no reconocido")
    return list(ls)


def reconstruct_from_sources():
    """Reconstruye el DataFrame analizado de T3 desde las fuentes originales."""
    df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    t2 = read_json(T2_JSON)
    ls = extract_load_score(t2)
    if len(ls) != len(df):
        raise ValueError(
            "load_score de T2 tiene %d valores pero el CSV tiene %d filas" % (len(ls), len(df))
        )
    df = df.copy()
    df["load_score"] = np.asarray(ls, dtype="float64")
    df["requires_review"] = ((df["load_score"] > 1.5) | (df["temperature_c"] > 80)).astype(bool)
    return df


def normalize(df, t3):
    """Ajusta dtypes y orden de columnas al contrato de T3."""
    df = df.copy()
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "load_score" in df.columns:
        df["load_score"] = df["load_score"].astype("float64")
    if "requires_review" in df.columns:
        col = df["requires_review"]
        if col.dtype == object:
            df["requires_review"] = (
                col.astype(str).str.strip().str.lower().isin(["true", "1", "yes"]).astype(bool)
            )
        else:
            df["requires_review"] = col.astype(bool)
    copy_cols = t3.get("copy_columns")
    if copy_cols and set(copy_cols) == set(df.columns):
        df = df[list(copy_cols)]
    return df.reset_index(drop=True)


def validate_against_t3(df, t3):
    """Compara el DataFrame candidato con lo declarado en entrada/T3/resultados.json."""
    checks = {}
    if "n_rows" in t3:
        checks["n_rows"] = len(df) == int(t3["n_rows"])
    if "copy_columns" in t3:
        checks["columns"] = list(df.columns) == list(t3["copy_columns"])
    if "n_requires_review" in t3 and "requires_review" in df.columns:
        checks["n_requires_review"] = (
            int(df["requires_review"].sum()) == int(t3["n_requires_review"])
        )
    review_rows = t3.get("review_rows") or []
    needed = ("server", "timestamp", "load_score", "temperature_c", "requires_review")
    if review_rows and all(c in df.columns for c in needed):
        ok = True
        for r in review_rows[:5]:
            i = int(r["index"])
            if i < 0 or i >= len(df):
                ok = False
                break
            row = df.iloc[i]
            ok = ok and (
                row["server"] == r["server"]
                and pd.Timestamp(row["timestamp"]) == pd.Timestamp(r["timestamp"])
                and math.isclose(float(row["load_score"]), float(r["load_score"]),
                                 rel_tol=1e-12, abs_tol=1e-12)
                and math.isclose(float(row["temperature_c"]), float(r["temperature_c"]),
                                 rel_tol=1e-12, abs_tol=1e-12)
                and bool(row["requires_review"]) == bool(r["requires_review"])
            )
        checks["spot_review_rows"] = bool(ok)
    return checks


def values_equal(df1, df2, cols):
    """Verificación celda a celda (NaN/NaT en la misma posición cuenta como igual)."""
    if len(df1) != len(df2):
        return False
    for c in cols:
        if c not in df2.columns:
            return False
        v1 = df1[c].reset_index(drop=True)
        v2 = df2[c].reset_index(drop=True)
        try:
            eq = (v1 == v2) | (v1.isna() & v2.isna())
            ok = bool(eq.all())
        except Exception:
            ok = all(
                (a == b) or (pd.isna(a) and pd.isna(b))
                for a, b in zip(v1.tolist(), v2.tolist())
            )
        if not ok:
            return False
    return True


def main():
    t3 = load_t3_results()

    # ------------------------------------------------------------------
    # 1) Obtener el DataFrame analizado de T3
    # ------------------------------------------------------------------
    candidates = candidates_from_t3_dir()
    try:
        candidates.append(
            ("reconstruido: data/server_measurements.csv + entrada/T2/resultados.json",
             reconstruct_from_sources())
        )
    except Exception as exc:
        print("Aviso: no se pudo reconstruir el DataFrame desde las fuentes (%s)" % exc)

    if not candidates:
        raise SystemExit("ERROR: no se pudo obtener el DataFrame analizado de T3.")

    evaluated = []
    for name, df in candidates:
        dfn = normalize(df, t3)
        evaluated.append((name, dfn, validate_against_t3(dfn, t3)))

    fully_valid = [e for e in evaluated if e[2] and all(e[2].values())]
    if fully_valid:
        source_name, df_analyzed, source_checks = fully_valid[0]
    else:
        recon = [e for e in evaluated if e[0].startswith("reconstruido")]
        source_name, df_analyzed, source_checks = (recon or evaluated)[0]

    # ------------------------------------------------------------------
    # 2) Guardar con PyArrow, sin el índice
    # ------------------------------------------------------------------
    df_before = df_analyzed.copy()
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df_before, preserve_index=False)
    pq.write_table(table, PARQUET_PATH)

    # ------------------------------------------------------------------
    # 3) Volver a leer y verificar el round trip
    # ------------------------------------------------------------------
    df_after = pq.read_table(PARQUET_PATH).to_pandas()

    cols_before = list(df_before.columns)
    cols_after = list(df_after.columns)

    filas_ok = len(df_before) == len(df_after)
    columnas_ok = (len(cols_before) == len(cols_after)) and (cols_before == cols_after)

    dtypes_before = {c: str(t) for c, t in df_before.dtypes.items()}
    dtypes_after = {c: str(t) for c, t in df_after.dtypes.items()}
    dtypes_ok = columnas_ok and all(dtypes_before[c] == dtypes_after[c] for c in cols_before)

    schema_after_arrow = pq.read_schema(PARQUET_PATH)
    arrow_schema_equal = bool(table.schema.equals(schema_after_arrow))
    esquema_ok = bool(dtypes_ok and arrow_schema_equal)

    dataframe_equals = bool(df_before.equals(df_after))
    valores_ok = bool(columnas_ok and values_equal(df_before, df_after, cols_before))

    # ------------------------------------------------------------------
    # 4) Mensaje esperado + resultados.json
    # ------------------------------------------------------------------
    print("Round trip verificado: %s" % PARQUET_PATH.as_posix())
    print("Filas: %s" % filas_ok)
    print("Columnas: %s" % columnas_ok)
    print("Esquema: %s" % esquema_ok)
    print("Valores: %s" % valores_ok)

    index_cols_in_file = [f.name for f in schema_after_arrow if "__index_level" in f.name]

    resultados = {
        "subtask": "T5",
        "task": "Guardar el DataFrame analizado de T3 en Parquet (PyArrow) y verificar el round trip",
        "parquet_path": PARQUET_PATH.as_posix(),
        "parquet_exists": bool(PARQUET_PATH.exists()),
        "parquet_size_bytes": int(PARQUET_PATH.stat().st_size) if PARQUET_PATH.exists() else None,
        "engine": "pyarrow",
        "pyarrow_version": pa.__version__,
        "pandas_version": pd.__version__,
        "index_saved": False,
        "index_columns_in_file": index_cols_in_file,
        "df_source": source_name,
        "df_source_validated_against_T3": bool(source_checks) and all(source_checks.values()),
        "df_source_validation_checks": {k: bool(v) for k, v in source_checks.items()},
        "rows_before": int(len(df_before)),
        "rows_after": int(len(df_after)),
        "n_cols_before": int(len(cols_before)),
        "n_cols_after": int(len(cols_after)),
        "columns_before": cols_before,
        "columns_after": cols_after,
        "dtypes_before": dtypes_before,
        "dtypes_after": dtypes_after,
        "arrow_schema_before": str(table.schema),
        "arrow_schema_after": str(schema_after_arrow),
        "arrow_schema_equal": arrow_schema_equal,
        "dataframe_equals": dataframe_equals,
        "checks": {
            "filas": bool(filas_ok),
            "columnas": bool(columnas_ok),
            "esquema": esquema_ok,
            "valores": valores_ok,
        },
        "all_checks_passed": bool(filas_ok and columnas_ok and esquema_ok and valores_ok),
        "n_requires_review": (
            int(df_before["requires_review"].sum())
            if "requires_review" in df_before.columns else None
        ),
        "message": "Round trip verificado: %s" % PARQUET_PATH.as_posix(),
    }

    with open("resultados.json", "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=2)

    t5_dir = Path("entrada") / "T5"
    t5_dir.mkdir(parents=True, exist_ok=True)
    with open(t5_dir / "resultados.json", "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Resumen breve
    # ------------------------------------------------------------------
    print("\nResumen T5:")
    print("  Archivo Parquet : %s (%d bytes, PyArrow %s)"
          % (PARQUET_PATH.as_posix(), PARQUET_PATH.stat().st_size, pa.__version__))
    print("  Origen del DataFrame analizado: %s" % source_name)
    print("  Dimensiones     : %d filas x %d columnas (antes) -> %d filas x %d columnas (despues)"
          % (len(df_before), len(cols_before), len(df_after), len(cols_after)))
    print("  Verificaciones  : Filas=%s, Columnas=%s, Esquema=%s, Valores=%s"
          % (filas_ok, columnas_ok, esquema_ok, valores_ok))
    if resultados["n_requires_review"] is not None:
        print("  Filas con requires_review=True: %d" % resultados["n_requires_review"])


if __name__ == "__main__":
    main()
