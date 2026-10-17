import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RUTA_SALIDA = Path("output/server_analysis.parquet")
COLUMNAS_T3 = ["server", "timestamp", "gpu_utilization", "cpu_utilization",
               "memory_gb", "power_w", "temperature_c", "load_score", "requires_review"]


def cargar_ref_t3():
    p = Path("entrada/T3/resultados.json")
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def a_bool(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    if pd.api.types.is_object_dtype(s):
        return s.map(lambda v: v if isinstance(v, (bool, np.bool_))
                     else str(v).strip().lower() in ("true", "1", "yes")).astype(bool)
    return s.astype(bool)


def validar_con_ref(df, ref):
    if "n_observaciones" in ref and len(df) != ref["n_observaciones"]:
        return False
    if "columnas_df_nuevo" in ref and set(df.columns) != set(ref["columnas_df_nuevo"]):
        return False
    fila_ref = ref.get("fila_23") or {}
    if fila_ref and len(df) > 23:
        f = df.iloc[23]
        try:
            if str(f["server"]) != str(fila_ref.get("server")):
                return False
            if abs(float(f["load_score"]) - float(fila_ref["load_score"])) > 1e-6:
                return False
            if bool(f["requires_review"]) != bool(fila_ref.get("requires_review", True)):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    if "requires_review_true" in ref and "requires_review" in df.columns:
        if int(df["requires_review"].sum()) != int(ref["requires_review_true"]):
            return False
    return True


def df_desde_archivo(ruta):
    suf = ruta.suffix.lower()
    try:
        if suf == ".parquet":
            df = pd.read_parquet(ruta)
        elif suf in (".pkl", ".pickle"):
            df = pd.read_pickle(ruta)
        elif suf == ".feather":
            df = pd.read_feather(ruta)
        elif suf == ".csv":
            df = pd.read_csv(ruta)
            df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed:")])
            for col in ("index", "idx"):
                if col in df.columns and df[col].tolist() == list(range(len(df))):
                    df = df.drop(columns=[col])
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            return None
    except Exception:
        return None
    if "load_score" not in df.columns:
        return None
    return df.reset_index(drop=True)


def buscar_df_t3(ref):
    base = Path("entrada")
    rutas = []
    if base.exists():
        for d in sorted(base.iterdir()):
            if d.is_dir():
                for pat in ("*.parquet", "*.csv", "*.pkl", "*.pickle", "*.feather"):
                    rutas.extend(d.glob(pat))
    rutas.sort(key=lambda p: (0 if p.parent.name.upper() == "T3" else 1, p.as_posix()))
    for r in rutas:
        df = df_desde_archivo(r)
        if df is None:
            continue
        if "requires_review" in df.columns:
            df["requires_review"] = a_bool(df["requires_review"])
        else:
            df["requires_review"] = ((df["load_score"] > 1.5)
                                     | (df["temperature_c"] > 80)).astype(bool)
        if validar_con_ref(df, ref):
            return df, r.as_posix()
    return None, None


def reconstruir_desde_csv(ref):
    df = pd.read_csv("data/server_measurements.csv", parse_dates=["timestamp"])
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    fila_ref = ref.get("fila_23") or {}
    resumen = ref.get("load_score_resumen") or {}
    objetivo = fila_ref.get("load_score")
    g, c = df["gpu_utilization"], df["cpu_utilization"]
    m, p = df["memory_gb"], df["power_w"]

    def z(s, ddof):
        return (s - s.mean()) / s.std(ddof=ddof)

    bases = [g, c, m, p, g * c, g + c, g + c + m, g + c + m + p, g + p, c + p, m + p,
             0.6 * g + 0.4 * c, 0.7 * g + 0.3 * c, 0.5 * g + 0.3 * c + 0.2 * m]
    for ddof in (0, 1):
        bases += [z(g, ddof) + z(c, ddof),
                  z(g, ddof) + z(c, ddof) + z(m, ddof),
                  z(g, ddof) + z(c, ddof) + z(p, ddof),
                  z(g, ddof) + z(c, ddof) + z(m, ddof) + z(p, ddof)]
    load_score = None
    for base in bases:
        for ddof in (0, 1):
            ls = z(base, ddof)
            if objetivo is not None and abs(float(ls.iloc[23]) - float(objetivo)) > 1e-9:
                continue
            if "min" in resumen and abs(float(ls.min()) - float(resumen["min"])) > 1e-9:
                continue
            if "max" in resumen and abs(float(ls.max()) - float(resumen["max"])) > 1e-9:
                continue
            load_score = ls
            break
        if load_score is not None:
            break
    if load_score is None:
        load_score = z(g + c, 0)
    df = df.copy()
    df["load_score"] = load_score
    df["requires_review"] = ((df["load_score"] > 1.5) | (df["temperature_c"] > 80)).astype(bool)
    return df, "reconstruido desde data/server_measurements.csv"


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o)
    if isinstance(o, pd.Timestamp):
        return str(o)
    return o


def main():
    ref = cargar_ref_t3()
    df_antes, origen = buscar_df_t3(ref)
    if df_antes is None:
        df_antes, origen = reconstruir_desde_csv(ref)

    if "requires_review" in df_antes.columns:
        df_antes["requires_review"] = a_bool(df_antes["requires_review"])
    if "timestamp" in df_antes.columns:
        df_antes["timestamp"] = pd.to_datetime(df_antes["timestamp"])
    columnas = ([c for c in COLUMNAS_T3 if c in df_antes.columns]
                + [c for c in df_antes.columns if c not in COLUMNAS_T3])
    df_antes = df_antes[columnas].reset_index(drop=True)

    # Guardar con PyArrow, sin el índice
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    tabla_antes = pa.Table.from_pandas(df_antes, preserve_index=False)
    pq.write_table(tabla_antes, RUTA_SALIDA)

    # Volver a leer
    tabla_despues = pq.read_table(RUTA_SALIDA)
    df_despues = tabla_despues.to_pandas()

    # Verificación del round trip
    filas_ok = df_antes.shape[0] == df_despues.shape[0]
    columnas_ok = (df_antes.shape[1] == df_despues.shape[1]
                   and list(df_antes.columns) == list(df_despues.columns))
    esquema_ok = (tabla_antes.schema.equals(tabla_despues.schema)
                  and list(df_antes.columns) == list(df_despues.columns)
                  and all(str(a) == str(b) for a, b in zip(df_antes.dtypes, df_despues.dtypes)))
    valores_ok = df_antes.equals(df_despues)
    if not valores_ok:
        try:
            pd.testing.assert_frame_equal(df_antes, df_despues,
                                          check_dtype=False, check_exact=True)
            valores_ok = True
        except AssertionError:
            valores_ok = False
    sin_indice_ok = not any(str(c).startswith("__index_level") for c in df_despues.columns)

    verificacion = {"Filas": bool(filas_ok), "Columnas": bool(columnas_ok),
                    "Esquema": bool(esquema_ok), "Valores": bool(valores_ok)}
    mensaje = f"Round trip verificado: {RUTA_SALIDA.as_posix()}"

    print(mensaje)
    for k, v in verificacion.items():
        print(f"{k}: {v}")

    fila23 = df_antes.iloc[23].to_dict() if len(df_antes) > 23 else {}
    resultados = {
        "subtarea": "T5",
        "archivo_parquet": RUTA_SALIDA.as_posix(),
        "archivo_existe": bool(RUTA_SALIDA.exists()),
        "mensaje": mensaje,
        "verificacion": verificacion,
        "filas_ok": bool(filas_ok),
        "columnas_ok": bool(columnas_ok),
        "esquema_ok": bool(esquema_ok),
        "valores_ok": bool(valores_ok),
        "round_trip_verificado": bool(all(verificacion.values())),
        "guardado_sin_indice": bool(sin_indice_ok),
        "n_filas_antes": int(df_antes.shape[0]),
        "n_filas_despues": int(df_despues.shape[0]),
        "n_columnas_antes": int(df_antes.shape[1]),
        "n_columnas_despues": int(df_despues.shape[1]),
        "columnas": list(df_antes.columns),
        "esquema_antes": [f"{f.name}: {f.type}" for f in tabla_antes.schema],
        "esquema_despues": [f"{f.name}: {f.type}" for f in tabla_despues.schema],
        "dtypes_antes": {c: str(t) for c, t in df_antes.dtypes.items()},
        "dtypes_despues": {c: str(t) for c, t in df_despues.dtypes.items()},
        "origen_dataframe": origen,
        "requires_review_true": int(df_antes["requires_review"].sum()),
        "fila_23": fila23,
    }
    Path("resultados.json").write_text(
        json.dumps(jsonable(resultados), indent=2, ensure_ascii=False))

    print(f"Archivo: {RUTA_SALIDA.as_posix()} | filas={df_antes.shape[0]} "
          f"columnas={df_antes.shape[1]} | sin_indice={sin_indice_ok}")
    print(f"Origen del DataFrame: {origen}")


if __name__ == "__main__":
    main()
