import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CSV = Path("data/server_measurements.csv")
RUTA_PARQUET = Path("output/server_analysis.parquet")
COLS_Z = ["gpu_utilization", "cpu_utilization", "memory_gb"]
PESOS = np.array([0.50, 0.30, 0.20])  # gpu, cpu, memory (según enunciado)
FILA23 = {"server": "AI-SRV-01", "timestamp": "2026-08-24 09:55:00",
          "load_score": 3.136218, "temperature_c": 66.97, "requires_review": True}


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
    # --- T2+T3: construir el DataFrame analizado ---
    # load_score = zscores(gpu, cpu, memory) @ [0.50, 0.30, 0.20]
    df = pd.read_csv(CSV, parse_dates=["timestamp"])
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # ddof del z-score fijado comprobando contra el valor que el enunciado
    # exige para la fila 23 (3.136218); sin fórmulas alternativas ni archivos externos.
    candidatos = {}
    for ddof in (0, 1):
        z = df[COLS_Z].apply(lambda s, d=ddof: (s - s.mean()) / s.std(ddof=d))
        candidatos[ddof] = z.to_numpy() @ PESOS
    ddof = min(candidatos, key=lambda d: abs(candidatos[d][23] - FILA23["load_score"]))
    load_score = pd.Series(candidatos[ddof], index=df.index)

    df_nuevo = df.copy()
    df_nuevo["load_score"] = load_score
    df_nuevo["requires_review"] = ((df_nuevo["load_score"] > 1.5)
                                   | (df_nuevo["temperature_c"] > 80)).astype(bool)

    # Verificación contra el enunciado (fila de índice 23)
    f23 = df_nuevo.iloc[23]
    fila23_ok = bool(f23["server"] == FILA23["server"]
                     and f23["timestamp"] == pd.Timestamp(FILA23["timestamp"])
                     and abs(float(f23["load_score"]) - FILA23["load_score"]) <= 5e-7
                     and abs(float(f23["temperature_c"]) - FILA23["temperature_c"]) <= 1e-9
                     and bool(f23["requires_review"]) == FILA23["requires_review"])

    # --- T5: guardar con PyArrow, sin el índice ---
    RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    tabla_antes = pa.Table.from_pandas(df_nuevo, preserve_index=False)
    pq.write_table(tabla_antes, RUTA_PARQUET)

    # --- Round trip: volver a leer y verificar ---
    tabla_despues = pq.read_table(RUTA_PARQUET)
    df_despues = tabla_despues.to_pandas()

    filas_ok = df_nuevo.shape[0] == df_despues.shape[0]
    columnas_ok = (df_nuevo.shape[1] == df_despues.shape[1]
                   and list(df_nuevo.columns) == list(df_despues.columns))
    esquema_ok = (tabla_antes.schema.equals(tabla_despues.schema)
                  and [str(t) for t in df_nuevo.dtypes] == [str(t) for t in df_despues.dtypes])
    valores_ok = df_nuevo.equals(df_despues)
    if not valores_ok:
        try:
            pd.testing.assert_frame_equal(df_nuevo, df_despues, check_exact=True)
            valores_ok = True
        except AssertionError:
            valores_ok = False
    sin_indice = not any(str(c).startswith("__index_level") for c in df_despues.columns)

    verificacion = {"Filas": bool(filas_ok), "Columnas": bool(columnas_ok),
                    "Esquema": bool(esquema_ok), "Valores": bool(valores_ok)}
    mensaje = f"Round trip verificado: {RUTA_PARQUET.as_posix()}"

    print(mensaje)
    for k, v in verificacion.items():
        print(f"{k}: {v}")

    resultados = {
        "subtarea": "T5",
        "archivo_parquet": RUTA_PARQUET.as_posix(),
        "archivo_existe": bool(RUTA_PARQUET.exists()),
        "mensaje": mensaje,
        "verificacion": verificacion,
        "filas_ok": bool(filas_ok),
        "columnas_ok": bool(columnas_ok),
        "esquema_ok": bool(esquema_ok),
        "valores_ok": bool(valores_ok),
        "round_trip_verificado": bool(all(verificacion.values())),
        "guardado_sin_indice": bool(sin_indice),
        "n_filas_antes": int(df_nuevo.shape[0]),
        "n_filas_despues": int(df_despues.shape[0]),
        "n_columnas_antes": int(df_nuevo.shape[1]),
        "n_columnas_despues": int(df_despues.shape[1]),
        "columnas": list(df_nuevo.columns),
        "esquema_antes": [f"{f.name}: {f.type}" for f in tabla_antes.schema],
        "esquema_despues": [f"{f.name}: {f.type}" for f in tabla_despues.schema],
        "dtypes_antes": {c: str(t) for c, t in df_nuevo.dtypes.items()},
        "dtypes_despues": {c: str(t) for c, t in df_despues.dtypes.items()},
        "formula_load_score": "zscores(gpu_utilization, cpu_utilization, memory_gb) @ [0.50, 0.30, 0.20]",
        "zscore_ddof": int(ddof),
        "fila_23": df_nuevo.iloc[23].to_dict(),
        "fila_23_coincide_enunciado": fila23_ok,
        "load_score_resumen": {"min": float(df_nuevo["load_score"].min()),
                               "max": float(df_nuevo["load_score"].max()),
                               "mean": float(df_nuevo["load_score"].mean())},
        "requires_review_true": int(df_nuevo["requires_review"].sum()),
        "requires_review_true_referencia_T3": 49,
        "requires_review_true_por_servidor": {
            k: int(v) for k, v in
            df_nuevo.loc[df_nuevo["requires_review"], "server"].value_counts().sort_index().items()},
    }
    Path("resultados.json").write_text(
        json.dumps(jsonable(resultados), indent=2, ensure_ascii=False))

    print(f"Parquet: {RUTA_PARQUET.as_posix()} | filas={df_nuevo.shape[0]} "
          f"columnas={df_nuevo.shape[1]} | sin_indice={sin_indice}")
    print(f"load_score fila 23 = {float(f23['load_score'])} "
          f"(enunciado: {FILA23['load_score']}) | coincide={fila23_ok}")
    print(f"requires_review=True: {int(df_nuevo['requires_review'].sum())}")


if __name__ == "__main__":
    main()
