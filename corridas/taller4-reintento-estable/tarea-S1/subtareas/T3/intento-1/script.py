#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 - Identificar observaciones para revisión.

Crea una copia del DataFrame original (sin modificarlo) y agrega:
  * load_score      : indicador obtenido en T2 (zscores @ [0.50, 0.30, 0.20]).
  * requires_review : bool, True si load_score > 1.5 o temperature_c > 80.
Muestra las filas marcadas para revisión, verifica la fila 23 contra el
resultado esperado y escribe el parquet exigido: output/server_analysis.parquet.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RUTA_CSV = "data/server_measurements.csv"
RUTA_T2_JSON = "entrada/T2/resultados.json"
RUTA_PARQUET = "output/server_analysis.parquet"
CARPETA_T3 = "entrada/T3"

FEATURES = ["gpu_utilization", "cpu_utilization", "memory_gb"]
PESOS_T2 = np.array([0.50, 0.30, 0.20])
UMBRAL_LOAD = 1.5
UMBRAL_TEMP = 80.0


def _zscore_load_score(X, means, stds, pesos):
    """Z-score vectorizado (sin for sobre observaciones) + combinación lineal."""
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    pesos = np.asarray(pesos, dtype=float)
    stds_safe = np.where(stds == 0, 1.0, stds)  # evita división por cero
    Z = (X - means) / stds_safe
    return Z @ pesos


def obtener_load_score(df):
    """Recupera el load_score calculado en T2; si no está disponible, lo recalcula igual."""
    if Path(RUTA_T2_JSON).is_file():
        with open(RUTA_T2_JSON, "r", encoding="utf-8") as fh:
            t2 = json.load(fh)
        lista = t2.get("load_score")
        if lista is not None and len(lista) == len(df):
            return np.asarray(lista, dtype=float), "lista_load_score_de_T2"
        if all(k in t2 for k in ("means", "stds", "pesos")):
            X = df[FEATURES].to_numpy(dtype=float)
            return (_zscore_load_score(X, t2["means"], t2["stds"], t2["pesos"]),
                    "recalculado_con_means_stds_pesos_de_T2")
    # Último recurso: mismos pasos que T2 a partir de los datos
    X = df[FEATURES].to_numpy(dtype=float)
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    return _zscore_load_score(X, means, stds, PESOS_T2), "recalculado_desde_datos"


def a_tipos_json(obj):
    """Convierte tipos de numpy/pandas a tipos nativos serializables a JSON."""
    if isinstance(obj, dict):
        return {k: a_tipos_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [a_tipos_json(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def main():
    # ------------------------------------------------------------------
    # 1) DataFrame original (timestamp leído/convertido como fecha-hora)
    # ------------------------------------------------------------------
    df_original = pd.read_csv(RUTA_CSV)
    df_original.columns = [str(c).strip() for c in df_original.columns]
    df_original["timestamp"] = pd.to_datetime(df_original["timestamp"])

    # ------------------------------------------------------------------
    # 2) load_score obtenido en T2
    # ------------------------------------------------------------------
    load_score, origen_ls = obtener_load_score(df_original)
    assert load_score.shape == (len(df_original),), "load_score debe tener forma (n,)"

    # ------------------------------------------------------------------
    # 3) Copia del original + columnas nuevas (el original NO se modifica)
    # ------------------------------------------------------------------
    df_new = df_original.copy()
    df_new["load_score"] = load_score
    df_new["requires_review"] = (
        (df_new["load_score"] > UMBRAL_LOAD) | (df_new["temperature_c"] > UMBRAL_TEMP)
    ).astype(bool)

    assert "load_score" not in df_original.columns, "el original debe quedar intacto"
    assert "requires_review" not in df_original.columns, "el original debe quedar intacto"
    assert df_new["requires_review"].dtype == bool, "requires_review debe ser de tipo bool"

    # ------------------------------------------------------------------
    # 4) Filas marcadas para revisión (mostrar)
    # ------------------------------------------------------------------
    cols_show = ["server", "timestamp", "load_score", "temperature_c", "requires_review"]
    df_revisar = df_new.loc[df_new["requires_review"], cols_show]

    print("Filas marcadas para revisión (requires_review == True):")
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                           "display.width", 200):
        print(df_revisar.to_string())

    mask_load = df_new["load_score"] > UMBRAL_LOAD
    mask_temp = df_new["temperature_c"] > UMBRAL_TEMP

    # ------------------------------------------------------------------
    # 5) Verificación de la fila 23 contra el resultado esperado
    # ------------------------------------------------------------------
    f23 = df_new.iloc[23]
    fila_23 = {
        "indice": int(df_new.index[23]),
        "server": str(f23["server"]),
        "timestamp": f23["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        "load_score": float(f23["load_score"]),
        "temperature_c": float(f23["temperature_c"]),
        "requires_review": bool(f23["requires_review"]),
    }
    esperado_23 = {
        "server": "AI-SRV-01",
        "timestamp": "2026-08-24 09:55:00",
        "load_score_redondeado_6dec": 3.136218,
        "temperature_c": 66.97,
        "requires_review": True,
    }
    verif_23 = {
        "server_ok": fila_23["server"] == esperado_23["server"],
        "timestamp_ok": fila_23["timestamp"] == esperado_23["timestamp"],
        "load_score_ok": round(fila_23["load_score"], 6) == esperado_23["load_score_redondeado_6dec"],
        "temperature_c_ok": round(fila_23["temperature_c"], 2) == esperado_23["temperature_c"],
        "requires_review_ok": fila_23["requires_review"] == esperado_23["requires_review"],
    }
    fila_23_coincide = bool(all(verif_23.values()))

    # ------------------------------------------------------------------
    # 6) Parquet exigido por el enunciado + copia para la subtarea T4
    # ------------------------------------------------------------------
    Path(RUTA_PARQUET).parent.mkdir(parents=True, exist_ok=True)
    df_new.to_parquet(RUTA_PARQUET, index=False)

    Path(CARPETA_T3).mkdir(parents=True, exist_ok=True)
    df_new.to_parquet(str(Path(CARPETA_T3) / "df_enriquecido.parquet"), index=False)

    # ------------------------------------------------------------------
    # 7) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T3_requires_review",
        "archivo_datos": RUTA_CSV,
        "origen_load_score": origen_ls,
        "n_observaciones": int(len(df_new)),
        "columnas_resultado": list(df_new.columns),
        "columnas_agregadas": ["load_score", "requires_review"],
        "requires_review_dtype": str(df_new["requires_review"].dtype),
        "original_sin_modificar": bool("load_score" not in df_original.columns
                                       and "requires_review" not in df_original.columns),
        "regla_revision": "load_score > 1.5 o temperature_c > 80",
        "umbral_load_score": UMBRAL_LOAD,
        "umbral_temperature_c": UMBRAL_TEMP,
        "n_requires_review": int(df_new["requires_review"].sum()),
        "n_por_load_score": int(mask_load.sum()),
        "n_por_temperature_c": int(mask_temp.sum()),
        "n_por_ambas_condiciones": int((mask_load & mask_temp).sum()),
        "indices_requires_review": [int(i) for i in
                                    df_new.index[df_new["requires_review"].to_numpy()]],
        "filas_requires_review": a_tipos_json(df_revisar.to_dict(orient="records")),
        "fila_23": fila_23,
        "fila_23_esperada": esperado_23,
        "verificacion_fila_23": verif_23,
        "fila_23_coincide": fila_23_coincide,
        "dataframe_resultado": a_tipos_json(df_new.to_dict(orient="records")),
        "archivo_parquet": RUTA_PARQUET,
    }
    with open("resultados.json", "w", encoding="utf-8") as fh:
        json.dump(a_tipos_json(resultados), fh, indent=2, ensure_ascii=False)
    with open(str(Path(CARPETA_T3) / "resultados.json"), "w", encoding="utf-8") as fh:
        json.dump(a_tipos_json(resultados), fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 8) Resumen breve
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESUMEN T3 - Identificación de observaciones para revisión")
    print("=" * 70)
    print(f"Observaciones: {len(df_new)} | Columnas agregadas: load_score, "
          f"requires_review (dtype={df_new['requires_review'].dtype})")
    print(f"Regla: load_score > {UMBRAL_LOAD} o temperature_c > {UMBRAL_TEMP}")
    print(f"Filas marcadas: {int(df_new['requires_review'].sum())} "
          f"(por load_score: {int(mask_load.sum())}, por temperatura: {int(mask_temp.sum())}, "
          f"por ambas: {int((mask_load & mask_temp).sum())})")
    print(f"Fila 23 -> {fila_23['server']} | {fila_23['timestamp']} | "
          f"load_score={fila_23['load_score']:.6f} | "
          f"temperature_c={fila_23['temperature_c']} | requires_review={fila_23['requires_review']}")
    print(f"Verificación fila 23: {'OK' if fila_23_coincide else 'FALLO'} -> {verif_23}")
    print(f"Original sin modificar: {'load_score' not in df_original.columns and 'requires_review' not in df_original.columns}")
    print(f"Parquet escrito: {RUTA_PARQUET}")


if __name__ == "__main__":
    main()
