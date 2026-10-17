#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5 - Guardar y verificar el resultado (round trip Parquet con PyArrow).

1) Carga el DataFrame analizado de T3 (300 filas x 9 columnas, con load_score
   y requires_review).
2) Lo guarda en output/server_analysis.parquet con PyArrow, SIN el índice.
3) Lo vuelve a leer y verifica, como mínimo: filas antes == filas después,
   columnas antes == columnas después, esquema antes == esquema después y
   valores iguales.
4) Imprime el mensaje de round trip y escribe resultados.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RUTA_PARQUET = "output/server_analysis.parquet"
RUTA_RESULTADOS = "resultados.json"
RUTA_CSV = "data/server_measurements.csv"

COLUMNAS_ESPERADAS = [
    "server", "timestamp", "gpu_utilization", "cpu_utilization",
    "memory_gb", "power_w", "temperature_c", "load_score", "requires_review",
]
N_ESPERADO = 300
UMBRAL_LOAD_SCORE = 1.5
UMBRAL_TEMPERATURA = 80.0


# ---------------------------------------------------------------------------
# Normalización del DataFrame analizado de T3
# ---------------------------------------------------------------------------

def _preparar(df):
    """Deja el DataFrame con las 9 columnas de T3, dtypes correctos e índice 0..n-1."""
    faltantes = [c for c in COLUMNAS_ESPERADAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas del DataFrame de T3: {faltantes}")
    df = df.loc[:, COLUMNAS_ESPERADAS].copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if not pd.api.types.is_bool_dtype(df["requires_review"]):
        if pd.api.types.is_object_dtype(df["requires_review"]):
            df["requires_review"] = (
                df["requires_review"].astype(str).str.strip().str.lower()
                .isin(["true", "1", "t", "yes"])
            )
        else:
            df["requires_review"] = df["requires_review"].astype(bool)
    for col in ("gpu_utilization", "cpu_utilization", "memory_gb",
                "power_w", "temperature_c", "load_score"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype("float64")
    return df.reset_index(drop=True)


def _leer_tabla(ruta):
    sufijo = ruta.suffix.lower()
    if sufijo == ".parquet":
        return pd.read_parquet(ruta)
    if sufijo == ".csv":
        return pd.read_csv(ruta)
    if sufijo in (".pickle", ".pkl"):
        return pd.read_pickle(ruta)
    if sufijo == ".feather":
        return pd.read_feather(ruta)
    raise ValueError(f"Formato no soportado: {ruta}")


# ---------------------------------------------------------------------------
# Carga del DataFrame analizado de T3 (archivos de entrada/T3/, JSON o reconstrucción)
# ---------------------------------------------------------------------------

def _desde_archivos_t3():
    dir_t3 = Path("entrada") / "T3"
    if not dir_t3.is_dir():
        return None
    for patron in ("*.parquet", "*.csv", "*.pickle", "*.pkl", "*.feather"):
        for ruta in sorted(dir_t3.glob(patron)):
            try:
                df = _leer_tabla(ruta)
            except Exception:
                continue
            if (isinstance(df, pd.DataFrame)
                    and len(df) == N_ESPERADO
                    and set(COLUMNAS_ESPERADAS).issubset(df.columns)):
                return _preparar(df), f"entrada/T3/{ruta.name}"
    return None


def _buscar_registros(obj, n):
    """Busca recursivamente una lista de n dicts con las columnas clave de T3."""
    claves_min = {"server", "timestamp", "load_score", "requires_review"}
    if isinstance(obj, list):
        if (len(obj) == n and obj and isinstance(obj[0], dict)
                and claves_min.issubset(obj[0].keys())):
            return obj
        for elem in obj:
            hallado = _buscar_registros(elem, n)
            if hallado is not None:
                return hallado
    elif isinstance(obj, dict):
        for clave in sorted(obj):
            hallado = _buscar_registros(obj[clave], n)
            if hallado is not None:
                return hallado
    return None


def _desde_json_t3():
    ruta = Path("entrada") / "T3" / "resultados.json"
    if not ruta.is_file():
        return None
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except Exception:
        return None
    registros = _buscar_registros(datos, N_ESPERADO)
    if not registros:
        return None
    try:
        df = pd.DataFrame(registros)
    except Exception:
        return None
    if len(df) == N_ESPERADO and set(COLUMNAS_ESPERADAS).issubset(df.columns):
        return _preparar(df), "entrada/T3/resultados.json (tabla embebida)"
    return None


def _listas_numericas(obj, n, prefijo=""):
    """Recorre un JSON y devuelve [(ruta, lista_de_n_numeros)] encontrados."""
    halladas = []
    if isinstance(obj, list):
        if len(obj) == n and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj):
            halladas.append((prefijo or "<raiz>", [float(v) for v in obj]))
        else:
            for i, elem in enumerate(obj):
                halladas.extend(_listas_numericas(elem, n, f"{prefijo}[{i}]"))
    elif isinstance(obj, dict):
        if len(obj) == n:
            try:
                pares_ordenados = sorted(
                    (int(k), float(v)) for k, v in obj.items()
                    if not isinstance(v, bool)
                )
                if (len(pares_ordenados) == n
                        and [k for k, _ in pares_ordenados] == list(range(n))):
                    halladas.append((prefijo or "<raiz>",
                                     [v for _, v in pares_ordenados]))
                    return halladas
            except (TypeError, ValueError):
                pass
        for clave, valor in obj.items():
            nueva_ruta = f"{prefijo}.{clave}" if prefijo else clave
            halladas.extend(_listas_numericas(valor, n, nueva_ruta))
    return halladas


def _pares_conocidos_load_score():
    """Pares (índice_original, load_score) conocidos a partir de los resultados de T3."""
    ruta = Path("entrada") / "T3" / "resultados.json"
    if not ruta.is_file():
        return []
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(datos, dict):
        return []
    indices = datos.get("indices_requires_review")
    filas = datos.get("filas_requires_review")
    if not (isinstance(indices, list) and isinstance(filas, list)
            and len(indices) == len(filas) and len(indices) > 0):
        return []
    pares = []
    for i, fila in zip(indices, filas):
        if isinstance(fila, dict) and "load_score" in fila:
            try:
                pares.append((int(i), float(fila["load_score"])))
            except (TypeError, ValueError):
                continue
    return pares


def _coincide(valores, pares, tol=1e-9):
    """Comprueba valores[i] == valor_conocido para todos los pares (i, valor)."""
    if not pares:
        return False
    try:
        arr = np.asarray(valores, dtype="float64")
        idx = np.array([i for i, _ in pares], dtype=int)
        ref = np.array([v for _, v in pares], dtype="float64")
        if arr.ndim != 1 or len(arr) == 0 or idx.min() < 0 or idx.max() >= len(arr):
            return False
        return bool(np.allclose(arr[idx], ref, rtol=0.0, atol=tol))
    except Exception:
        return False


def _candidatos_load_score(df_base):
    """Fórmulas plausibles de load_score (T2) para calibrar con los valores de T3."""
    candidatos = []
    for col in ("gpu_utilization", "cpu_utilization", "memory_gb",
                "power_w", "temperature_c"):
        if col not in df_base.columns:
            continue
        x = df_base[col].to_numpy(dtype="float64")
        for ddof in (0, 1):
            sd = x.std(ddof=ddof)
            if sd > 0:
                candidatos.append((f"z-score de {col} (ddof={ddof})",
                                   (x - x.mean()) / sd))
    if {"gpu_utilization", "cpu_utilization", "memory_gb"}.issubset(df_base.columns):
        gpu = df_base["gpu_utilization"].to_numpy(dtype="float64")
        cpu = df_base["cpu_utilization"].to_numpy(dtype="float64")
        mem = df_base["memory_gb"].to_numpy(dtype="float64")
        for ddof in (0, 1):
            comp = gpu + cpu + mem
            sd = comp.std(ddof=ddof)
            if sd > 0:
                candidatos.append((f"z-score de (gpu+cpu+mem) (ddof={ddof})",
                                   (comp - comp.mean()) / sd))
            zs = [(a - a.mean()) / a.std(ddof=ddof) for a in (gpu, cpu, mem)]
            candidatos.append((f"media de z-scores de gpu, cpu y mem (ddof={ddof})",
                               (zs[0] + zs[1] + zs[2]) / 3.0))
    return candidatos


def _obtener_load_score(df_base):
    """Obtiene la columna load_score de T2: lista guardada en entrada/T2 o calibrada."""
    n = len(df_base)
    pares = _pares_conocidos_load_score()

    ruta_t2 = Path("entrada") / "T2" / "resultados.json"
    if ruta_t2.is_file():
        try:
            datos_t2 = json.loads(ruta_t2.read_text(encoding="utf-8"))
        except Exception:
            datos_t2 = None
        if datos_t2 is not None:
            listas = _listas_numericas(datos_t2, n)
            listas.sort(key=lambda t: (0 if "load" in t[0].lower() else 1,))
            for ruta_campo, lista in listas:
                if not pares or _coincide(lista, pares):
                    return (np.asarray(lista, dtype="float64"),
                            f"lista load_score de entrada/T2/resultados.json ({ruta_campo})")

    if pares:
        for descripcion, candidato in _candidatos_load_score(df_base):
            if _coincide(candidato, pares):
                return candidato, f"reconstruido y calibrado con T3: {descripcion}"

    x = df_base["gpu_utilization"].to_numpy(dtype="float64")
    return ((x - x.mean()) / x.std(ddof=0),
            "z-score de gpu_utilization (ddof=0), estimación sin T2")


def _reconstruir_desde_csv():
    """Reconstruye el DataFrame de T3: CSV original + load_score + requires_review."""
    df = pd.read_csv(RUTA_CSV)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    load_score, origen_ls = _obtener_load_score(df)
    df["load_score"] = load_score
    df["requires_review"] = ((df["load_score"] > UMBRAL_LOAD_SCORE)
                             | (df["temperature_c"] > UMBRAL_TEMPERATURA)).astype(bool)
    origen = f"reconstruido desde {RUTA_CSV} (load_score: {origen_ls})"
    return _preparar(df), origen


def cargar_dataframe_t3():
    hallado = _desde_archivos_t3() or _desde_json_t3()
    if hallado is not None:
        return hallado
    return _reconstruir_desde_csv()


# ---------------------------------------------------------------------------
# Verificación del round trip
# ---------------------------------------------------------------------------

def _valores_iguales(df_antes, df_despues):
    """Comparación estricta (valores y dtypes) entre el DataFrame antes y después."""
    a = df_antes.reset_index(drop=True)
    b = df_despues.reset_index(drop=True)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    try:
        pd.testing.assert_frame_equal(
            a, b,
            check_dtype=True,
            check_index_type=True,
            check_column_type=True,
            check_frame_type=True,
            check_names=True,
            check_exact=True,
        )
        return True
    except AssertionError:
        return False


def main():
    df_antes, origen = cargar_dataframe_t3()

    # 1) Guardar con PyArrow, SIN el índice
    ruta_salida = Path(RUTA_PARQUET)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    tabla = pa.Table.from_pandas(df_antes, preserve_index=False)
    try:
        pq.write_table(tabla, RUTA_PARQUET, version="2.6")
    except (ValueError, TypeError):
        pq.write_table(tabla, RUTA_PARQUET)

    # 2) Volver a leer
    df_despues = pq.read_table(RUTA_PARQUET).to_pandas()

    # 3) Verificaciones mínimas del round trip
    filas_antes, filas_despues = len(df_antes), len(df_despues)
    filas_ok = filas_antes == filas_despues

    cols_antes = list(df_antes.columns)
    cols_despues = list(df_despues.columns)
    columnas_ok = (len(cols_antes) == len(cols_despues)
                   and cols_antes == cols_despues)

    esquema_antes = pa.Schema.from_pandas(df_antes, preserve_index=False)
    esquema_despues = pq.read_schema(RUTA_PARQUET)
    esquema_ok = bool(esquema_antes.equals(esquema_despues))

    valores_ok = _valores_iguales(df_antes, df_despues)

    sin_indice = (len(esquema_despues) == len(COLUMNAS_ESPERADAS)
                  and not any(str(f.name).startswith("__index_level")
                              for f in esquema_despues))

    # 4) Mensaje esperado por el enunciado
    print(f"Round trip verificado: {RUTA_PARQUET}")
    print(f"Filas: {filas_ok}")
    print(f"Columnas: {columnas_ok}")
    print(f"Esquema: {esquema_ok}")
    print(f"Valores: {valores_ok}")

    # 5) resultados.json (contrato de la subtarea)
    resultados = {
        "subtarea": "T5_round_trip_parquet",
        "archivo_parquet": RUTA_PARQUET,
        "archivo_existe": bool(ruta_salida.is_file()),
        "origen_dataframe": origen,
        "metodo_guardado": "pyarrow (pa.Table.from_pandas preserve_index=False + pq.write_table)",
        "sin_indice": bool(sin_indice),
        "filas_antes": int(filas_antes),
        "filas_despues": int(filas_despues),
        "filas_iguales": bool(filas_ok),
        "columnas_antes": int(len(cols_antes)),
        "columnas_despues": int(len(cols_despues)),
        "nombres_columnas_antes": cols_antes,
        "nombres_columnas_despues": cols_despues,
        "columnas_iguales": bool(columnas_ok),
        "esquema_antes": str(esquema_antes),
        "esquema_despues": str(esquema_despues),
        "esquema_igual": esquema_ok,
        "valores_iguales": bool(valores_ok),
        "round_trip_verificado": bool(filas_ok and columnas_ok and esquema_ok and valores_ok),
        "mensaje_round_trip": {
            "archivo": RUTA_PARQUET,
            "Filas": bool(filas_ok),
            "Columnas": bool(columnas_ok),
            "Esquema": esquema_ok,
            "Valores": bool(valores_ok),
        },
        "n_requires_review": int(df_antes["requires_review"].sum()),
    }
    if len(df_antes) > 23:
        resultados["fila_23"] = {
            "server": str(df_antes["server"].iloc[23]),
            "timestamp": str(df_antes["timestamp"].iloc[23]),
            "load_score": float(df_antes["load_score"].iloc[23]),
            "temperature_c": float(df_antes["temperature_c"].iloc[23]),
            "requires_review": bool(df_antes["requires_review"].iloc[23]),
        }

    Path(RUTA_RESULTADOS).write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 6) Resumen breve
    print(f"Origen del DataFrame analizado: {origen}")
    print(f"Parquet escrito: {RUTA_PARQUET} "
          f"({ruta_salida.stat().st_size} bytes, "
          f"{filas_despues} filas x {len(cols_despues)} columnas, sin índice)")
    print(f"resultados.json escrito en: {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
