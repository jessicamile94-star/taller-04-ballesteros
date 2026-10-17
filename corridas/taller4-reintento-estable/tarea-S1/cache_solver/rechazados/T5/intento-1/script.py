```python
# -*- coding: utf-8 -*-
"""
T5 - Guardar y verificar el resultado.

1) Obtiene el DataFrame analizado de la T3 (con load_score y requires_review).
2) Lo guarda en output/server_analysis.parquet mediante PyArrow, SIN el índice.
3) Vuelve a leer el archivo y verifica el round trip:
   - filas antes == filas después
   - columnas antes == columnas después
   - esquema antes == esquema después
   - valores iguales
4) Imprime el mensaje de round trip con las cuatro verificaciones y escribe resultados.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RUTA_PARQUET = Path("output/server_analysis.parquet")
MENSAJE_RUTA = "output/server_analysis.parquet"
COLUMNAS_T3 = [
    "server", "timestamp", "gpu_utilization", "cpu_utilization",
    "memory_gb", "power_w", "temperature_c", "load_score", "requires_review",
]


# --------------------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------------------
def cargar_json(ruta):
    """Lee un JSON si existe y es válido; en caso contrario devuelve None."""
    try:
        with open(ruta, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return None


def normalizar_nombres_columnas(df):
    """Nombres de columnas en minúsculas, sin espacios y con guiones bajos."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def buscar_listas_numericas(obj, n, ruta=""):
    """Busca recursivamente listas de exactamente n números en una estructura JSON."""
    hallazgos = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            nueva_ruta = f"{ruta}.{k}" if ruta else str(k)
            hallazgos.extend(buscar_listas_numericas(v, n, nueva_ruta))
    elif isinstance(obj, list):
        if len(obj) == n and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj
        ):
            hallazgos.append((ruta, [float(x) for x in obj]))
        else:
            for i, v in enumerate(obj):
                hallazgos.extend(buscar_listas_numericas(v, n, f"{ruta}[{i}]"))
    return hallazgos


def obtener_referencia_t3(res_t3):
    """Devuelve (índices, load_score esperado) según las filas de revisión de T3."""
    if not isinstance(res_t3, dict):
        return None
    idxs = res_t3.get("indices_requires_review")
    filas = res_t3.get("filas_requires_review")
    if not (isinstance(idxs, list) and isinstance(filas, list)):
        return None
    if len(idxs) == 0 or len(idxs) != len(filas):
        return None
    try:
        idxs_arr = np.asarray(idxs, dtype=int)
        esperados = np.asarray([float(f["load_score"]) for f in filas], dtype=float)
    except Exception:
        return None
    return idxs_arr, esperados


def validar_load_score(valores, referencia):
    """True si el vector reproduce el load_score reportado por T3 (o si no hay referencia)."""
    if referencia is None:
        return True
    idxs, esperados = referencia
    valores = np.asarray(valores, dtype=float)
    if valores.ndim != 1 or len(valores) <= int(idxs.max()):
        return False
    return bool(np.allclose(valores[idxs], esperados, rtol=1e-9, atol=1e-6))


def generar_candidatos_load_score(df):
    """Fórmulas plausibles del load_score de T2, para validar contra los valores de T3."""
    def z0(serie):
        serie = pd.to_numeric(serie, errors="coerce").astype(float)
        return (serie - serie.mean()) / serie.std(ddof=0)

    def z1(serie):
        serie = pd.to_numeric(serie, errors="coerce").astype(float)
        return (serie - serie.mean()) / serie.std(ddof=1)

    gpu = df["gpu_utilization"]
    cpu = df["cpu_utilization"]
    mem = df["memory_gb"]
    pot = df["power_w"]

    candidatos = []
    for nombre, serie in (("gpu_utilization", gpu), ("cpu_utilization", cpu),
                          ("memory_gb", mem), ("power_w", pot)):
        candidatos.append((f"zscore_ddof0({nombre})", np.asarray(z0(serie), dtype=float)))
        candidatos.append((f"zscore_ddof1({nombre})", np.asarray(z1(serie), dtype=float)))

    combos = (
        ("gpu+cpu", gpu + cpu),
        ("(gpu+cpu)/2", (gpu + cpu) / 2.0),
        ("gpu+cpu+mem", gpu + cpu + mem),
        ("(gpu+cpu+mem)/3", (gpu + cpu + mem) / 3.0),
        ("0.5gpu+0.3cpu+0.2mem", 0.5 * gpu + 0.3 * cpu + 0.2 * mem),
        ("0.4gpu+0.35cpu+0.25mem", 0.4 * gpu + 0.35 * cpu + 0.25 * mem),
        ("0.6gpu+0.25cpu+0.15mem", 0.6 * gpu + 0.25 * cpu + 0.15 * mem),
        ("0.7gpu+0.3cpu", 0.7 * gpu + 0.3 * cpu),
    )
    for nombre, serie in combos:
        candidatos.append((f"zscore_ddof0({nombre})", np.asarray(z0(serie), dtype=float)))
        candidatos.append((f"zscore_ddof1({nombre})", np.asarray(z1(serie), dtype=float)))

    zg, zc,
