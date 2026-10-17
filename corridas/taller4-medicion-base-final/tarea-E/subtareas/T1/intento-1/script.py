#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUBTAREA T1 — Estadística descriptiva
Calcula media, mediana y desviación estándar muestral (ddof=1) de las
observaciones [12, 15, 18, 20, 25], con unidades de observación e
interpretación breve de cada medida. La nota administrativa se ignora:
no forma parte del análisis estadístico.

Salida: resultados.json (contrato de la subtarea).
"""

import json
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------
# 1. Datos: observaciones de la Tarea 1 (sin ninguna nota ajena)
# ------------------------------------------------------------------
observaciones = np.array([12, 15, 18, 20, 25], dtype=float)
n = observaciones.size
UNIDADES = "unidades de observación (u)"

# ------------------------------------------------------------------
# 2. Cálculos por ejecución de código en sandbox
# ------------------------------------------------------------------
media = float(np.mean(observaciones))                       # media aritmética
mediana = float(np.median(observaciones))                   # mediana (percentil 50)
desv_muestral = float(np.std(observaciones, ddof=1))        # ddof=1 => denominador (n-1)

# Verificación manual de la desviación muestral (transparencia/auditoría)
suma_cuadrados = float(np.sum((observaciones - media) ** 2))  # 98.0
var_muestral = suma_cuadrados / (n - 1)                       # 24.5
assert abs(var_muestral - desv_muestral ** 2) < 1e-12, "Inconsistencia en la varianza muestral"

# ------------------------------------------------------------------
# 3. Resultados con unidades e interpretación (contrato resultados.json)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T1 — Estadística descriptiva",
    "observaciones": [12, 15, 18, 20, 25],
    "n": int(n),
    "unidades_de_observacion": UNIDADES,
    "media": {
        "valor": media,
        "unidades": UNIDADES,
        "interpretacion": (
            "La media aritmética es 18.0 unidades de observación: es el centro de gravedad "
            "del conjunto, el valor que resultaría si las 5 observaciones se repartieran por "
            "igual. Es sensible a valores extremos."
        ),
    },
    "mediana": {
        "valor": mediana,
        "unidades": UNIDADES,
        "interpretacion": (
            "La mediana es 18.0 unidades de observación: con las observaciones ordenadas "
            "(12, 15, 18, 20, 25) es el valor central, dejando 2 observaciones por debajo y "
            "2 por encima. Es robusta: no cambia ante valores extremos."
        ),
    },
    "desviacion_estandar_muestral": {
        "valor": desv_muestral,
        "unidades": UNIDADES,
        "interpretacion": (
            "La desviación estándar muestral (denominador n-1 = 4) es aproximadamente "
            "4.9497 unidades de observación: mide la dispersión típica de las observaciones "
            "respecto de la media (18.0). Usar n-1 corrige el sesgo al estimar la dispersión "
            "poblacional a partir de una muestra pequeña."
        ),
    },
    "detalles_calculo": {
        "suma": float(np.sum(observaciones)),
        "suma_cuadrados_de_desviaciones": suma_cuadrados,
        "varianza_muestral": var_muestral,
        "denominador": int(n - 1),
        "formula_desviacion": "sqrt( sum((x_i - media)^2) / (n - 1) )",
    },
    "nota": (
        "La nota administrativa mencionada en el enunciado se ignora: no forma parte del "
        "análisis estadístico y no se incluye en estos resultados."
    ),
}

# ------------------------------------------------------------------
# 4. Escritura del contrato
# ------------------------------------------------------------------
out = Path("resultados.json")
out.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")

# ------------------------------------------------------------------
# 5. Resumen breve
# ------------------------------------------------------------------
print("=== T1 — Estadística descriptiva ===")
print(f"Observaciones: {observaciones.tolist()}  (n = {n})")
print(f"Media                          = {media:.6f} {UNIDADES}")
print(f"Mediana                        = {mediana:.6f} {UNIDADES}")
print(f"Desviación estándar muestral   = {desv_muestral:.6f} {UNIDADES}  (ddof=1)")
print(f"Varianza muestral              = {var_muestral:.6f}")
print(f"Resultados escritos en: {out.resolve()}")
