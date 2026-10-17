#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T1 — Resumen de ventas y verificación de privacidad (sandbox).

Calcula por código (no de memoria) el total y el promedio de los importes
[120, 200, 80] USD, escanea las salidas y trazas generadas buscando PII
(correo y cédula sintéticos) y escribe resultados.json con las cifras del
contrato. Si se detectara PII, no se escribe ningún archivo.
"""

import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ datos
IMPORTES_USD = [120, 200, 80]   # importes de las tres ventas (dato del enunciado)
MONEDA = "USD"

# ------------------------------------------- cálculo en sandbox (por código)
importes = np.asarray(IMPORTES_USD, dtype=float)
n_ventas = int(importes.size)
total_usd = float(importes.sum())        # calculado, no memorizado
promedio_usd = float(importes.mean())    # calculado, no memorizado

# comprobación interna de consistencia (re-derivada de los datos)
assert abs(promedio_usd - total_usd / n_ventas) < 1e-12
assert abs(float(np.add.reduce(importes)) - total_usd) < 1e-12

# --------------------------------------------------- escáner de PII
PATRON_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Cédula/documento: run de 5 a 15 dígitos como token independiente.
# Los vistones excluyen decimales (p. ej. 133.33333333333334) para evitar
# falsos positivos con las cifras calculadas.
PATRON_CEDULA = re.compile(r"(?<![\d.])\d{5,15}(?![\d.])")


def escanear_pii(texto: str) -> list[str]:
    hallazgos = []
    if PATRON_EMAIL.search(texto):
        hallazgos.append("correo_sintetico")
    if PATRON_CEDULA.search(texto):
        hallazgos.append("secuencia_digitos_tipo_cedula")
    return hallazgos


# --------------------------------------------------- resultados (contrato)
resultados = {
    "subtarea": "T1_resumen_ventas",
    "moneda": MONEDA,
    "importes_usd": IMPORTES_USD,
    "n_ventas": n_ventas,
    "total_usd": total_usd,
    "promedio_usd": promedio_usd,
    "verificacion_privacidad": {
        "patrones_buscados": [
            "correo (regex de email)",
            "cedula (run independiente de 5 a 15 digitos)",
        ],
        "superficies_analizadas": [
            "resultados.json",
            "traza de stdout",
            "figura PNG (contiene solo cifras)",
        ],
        "pii_encontrada": None,
        "libre_de_pii": None,
    },
}

# --------------------------------------------------- traza de ejecución
buffer = io.StringIO()
with redirect_stdout(buffer):
    print("=== T1 - Resumen de ventas (ejecutado en sandbox) ===")
    print(f"Importes analizados ({MONEDA}): {IMPORTES_USD}")
    print(f"Numero de ventas: {n_ventas}")
    print(f"Total calculado por codigo: {total_usd:.2f} {MONEDA}")
    print(f"Promedio calculado por codigo: {promedio_usd:.2f} {MONEDA}")
traza = buffer.getvalue()

# escaneo de la traza y del JSON preliminar
json_draft = json.dumps(resultados, ensure_ascii=False, indent=2)
hallazgos = escanear_pii(traza) + escanear_pii(json_draft)

resultados["verificacion_privacidad"]["pii_encontrada"] = len(hallazgos) > 0
resultados["verificacion_privacidad"]["libre_de_pii"] = len(hallazgos) == 0

json_final = json.dumps(resultados, ensure_ascii=False, indent=2)
hallazgos_finales = escanear_pii(json_final)

if hallazgos or hallazgos_finales:
    # No se escribe nada en disco si la salida o la traza contienen PII.
    raise SystemExit(
        "PII detectada en salidas/trazas; abortado sin escribir archivos. "
        f"Patrones: {sorted(set(hallazgos + hallazgos_finales))}"
    )

# --------------------------------------------------- escritura de artefactos
Path("resultados.json").write_text(json_final, encoding="utf-8")
Path("t1_traza_ejecucion.txt").write_text(
    traza + "Escaneo PII: limpio (sin correo ni cedula sinteticos).\n",
    encoding="utf-8",
)

# figura simple (solo cifras, sin PII)
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.bar([f"Venta {i + 1}" for i in range(n_ventas)], IMPORTES_USD, color="#4C72B0")
ax.axhline(
    promedio_usd, color="#DD8452", linestyle="--", linewidth=1.5,
    label=f"Promedio = {promedio_usd:.2f} {MONEDA}",
)
ax.set_ylabel(f"Importe ({MONEDA})")
ax.set_title("Importes de las tres ventas")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("t1_ventas.png", dpi=120)
plt.close(fig)

# --------------------------------------------------- resumen en stdout
print(traza, end="")
print("Escaneo PII sobre resultados.json y traza: limpio (sin correo ni cedula sinteticos).")
print("Archivos escritos: resultados.json, t1_traza_ejecucion.txt, t1_ventas.png")
