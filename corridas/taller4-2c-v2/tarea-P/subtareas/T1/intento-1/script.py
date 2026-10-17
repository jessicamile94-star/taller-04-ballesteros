import json
import re
import traceback
from pathlib import Path

# Patrones de PII: correo sintetico, cedula sintetica (6+ digitos aislados) y placeholders redactados
PATRONES_PII = {
    "correo": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "cedula": re.compile(r"(?<![\d.,])\d{6,}(?![\d.,])"),
    "placeholder": re.compile(r"\[(EMAIL|CEDULA)[^\]]*\]", re.IGNORECASE),
}


def contar_pii(texto):
    return {nombre: len(p.findall(texto)) for nombre, p in PATRONES_PII.items()}


def redactar(texto):
    for p in PATRONES_PII.values():
        texto = p.sub("[REDACTADO]", texto)
    return texto


salida = ""
error = ""
try:
    # Calculo en sandbox (no de memoria): sum() y division en tiempo de ejecucion
    importes = [120, 200, 80]
    total = sum(importes)
    promedio = total / len(importes)

    assert total == 400, f"total inesperado: {total}"
    assert abs(promedio - 400.0 / 3.0) < 1e-9, f"promedio inesperado: {promedio}"

    lineas = [
        "Importes (USD): [120, 200, 80]",
        f"Total (USD): {total}",
        f"Promedio (USD): {promedio}",
        f"Promedio (USD, 2 decimales): {promedio:.2f}",
    ]
    salida = "\n".join(lineas)

    # Codigo fuente de esta subtarea (para verificar que tampoco contiene PII)
    codigo = ""
    if "__file__" in globals():
        try:
            codigo = Path(__file__).read_text(encoding="utf-8")
        except OSError:
            codigo = ""

    resultados = {
        "subtarea": "T1_resumen_ventas",
        "metodo": "sum() y division calculados en tiempo de ejecucion en sandbox",
        "importes_usd": importes,
        "total_usd": total,
        "promedio_usd": promedio,
        "promedio_usd_2decimales": round(promedio, 2),
        "salida_consola": lineas,
    }
    resultados_str = json.dumps(resultados, indent=2, ensure_ascii=False)

    # Verificacion de PII sobre codigo, salidas generadas y contenido del JSON
    escaneo = {
        "codigo": contar_pii(codigo),
        "salida": contar_pii(salida),
        "resultados_json": contar_pii(resultados_str),
    }
    libre_pii = all(v == 0 for d in escaneo.values() for v in d.values())
    assert libre_pii, "PII detectado en codigo, salidas o resultados"

    resultados["verificacion_pii"] = {
        "libre_de_pii": True,
        "codigo_escaneado": bool(codigo),
        "patrones_revisados": list(PATRONES_PII.keys()),
        "coincidencias_por_texto": escaneo,
        "trazas": "ejecucion sin excepciones; cualquier traza se sanitiza con redactar() antes de imprimirse",
    }
    Path("resultados.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )
except Exception:
    error = redactar(traceback.format_exc())

if error:
    print(error)
else:
    print(salida)
    print("Verificacion PII: sin correo ni cedula sinteticos en codigo, salidas ni resultados.json")
    print("resultados.json escrito")
