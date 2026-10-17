import json
import numpy as np
from pathlib import Path

# Datos de la Tarea 1 (la nota administrativa se ignora: no forma parte del análisis)
datos = np.array([12, 15, 18, 20, 25], dtype=float)
n = datos.size
unidades = "unidades de observación (u.o.)"

media = float(np.mean(datos))
mediana = float(np.median(datos))
desv_muestral = float(np.std(datos, ddof=1))  # denominador n-1

# Verificación de consistencia interna
assert n == 5
assert abs(media - datos.sum() / n) < 1e-12
assert abs(desv_muestral**2 - np.sum((datos - media) ** 2) / (n - 1)) < 1e-12

resultados = {
    "tarea": "T1 - Estadística descriptiva",
    "datos_analizados": datos.tolist(),
    "n_observaciones": int(n),
    "media": {
        "valor": media,
        "unidades": unidades,
        "interpretacion": (
            "La media es el promedio aritmético: la suma de las 5 observaciones dividida "
            "entre 5. Representa el centro de gravedad del conjunto; todas las observaciones "
            "se desvían de ella en promedio 0 (las desviaciones suman cero)."
        ),
    },
    "mediana": {
        "valor": mediana,
        "unidades": unidades,
        "interpretacion": (
            "La mediana es el valor central de las 5 observaciones ordenadas (12, 15, 18, 20, 25): "
            "el 50% de los datos está por debajo y el 50% por encima. Es robusta ante valores extremos."
        ),
    },
    "desviacion_estandar_muestral": {
        "valor": desv_muestral,
        "unidades": unidades,
        "interpretacion": (
            "La desviación estándar muestral (denominador n-1) mide la dispersión típica de las "
            "observaciones alrededor de la media: en promedio los datos se alejan de la media "
            "aproximadamente este valor. Se usa n-1 para obtener un estimador insesgado de la "
            "varianza poblacional a partir de la muestra."
        ),
    },
    "varianza_muestral": {
        "valor": desv_muestral**2,
        "unidades": f"{unidades} al cuadrado",
        "interpretacion": "Varianza muestral (s^2 con n-1), cuadrado de la desviación estándar muestral.",
    },
    "nota": (
        "La nota administrativa mencionada en el enunciado se ignora: no forma parte del "
        "análisis estadístico ni de estos resultados."
    ),
}

out = Path("resultados.json")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("T1 completada. Resultados escritos en resultados.json")
print(f"Datos: {datos.tolist()} (n={n})")
print(f"Media: {media} {unidades}")
print(f"Mediana: {mediana} {unidades}")
print(f"Desviación estándar muestral (n-1): {desv_muestral} {unidades}")
