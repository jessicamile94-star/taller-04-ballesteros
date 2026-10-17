"""T1 - Representar las mediciones de servidores.

Carga data/server_measurements.csv (timestamp con parse_dates), implementa la
clase ServerMeasurements (matriz NumPy 2D + feature_names como tupla, con
validaciones que lanzan ValueError) y la función build_measurements(df).
Imprime el repr del objeto construido y guarda los resultados en resultados.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")


class ServerMeasurements:
    """Almacena una matriz NumPy bidimensional (values) y una tupla con los
    nombres de las características (feature_names).

    Lanza ValueError si:
      - la matriz no es bidimensional,
      - el número de columnas no coincide con la cantidad de nombres,
      - los datos no son numéricos.
    """

    def __init__(self, values, feature_names):
        arr = np.asarray(values)

        # Validación 1: la matriz debe ser bidimensional
        if arr.ndim != 2:
            raise ValueError(
                f"values debe ser una matriz bidimensional; se recibió ndim={arr.ndim}."
            )

        # Validación 2: los datos deben ser numéricos
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(
                f"values debe contener datos numéricos; se recibió dtype={arr.dtype}."
            )

        # Validación 3: número de columnas == cantidad de nombres
        names = tuple(feature_names)
        if arr.shape[1] != len(names):
            raise ValueError(
                f"El número de columnas de values ({arr.shape[1]}) no coincide "
                f"con la cantidad de feature_names ({len(names)})."
            )

        self.values = arr
        self.feature_names = names

    def __repr__(self):
        return (
            f"ServerMeasurements(shape={self.values.shape}, "
            f"features={self.feature_names})"
        )


def build_measurements(df):
    """Selecciona gpu_utilization, cpu_utilization y memory_gb (en ese orden),
    las convierte a una matriz NumPy de tipo flotante y retorna un objeto
    ServerMeasurements."""
    values = df[list(FEATURES)].to_numpy(dtype=float)
    return ServerMeasurements(values, FEATURES)


def _lanza_valueerror(fn):
    """Retorna True si fn lanza ValueError, False en caso contrario."""
    try:
        fn()
        return False
    except ValueError:
        return True


def main():
    # 1) Cargar datos (timestamp con parse_dates)
    df = pd.read_csv("data/server_measurements.csv", parse_dates=["timestamp"])

    # 2) Construir el objeto de mediciones
    measurements = build_measurements(df)

    # 3) Imprimir el repr (salida esperada exacta)
    print(repr(measurements))

    # 4) Verificar que las validaciones lanzan ValueError en los casos inválidos
    validaciones = {
        "matriz_1d_lanza_valueerror": _lanza_valueerror(
            lambda: ServerMeasurements(np.array([1.0, 2.0, 3.0]), FEATURES)
        ),
        "matriz_3d_lanza_valueerror": _lanza_valueerror(
            lambda: ServerMeasurements(np.zeros((2, 2, 2)), ("a", "b"))
        ),
        "columnas_no_coinciden_lanza_valueerror": _lanza_valueerror(
            lambda: ServerMeasurements(np.zeros((5, 2)), FEATURES)
        ),
        "datos_no_numericos_lanza_valueerror": _lanza_valueerror(
            lambda: ServerMeasurements(
                np.array([["a", "b", "c"], ["d", "e", "f"]]), FEATURES
            )
        ),
    }

    # 5) Guardar resultados.json (contrato de la subtarea)
    values = measurements.values
    resumen = {}
    for i, name in enumerate(measurements.feature_names):
        col = values[:, i]
        resumen[name] = {
            "media": float(np.mean(col)),
            "desviacion_estandar": float(np.std(col)),
            "minimo": float(np.min(col)),
            "maximo": float(np.max(col)),
        }

    resultados = {
        "subtarea": "T1_representar_mediciones",
        "archivo_datos": "data/server_measurements.csv",
        "filas_csv": int(len(df)),
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "feature_names": list(measurements.feature_names),
        "dtype_values": str(values.dtype),
        "repr": repr(measurements),
        "validaciones_valueerror": validaciones,
        "resumen_estadistico": resumen,
    }
    Path("resultados.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 6) Resumen breve
    print("\nResumen T1:")
    print(f"  Filas leídas del CSV: {len(df)}")
    print(f"  Objeto construido: {measurements!r}")
    print(f"  dtype de values: {values.dtype}")
    print(f"  Validaciones ValueError correctas: {validaciones}")
    print("  Resultados guardados en resultados.json")


if __name__ == "__main__":
    main()
