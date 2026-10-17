"""T1 - Representar las mediciones.

Carga data/server_measurements.csv (parse_dates en timestamp), implementa la
clase ServerMeasurements (values bidimensional, feature_names, validaciones
ValueError y __repr__) y la función build_measurements(df), que selecciona
gpu_utilization, cpu_utilization, memory_gb en ese orden, las convierte a
flotante y retorna el objeto. Imprime el repr y escribe resultados.json.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path("data/server_measurements.csv")
RESULTS_PATH = Path("resultados.json")
FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")


class ServerMeasurements:
    """Almacena una matriz NumPy bidimensional (values) y una tupla con los
    nombres de las características (feature_names)."""

    def __init__(self, values, feature_names):
        arr = np.asarray(values)

        # Validación 1: la matriz debe ser bidimensional.
        if arr.ndim != 2:
            raise ValueError(
                f"La matriz de valores debe ser bidimensional; se recibió ndim={arr.ndim}."
            )

        names = tuple(feature_names)

        # Validación 2: el número de columnas debe coincidir con la cantidad de nombres.
        if arr.shape[1] != len(names):
            raise ValueError(
                f"El número de columnas ({arr.shape[1]}) no coincide con la "
                f"cantidad de nombres de características ({len(names)})."
            )

        # Validación 3: los datos deben ser numéricos.
        try:
            arr = arr.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Los datos deben ser numéricos.") from exc

        self.values = arr
        self.feature_names = names

    def __repr__(self):
        return (
            f"ServerMeasurements(shape={self.values.shape}, "
            f"features={self.feature_names!r})"
        )


def build_measurements(df):
    """Selecciona las tres características, las convierte a una matriz NumPy
    de tipo flotante y retorna un objeto ServerMeasurements."""
    matrix = df[list(FEATURES)].to_numpy(dtype=float)
    return ServerMeasurements(matrix, FEATURES)


# Alias como método estático (misma funcionalidad, llamada opcional alternativa).
ServerMeasurements.build_measurements = staticmethod(build_measurements)


def md5_of_file(path):
    """MD5 hex de un archivo (para verificar que el CSV de entrada no cambia)."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raises_value_error(factory):
    """Retorna True si factory() lanza ValueError."""
    try:
        factory()
    except ValueError:
        return True
    return False


def main():
    csv_md5_before = md5_of_file(CSV_PATH)

    # Carga del CSV con parse_dates en timestamp (solo lectura, nunca se escribe).
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])

    # Construcción del objeto e impresión del repr.
    measurements = build_measurements(df)
    measurements_repr = repr(measurements)
    print(measurements_repr)

    # Pruebas de las tres validaciones (cada una debe lanzar ValueError).
    validation_tests = {
        "matriz_no_bidimensional_lanza_valueerror": raises_value_error(
            lambda: ServerMeasurements(np.array([1.0, 2.0, 3.0]), ("a",))
        ),
        "num_columnas_distinto_de_num_nombres_lanza_valueerror": raises_value_error(
            lambda: ServerMeasurements(np.zeros((3, 2)), ("a", "b", "c"))
        ),
        "datos_no_numericos_lanza_valueerror": raises_value_error(
            lambda: ServerMeasurements(np.array([["x", "y"], ["z", "w"]]), ("a", "b"))
        ),
    }

    csv_md5_after = md5_of_file(CSV_PATH)

    expected_repr = (
        "ServerMeasurements(shape=(300, 3), "
        "features=('gpu_utilization', 'cpu_utilization', 'memory_gb'))"
    )

    results = {
        "subtask": "T1",
        "repr": measurements_repr,
        "repr_matches_expected": measurements_repr == expected_repr,
        "shape": [int(n) for n in measurements.values.shape],
        "feature_names": list(measurements.feature_names),
        "values_dtype": str(measurements.values.dtype),
        "values_is_numpy_2d": isinstance(measurements.values, np.ndarray)
        and measurements.values.ndim == 2,
        "feature_names_is_tuple": isinstance(measurements.feature_names, tuple),
        "n_rows": int(measurements.values.shape[0]),
        "n_columns": int(measurements.values.shape[1]),
        "dataframe_shape": [int(n) for n in df.shape],
        "dataframe_columns": [str(c) for c in df.columns],
        "timestamp_dtype": str(df["timestamp"].dtype),
        "n_servers": int(df["server"].nunique()),
        "first_row_values": [float(v) for v in measurements.values[0]],
        "column_means": {
            name: float(measurements.values[:, i].mean())
            for i, name in enumerate(measurements.feature_names)
        },
        "validation_tests": validation_tests,
        "all_validations_raise_valueerror": all(validation_tests.values()),
        "csv_md5_before": csv_md5_before,
        "csv_md5_after": csv_md5_after,
        "csv_unmodified": csv_md5_before == csv_md5_after,
    }

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Resumen breve
    print("Validaciones (True = lanzan ValueError):", validation_tests)
    print("repr coincide con el esperado:", results["repr_matches_expected"])
    print("CSV de entrada sin modificar:", results["csv_unmodified"])
    print(f"Resultados escritos en {RESULTS_PATH}")


if __name__ == "__main__":
    main()
