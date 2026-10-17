import json

import numpy as np
import pandas as pd

EXPECTED_REPR = (
    "ServerMeasurements(shape=(300, 3), "
    "features=('gpu_utilization', 'cpu_utilization', 'memory_gb'))"
)
FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")


class ServerMeasurements:
    """Almacena una matriz NumPy 2D y los nombres de las características."""

    def __init__(self, values, feature_names):
        arr = np.asarray(values)
        if arr.ndim != 2:
            raise ValueError("values debe ser una matriz NumPy bidimensional")
        names = tuple(feature_names)
        if arr.shape[1] != len(names):
            raise ValueError("el número de columnas no coincide con la cantidad de nombres")
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError("los datos no son numéricos")
        self.values = arr
        self.feature_names = names

    def __repr__(self):
        return (
            f"ServerMeasurements(shape={self.values.shape}, "
            f"features={self.feature_names})"
        )


def build_measurements(df):
    """Selecciona las tres características, las convierte a flotante y retorna el objeto."""
    values = df[list(FEATURES)].to_numpy(dtype=float)
    return ServerMeasurements(values, FEATURES)


def main():
    df = pd.read_csv("data/server_measurements.csv", parse_dates=["timestamp"])

    # Validación de los tres casos inválidos: cada uno debe lanzar ValueError
    casos_invalidos = {
        "no_bidimensional": (np.array([1.0, 2.0, 3.0]), ("a", "b", "c")),
        "columnas_vs_nombres": (np.zeros((5, 4)), ("a", "b", "c")),
        "no_numerico": (np.array([["x", "y", "z"]] * 5), ("a", "b", "c")),
    }
    validaciones = {}
    for nombre, (vals, names) in casos_invalidos.items():
        try:
            ServerMeasurements(vals, names)
            validaciones[nombre] = False
        except ValueError:
            validaciones[nombre] = True

    # Caso válido no debe lanzar error
    try:
        obj_ok = ServerMeasurements(np.zeros((2, 3)), ("a", "b", "c"))
        caso_valido_ok = obj_ok.values.shape == (2, 3) and isinstance(obj_ok.feature_names, tuple)
    except ValueError:
        caso_valido_ok = False

    # Ejecución principal
    measurements = build_measurements(df)
    repr_obtenido = repr(measurements)

    results = {
        "repr_obtenido": repr_obtenido,
        "repr_esperado": EXPECTED_REPR,
        "repr_coincide": repr_obtenido == EXPECTED_REPR,
        "forma": list(measurements.values.shape),
        "feature_names": list(measurements.feature_names),
        "dtype_values": str(measurements.values.dtype),
        "n_filas_csv": int(len(df)),
        "timestamp_es_datetime": bool(pd.api.types.is_datetime64_any_dtype(df["timestamp"])),
        "validaciones_lanzan_valueerror": validaciones,
        "caso_valido_sin_error": caso_valido_ok,
        "medias_por_feature": {
            f: float(measurements.values[:, i].mean()) for i, f in enumerate(FEATURES)
        },
    }
    with open("resultados.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print("repr obtenido:", repr_obtenido)
    print("coincide con el esperado:", results["repr_coincide"])
    print("validaciones ValueError (no 2D / columnas / no numérico):", validaciones)
    print("forma:", results["forma"], "| dtype:", results["dtype_values"])


if __name__ == "__main__":
    main()
