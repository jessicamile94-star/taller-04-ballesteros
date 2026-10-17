"""T1: Carga de Breast Cancer Wisconsin (Diagnostic) y división 70/30 estratificada.

Reporta ejemplos, atributos y conteos por clase; crea la única división
entrenamiento/prueba (70%/30%, estratificada, random_state=42) que se usará
en toda la tarea y la guarda en CSV para las subtareas siguientes.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def main():
    # ------------------------------------------------------------------
    # 1) Carga del conjunto de datos
    # ------------------------------------------------------------------
    datos = load_breast_cancer()
    X = pd.DataFrame(datos.data, columns=list(datos.feature_names))
    y = pd.Series(datos.target, name="target")

    n_ejemplos, n_atributos = X.shape
    nombres_clases = list(datos.target_names)  # ['malignant', 'benign']

    conteo_total = np.bincount(y.to_numpy())
    conteo_total_dict = {
        nombres_clases[c]: int(conteo_total[c]) for c in range(len(nombres_clases))
    }

    # ------------------------------------------------------------------
    # 2) División 70/30 estratificada por clase, random_state=42
    #    (única división de la tarea; la prueba no se usa para ajustar nada)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    conteo_train = np.bincount(y_train.to_numpy())
    conteo_test = np.bincount(y_test.to_numpy())
    conteo_train_dict = {
        nombres_clases[c]: int(conteo_train[c]) for c in range(len(nombres_clases))
    }
    conteo_test_dict = {
        nombres_clases[c]: int(conteo_test[c]) for c in range(len(nombres_clases))
    }

    prop_total = {k: v / n_ejemplos for k, v in conteo_total_dict.items()}
    prop_train = {k: v / len(y_train) for k, v in conteo_train_dict.items()}
    prop_test = {k: v / len(y_test) for k, v in conteo_test_dict.items()}

    # Verificación de la estratificación: proporciones por clase casi idénticas
    max_dif = max(abs(prop_train[k] - prop_total[k]) for k in nombres_clases)
    max_dif = max(max_dif, max(abs(prop_test[k] - prop_total[k]) for k in nombres_clases))
    estratificada = bool(max_dif < 0.01)

    # ------------------------------------------------------------------
    # 3) Guardar la división para las subtareas siguientes (rutas relativas)
    # ------------------------------------------------------------------
    def guardar(df_x: pd.DataFrame, serie_y: pd.Series, nombre: str) -> str:
        df = df_x.copy()
        df["target"] = serie_y.to_numpy()
        df["indice_original"] = df_x.index.to_numpy()
        df.to_csv(nombre, index=False)
        return nombre

    archivos = []
    archivos.append(guardar(X, y, "t1_datos_completos.csv"))
    archivos.append(guardar(X_train, y_train, "t1_entrenamiento.csv"))
    archivos.append(guardar(X_test, y_test, "t1_prueba.csv"))

    # ------------------------------------------------------------------
    # 4) Figura de verificación de la estratificación
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x_pos = np.arange(len(nombres_clases))
    ancho = 0.25
    ax.bar(x_pos - ancho, [conteo_total_dict[c] for c in nombres_clases], ancho,
           label=f"Total (n={n_ejemplos})")
    ax.bar(x_pos, [conteo_train_dict[c] for c in nombres_clases], ancho,
           label=f"Entrenamiento 70% (n={len(y_train)})")
    ax.bar(x_pos + ancho, [conteo_test_dict[c] for c in nombres_clases], ancho,
           label=f"Prueba 30% (n={len(y_test)})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(nombres_clases)
    ax.set_ylabel("Número de casos")
    ax.set_title("Breast Cancer Wisconsin: conteos por clase\n(70/30 estratificado, random_state=42)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("t1_distribucion_clases.png", dpi=120)
    plt.close(fig)
    archivos.append("t1_distribucion_clases.png")

    # ------------------------------------------------------------------
    # 5) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T1",
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "total_ejemplos": int(n_ejemplos),
        "numero_atributos": int(n_atributos),
        "nombres_atributos": list(datos.feature_names),
        "nombres_clases": nombres_clases,
        "codificacion_clases": {nombres_clases[0]: 0, nombres_clases[1]: 1},
        "conteo_clases_total": conteo_total_dict,
        "conteo_clases_entrenamiento": conteo_train_dict,
        "conteo_clases_prueba": conteo_test_dict,
        "proporcion_clases_total": prop_total,
        "proporcion_clases_entrenamiento": prop_train,
        "proporcion_clases_prueba": prop_test,
        "tamano_entrenamiento": int(len(y_train)),
        "tamano_prueba": int(len(y_test)),
        "porcentaje_entrenamiento": 100.0 * len(y_train) / n_ejemplos,
        "porcentaje_prueba": 100.0 * len(y_test) / n_ejemplos,
        "estratificacion": {
            "estratificada_por_clase": True,
            "max_diferencia_proporciones": float(max_dif),
            "verificada": estratificada,
        },
        "parametros_division": {
            "test_size": 0.30,
            "random_state": 42,
            "stratify": "target",
        },
        "archivos_generados": archivos,
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 6) Resumen breve
    # ------------------------------------------------------------------
    print("=== T1: Breast Cancer Wisconsin (Diagnostic) ===")
    print(f"Ejemplos totales: {n_ejemplos} | Atributos: {n_atributos}")
    print(f"Conteo por clase (total): {conteo_total_dict}")
    print(f"Entrenamiento (70%): {len(y_train)} ejemplos -> {conteo_train_dict}")
    print(f"Prueba (30%): {len(y_test)} ejemplos -> {conteo_test_dict}")
    print(f"Proporciones entrenamiento: {prop_train}")
    print(f"Proporciones prueba:        {prop_test}")
    print(f"Estratificación verificada: {estratificada} "
          f"(máx. diferencia de proporciones = {max_dif:.6f})")
    print("División guardada en t1_entrenamiento.csv / t1_prueba.csv "
          "(random_state=42, única división de la tarea).")
    print("Archivos generados:", ", ".join(archivos))


if __name__ == "__main__":
    main()
