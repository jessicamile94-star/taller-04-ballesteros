# T1: Carga de Breast Cancer Wisconsin (Diagnostic), reporte descriptivo
# y creación de la división única 70/30 estratificada (random_state=42).

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.30


def to_py(obj):
    """Convierte tipos de numpy a tipos nativos para JSON."""
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main():
    # ------------------------------------------------------------------
    # 1) Carga del conjunto de datos
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=list(data.feature_names))
    y = pd.Series(data.target.astype(int), name="target")
    target_names = [str(t) for t in data.target_names]
    nombre_clase = {int(i): target_names[int(i)] for i in range(len(target_names))}

    n_ejemplos = int(X.shape[0])
    n_atributos = int(X.shape[1])

    conteo_clases = {
        nombre_clase[int(c)]: int((y == c).sum()) for c in sorted(y.unique())
    }
    proporcion_clases = {k: v / n_ejemplos for k, v in conteo_clases.items()}

    # ------------------------------------------------------------------
    # 2) División única 70/30 estratificada por clase, random_state=42
    #    (esta división no se vuelve a hacer en ninguna otra subtarea)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    n_train = int(X_train.shape[0])
    n_test = int(X_test.shape[0])

    conteo_train = {
        nombre_clase[int(c)]: int((y_train == c).sum()) for c in sorted(y_train.unique())
    }
    conteo_test = {
        nombre_clase[int(c)]: int((y_test == c).sum()) for c in sorted(y_test.unique())
    }
    prop_train = {k: v / n_train for k, v in conteo_train.items()}
    prop_test = {k: v / n_test for k, v in conteo_test.items()}

    # Verificación de la estratificación: las proporciones por clase en
    # entrenamiento y prueba deben coincidir con las globales.
    estrat_ok = all(
        abs(prop_train[k] - proporcion_clases[k]) < 0.01
        and abs(prop_test[k] - proporcion_clases[k]) < 0.01
        for k in conteo_clases
    )

    # ------------------------------------------------------------------
    # 3) Guardar la división para las subtareas siguientes
    # ------------------------------------------------------------------
    idx_train = list(X_train.index)
    idx_test = list(X_test.index)

    train_df = X_train.copy()
    train_df.insert(0, "indice_original", idx_train)
    train_df["target"] = y_train.values
    train_df["target_name"] = [nombre_clase[int(t)] for t in y_train.values]
    train_df = train_df.reset_index(drop=True)
    train_df.to_csv("T1_entrenamiento.csv", index=False)

    test_df = X_test.copy()
    test_df.insert(0, "indice_original", idx_test)
    test_df["target"] = y_test.values
    test_df["target_name"] = [nombre_clase[int(t)] for t in y_test.values]
    test_df = test_df.reset_index(drop=True)
    test_df.to_csv("T1_prueba.csv", index=False)

    # ------------------------------------------------------------------
    # 4) Figura: distribución de clases global / entrenamiento / prueba
    # ------------------------------------------------------------------
    etiquetas = list(conteo_clases.keys())
    x_pos = np.arange(len(etiquetas))
    ancho = 0.27

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x_pos - ancho, [conteo_clases[k] for k in etiquetas], ancho,
           label=f"Total (n={n_ejemplos})", color="#4C72B0")
    ax.bar(x_pos, [conteo_train[k] for k in etiquetas], ancho,
           label=f"Entrenamiento (n={n_train})", color="#DD8452")
    ax.bar(x_pos + ancho, [conteo_test[k] for k in etiquetas], ancho,
           label=f"Prueba (n={n_test})", color="#55A868")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(etiquetas)
    ax.set_ylabel("Número de casos")
    ax.set_title("Breast Cancer Wisconsin (Diagnostic): distribución de clases\n"
                 "División 70/30 estratificada (random_state=42)")
    ax.legend()
    for bars in ax.containers:
        ax.bar_label(bars, fontsize=8)
    fig.tight_layout()
    plt.savefig("T1_distribucion_clases.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T1",
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "fuente": "sklearn.datasets.load_breast_cancer",
        "total_ejemplos": n_ejemplos,
        "numero_atributos": n_atributos,
        "nombres_atributos": list(data.feature_names),
        "nombres_clases": target_names,
        "conteo_clases_total": conteo_clases,
        "proporcion_clases_total": proporcion_clases,
        "division": {
            "test_size": TEST_SIZE,
            "train_size": 1.0 - TEST_SIZE,
            "estratificado_por_clase": True,
            "random_state": RANDOM_STATE,
            "n_entrenamiento": n_train,
            "n_prueba": n_test,
            "porcentaje_entrenamiento": n_train / n_ejemplos,
            "porcentaje_prueba": n_test / n_ejemplos,
            "conteo_clases_entrenamiento": conteo_train,
            "conteo_clases_prueba": conteo_test,
            "proporcion_clases_entrenamiento": prop_train,
            "proporcion_clases_prueba": prop_test,
            "estratificacion_verificada": bool(estrat_ok),
        },
        "archivos_generados": {
            "entrenamiento": "T1_entrenamiento.csv",
            "prueba": "T1_prueba.csv",
            "figura": "T1_distribucion_clases.png",
        },
        "nota": ("Esta división 70/30 estratificada con random_state=42 es la única "
                 "usada en toda la tarea; el conjunto de prueba no se usa para ajustar "
                 "nada (ni escaladores ni selección de atributos)."),
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(to_py(resultados), f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 6) Resumen en consola
    # ------------------------------------------------------------------
    print("=== T1: Breast Cancer Wisconsin (Diagnostic) ===")
    print(f"Total de ejemplos: {n_ejemplos}")
    print(f"Número de atributos: {n_atributos}")
    print(f"Conteo por clase (total): {conteo_clases}")
    print(f"División 70/30 estratificada (random_state={RANDOM_STATE}):")
    print(f"  Entrenamiento: {n_train} ejemplos ({n_train / n_ejemplos:.1%}) -> {conteo_train}")
    print(f"  Prueba:        {n_test} ejemplos ({n_test / n_ejemplos:.1%}) -> {conteo_test}")
    print(f"Estratificación verificada: {estrat_ok}")
    print("Archivos escritos: resultados.json, T1_entrenamiento.csv, "
          "T1_prueba.csv, T1_distribucion_clases.png")


if __name__ == "__main__":
    main()
