"""T1: Carga del dataset Breast Cancer Wisconsin (Diagnostic) y división 70/30 estratificada.

Reporta número de ejemplos, número de atributos y conteo de casos por clase.
Crea la única división entrenamiento/prueba (70%/30%, estratificada por clase,
random_state=42) que se usará en toda la tarea, y la guarda para las subtareas
siguientes.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.30


def main():
    # ------------------------------------------------------------------
    # 1) Carga del conjunto de datos
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X, y = data.data, data.target
    n_ejemplos, n_atributos = X.shape
    nombres_clases = [str(c) for c in data.target_names]

    conteo_total = {
        nombres_clases[i]: int(np.sum(y == i)) for i in range(len(nombres_clases))
    }

    # ------------------------------------------------------------------
    # 2) División 70/30 estratificada por clase, random_state=42.
    #    Se hace sobre los índices para poder guardarla y reutilizarla
    #    exactamente igual en las subtareas siguientes.
    # ------------------------------------------------------------------
    indices = np.arange(n_ejemplos)
    idx_train, idx_test = train_test_split(
        indices,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    conteo_train = {
        nombres_clases[i]: int(np.sum(y_train == i)) for i in range(len(nombres_clases))
    }
    conteo_test = {
        nombres_clases[i]: int(np.sum(y_test == i)) for i in range(len(nombres_clases))
    }

    # ------------------------------------------------------------------
    # 3) Contrato de resultados: resultados.json
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T1_carga_y_division",
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "n_ejemplos_total": int(n_ejemplos),
        "n_atributos": int(n_atributos),
        "nombres_atributos": [str(f) for f in data.feature_names],
        "nombres_clases": nombres_clases,
        "mapa_etiqueta": {nombres_clases[i]: int(i) for i in range(len(nombres_clases))},
        "conteo_clases_total": conteo_total,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "estratificado": True,
        "n_entrenamiento": int(X_train.shape[0]),
        "n_prueba": int(X_test.shape[0]),
        "fraccion_entrenamiento": float(X_train.shape[0] / n_ejemplos),
        "fraccion_prueba": float(X_test.shape[0] / n_ejemplos),
        "conteo_clases_entrenamiento": conteo_train,
        "conteo_clases_prueba": conteo_test,
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 4) Guardar la división para las subtareas siguientes
    # ------------------------------------------------------------------
    np.savez_compressed(
        "division_70_30.npz",
        X_entrenamiento=X_train,
        y_entrenamiento=y_train,
        X_prueba=X_test,
        y_prueba=y_test,
        indices_entrenamiento=idx_train,
        indices_prueba=idx_test,
    )
    orden = np.concatenate([idx_train, idx_test])
    pd.DataFrame(
        {
            "indice_original": orden,
            "conjunto": ["entrenamiento"] * len(idx_train) + ["prueba"] * len(idx_test),
            "clase": y[orden],
        }
    ).to_csv("indices_division_70_30.csv", index=False)

    # ------------------------------------------------------------------
    # 5) Figura: distribución de clases en total/entrenamiento/prueba
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_pos = np.arange(len(nombres_clases))
    ancho = 0.27
    ax.bar(x_pos - ancho, [conteo_total[c] for c in nombres_clases], ancho, label="Total")
    ax.bar(x_pos, [conteo_train[c] for c in nombres_clases], ancho, label="Entrenamiento (70%)")
    ax.bar(x_pos + ancho, [conteo_test[c] for c in nombres_clases], ancho, label="Prueba (30%)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(nombres_clases)
    ax.set_ylabel("Número de casos")
    ax.set_title("Breast Cancer Wisconsin: distribución de clases\n(70/30 estratificada, random_state=42)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("t1_distribucion_clases.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6) Resumen breve
    # ------------------------------------------------------------------
    print("=== T1: Carga y división 70/30 estratificada ===")
    print(f"Dataset: Breast Cancer Wisconsin (Diagnostic)")
    print(f"Ejemplos totales: {n_ejemplos}")
    print(f"Atributos: {n_atributos}")
    print(f"Conteo por clase (total): {conteo_total}")
    print(f"Entrenamiento: {X_train.shape[0]} ejemplos -> {conteo_train}")
    print(f"Prueba: {X_test.shape[0]} ejemplos -> {conteo_test}")
    print(
        "Archivos generados: resultados.json, division_70_30.npz, "
        "indices_division_70_30.csv, t1_distribucion_clases.png"
    )


if __name__ == "__main__":
    main()
