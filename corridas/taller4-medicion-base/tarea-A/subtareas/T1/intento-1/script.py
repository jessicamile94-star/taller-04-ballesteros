"""T1 — Carga del dataset Breast Cancer Wisconsin (Diagnostic) y creación de la
división entrenamiento/prueba 70%/30% estratificada por clase (random_state=42).

Contrato de salida: resultados.json (cifras pedidas) + division_T1.npz (la
división queda disponible para las subtareas siguientes) + figura PNG de
verificación de la estratificación.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

SEMILLA = 42
TEST_SIZE = 0.30
TOLERANCIA_ESTRAT = 0.01  # diferencia máxima admisible entre proporciones de clase


def conteos_por_clase(y: np.ndarray, nombres: list) -> dict:
    valores, conteos = np.unique(y, return_counts=True)
    return {nombres[int(v)]: int(c) for v, c in zip(valores, conteos)}


def proporciones(conteo: dict, total: int) -> dict:
    return {k: v / total for k, v in conteo.items()}


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Carga del conjunto de datos
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X, y = data.data, data.target
    nombres_clases = [str(s) for s in data.target_names]        # ['malignant', 'benign']
    nombres_atributos = [str(s) for s in data.feature_names]    # 30 atributos

    n_ejemplos = int(X.shape[0])
    n_atributos = int(X.shape[1])
    conteo_total = conteos_por_clase(y, nombres_clases)

    # ------------------------------------------------------------------
    # 2) División 70/30 estratificada por clase, random_state=42
    #    (única división usada en toda la tarea; la prueba no se toca)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEMILLA
    )

    conteo_train = conteos_por_clase(y_train, nombres_clases)
    conteo_test = conteos_por_clase(y_test, nombres_clases)

    prop_total = proporciones(conteo_total, n_ejemplos)
    prop_train = proporciones(conteo_train, int(len(y_train)))
    prop_test = proporciones(conteo_test, int(len(y_test)))

    diferencias = [
        max(abs(prop_train[k] - prop_total[k]), abs(prop_test[k] - prop_total[k]))
        for k in prop_total
    ]
    max_dif = float(max(diferencias))
    estrat_verificada = bool(max_dif < TOLERANCIA_ESTRAT)

    resultados = {
        "subtarea": "T1",
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "n_ejemplos_total": n_ejemplos,
        "n_atributos": n_atributos,
        "nombres_clases": nombres_clases,
        "codificacion_clases": {nombres_clases[0]: 0, nombres_clases[1]: 1},
        "conteo_clases_total": conteo_total,
        "proporcion_clases_total": prop_total,
        "tamano_entrenamiento": int(len(y_train)),
        "tamano_prueba": int(len(y_test)),
        "fraccion_entrenamiento": float(len(y_train) / n_ejemplos),
        "fraccion_prueba": float(len(y_test) / n_ejemplos),
        "conteo_clases_entrenamiento": conteo_train,
        "conteo_clases_prueba": conteo_test,
        "proporcion_clases_entrenamiento": prop_train,
        "proporcion_clases_prueba": prop_test,
        "max_diferencia_proporciones_vs_total": max_dif,
        "estratificacion_verificada": estrat_verificada,
        "parametros_division": {
            "test_size": TEST_SIZE,
            "train_size": 1.0 - TEST_SIZE,
            "estratificar_por_clase": True,
            "random_state": SEMILLA,
        },
        "archivo_division": "division_T1.npz",
    }

    # ------------------------------------------------------------------
    # 3) Escritura del contrato resultados.json (carpeta actual y copia
    #    en entrada/T1/ para las subtareas siguientes)
    # ------------------------------------------------------------------
    texto_json = json.dumps(resultados, indent=2, ensure_ascii=False)
    Path("resultados.json").write_text(texto_json, encoding="utf-8")

    carpeta_salida = Path("entrada") / "T1"
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    (carpeta_salida / "resultados.json").write_text(texto_json, encoding="utf-8")

    # ------------------------------------------------------------------
    # 4) Guardado de la división para las subtareas siguientes
    # ------------------------------------------------------------------
    np.savez(
        "division_T1.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=np.array(nombres_atributos, dtype=object),
        target_names=np.array(nombres_clases, dtype=object),
    )
    np.savez(
        carpeta_salida / "division_T1.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=np.array(nombres_atributos, dtype=object),
        target_names=np.array(nombres_clases, dtype=object),
    )

    # ------------------------------------------------------------------
    # 5) Figura de verificación de la estratificación
    # ------------------------------------------------------------------
    conjuntos = [
        (f"Total ({n_ejemplos})", conteo_total),
        (f"Entrenamiento 70% ({len(y_train)})", conteo_train),
        (f"Prueba 30% ({len(y_test)})", conteo_test),
    ]
    fig, ejes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (titulo, conteo) in zip(ejes, conjuntos):
        valores = [conteo.get(k, 0) for k in nombres_clases]
        ax.bar(nombres_clases, valores, color=["#c44e52", "#4c72b0"])
        ax.set_title(titulo)
        ax.set_ylabel("Número de casos")
        ax.set_ylim(0, max(valores) * 1.15)
        for i, v in enumerate(valores):
            ax.text(i, v, str(v), ha="center", va="bottom")
    fig.suptitle(
        "T1: distribución de clases — división 70/30 estratificada (random_state=42)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig("T1_distribucion_clases.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6) Resumen
    # ------------------------------------------------------------------
    print("=== T1: Breast Cancer Wisconsin (Diagnostic) ===")
    print(f"Ejemplos totales : {n_ejemplos}")
    print(f"Atributos        : {n_atributos}")
    print(f"Clases           : {conteo_total}")
    print(f"Entrenamiento 70%: {len(y_train)} -> {conteo_train}")
    print(f"Prueba 30%       : {len(y_test)} -> {conteo_test}")
    print(f"Estratificación verificada: {estrat_verificada} "
          f"(máx. diferencia de proporciones vs total = {max_dif:.4f})")
    print("Archivos escritos: resultados.json, division_T1.npz, "
          "T1_distribucion_clases.png, entrada/T1/ (copias)")


if __name__ == "__main__":
    main()
