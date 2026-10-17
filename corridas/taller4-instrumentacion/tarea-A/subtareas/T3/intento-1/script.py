#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 — Curva de aprendizaje: Naive Bayes gaussiano (generativo) vs. regresión
logística (discriminativo) sobre la división 70/30 de la Parte 1.

- Submuestras estratificadas del 5 %, 10 %, 25 %, 50 % y 100 % del conjunto
  de entrenamiento (random_state=42).
- Cada modelo se evalúa SIEMPRE sobre el mismo conjunto de prueba de la
  Parte 1 (171 ejemplos), que nunca se usa para ajustar nada.
- Figura: T3_curva_aprendizaje.png (exactitud vs. nº de ejemplos, 2 curvas).
- Cifras de cada punto: resultados.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------------------------ constantes --------------------------------
SEED = 42
FRACCIONES = [0.05, 0.10, 0.25, 0.50, 1.00]
FIGURA_PNG = "T3_curva_aprendizaje.png"
TABLA_CSV = "T3_curva_aprendizaje.csv"

FEATURE_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
]
CLASS_NAMES = ["malignant", "benign"]  # codificación de sklearn: 0=malignant, 1=benign

# Exactitud de la Parte 2 (T2) con el 100 % del entrenamiento (verificación)
T2_ACC_NB = 0.935672514619883
T2_ACC_LR = 0.9883040935672515


# ------------------------------- funciones --------------------------------
def cargar_split(ruta_csv):
    """Lee un CSV de la división de la Parte 1 y devuelve (X, y), y en {0,1}."""
    df = pd.read_csv(ruta_csv)
    candidatos = ["target", "target_name", "clase", "class", "diagnosis", "label", "y"]
    col_obj = next((c for c in candidatos if c in df.columns), df.columns[-1])
    serie = df[col_obj]
    if serie.dtype == object:
        y = serie.map({n: i for i, n in enumerate(CLASS_NAMES)})
        if y.isna().any():
            raise ValueError(f"No se pudo codificar la columna objetivo '{col_obj}'.")
        y = y.astype(int).to_numpy()
    else:
        y = serie.astype(int).to_numpy()
    cols_atrib = [c for c in FEATURE_NAMES if c in df.columns]
    if len(cols_atrib) == len(FEATURE_NAMES):
        X = df[cols_atrib].to_numpy(dtype=float)
    else:
        excluidas = {"target", "target_name", "id", "index", "Unnamed: 0", col_obj}
        atrib = [c for c in df.columns
                 if c not in excluidas and pd.api.types.is_numeric_dtype(df[c])]
        if not atrib:
            raise ValueError(f"No se identificaron columnas de atributos en {ruta_csv}.")
        X = df[atrib].to_numpy(dtype=float)
    return X, y


def obtener_datos():
    """Reutiliza la división de la Parte 1 (T1); si no está disponible, la
    regenera de forma idéntica (70/30 estratificada, random_state=42)."""
    ruta_tr = Path("entrada/T1/T1_entrenamiento.csv")
    ruta_te = Path("entrada/T1/T1_prueba.csv")
    if ruta_tr.exists() and ruta_te.exists():
        try:
            X_tr, y_tr = cargar_split(ruta_tr)
            X_te, y_te = cargar_split(ruta_te)
            if X_tr.shape[1] == X_te.shape[1] and len(y_tr) > 0 and len(y_te) > 0:
                fuente = ("división de la Parte 1 reutilizada desde "
                          "entrada/T1/T1_entrenamiento.csv y entrada/T1/T1_prueba.csv")
                return X_tr, y_tr, X_te, y_te, fuente
        except Exception:
            pass
    data = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=0.3, stratify=data.target,
        random_state=SEED)
    fuente = ("división regenerada de forma idéntica a la Parte 1 "
              "(load_breast_cancer, 70/30 estratificada, random_state=42)")
    return X_tr, y_tr, X_te, y_te, fuente


def main():
    X_tr, y_tr, X_te, y_te, fuente = obtener_datos()
    n_tr_total, n_te = int(len(y_tr)), int(len(y_te))

    puntos = []
    for frac in FRACCIONES:
        if frac >= 1.0:
            X_sub, y_sub = X_tr, y_tr
        else:
            # Submuestra estratificada del entrenamiento (nunca de la prueba)
            X_sub, _, y_sub, _ = train_test_split(
                X_tr, y_tr, train_size=frac, stratify=y_tr, random_state=SEED)
        n_sub = int(len(y_sub))

        # Escalador ajustado SOLO con la submuestra de entrenamiento de este
        # punto; el conjunto de prueba jamás participa en ningún ajuste.
        scaler = StandardScaler().fit(X_sub)
        X_sub_s = scaler.transform(X_sub)
        X_te_s = scaler.transform(X_te)

        nb = GaussianNB().fit(X_sub_s, y_sub)
        acc_nb = float(accuracy_score(y_te, nb.predict(X_te_s)))

        lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_sub_s, y_sub)
        acc_lr = float(accuracy_score(y_te, lr.predict(X_te_s)))

        vals, conts = np.unique(y_sub, return_counts=True)
        conteo = {CLASS_NAMES[int(v)]: int(c) for v, c in zip(vals, conts)}

        puntos.append({
            "fraccion_entrenamiento": frac,
            "n_ejemplos_entrenamiento": n_sub,
            "conteo_clases_submuestra": conteo,
            "accuracy_naive_bayes": acc_nb,
            "accuracy_regresion_logistica": acc_lr,
        })
        print(f"  {frac:>4.0%} del entrenamiento -> n={n_sub:>3} | "
              f"NB={acc_nb:.4f} | RL={acc_lr:.4f}")

    ns = [p["n_ejemplos_entrenamiento"] for p in puntos]
    accs_nb = [p["accuracy_naive_bayes"] for p in puntos]
    accs_lr = [p["accuracy_regresion_logistica"] for p in puntos]

    # ------------------------------ figura ---------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(ns, accs_nb, marker="o", color="tab:blue",
            label="Naive Bayes gaussiano (generativo)")
    ax.plot(ns, accs_lr, marker="s", color="tab:red",
            label="Regresión logística (discriminativo)")
    for x, a in zip(ns, accs_nb):
        ax.annotate(f"{a:.3f}", (x, a), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8, color="tab:blue")
    for x, a in zip(ns, accs_lr):
        ax.annotate(f"{a:.3f}", (x, a), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="tab:red")
    ax.set_xticks(ns)
    y_min = min(accs_nb + accs_lr)
    ax.set_ylim(max(0.5, y_min - 0.07), 1.015)
    ax.set_xlabel("Número de ejemplos de entrenamiento")
    ax.set_ylabel("Exactitud (accuracy) en el conjunto de prueba")
    ax.set_title("Curva de aprendizaje: generativo vs. discriminativo\n"
                 "Breast Cancer (diagnóstico) — prueba fija de la Parte 1 "
                 f"({n_te} ejemplos)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURA_PNG, dpi=120)
    plt.close(fig)

    # --------------------------- tabla CSV extra ---------------------------
    pd.DataFrame({
        "fraccion_entrenamiento": FRACCIONES,
        "n_ejemplos_entrenamiento": ns,
        "accuracy_naive_bayes": accs_nb,
        "accuracy_regresion_logistica": accs_lr,
    }).to_csv(TABLA_CSV, index=False)

    # --------------------------- resultados.json ---------------------------
    resultados = {
        "subtarea": "T3",
        "descripcion": ("Curva de aprendizaje: Naive Bayes gaussiano y regresión "
                        "logística entrenados con 5 %, 10 %, 25 %, 50 % y 100 % del "
                        "conjunto de entrenamiento de la Parte 1 (submuestras "
                        "estratificadas, random_state=42) y evaluados siempre sobre "
                        "el mismo conjunto de prueba de la Parte 1."),
        "datos": {
            "fuente": fuente,
            "division_reutilizada_de_parte_1": True,
            "random_state": SEED,
            "n_entrenamiento_total": n_tr_total,
            "n_prueba": n_te,
            "n_atributos": int(X_tr.shape[1]),
            "conjunto_prueba_fijo_en_todos_los_puntos": True,
        },
        "protocolo": {
            "fracciones": FRACCIONES,
            "submuestreo": ("train_test_split(train_size=fraccion, stratify=y_train, "
                            "random_state=42); para el 100 % se usa el entrenamiento "
                            "completo"),
            "submuestreo_estratificado": True,
            "random_state_submuestreo": SEED,
            "escalador": ("StandardScaler ajustado en cada punto SOLO con la "
                          "submuestra de entrenamiento correspondiente; el conjunto "
                          "de prueba solo se transforma"),
            "modelos": {
                "naive_bayes_gaussiano": "GaussianNB (por defecto)",
                "regresion_logistica": "LogisticRegression(max_iter=1000, random_state=42)",
            },
            "metrica": "accuracy sobre el conjunto de prueba de la Parte 1",
        },
        "curva_aprendizaje": {
            "fracciones": FRACCIONES,
            "n_ejemplos_entrenamiento": ns,
            "accuracy_naive_bayes": accs_nb,
            "accuracy_regresion_logistica": accs_lr,
        },
        "tabla_puntos": puntos,
        "verificacion_100pct_vs_T2": {
            "accuracy_naive_bayes_T2": T2_ACC_NB,
            "accuracy_naive_bayes_T3_100pct": accs_nb[-1],
            "coincide_naive_bayes": bool(abs(accs_nb[-1] - T2_ACC_NB) < 1e-12),
            "accuracy_regresion_logistica_T2": T2_ACC_LR,
            "accuracy_regresion_logistica_T3_100pct": accs_lr[-1],
            "coincide_regresion_logistica": bool(abs(accs_lr[-1] - T2_ACC_LR) < 1e-12),
        },
        "archivos_generados": {
            "figura": FIGURA_PNG,
            "tabla_csv": TABLA_CSV,
            "resultados": "resultados.json",
        },
        "nota": ("El conjunto de prueba no se usa para ajustar nada: en cada punto el "
                 "escalador se ajusta únicamente con la submuestra de entrenamiento y "
                 "la prueba solo se transforma y evalúa. Con el 100 % del entrenamiento "
                 "los resultados reproducen los de la Parte 2 (T2)."),
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------- resumen --------------------------------
    print("\n=== T3 — Curva de aprendizaje (evaluación sobre la prueba fija de la Parte 1) ===")
    print(f"{'fracción':>9} | {'n entrenamiento':>15} | {'NB':>7} | {'RL':>7}")
    for p in puntos:
        print(f"{p['fraccion_entrenamiento']:>8.0%} | "
              f"{p['n_ejemplos_entrenamiento']:>15} | "
              f"{p['accuracy_naive_bayes']:>7.4f} | "
              f"{p['accuracy_regresion_logistica']:>7.4f}")
    print(f"\nFigura guardada en:  {FIGURA_PNG}")
    print(f"Tabla guardada en:   {TABLA_CSV}")
    print("Cifras completas en: resultados.json")


if __name__ == "__main__":
    main()
