# -*- coding: utf-8 -*-
"""
T3 — Parte 3: La curva de aprendizaje.

Sobre la MISMA división 70/30 estratificada de la Parte 1 (random_state=42),
entrena un Naive Bayes gaussiano y una regresión logística (atributos
estandarizados; el escalador se ajusta solo con los datos de entrenamiento de
cada punto) con el 5 %, 10 %, 25 %, 50 % y 100 % del conjunto de entrenamiento
(submuestras estratificadas con random_state=42). Cada modelo se evalúa
SIEMPRE sobre el mismo conjunto de prueba de la Parte 1.

Salidas (carpeta actual):
  - curva_aprendizaje_T3.png : exactitud vs. número de ejemplos de entrenamiento,
                               una curva por modelo (5 tamaños cada una)
  - tabla_T3.csv             : tabla de resultados por (fracción, modelo)
  - resultados.json          : contrato con todas las cifras de la subtarea
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
from sklearn.metrics import accuracy_score, f1_score

SEMILLA = 42
FRACCIONES = [0.05, 0.10, 0.25, 0.50, 1.00]
MODELO_NB = "Naive Bayes gaussiano"
MODELO_LR = "Regresión logística"
NOMBRE_FIGURA = "curva_aprendizaje_T3.png"
NOMBRE_TABLA = "tabla_T3.csv"
NOMBRE_RESULTADOS = "resultados.json"


def recrear_division_parte1():
    """División 70/30 estratificada de la Parte 1 (determinista, random_state=42)."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, train_size=0.7, stratify=y, random_state=SEMILLA
    )
    return X, y, X_train, X_test, y_train, y_test


def validar_division_contra_T1(y, y_train, y_test):
    """Best-effort: compara la división recreada con la guardada por T1 (si existe)."""
    ruta = Path("entrada") / "T1" / "division_T1.npz"
    if not ruta.exists():
        return "archivo_no_disponible"
    try:
        npz = np.load(ruta)
        claves = set(npz.files)
        if {"y_train", "y_test"} <= claves:
            ok = np.array_equal(np.asarray(npz["y_train"]), y_train) and np.array_equal(
                np.asarray(npz["y_test"]), y_test
            )
            return "coincide" if ok else "no_coincide"
        if {"train_idx", "test_idx"} <= claves:
            ok = np.array_equal(y[np.asarray(npz["train_idx"])], y_train) and np.array_equal(
                y[np.asarray(npz["test_idx"])], y_test
            )
            return "coincide" if ok else "no_coincide"
        return "claves_no_reconocidas"
    except Exception:
        return "error_al_leer"


def main():
    # ------------------------------------------------------------------ #
    # 1) Misma división de la Parte 1: 70/30 estratificada, random_state=42
    # ------------------------------------------------------------------ #
    X, y, X_train, X_test, y_train, y_test = recrear_division_parte1()
    n_train_total = int(len(y_train))
    n_prueba = int(len(y_test))
    validacion_t1 = validar_division_contra_T1(y, y_train, y_test)

    # ------------------------------------------------------------------ #
    # 2) Submuestras estratificadas (5/10/25/50/100 %) + entrenamiento y
    #    evaluación fija sobre el conjunto de prueba de la Parte 1
    # ------------------------------------------------------------------ #
    filas = []
    submuestras = []
    curvas = {MODELO_NB: {"n": [], "acc": []}, MODELO_LR: {"n": [], "acc": []}}

    for frac in FRACCIONES:
        if frac >= 1.0:
            # 100 %: entrenamiento completo (idéntico al ajuste de la Parte 2)
            X_sub, y_sub = X_train, y_train
        else:
            # Submuestra ESTRATIFICADA del entrenamiento, random_state=42
            X_sub, _, y_sub, _ = train_test_split(
                X_train,
                y_train,
                train_size=frac,
                stratify=y_train,
                random_state=SEMILLA,
            )

        n_sub = int(len(y_sub))
        submuestras.append(
            {
                "fraccion": frac,
                "n_ejemplos_entrenamiento": n_sub,
                "conteo_clases_submuestra": {
                    "malignant": int(np.sum(y_sub == 0)),
                    "benign": int(np.sum(y_sub == 1)),
                },
            }
        )

        # Escalador ajustado SOLO con la submuestra de entrenamiento; el
        # conjunto de prueba únicamente se transforma (no ajusta nada).
        scaler = StandardScaler().fit(X_sub)
        X_sub_std = scaler.transform(X_sub)
        X_test_std = scaler.transform(X_test)

        # Modelos exactamente como en la Parte 2
        nb = GaussianNB()
        nb.fit(X_sub_std, y_sub)
        y_pred_nb = nb.predict(X_test_std)

        lr = LogisticRegression(max_iter=5000, random_state=SEMILLA)
        lr.fit(X_sub_std, y_sub)
        y_pred_lr = lr.predict(X_test_std)

        for modelo, y_pred in ((MODELO_NB, y_pred_nb), (MODELO_LR, y_pred_lr)):
            acc = float(accuracy_score(y_test, y_pred))
            f1m = float(f1_score(y_test, y_pred, average="macro"))
            filas.append(
                {
                    "fraccion": frac,
                    "n_ejemplos_entrenamiento": n_sub,
                    "modelo": modelo,
                    "accuracy_prueba": acc,
                    "f1_macro_prueba": f1m,
                }
            )
            curvas[modelo]["n"].append(n_sub)
            curvas[modelo]["acc"].append(acc)

    df = pd.DataFrame(filas)

    # ------------------------------------------------------------------ #
    # 3) Figura: exactitud vs. número de ejemplos de entrenamiento
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    estilos = {
        MODELO_NB: {"marker": "o", "color": "tab:blue"},
        MODELO_LR: {"marker": "s", "color": "tab:orange"},
    }
    for modelo in (MODELO_NB, MODELO_LR):
        est = estilos[modelo]
        ns, accs = curvas[modelo]["n"], curvas[modelo]["acc"]
        ax.plot(ns, accs, marker=est["marker"], color=est["color"],
                linewidth=1.8, markersize=6, label=modelo)
        dy = 8 if modelo == MODELO_NB else -15
        for n_pto, a in zip(ns, accs):
            ax.annotate(
                f"{a:.3f}",
                (n_pto, a),
                textcoords="offset points",
                xytext=(0, dy),
                ha="center",
                fontsize=8,
                color=est["color"],
            )

    todos_acc = [a for m in (MODELO_NB, MODELO_LR) for a in curvas[m]["acc"]]
    ax.set_ylim(max(0.0, min(todos_acc) - 0.08), min(1.0, max(todos_acc) + 0.06) + 0.005)
    ax.set_xticks(curvas[MODELO_NB]["n"])
    ax.set_xlabel("Número de ejemplos de entrenamiento")
    ax.set_ylabel("Exactitud (accuracy) en el conjunto de prueba")
    ax.set_title(
        "Curva de aprendizaje — Naive Bayes gaussiano vs. regresión logística\n"
        "Breast Cancer Wisconsin · división 70/30 de la Parte 1 · evaluación fija en prueba"
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig(NOMBRE_FIGURA, dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 4) Tabla CSV y tabla Markdown
    # ------------------------------------------------------------------ #
    df.to_csv(NOMBRE_TABLA, index=False)
    cols_md = ["fraccion", "n_ejemplos_entrenamiento", "modelo", "accuracy_prueba"]
    lineas = ["| " + " | ".join(cols_md) + " |", "|" + "---|" * len(cols_md)]
    for _, r in df[cols_md].iterrows():
        lineas.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    tabla_md = "\n".join(lineas)

    # ------------------------------------------------------------------ #
    # 5) Verificación del punto 100 % contra la Parte 2 (si está disponible)
    # ------------------------------------------------------------------ #
    verificacion_t2 = None
    ruta_t2 = Path("entrada") / "T2" / "resultados.json"
    if ruta_t2.exists():
        try:
            t2 = json.loads(ruta_t2.read_text(encoding="utf-8"))
            acc_100 = {r["modelo"]: r["accuracy_prueba"] for r in filas if r["fraccion"] == 1.0}
            verificacion_t2 = {
                "accuracy_NB_parte2": t2["modelos"][MODELO_NB]["accuracy_prueba"],
                "accuracy_LR_parte2": t2["modelos"][MODELO_LR]["accuracy_prueba"],
                "accuracy_NB_T3_100pct": acc_100[MODELO_NB],
                "accuracy_LR_T3_100pct": acc_100[MODELO_LR],
                "coincide_NB": bool(
                    abs(acc_100[MODELO_NB] - t2["modelos"][MODELO_NB]["accuracy_prueba"]) < 1e-12
                ),
                "coincide_LR": bool(
                    abs(acc_100[MODELO_LR] - t2["modelos"][MODELO_LR]["accuracy_prueba"]) < 1e-12
                ),
            }
        except Exception:
            verificacion_t2 = "error_al_leer"

    # Observaciones factuales de la curva (insumo para la Parte 4)
    acc_nb = curvas[MODELO_NB]["acc"]
    acc_lr = curvas[MODELO_LR]["acc"]
    diferencias = [a - b for a, b in zip(acc_nb, acc_lr)]
    observaciones = {
        "mejor_modelo_en_5pct": MODELO_NB if diferencias[0] > 0 else MODELO_LR,
        "mejor_modelo_en_100pct": MODELO_NB if diferencias[-1] > 0 else MODELO_LR,
        "hay_cruce_entre_curvas": bool(
            any(d1 * d2 < 0 for d1, d2 in zip(diferencias[:-1], diferencias[1:]))
        ),
    }

    # ------------------------------------------------------------------ #
    # 6) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------ #
    resultados = {
        "subtarea": "T3",
        "descripcion": (
            "Curva de aprendizaje sobre la misma división 70/30 estratificada de la "
            "Parte 1: Naive Bayes gaussiano y regresión logística entrenados con el "
            "5 %, 10 %, 25 %, 50 % y 100 % del conjunto de entrenamiento (submuestras "
            "estratificadas, random_state=42) y evaluados siempre sobre el mismo "
            "conjunto de prueba de la Parte 1."
        ),
        "fuente_division": (
            "recreada de forma determinista con train_test_split(test_size=0.3, "
            "train_size=0.7, stratify=y, random_state=42): idéntica a la de la Parte 1"
        ),
        "validacion_division_contra_T1": validacion_t1,
        "parametros_division": {
            "test_size": 0.3,
            "train_size": 0.7,
            "estratificar_por_clase": True,
            "random_state": 42,
        },
        "parametros_submuestreo": {
            "fracciones": FRACCIONES,
            "estratificado": True,
            "random_state": 42,
            "metodo": (
                "train_test_split(train_size=fraccion, stratify=y_train, "
                "random_state=42); para el 100 % se usa el entrenamiento completo"
            ),
        },
        "n_entrenamiento_total": n_train_total,
        "n_prueba": n_prueba,
        "preprocesamiento": {
            "escalador": "StandardScaler",
            "atributos_estandarizados_para": [MODELO_NB, MODELO_LR],
            "ajustado_solo_con": "cada submuestra de entrenamiento (nunca con el conjunto de prueba)",
            "nota": (
                "En el punto 100 % el escalador coincide con el de la Parte 2 "
                "(ajustado con el entrenamiento completo). El conjunto de prueba "
                "solo se transforma y nunca se usa para ajustar nada."
            ),
        },
        "modelos": {
            MODELO_NB: "sklearn.naive_bayes.GaussianNB (parámetros por defecto), atributos estandarizados",
            MODELO_LR: "sklearn.linear_model.LogisticRegression(max_iter=5000, random_state=42), atributos estandarizados",
        },
        "submuestras": submuestras,
        "puntos": filas,
        "curvas": {
            m: {
                "n_ejemplos_entrenamiento": curvas[m]["n"],
                "accuracy_prueba": curvas[m]["acc"],
            }
            for m in (MODELO_NB, MODELO_LR)
        },
        "observaciones": observaciones,
        "tabla": {
            "columnas": [
                "fraccion",
                "n_ejemplos_entrenamiento",
                "modelo",
                "accuracy_prueba",
                "f1_macro_prueba",
            ],
            "filas": filas,
        },
        "tabla_markdown": tabla_md,
        "verificacion_punto_100_contra_T2": verificacion_t2,
        "figura": NOMBRE_FIGURA,
        "archivos_generados": [NOMBRE_FIGURA, NOMBRE_TABLA, NOMBRE_RESULTADOS],
    }
    Path(NOMBRE_RESULTADOS).write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------------ #
    # 7) Resumen
    # ------------------------------------------------------------------ #
    print("=" * 72)
    print("T3 — Curva de aprendizaje (Parte 3)")
    print("=" * 72)
    print(
        f"División Parte 1: {n_train_total} entrenamiento / {n_prueba} prueba "
        f"(validación contra T1: {validacion_t1})"
    )
    print("\nExactitud en el conjunto de prueba (fijo) por tamaño de entrenamiento:")
    print(df.to_string(index=False))
    print(f"\nFigura guardada: {NOMBRE_FIGURA}")
    print(f"Tabla guardada:  {NOMBRE_TABLA}")
    print(f"Resultados:      {NOMBRE_RESULTADOS}")
    if isinstance(verificacion_t2, dict):
        print(
            "Punto 100 % vs Parte 2 -> NB coincide: "
            f"{verificacion_t2['coincide_NB']} | LR coincide: {verificacion_t2['coincide_LR']}"
        )


if __name__ == "__main__":
    main()
