# -*- coding: utf-8 -*-
"""
T3 — Curva de aprendizaje: Naive Bayes gaussiano vs. regresión logística.

Sobre la MISMA división 70/30 estratificada (random_state=42) de la Parte 1,
entrena los dos modelos de la Parte 2 con 5%, 10%, 25%, 50% y 100% del conjunto
de entrenamiento (submuestras estratificadas, random_state=42), evalúa cada uno
SIEMPRE sobre el mismo conjunto de prueba, dibuja la exactitud frente al número
de ejemplos de entrenamiento (una curva por modelo) y guarda la figura como PNG.
Todas las cifras se escriben en resultados.json.
"""

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FRACCIONES = [0.05, 0.10, 0.25, 0.50, 1.00]
RUTA_FIGURA = "t3_curva_aprendizaje.png"
RUTA_TABLA = "t3_curva_aprendizaje.csv"
RUTA_RESULTADOS = "resultados.json"

# Recuento esperado de la división de la Parte 1 (según entrada/T1/resultados.json)
ESPERADO = {
    "n_train": 398, "n_test": 171,
    "train": {0: 148, 1: 250},   # malignant=0, benign=1
    "test": {0: 64, 1: 107},
}


def _leer_csv_division(ruta):
    """Lee un CSV de la Parte 1 y devuelve (X, y) con malignant=0, benign=1."""
    df = pd.read_csv(ruta)
    candidatos = ["target", "diagnosis", "clase", "class", "label", "y", "objetivo"]
    col = next((c for c in candidatos if c in df.columns), None)
    if col is None:
        return None
    y_raw = df[col]
    if y_raw.dtype == object:
        mapa = {"malignant": 0, "benign": 1}
        if not set(y_raw.unique()).issubset(mapa):
            return None
        y = y_raw.map(mapa).to_numpy()
    else:
        y = y_raw.to_numpy()
    X = df.drop(columns=[col]).select_dtypes(include=[np.number])
    if X.shape[1] != 30:
        return None
    return X.to_numpy(dtype=float), y.astype(int)


def obtener_division():
    """Devuelve (X_train, X_test, y_train, y_test, origen) de la Parte 1."""
    # Intento 1: usar los archivos generados por T1
    try:
        r1 = _leer_csv_division("entrada/T1/t1_entrenamiento.csv")
        r2 = _leer_csv_division("entrada/T1/t1_prueba.csv")
        if r1 is not None and r2 is not None:
            X_tr, y_tr = r1
            X_te, y_te = r2
            ok = (
                X_tr.shape[0] == ESPERADO["n_train"]
                and X_te.shape[0] == ESPERADO["n_test"]
                and int((y_tr == 0).sum()) == ESPERADO["train"][0]
                and int((y_tr == 1).sum()) == ESPERADO["train"][1]
                and int((y_te == 0).sum()) == ESPERADO["test"][0]
                and int((y_te == 1).sum()) == ESPERADO["test"][1]
            )
            if ok:
                return (X_tr, X_te, y_tr, y_te,
                        "entrada/T1 (t1_entrenamiento.csv / t1_prueba.csv)")
    except Exception:
        pass
    # Intento 2: reproducción determinista de la división de la Parte 1
    datos = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(
        datos.data, datos.target, test_size=0.3,
        random_state=RANDOM_STATE, stratify=datos.target)
    return (X_tr, X_te, y_tr, y_te,
            "reproducida: load_breast_cancer + train_test_split"
            "(test_size=0.3, random_state=42, stratify=y)")


def main():
    X_tr, X_te, y_tr, y_te, origen = obtener_division()
    n_train_total, n_test = int(X_tr.shape[0]), int(X_te.shape[0])

    tamanos, acc_nb, acc_lr, f1_nb, f1_lr = [], [], [], [], []
    detalle, filas, puntos = [], [], []

    for frac in FRACCIONES:
        # Submuestra estratificada del entrenamiento (misma para los dos modelos)
        if frac >= 1.0:
            Xs, ys = X_tr, y_tr
        else:
            Xs, _, ys, _ = train_test_split(
                X_tr, y_tr, train_size=frac,
                stratify=y_tr, random_state=RANDOM_STATE)
        ys = np.asarray(ys)
        n = int(ys.size)

        # Escalador ajustado SOLO con la submuestra de entrenamiento de este punto
        # (el conjunto de prueba nunca participa en el ajuste)
        scaler = StandardScaler().fit(Xs)
        Xs_std = scaler.transform(Xs)
        Xte_std = scaler.transform(X_te)

        nb = GaussianNB().fit(Xs_std, ys)
        lr = LogisticRegression(max_iter=5000,
                                random_state=RANDOM_STATE).fit(Xs_std, ys)

        p_nb = nb.predict(Xte_std)
        p_lr = lr.predict(Xte_std)
        a_nb = float(accuracy_score(y_te, p_nb))
        a_lr = float(accuracy_score(y_te, p_lr))
        f_nb = float(f1_score(y_te, p_nb, average="macro"))
        f_lr = float(f1_score(y_te, p_lr, average="macro"))

        tamanos.append(n)
        acc_nb.append(a_nb)
        acc_lr.append(a_lr)
        f1_nb.append(f_nb)
        f1_lr.append(f_lr)
        detalle.append({
            "porcentaje": float(frac * 100),
            "n_entrenamiento": n,
            "n_malignant": int((ys == 0).sum()),
            "n_benign": int((ys == 1).sum()),
        })
        puntos.append({
            "porcentaje": float(frac * 100),
            "n_entrenamiento": n,
            "accuracy_naive_bayes_gaussiano": a_nb,
            "accuracy_regresion_logistica": a_lr,
            "f1_macro_naive_bayes_gaussiano": f_nb,
            "f1_macro_regresion_logistica": f_lr,
        })
        for modelo, a, f in (("Naive Bayes gaussiano", a_nb, f_nb),
                             ("Regresión logística", a_lr, f_lr)):
            filas.append({
                "porcentaje_entrenamiento": float(frac * 100),
                "n_entrenamiento": n,
                "modelo": modelo,
                "accuracy_prueba": a,
                "f1_macro_prueba": f,
            })

    # Verificación de consistencia con la Parte 2 (el punto 100 % debe coincidir)
    try:
        with open("entrada/T2/resultados.json", "r", encoding="utf-8") as fh:
            t2 = json.load(fh)
        verif = {
            "accuracy_100pct_naive_bayes_gaussiano": acc_nb[-1],
            "accuracy_T2_naive_bayes_gaussiano": t2.get("accuracy_naive_bayes_gaussiano"),
            "accuracy_100pct_regresion_logistica": acc_lr[-1],
            "accuracy_T2_regresion_logistica": t2.get("accuracy_regresion_logistica"),
            "coincide_con_parte_2": bool(
                np.isclose(acc_nb[-1], float(t2["accuracy_naive_bayes_gaussiano"]))
                and np.isclose(acc_lr[-1], float(t2["accuracy_regresion_logistica"]))
            ),
        }
    except Exception:
        verif = {"nota": "entrada/T2/resultados.json no disponible; verificación omitida."}

    # Tabla CSV de apoyo
    pd.DataFrame(filas).to_csv(RUTA_TABLA, index=False)

    # Figura: curva de aprendizaje (una curva por modelo, prueba fija)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(tamanos, acc_nb, "o-", color="tab:blue",
            label="Naive Bayes gaussiano (generativo)")
    ax.plot(tamanos, acc_lr, "s-", color="tab:orange",
            label="Regresión logística (discriminativo)")
    for x, y in zip(tamanos, acc_nb):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8, color="tab:blue")
    for x, y in zip(tamanos, acc_lr):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color="tab:orange")
    etiquetas = [f"{n}\n({d['porcentaje']:.0f}%)"
                 for n, d in zip(tamanos, detalle)]
    ax.set_xticks(tamanos)
    ax.set_xticklabels(etiquetas)
    ax.set_xlabel("Número de ejemplos de entrenamiento")
    ax.set_ylabel("Exactitud (accuracy) en el conjunto de prueba")
    ax.set_title("Curva de aprendizaje: Naive Bayes vs. regresión logística\n"
                 f"Breast Cancer Wisconsin — división 70/30 (prueba fija, n={n_test})")
    ymin = max(0.0, min(min(acc_nb), min(acc_lr)) - 0.06)
    ax.set_ylim(ymin, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RUTA_FIGURA, dpi=120)
    plt.close(fig)

    resultados = {
        "subtarea": "T3",
        "descripcion": (
            "Curva de aprendizaje sobre la división 70/30 de la Parte 1: Naive Bayes "
            "gaussiano y regresión logística entrenados con 5%, 10%, 25%, 50% y 100% "
            "del entrenamiento (submuestras estratificadas, random_state=42) y "
            "evaluados siempre sobre el mismo conjunto de prueba."
        ),
        "division_usada": {
            "origen": origen,
            "test_size": 0.3,
            "random_state": RANDOM_STATE,
            "estratificada": True,
            "tamano_entrenamiento": n_train_total,
            "tamano_prueba": n_test,
        },
        "porcentajes_entrenamiento": [float(f * 100) for f in FRACCIONES],
        "submuestreo": {
            "metodo": "train_test_split(train_size=fraccion, stratify=y_train, random_state=42)",
            "random_state": RANDOM_STATE,
            "detalle_submuestras": detalle,
        },
        "preprocesamiento": {
            "escalador": "StandardScaler",
            "ajustado_solo_con_datos_de_entrenamiento": True,
            "nota": (
                "El escalador se ajusta en cada punto solo con la submuestra de "
                "entrenamiento correspondiente; el conjunto de prueba jamás participa "
                "en el ajuste. Con 100% coincide con el ajuste de la Parte 2."
            ),
        },
        "modelos": {
            "naive_bayes_gaussiano": {
                "clase": "sklearn.naive_bayes.GaussianNB",
                "hiperparametros": "por defecto",
            },
            "regresion_logistica": {
                "clase": "sklearn.linear_model.LogisticRegression",
                "hiperparametros": {"max_iter": 5000, "random_state": RANDOM_STATE},
            },
        },
        "curva_aprendizaje": {
            "n_entrenamiento": tamanos,
            "accuracy_naive_bayes_gaussiano": acc_nb,
            "accuracy_regresion_logistica": acc_lr,
            "f1_macro_naive_bayes_gaussiano": f1_nb,
            "f1_macro_regresion_logistica": f1_lr,
        },
        "accuracy_por_punto": puntos,
        "verificacion_con_parte_2": verif,
        "figura_png": RUTA_FIGURA,
        "archivos_generados": [RUTA_FIGURA, RUTA_TABLA, RUTA_RESULTADOS],
    }

    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=2)

    print("T3 — Curva de aprendizaje (evaluación fija sobre el conjunto de prueba)")
    print(f"División: {origen}")
    print(f"Entrenamiento: {n_train_total} ejemplos | Prueba: {n_test} ejemplos (fija)")
    print(f"{'%entren.':>9} {'n_train':>8} {'Acc NB':>9} {'Acc LR':>9}")
    for d, a1, a2 in zip(detalle, acc_nb, acc_lr):
        print(f"{d['porcentaje']:>8.0f}% {d['n_entrenamiento']:>8d} "
              f"{a1:>9.4f} {a2:>9.4f}")
    print(f"Figura guardada en: {RUTA_FIGURA}")
    print(f"Tabla guardada en:  {RUTA_TABLA}")
    print(f"Resultados en:      {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
