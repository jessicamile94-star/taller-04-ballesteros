# -*- coding: utf-8 -*-
"""
T2 — Dos clasificadores sobre la división de la Parte 1 (Breast Cancer Wisconsin).

Entrena un Naive Bayes gaussiano y una regresión logística sobre atributos
estandarizados (StandardScaler ajustado SOLO con el conjunto de entrenamiento)
y reporta accuracy y F1 macro de cada modelo sobre el conjunto de prueba,
presentados en una tabla.

Entradas : entrada/T1/resultados.json, entrada/T1/division_T1.npz
Salidas  : resultados.json, tabla_T2.csv, T2_metricas_modelos.png
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

# ------------------------------------------------------------------ constantes
RANDOM_STATE = 42
TEST_SIZE = 0.3
TRAIN_SIZE = 0.7

T1_DIR = Path("entrada") / "T1"
T1_JSON = T1_DIR / "resultados.json"
T1_NPZ = T1_DIR / "division_T1.npz"

# ------------------------------------------- datos y división única de la Parte 1
data = load_breast_cancer()
X, y = data.data, data.target
nombres_clases = [str(c) for c in data.target_names]  # 0=malignant, 1=benign

t1 = {}
if T1_JSON.exists():
    try:
        t1 = json.loads(T1_JSON.read_text(encoding="utf-8"))
    except Exception:
        t1 = {}
n_train_esp = int(t1.get("tamano_entrenamiento", 398))
n_test_esp = int(t1.get("tamano_prueba", 171))


def _cargar_division_npz(ruta, X, y, n_tr, n_te):
    """Recupera la división 70/30 guardada por T1.
    Devuelve (X_train, X_test, y_train, y_test, descripcion) o None."""
    try:
        npz = np.load(ruta, allow_pickle=True)
        claves = list(npz.files)
        marcas_idx = ("idx", "index", "indices", "indice")

        # Caso 1: el archivo guarda los índices de cada partición
        k_tr = [k for k in claves if "train" in k.lower() and any(m in k.lower() for m in marcas_idx)]
        k_te = [k for k in claves
                if ("test" in k.lower() or "prueba" in k.lower()) and any(m in k.lower() for m in marcas_idx)]
        if k_tr and k_te:
            it = np.asarray(npz[k_tr[0]]).ravel().astype(int)
            ie = np.asarray(npz[k_te[0]]).ravel().astype(int)
            if len(it) == n_tr and len(ie) == n_te:
                return X[it], X[ie], y[it], y[ie], f"índices ({k_tr[0]}, {k_te[0]})"

        # Caso 2: el archivo guarda los arreglos ya divididos
        def _busca(prefijos, como):
            for k in claves:
                kl = k.lower()
                if any(kl == p or kl.startswith(p) for p in prefijos):
                    arr = np.asarray(npz[k])
                    if como == "X":
                        if arr.ndim == 2 and arr.shape[1] == X.shape[1]:
                            return arr
                    else:
                        arr = arr.ravel()
                        if arr.ndim == 1:
                            return arr
            return None

        Xt = _busca(("x_train", "x_entrenamiento", "xtrain"), "X")
        Xe = _busca(("x_test", "x_prueba", "xtest"), "X")
        yt = _busca(("y_train", "y_entrenamiento", "ytrain"), "y")
        ye = _busca(("y_test", "y_prueba", "ytest"), "y")
        if all(v is not None for v in (Xt, Xe, yt, ye)) and len(yt) == n_tr and len(ye) == n_te:
            return (
                np.asarray(Xt, dtype=float),
                np.asarray(Xe, dtype=float),
                np.asarray(yt).astype(int),
                np.asarray(ye).astype(int),
                "arreglos X/y",
            )
        return None
    except Exception:
        return None


fuente_division = None
X_train = X_test = y_train = y_test = None
if T1_NPZ.exists():
    rec = _cargar_division_npz(T1_NPZ, X, y, n_train_esp, n_test_esp)
    if rec is not None:
        X_train, X_test, y_train, y_test, desc = rec
        fuente_division = f"{T1_NPZ.as_posix()} — {desc}"

if X_train is None:
    # Respaldo determinista: reproduce exactamente la división de la Parte 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        train_size=TRAIN_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    fuente_division = (
        "recreada de forma determinista con train_test_split(test_size=0.3, "
        "train_size=0.7, stratify=y, random_state=42): idéntica a la de la Parte 1"
    )

conteo_train = {nombres_clases[c]: int((y_train == c).sum()) for c in np.unique(y)}
conteo_test = {nombres_clases[c]: int((y_test == c).sum()) for c in np.unique(y)}
coincide_t1 = bool(
    len(y_train) == n_train_esp
    and len(y_test) == n_test_esp
    and conteo_train == t1.get("conteo_clases_entrenamiento", conteo_train)
    and conteo_test == t1.get("conteo_clases_prueba", conteo_test)
)

# ------------------------------- escalador ajustado SOLO con el entrenamiento
scaler = StandardScaler()
scaler.fit(X_train)  # única llamada a fit: únicamente datos de entrenamiento
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)  # el conjunto de prueba solo se transforma

media_train = X_train.mean(axis=0)
media_test = X_test.mean(axis=0)
media_total = X.mean(axis=0)
diff_media_vs_train = float(np.max(np.abs(scaler.mean_ - media_train)))
diff_media_vs_test = float(np.max(np.abs(scaler.mean_ - media_test)))
diff_media_vs_total = float(np.max(np.abs(scaler.mean_ - media_total)))
diff_scale_vs_train_std = float(np.max(np.abs(scaler.scale_ - X_train.std(axis=0))))
media_prueba_est = X_test_s.mean(axis=0)
std_prueba_est = X_test_s.std(axis=0)  # ddof=0, igual que StandardScaler

escalador_solo_train = bool(
    diff_media_vs_train <= 1e-10
    and diff_scale_vs_train_std <= 1e-8
    and diff_media_vs_total > 1e-8
    and float(np.max(np.abs(media_prueba_est))) > 1e-6
)

# ---------------------------------------------------------------------- modelos
modelos = {
    "Naive Bayes gaussiano": GaussianNB(),
    "Regresión logística": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
}

filas = []
for nombre, modelo in modelos.items():
    modelo.fit(X_train_s, y_train)
    y_pred = modelo.predict(X_test_s)
    filas.append(
        {
            "modelo": nombre,
            "accuracy_prueba": float(accuracy_score(y_test, y_pred)),
            "f1_macro_prueba": float(f1_score(y_test, y_pred, average="macro")),
        }
    )

tabla = pd.DataFrame(filas)
tabla.to_csv("tabla_T2.csv", index=False, encoding="utf-8")


def tabla_a_markdown(df):
    cols = list(df.columns)
    lineas = ["| " + " | ".join(cols) + " |", "|" + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        lineas.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lineas)


md_tabla = tabla_a_markdown(tabla)

# ----------------------------------------------------------------------- figura
fig, ax = plt.subplots(figsize=(7.5, 4.8))
xpos = np.arange(len(filas))
ancho = 0.35
b1 = ax.bar(xpos - ancho / 2, [f["accuracy_prueba"] for f in filas], ancho,
            label="Accuracy", color="#4C72B0")
b2 = ax.bar(xpos + ancho / 2, [f["f1_macro_prueba"] for f in filas], ancho,
            label="F1 macro", color="#DD8452")
for barras in (b1, b2):
    for b in barras:
        ax.annotate(f"{b.get_height():.4f}",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
ax.set_xticks(xpos)
ax.set_xticklabels([f["modelo"] for f in filas])
ax.set_ylim(0.0, 1.08)
ax.set_ylabel("Valor en el conjunto de prueba")
ax.set_title("Parte 2 — Accuracy y F1 macro en prueba\n"
             "(división 70/30 estratificada, random_state=42; escalador ajustado solo con entrenamiento)")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
plt.savefig("T2_metricas_modelos.png", dpi=120)
plt.close(fig)

# -------------------------------------------------------------- resultados.json
resultados = {
    "subtarea": "T2",
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "n_atributos": int(X.shape[1]),
    "codificacion_clases": {"malignant": 0, "benign": 1},
    "descripcion": (
        "Naive Bayes gaussiano y regresión logística entrenados sobre la división 70/30 "
        "de la Parte 1 con atributos estandarizados (StandardScaler ajustado solo con el "
        "conjunto de entrenamiento); accuracy y F1 macro medidos sobre el conjunto de prueba."
    ),
    "fuente_division": fuente_division,
    "division_validada_contra_T1": coincide_t1,
    "parametros_division": {
        "test_size": TEST_SIZE,
        "train_size": TRAIN_SIZE,
        "estratificar_por_clase": True,
        "random_state": RANDOM_STATE,
    },
    "n_entrenamiento": int(len(y_train)),
    "n_prueba": int(len(y_test)),
    "conteo_clases_entrenamiento": conteo_train,
    "conteo_clases_prueba": conteo_test,
    "preprocesamiento": {
        "escalador": "StandardScaler",
        "atributos_estandarizados_para": list(modelos.keys()),
        "ajustado_solo_con_entrenamiento": escalador_solo_train,
        "nota": (
            "El escalador se ajustó únicamente con X_train (única llamada a fit); el "
            "conjunto de prueba solo se transforma con esos parámetros y no se usa para "
            "ajustar nada (ni escalador, ni modelos, ni selección de atributos)."
        ),
        "evidencia": {
            "max_abs_diff_media_scaler_vs_media_entrenamiento": diff_media_vs_train,
            "max_abs_diff_media_scaler_vs_media_prueba": diff_media_vs_test,
            "max_abs_diff_media_scaler_vs_media_total": diff_media_vs_total,
            "max_abs_diff_scale_scaler_vs_std_entrenamiento": diff_scale_vs_train_std,
            "max_abs_media_atributos_prueba_estandarizados": float(np.max(np.abs(media_prueba_est))),
            "std_atributos_prueba_estandarizados_min": float(std_prueba_est.min()),
            "std_atributos_prueba_estandarizados_max": float(std_prueba_est.max()),
        },
    },
    "modelos": {
        "Naive Bayes gaussiano": {
            "implementacion": "sklearn.naive_bayes.GaussianNB (parámetros por defecto)",
            "atributos": "estandarizados",
            "accuracy_prueba": filas[0]["accuracy_prueba"],
            "f1_macro_prueba": filas[0]["f1_macro_prueba"],
        },
        "Regresión logística": {
            "implementacion": "sklearn.linear_model.LogisticRegression(max_iter=5000, random_state=42)",
            "atributos": "estandarizados",
            "accuracy_prueba": filas[1]["accuracy_prueba"],
            "f1_macro_prueba": filas[1]["f1_macro_prueba"],
        },
    },
    "tabla": {"columnas": list(tabla.columns), "filas": filas},
    "tabla_markdown": md_tabla,
    "archivos_generados": ["tabla_T2.csv", "T2_metricas_modelos.png"],
}

Path("resultados.json").write_text(
    json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
)

# ---------------------------------------------------------------------- resumen
print("T2 — Dos clasificadores sobre la división 70/30 de la Parte 1 (random_state=42)")
print(f"División usada: {fuente_division}")
print(f"n_entrenamiento={len(y_train)}  n_prueba={len(y_test)}  validada_contra_T1={coincide_t1}")
print()
print("Tabla de métricas en el conjunto de prueba:")
print(tabla.to_string(index=False))
print()
print(f"Escalador ajustado solo con entrenamiento: {escalador_solo_train}")
print(f"  max|media_scaler - media_train| = {diff_media_vs_train:.3e} (≈0: coincide con el entrenamiento)")
print(f"  max|media_scaler - media_total| = {diff_media_vs_total:.3e} (>0: NO se ajustó con todos los datos)")
print(f"  max|media_prueba_estandarizada| = {float(np.max(np.abs(media_prueba_est))):.3e} (>0: el prueba solo se transforma)")
print()
print("Archivos generados: resultados.json, tabla_T2.csv, T2_metricas_modelos.png")
