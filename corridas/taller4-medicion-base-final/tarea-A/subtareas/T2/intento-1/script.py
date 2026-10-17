# -*- coding: utf-8 -*-
"""
T2 — Parte 2: dos clasificadores sobre la división de la Parte 1.

Entrena un Naive Bayes gaussiano y una regresión logística (atributos
estandarizados; el escalador se ajusta SOLO con el entrenamiento) sobre la
división 70/30 estratificada con random_state=42 de la Parte 1 (T1), y mide
accuracy y F1 macro de cada uno sobre el conjunto de prueba.

Salidas:
  - resultados.json               (contrato: tabla con las 4 cifras + verificaciones)
  - tabla_parte2_metricas.csv     (tabla en CSV)
  - parte2_metricas_prueba.png    (figura complementaria)
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

SEED = 42
TEST_SIZE = 0.30
T1_DIR = Path("entrada/T1")

# ---------------------------------------------------------------------------
# 1. Recuperar la división de la Parte 1 (T1)
# ---------------------------------------------------------------------------
def cargar_split_de_T1():
    """Si T1 guardó los arreglos del split, reutilizarlos tal cual."""
    candidatos = {
        "X_train": ["X_train.npy", "X_entrenamiento.npy"],
        "X_test": ["X_test.npy", "X_prueba.npy"],
        "y_train": ["y_train.npy", "y_entrenamiento.npy"],
        "y_test": ["y_test.npy", "y_prueba.npy"],
    }
    arrs = {}
    for clave, opciones in candidatos.items():
        for nombre in opciones:
            p = T1_DIR / nombre
            if p.exists():
                arrs[clave] = np.load(p)
                break
    if len(arrs) == 4:
        return (arrs["X_train"], arrs["X_test"],
                arrs["y_train"], arrs["y_test"],
                "archivos guardados por T1 en entrada/T1")
    return None


split_t1 = cargar_split_de_T1()
if split_t1 is not None:
    X_train, X_test, y_train, y_test, origen_split = split_t1
else:
    # Reproducción determinista de la división de la Parte 1 con los mismos
    # parámetros reportados por T1 (load_breast_cancer, 70/30, estratificada,
    # random_state=42). No se usa el conjunto de prueba para nada más.
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target,
        test_size=TEST_SIZE, random_state=SEED, stratify=data.target,
    )
    origen_split = ("reproducida con los parámetros de T1: test_size=0.3, "
                    "random_state=42, estratificada por clase")

X_train = np.asarray(X_train, dtype=float)
X_test = np.asarray(X_test, dtype=float)
y_train = np.asarray(y_train).astype(int)
y_test = np.asarray(y_test).astype(int)

# Verificación contra resultados.json de T1 (misma división)
coincide_con_T1 = None
ruta_t1 = T1_DIR / "resultados.json"
if ruta_t1.exists():
    with open(ruta_t1, "r", encoding="utf-8") as fh:
        t1 = json.load(fh)
    coincide_con_T1 = bool(
        t1.get("n_entrenamiento") == int(X_train.shape[0])
        and t1.get("n_prueba") == int(X_test.shape[0])
        and t1.get("conteo_clases_entrenamiento", {}).get("malignant") == int((y_train == 0).sum())
        and t1.get("conteo_clases_entrenamiento", {}).get("benign") == int((y_train == 1).sum())
        and t1.get("conteo_clases_prueba", {}).get("malignant") == int((y_test == 0).sum())
        and t1.get("conteo_clases_prueba", {}).get("benign") == int((y_test == 1).sum())
    )

# ---------------------------------------------------------------------------
# 2. Estandarización: el escalador se ajusta ÚNICAMENTE con el entrenamiento
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit SOLO con entrenamiento
X_test_s = scaler.transform(X_test)         # test solo se transforma

# Evidencia de que el escalador no vio el conjunto de prueba:
dif_media_vs_train = float(np.max(np.abs(scaler.mean_ - X_train.mean(axis=0))))
dif_media_vs_test = float(np.max(np.abs(scaler.mean_ - X_test.mean(axis=0))))

# ---------------------------------------------------------------------------
# 3. Modelos: Naive Bayes gaussiano y regresión logística
# ---------------------------------------------------------------------------
modelos = {
    "Naive Bayes gaussiano": GaussianNB(),
    "Regresión logística": LogisticRegression(max_iter=1000, random_state=SEED),
}

filas = []
for nombre, modelo in modelos.items():
    modelo.fit(X_train_s, y_train)          # entrenamiento solo con train
    y_pred = modelo.predict(X_test_s)       # evaluación sobre prueba
    filas.append({
        "modelo": nombre,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    })

tabla = pd.DataFrame(filas)

# Tabla en Markdown (para el reporte)
lineas_md = ["| modelo | accuracy | f1_macro |", "|---|---|---|"]
for f in filas:
    lineas_md.append(f"| {f['modelo']} | {f['accuracy']:.6f} | {f['f1_macro']:.6f} |")
tabla_md = "\n".join(lineas_md)

# ---------------------------------------------------------------------------
# 4. Figura complementaria (barras agrupadas)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2))
x = np.arange(len(tabla))
ancho = 0.35
ax.bar(x - ancho / 2, tabla["accuracy"], ancho, label="Accuracy", color="#4C72B0")
ax.bar(x + ancho / 2, tabla["f1_macro"], ancho, label="F1 macro", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(tabla["modelo"])
ax.set_ylim(0.0, 1.08)
ax.set_ylabel("Valor en el conjunto de prueba")
ax.set_title("Parte 2 — Desempeño en prueba (división 70/30, random_state=42)")
for i, (a, f1m) in enumerate(zip(tabla["accuracy"], tabla["f1_macro"])):
    ax.text(i - ancho / 2, a + 0.015, f"{a:.3f}", ha="center", fontsize=9)
    ax.text(i + ancho / 2, f1m + 0.015, f"{f1m:.3f}", ha="center", fontsize=9)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("parte2_metricas_prueba.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Tabla en CSV y resultados.json (contrato)
# ---------------------------------------------------------------------------
tabla.to_csv("tabla_parte2_metricas.csv", index=False, encoding="utf-8")

metricas = {f["modelo"]: f for f in filas}
nb = metricas["Naive Bayes gaussiano"]
rl = metricas["Regresión logística"]

resultados = {
    "subtarea": "T2_dos_clasificadores",
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "division_usada": {
        "origen": origen_split,
        "coincide_con_T1": coincide_con_T1,
        "random_state": SEED,
        "test_size": TEST_SIZE,
        "estratificada": True,
        "n_entrenamiento": int(X_train.shape[0]),
        "n_prueba": int(X_test.shape[0]),
    },
    "escalador": {
        "tipo": "StandardScaler",
        "ajustado_solo_con_entrenamiento": True,
        "verificacion": {
            "max_dif_media_scaler_vs_media_entrenamiento": dif_media_vs_train,
            "max_dif_media_scaler_vs_media_prueba": dif_media_vs_test,
            "nota": ("La media del escalador coincide con la media del "
                     "entrenamiento (dif ~ 0) y difiere de la media de prueba, "
                     "evidencia de que se ajustó únicamente con entrenamiento."),
        },
    },
    "modelos": {
        "Naive Bayes gaussiano": {"clase": "sklearn.naive_bayes.GaussianNB",
                                  "atributos": "estandarizados"},
        "Regresión logística": {"clase": "sklearn.linear_model.LogisticRegression",
                                "max_iter": 1000, "random_state": SEED,
                                "atributos": "estandarizados"},
    },
    "tabla_metricas_prueba": filas,
    "accuracy_nb_gaussiano": nb["accuracy"],
    "f1_macro_nb_gaussiano": nb["f1_macro"],
    "accuracy_regresion_logistica": rl["accuracy"],
    "f1_macro_regresion_logistica": rl["f1_macro"],
    "mejor_modelo": {
        "por_accuracy": max(filas, key=lambda f: f["accuracy"])["modelo"],
        "por_f1_macro": max(filas, key=lambda f: f["f1_macro"])["modelo"],
    },
    "tabla_markdown": tabla_md,
    "archivos_generados": [
        "resultados.json",
        "tabla_parte2_metricas.csv",
        "parte2_metricas_prueba.png",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(resultados, fh, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# 6. Resumen
# ---------------------------------------------------------------------------
print("T2 — Parte 2: dos clasificadores sobre la división 70/30 de la Parte 1")
print(f"División: {origen_split} | coincide con T1: {coincide_con_T1}")
print(f"n_entrenamiento={X_train.shape[0]}  n_prueba={X_test.shape[0]}")
print(f"Escalador ajustado solo con entrenamiento: "
      f"dif media vs train={dif_media_vs_train:.2e}, vs test={dif_media_vs_test:.2e}")
print("\nTabla de métricas en el conjunto de prueba:")
print(tabla.to_string(index=False))
print("\nArchivos generados: resultados.json, tabla_parte2_metricas.csv, "
      "parte2_metricas_prueba.png")
