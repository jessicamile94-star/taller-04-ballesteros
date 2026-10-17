# -*- coding: utf-8 -*-
"""
T2 — Dos clasificadores sobre la división de la Parte 1 (Breast Cancer Wisconsin).

Entrena un Naive Bayes gaussiano y una regresión logística (atributos estandarizados;
el escalador se ajusta SOLO con el entrenamiento) y reporta accuracy y F1 macro de
cada uno sobre el conjunto de prueba, en una tabla.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

SEMILLA = 42

# ----------------------------------------------------------------------
# 1) Recuperar la división 70/30 estratificada (random_state=42) de la Parte 1
# ----------------------------------------------------------------------
t1_dir = Path("entrada") / "T1"
t1_res = None
if (t1_dir / "resultados.json").exists():
    t1_res = json.loads((t1_dir / "resultados.json").read_text(encoding="utf-8"))

nombres_atributos = list(t1_res.get("nombres_atributos", [])) if t1_res else []
if not nombres_atributos:
    nombres_atributos = list(load_breast_cancer().feature_names)


def decodificar_objetivo(serie):
    """Convierte la columna objetivo a enteros (0=malignant, 1=benign)."""
    if serie.dtype == object or str(serie.dtype) == "category" or serie.dtype == bool:
        s = serie.astype(str).str.strip().str.lower()
        enc = s.map({"malignant": 0, "benign": 1, "m": 0, "b": 1})
        if enc.isna().any():
            enc = pd.to_numeric(s, errors="coerce")
        vals = enc.to_numpy(dtype=float)
    else:
        vals = pd.to_numeric(serie, errors="coerce").to_numpy(dtype=float)
    if np.isnan(vals).any():
        raise ValueError("No se pudo decodificar la columna objetivo.")
    return vals.astype(int)


def cargar_csv(ruta, nombres_atributos):
    df = pd.read_csv(ruta)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]
    candidatos = ["target", "clase", "diagnostico", "diagnosis", "label", "objetivo", "y", "class"]
    col_obj = next((c for c in candidatos if c in df.columns), None)
    if col_obj is None:
        extras = [c for c in df.columns if c not in nombres_atributos]
        col_obj = extras[-1] if extras else df.columns[-1]
    y = decodificar_objetivo(df[col_obj])
    cols_atr = [c for c in df.columns if c != col_obj]
    comunes = [c for c in nombres_atributos if c in cols_atr]
    if len(comunes) >= max(1, int(0.9 * len(nombres_atributos))):
        cols_atr = comunes  # mismo orden de atributos que en la Parte 1
    X = df[cols_atr].to_numpy(dtype=float)
    return X, y


X_train = X_test = y_train = y_test = None
fuente_datos = None
for rt, rs in [
    (t1_dir / "T1_entrenamiento.csv", t1_dir / "T1_prueba.csv"),
    (Path("T1_entrenamiento.csv"), Path("T1_prueba.csv")),
]:
    if rt.exists() and rs.exists():
        X_train, y_train = cargar_csv(rt, nombres_atributos)
        X_test, y_test = cargar_csv(rs, nombres_atributos)
        fuente_datos = f"División de la Parte 1 reutilizada desde {rt} y {rs}"
        break

if X_train is None:
    # Respaldo: regenerar exactamente la misma división determinista de la Parte 1
    data = load_breast_cancer()
    X_full, y_full = data.data.astype(float), data.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.3, stratify=y_full, random_state=SEMILLA
    )
    fuente_datos = (
        "load_breast_cancer + train_test_split(test_size=0.3, stratify=y, "
        "random_state=42): misma división determinista de la Parte 1"
    )

coincide_con_T1 = None
if t1_res and "division" in t1_res:
    d1 = t1_res["division"]
    coincide_con_T1 = bool(
        len(y_train) == d1.get("n_entrenamiento") and len(y_test) == d1.get("n_prueba")
    )

# ----------------------------------------------------------------------
# 2) Estandarización: el escalador se ajusta SOLO con el entrenamiento
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_train_z = scaler.fit_transform(X_train)  # fit + transform SOLO con entrenamiento
X_test_z = scaler.transform(X_test)        # prueba: únicamente transform

dif_vs_train = float(np.max(np.abs(scaler.mean_ - X_train.mean(axis=0))))
dif_vs_test = float(np.max(np.abs(scaler.mean_ - X_test.mean(axis=0))))
ajustado_solo_train = bool(dif_vs_train < 1e-10 and dif_vs_test > 1e-12)
max_abs_media_z = float(np.max(np.abs(X_train_z.mean(axis=0))))
min_std_z = float(X_train_z.std(axis=0).min())
max_std_z = float(X_train_z.std(axis=0).max())

# ----------------------------------------------------------------------
# 3) Modelos (sin ajuste de hiperparámetros con el conjunto de prueba)
# ----------------------------------------------------------------------
nb = GaussianNB()  # por defecto
nb.fit(X_train_z, y_train)

logreg = LogisticRegression(max_iter=1000, random_state=SEMILLA)
logreg.fit(X_train_z, y_train)

# ----------------------------------------------------------------------
# 4) Evaluación sobre el conjunto de prueba
# ----------------------------------------------------------------------
def evaluar(modelo, X, y):
    pred = modelo.predict(X)
    return {
        "accuracy_test": float(accuracy_score(y, pred)),
        "f1_macro_test": float(f1_score(y, pred, average="macro")),
    }


etiquetas_modelos = {
    "Naive Bayes gaussiano (GaussianNB)": nb,
    "Regresión logística (LogisticRegression)": logreg,
}
metricas = {nombre: evaluar(m, X_test_z, y_test) for nombre, m in etiquetas_modelos.items()}

# ----------------------------------------------------------------------
# 5) Tabla de resultados
# ----------------------------------------------------------------------
filas_tabla = [
    {
        "modelo": nombre,
        "accuracy_test": float(v["accuracy_test"]),
        "f1_macro_test": float(v["f1_macro_test"]),
    }
    for nombre, v in metricas.items()
]
tabla = pd.DataFrame(filas_tabla)
tabla.to_csv("T2_tabla_metricas_test.csv", index=False, encoding="utf-8")

lineas_md = ["| Modelo | Accuracy (prueba) | F1 macro (prueba) |", "|---|---:|---:|"]
for nombre, v in metricas.items():
    lineas_md.append(f"| {nombre} | {v['accuracy_test']:.6f} | {v['f1_macro_test']:.6f} |")
tabla_md = "\n".join(lineas_md)

# ----------------------------------------------------------------------
# 6) Figura: barras de accuracy y F1 macro por modelo
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.6))
nombres_cortos = ["GaussianNB", "LogisticRegression"]
accs = [filas_tabla[0]["accuracy_test"], filas_tabla[1]["accuracy_test"]]
f1s = [filas_tabla[0]["f1_macro_test"], filas_tabla[1]["f1_macro_test"]]
x = np.arange(2)
b1 = ax.bar(x - 0.2, accs, 0.4, label="Accuracy (prueba)", color="#4C72B0")
b2 = ax.bar(x + 0.2, f1s, 0.4, label="F1 macro (prueba)", color="#DD8452")
for barras in (b1, b2):
    for b in barras:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.004,
            f"{b.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
ax.set_xticks(x)
ax.set_xticklabels(nombres_cortos)
ax.set_ylim(0.0, 1.08)
ax.set_ylabel("Valor en el conjunto de prueba")
ax.set_title(
    f"T2: accuracy y F1 macro en prueba\n"
    f"(división 70/30 estratificada, random_state={SEMILLA}, n_prueba={len(y_test)})"
)
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("T2_metricas_modelos.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 7) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T2",
    "descripcion": (
        "Naive Bayes gaussiano y regresión logística entrenados sobre la división 70/30 "
        "de la Parte 1 (atributos estandarizados; escalador ajustado solo con el "
        "entrenamiento) y evaluados con accuracy y F1 macro sobre el conjunto de prueba."
    ),
    "datos": {
        "fuente": fuente_datos,
        "division_reutilizada_de_parte_1": True,
        "test_size": 0.3,
        "estratificada_por_clase": True,
        "random_state": SEMILLA,
        "n_entrenamiento": int(X_train.shape[0]),
        "n_prueba": int(X_test.shape[0]),
        "n_atributos": int(X_train.shape[1]),
        "conteo_clases_entrenamiento": {
            "malignant": int((y_train == 0).sum()),
            "benign": int((y_train == 1).sum()),
        },
        "conteo_clases_prueba": {
            "malignant": int((y_test == 0).sum()),
            "benign": int((y_test == 1).sum()),
        },
        "coincide_con_T1": coincide_con_T1,
    },
    "preprocesamiento": {
        "escalador": "sklearn.preprocessing.StandardScaler",
        "ajustado_solo_con_entrenamiento": ajustado_solo_train,
        "conjunto_de_prueba_solo_transformado": True,
        "verificacion": {
            "max_abs_dif_media_escalador_vs_media_entrenamiento": dif_vs_train,
            "max_abs_dif_media_escalador_vs_media_prueba": dif_vs_test,
            "max_abs_media_atributos_escalados_entrenamiento": max_abs_media_z,
            "min_desv_est_atributos_escalados_entrenamiento": min_std_z,
            "max_desv_est_atributos_escalados_entrenamiento": max_std_z,
            "interpretacion": (
                "La media del escalador coincide exactamente con la media del entrenamiento "
                "y difiere de la media de la prueba: se ajustó únicamente con el entrenamiento."
            ),
        },
    },
    "modelos": {
        "naive_bayes_gaussiano": {
            "implementacion": "sklearn.naive_bayes.GaussianNB",
            "hiperparametros": "por defecto",
            "accuracy_test": filas_tabla[0]["accuracy_test"],
            "f1_macro_test": filas_tabla[0]["f1_macro_test"],
        },
        "regresion_logistica": {
            "implementacion": "sklearn.linear_model.LogisticRegression",
            "hiperparametros": {
                "max_iter": 1000,
                "random_state": SEMILLA,
                "resto": "por defecto",
            },
            "accuracy_test": filas_tabla[1]["accuracy_test"],
            "f1_macro_test": filas_tabla[1]["f1_macro_test"],
        },
    },
    "tabla": {
        "columnas": ["modelo", "accuracy_test", "f1_macro_test"],
        "filas": filas_tabla,
    },
    "tabla_markdown": tabla_md,
    "archivos_generados": {
        "tabla_csv": "T2_tabla_metricas_test.csv",
        "figura": "T2_metricas_modelos.png",
    },
    "nota": (
        f"El escalador se ajustó únicamente con los {X_train.shape[0]} ejemplos de "
        f"entrenamiento; el conjunto de prueba ({X_test.shape[0]} ejemplos) solo se "
        "transformó con ese escalador y se usó exclusivamente para la evaluación final. "
        "No se ajustó ningún hiperparámetro con el conjunto de prueba."
    ),
}

Path("resultados.json").write_text(
    json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
)

# ----------------------------------------------------------------------
# 8) Resumen
# ----------------------------------------------------------------------
print("=" * 64)
print("T2 — Naive Bayes gaussiano vs. regresión logística (conjunto de prueba)")
print("=" * 64)
print(tabla.to_string(index=False))
print("-" * 64)
print(f"Escalador ajustado solo con entrenamiento: {ajustado_solo_train}")
print(f"  max|media_escalador - media_entrenamiento| = {dif_vs_train:.3e}")
print(f"  max|media_escalador - media_prueba|        = {dif_vs_test:.3e}")
print(
    f"n_entrenamiento={X_train.shape[0]}, n_prueba={X_test.shape[0]}, "
    f"coincide_con_T1={coincide_con_T1}"
)
print("Archivos generados: resultados.json, T2_tabla_metricas_test.csv, T2_metricas_modelos.png")
