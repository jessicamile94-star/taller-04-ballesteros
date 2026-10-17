# T2 — Naive Bayes gaussiano vs. regresión logística sobre la división 70/30 de la Parte 1
# Métricas: accuracy y F1 macro en el conjunto de prueba, presentadas en una tabla.
# El escalador (StandardScaler) se ajusta ÚNICAMENTE con el conjunto de entrenamiento.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ----------------------------------------------------------------------
# 1. Cargar la división de la Parte 1 (entrada/T1)
# ----------------------------------------------------------------------
t1_dir = Path("entrada") / "T1"
with open(t1_dir / "resultados.json", "r", encoding="utf-8") as f:
    t1 = json.load(f)

df_train = pd.read_csv(t1_dir / "t1_entrenamiento.csv")
df_test = pd.read_csv(t1_dir / "t1_prueba.csv")

# Descartar posibles columnas de índice sin nombre
df_train = df_train.loc[:, ~df_train.columns.str.startswith("Unnamed")]
df_test = df_test.loc[:, ~df_test.columns.str.startswith("Unnamed")]

target_col = "target" if "target" in df_train.columns else df_train.columns[-1]

# Usar los nombres de atributos registrados en T1 si están disponibles
nombres_atributos = t1.get("nombres_atributos", [])
if nombres_atributos and all(c in df_train.columns for c in nombres_atributos):
    feature_cols = list(nombres_atributos)
else:
    feature_cols = [c for c in df_train.columns if c != target_col]

assert len(feature_cols) == t1["numero_atributos"], (
    f"Se esperaban {t1['numero_atributos']} atributos, se encontraron {len(feature_cols)}"
)

X_train = df_train[feature_cols].to_numpy(dtype=float)
y_train = df_train[target_col].to_numpy(dtype=int)
X_test = df_test[feature_cols].to_numpy(dtype=float)
y_test = df_test[target_col].to_numpy(dtype=int)

print(f"División de la Parte 1 reutilizada: {X_train.shape[0]} entrenamiento / "
      f"{X_test.shape[0]} prueba (70/30 estratificada, random_state=42)")

# ----------------------------------------------------------------------
# 2. Estandarización: el escalador se ajusta SOLO con el entrenamiento
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)  # fit + transform SOLO con train
X_test_std = scaler.transform(X_test)        # el test SOLO se transforma

verif = {
    "max_abs_media_atributos_entrenamiento_estandarizado": float(
        np.max(np.abs(X_train_std.mean(axis=0)))
    ),
    "max_abs_std_menos_1_atributos_entrenamiento_estandarizado": float(
        np.max(np.abs(X_train_std.std(axis=0, ddof=0) - 1.0))
    ),
    "max_abs_media_atributos_prueba_estandarizado": float(
        np.max(np.abs(X_test_std.mean(axis=0)))
    ),
    "nota": (
        "La media del entrenamiento estandarizado es ~0 y su desviación estándar ~1, "
        "lo que confirma que el escalador se ajustó solo con el entrenamiento; la media "
        "del test estandarizado NO es exactamente 0, prueba de que el test no participó "
        "en el ajuste del escalador."
    ),
}

# ----------------------------------------------------------------------
# 3. Entrenamiento de los dos clasificadores
# ----------------------------------------------------------------------
gnb = GaussianNB()
gnb.fit(X_train_std, y_train)
pred_gnb = gnb.predict(X_test_std)

logreg = LogisticRegression(max_iter=5000, random_state=42)
logreg.fit(X_train_std, y_train)
pred_lr = logreg.predict(X_test_std)

# ----------------------------------------------------------------------
# 4. Métricas sobre el conjunto de prueba
# ----------------------------------------------------------------------
acc_gnb = float(accuracy_score(y_test, pred_gnb))
f1_gnb = float(f1_score(y_test, pred_gnb, average="macro"))
acc_lr = float(accuracy_score(y_test, pred_lr))
f1_lr = float(f1_score(y_test, pred_lr, average="macro"))

tabla = pd.DataFrame(
    {
        "modelo": ["Naive Bayes gaussiano", "Regresión logística"],
        "accuracy_prueba": [acc_gnb, acc_lr],
        "f1_macro_prueba": [f1_gnb, f1_lr],
    }
)
tabla.to_csv("t2_tabla_metricas.csv", index=False)

# ----------------------------------------------------------------------
# 5. Figuras (PNG en la carpeta actual)
# ----------------------------------------------------------------------
# 5a. Gráfico de barras
fig, ax = plt.subplots(figsize=(7, 4.5))
xpos = np.arange(2)
width = 0.35
b1 = ax.bar(xpos - width / 2, [acc_gnb, acc_lr], width, label="Accuracy", color="#4C72B0")
b2 = ax.bar(xpos + width / 2, [f1_gnb, f1_lr], width, label="F1 macro", color="#DD8452")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", fontsize=9)
ax.set_xticks(xpos)
ax.set_xticklabels(["Naive Bayes\ngaussiano", "Regresión\nlogística"])
ax.set_ylim(0, 1.08)
ax.set_ylabel("Valor en el conjunto de prueba")
ax.set_title("Parte 2 — Accuracy y F1 macro en prueba (división 70/30, random_state=42)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("t2_metricas_modelos.png", dpi=120)
plt.close(fig)

# 5b. Tabla como PNG
fig2, ax2 = plt.subplots(figsize=(7, 2.6))
ax2.axis("off")
celdas = [
    ["Naive Bayes gaussiano", f"{acc_gnb:.4f}", f"{f1_gnb:.4f}"],
    ["Regresión logística", f"{acc_lr:.4f}", f"{f1_lr:.4f}"],
]
tbl = ax2.table(cellText=celdas,
                colLabels=["Modelo", "Accuracy (prueba)", "F1 macro (prueba)"],
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 1.9)
ax2.set_title("Parte 2 — Métricas en el conjunto de prueba", fontsize=12)
fig2.savefig("t2_tabla_metricas.png", dpi=120, bbox_inches="tight")
plt.close(fig2)

# ----------------------------------------------------------------------
# 6. resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T2",
    "descripcion": (
        "Naive Bayes gaussiano y regresión logística entrenados sobre la división 70/30 "
        "de la Parte 1 (atributos estandarizados, escalador ajustado solo con el "
        "entrenamiento); accuracy y F1 macro medidas sobre el conjunto de prueba."
    ),
    "division_usada": {
        "origen": "entrada/T1 (Parte 1)",
        "test_size": 0.3,
        "random_state": 42,
        "estratificada": True,
        "tamano_entrenamiento": int(X_train.shape[0]),
        "tamano_prueba": int(X_test.shape[0]),
    },
    "preprocesamiento": {
        "escalador": "StandardScaler",
        "ajustado_solo_con_entrenamiento": True,
        "verificacion": verif,
    },
    "modelos": {
        "naive_bayes_gaussiano": {
            "clase": "sklearn.naive_bayes.GaussianNB",
            "hiperparametros": "por defecto",
        },
        "regresion_logistica": {
            "clase": "sklearn.linear_model.LogisticRegression",
            "hiperparametros": {"max_iter": 5000, "random_state": 42},
        },
    },
    "tabla_metricas_prueba": [
        {"modelo": "Naive Bayes gaussiano", "accuracy": acc_gnb, "f1_macro": f1_gnb},
        {"modelo": "Regresión logística", "accuracy": acc_lr, "f1_macro": f1_lr},
    ],
    "accuracy_naive_bayes_gaussiano": acc_gnb,
    "f1_macro_naive_bayes_gaussiano": f1_gnb,
    "accuracy_regresion_logistica": acc_lr,
    "f1_macro_regresion_logistica": f1_lr,
    "matrices_de_confusion_prueba": {
        "clases": t1["nombres_clases"],
        "naive_bayes_gaussiano": confusion_matrix(y_test, pred_gnb).tolist(),
        "regresion_logistica": confusion_matrix(y_test, pred_lr).tolist(),
    },
    "archivos_generados": [
        "resultados.json",
        "t2_tabla_metricas.csv",
        "t2_tabla_metricas.png",
        "t2_metricas_modelos.png",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------
# 7. Resumen
# ----------------------------------------------------------------------
print("\n=== T2 — Tabla de métricas en el conjunto de prueba ===")
print(tabla.to_string(index=False))
print(
    "\nVerificación del escalador (ajustado solo con entrenamiento): "
    f"max |media train est.| = {verif['max_abs_media_atributos_entrenamiento_estandarizado']:.2e}, "
    f"max |std train est. - 1| = {verif['max_abs_std_menos_1_atributos_entrenamiento_estandarizado']:.2e}, "
    f"max |media test est.| = {verif['max_abs_media_atributos_prueba_estandarizado']:.2e} (≠ 0, el test no se usó para ajustar)"
)
print("Archivos generados: resultados.json, t2_tabla_metricas.csv, "
      "t2_tabla_metricas.png, t2_metricas_modelos.png")
