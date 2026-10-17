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

# 1) Misma división de la Parte 1 (T1): 70/30 estratificada por clase, random_state=42
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

t1_path = Path("entrada/T1/resultados.json")
if t1_path.exists():
    t1 = json.loads(t1_path.read_text())
    assert len(X_train) == t1["tamano_entrenamiento"]
    assert len(X_test) == t1["tamano_prueba"]
    assert np.bincount(y_test).tolist() == [
        t1["conteo_por_clase_prueba"]["malignant"],
        t1["conteo_por_clase_prueba"]["benign"],
    ]

# 2) Estandarización: el escalador se ajusta SOLO con el conjunto de entrenamiento
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

media_train = X_train_s.mean(axis=0)
media_test = X_test_s.mean(axis=0)
max_abs_media_train = float(np.max(np.abs(media_train)))
max_abs_media_test = float(np.max(np.abs(media_test)))
escalador_ok = bool(max_abs_media_train < 1e-10 and max_abs_media_test > 1e-6)

# 3) Entrenamiento y evaluación sobre el conjunto de prueba
modelos = {
    "Naive Bayes gaussiano": GaussianNB(),
    "Regresión logística": LogisticRegression(max_iter=1000, random_state=42),
}
filas = []
metricas = {}
for nombre, modelo in modelos.items():
    modelo.fit(X_train_s, y_train)
    pred = modelo.predict(X_test_s)
    acc = float(accuracy_score(y_test, pred))
    f1m = float(f1_score(y_test, pred, average="macro"))
    filas.append({"modelo": nombre, "accuracy": acc, "f1_macro": f1m})
    metricas[nombre] = {"accuracy": acc, "f1_macro": f1m}

tabla = pd.DataFrame(filas)

# 4) Contrato: resultados.json con las 4 cifras y la confirmación del escalador
resultados = {
    "subtarea": "T2",
    "division": "70/30 estratificada por clase, random_state=42 (misma de la Parte 1/T1)",
    "tamano_entrenamiento": int(len(X_train)),
    "tamano_prueba": int(len(X_test)),
    "escalador": "StandardScaler ajustado únicamente con X_train",
    "escalador_ajustado_solo_con_entrenamiento": escalador_ok,
    "max_abs_media_atributos_escalados_entrenamiento": max_abs_media_train,
    "max_abs_media_atributos_escalados_prueba": max_abs_media_test,
    "tabla": filas,
    "accuracy_naive_bayes_gaussiano": metricas["Naive Bayes gaussiano"]["accuracy"],
    "f1_macro_naive_bayes_gaussiano": metricas["Naive Bayes gaussiano"]["f1_macro"],
    "accuracy_regresion_logistica": metricas["Regresión logística"]["accuracy"],
    "f1_macro_regresion_logistica": metricas["Regresión logística"]["f1_macro"],
}
Path("resultados.json").write_text(
    json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
)

# 5) Tabla como figura PNG
fig, ax = plt.subplots(figsize=(6.5, 2.6))
ax.axis("off")
tbl = ax.table(
    cellText=[[f"{r['accuracy']:.4f}", f"{r['f1_macro']:.4f}"] for r in filas],
    rowLabels=[r["modelo"] for r in filas],
    colLabels=["Accuracy", "F1 macro"],
    cellLoc="center",
    loc="center",
)
tbl.scale(1, 1.8)
ax.set_title("Parte 2 — Accuracy y F1 macro en el conjunto de prueba (n=171)", pad=14)
plt.savefig("tabla_parte2.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# 6) Resumen
print(tabla.to_string(index=False))
print(f"Escalador ajustado solo con entrenamiento: {escalador_ok}")
print(f"max|media| escalado train={max_abs_media_train:.2e}  test={max_abs_media_test:.2e}")
print("Resultados escritos en resultados.json; tabla en tabla_parte2.png")
