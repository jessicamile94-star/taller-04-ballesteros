import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

SEMILLA = 42
FRACCIONES = [0.05, 0.10, 0.25, 0.50, 1.00]
FIGURA = "curva_aprendizaje.png"

# --- Misma división de la Parte 1 (T1): 70/30 estratificada por clase, random_state=42 ---
datos = load_breast_cancer()
X, y = datos.data, datos.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEMILLA
)

t1_path = Path("entrada/T1/resultados.json")
if t1_path.exists():
    t1 = json.loads(t1_path.read_text(encoding="utf-8"))
    assert len(X_train) == t1["tamano_entrenamiento"]
    assert len(X_test) == t1["tamano_prueba"]

# --- Curva de aprendizaje: submuestras estratificadas del entrenamiento, prueba fija ---
curva = []
for frac in FRACCIONES:
    if frac >= 1.0:
        X_sub, y_sub = X_train, y_train
    else:
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train, train_size=frac, stratify=y_train, random_state=SEMILLA
        )
    escalador = StandardScaler().fit(X_sub)  # ajustado solo con el entrenamiento de este punto
    X_sub_z = escalador.transform(X_sub)
    X_test_z = escalador.transform(X_test)  # la prueba nunca se usa para ajustar
    nb = GaussianNB().fit(X_sub_z, y_sub)
    lr = LogisticRegression(max_iter=5000).fit(X_sub_z, y_sub)
    acc_nb = float(accuracy_score(y_test, nb.predict(X_test_z)))
    acc_lr = float(accuracy_score(y_test, lr.predict(X_test_z)))
    cnt = np.bincount(np.asarray(y_sub), minlength=2)
    curva.append({
        "fraccion": frac,
        "n_ejemplos_entrenamiento": int(len(y_sub)),
        "conteo_clases_submuestra": {"malignant": int(cnt[0]), "benign": int(cnt[1])},
        "accuracy_naive_bayes_gaussiano": acc_nb,
        "accuracy_regresion_logistica": acc_lr,
    })

n_vals = [p["n_ejemplos_entrenamiento"] for p in curva]
acc_nb_vals = [p["accuracy_naive_bayes_gaussiano"] for p in curva]
acc_lr_vals = [p["accuracy_regresion_logistica"] for p in curva]

# --- Verificación de consistencia con T2 en el punto 100% ---
coincide_t2 = None
t2_path = Path("entrada/T2/resultados.json")
if t2_path.exists():
    t2 = json.loads(t2_path.read_text(encoding="utf-8"))
    coincide_t2 = {
        "naive_bayes_gaussiano": bool(abs(curva[-1]["accuracy_naive_bayes_gaussiano"] - t2["accuracy_naive_bayes_gaussiano"]) < 1e-12),
        "regresion_logistica": bool(abs(curva[-1]["accuracy_regresion_logistica"] - t2["accuracy_regresion_logistica"]) < 1e-12),
    }

# --- Figura: exactitud vs número de ejemplos de entrenamiento, una curva por modelo ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n_vals, acc_nb_vals, "o-", color="tab:blue", label="Naive Bayes gaussiano")
ax.plot(n_vals, acc_lr_vals, "s-", color="tab:orange", label="Regresión logística")
for x, v in zip(n_vals, acc_nb_vals):
    ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
for x, v in zip(n_vals, acc_lr_vals):
    ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
ax.set_xticks(n_vals)
ax.set_ylim(min(min(acc_nb_vals), min(acc_lr_vals)) - 0.09, 1.01)
ax.set_xlabel("Número de ejemplos de entrenamiento")
ax.set_ylabel("Exactitud (accuracy) en el conjunto de prueba")
ax.set_title("Curva de aprendizaje — Breast Cancer (división 70/30, random_state=42)")
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIGURA, dpi=120)
plt.close(fig)

# --- Tabla de exactitud de cada punto ---
tabla = []
for p in curva:
    tabla.append({"modelo": "Naive Bayes gaussiano", "fraccion": p["fraccion"],
                  "n_ejemplos_entrenamiento": p["n_ejemplos_entrenamiento"],
                  "accuracy": p["accuracy_naive_bayes_gaussiano"]})
    tabla.append({"modelo": "Regresión logística", "fraccion": p["fraccion"],
                  "n_ejemplos_entrenamiento": p["n_ejemplos_entrenamiento"],
                  "accuracy": p["accuracy_regresion_logistica"]})

resultados = {
    "subtarea": "T3",
    "descripcion": "Curva de aprendizaje: los dos modelos de la Parte 2 entrenados con 5%, 10%, 25%, 50% y 100% del entrenamiento (submuestras estratificadas, random_state=42) y evaluados siempre sobre el mismo conjunto de prueba de la Parte 1.",
    "division": "70/30 estratificada por clase, random_state=42 (misma de la Parte 1/T1)",
    "tamano_entrenamiento_total": int(len(y_train)),
    "tamano_prueba": int(len(y_test)),
    "conjunto_prueba_fijo": True,
    "fracciones_entrenamiento": FRACCIONES,
    "submuestreo_estratificado": True,
    "random_state_submuestreo": SEMILLA,
    "escalador": "StandardScaler ajustado solo con la submuestra de entrenamiento de cada punto (nunca con la prueba)",
    "n_ejemplos_entrenamiento": n_vals,
    "accuracy_naive_bayes_gaussiano": acc_nb_vals,
    "accuracy_regresion_logistica": acc_lr_vals,
    "curva": curva,
    "tabla": tabla,
    "figura": FIGURA,
}
if coincide_t2 is not None:
    resultados["coincide_con_T2_en_punto_100"] = coincide_t2

Path("resultados.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")

print("T3 — Curva de aprendizaje (evaluación fija sobre el conjunto de prueba de la Parte 1)")
print(f"{'fraccion':>9} {'n_train':>8} {'acc NB':>10} {'acc LR':>10}")
for p in curva:
    print(f"{p['fraccion']:>9.2f} {p['n_ejemplos_entrenamiento']:>8d} "
          f"{p['accuracy_naive_bayes_gaussiano']:>10.4f} {p['accuracy_regresion_logistica']:>10.4f}")
if coincide_t2 is not None:
    print("Consistencia con T2 al 100%:", coincide_t2)
print(f"Figura guardada: {FIGURA}")
