# -*- coding: utf-8 -*-
"""
T3 — Curva de aprendizaje: Naive Bayes gaussiano vs Regresión logística.

- Misma división de la Parte 1: load_breast_cancer, 70/30 estratificado, random_state=42.
- Escalador StandardScaler ajustado SOLO con el entrenamiento completo (como en la Parte 2).
- Submuestras estratificadas del entrenamiento: 5%, 10%, 25%, 50% y 100% (random_state=42).
- Cada modelo se entrena desde cero en cada submuestra y se evalúa SIEMPRE sobre el mismo
  conjunto de prueba de la Parte 1 (la prueba nunca se usa para ajustar nada).
- Salidas: resultados.json, parte3_curva_aprendizaje.csv, parte3_curva_aprendizaje.png
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

RANDOM_STATE = 42
FRACCIONES = [0.05, 0.10, 0.25, 0.50, 1.00]
FIG_PNG = "parte3_curva_aprendizaje.png"
CSV_SALIDA = "parte3_curva_aprendizaje.csv"
RUTA_RESULTADOS = "resultados.json"
NOMBRES_MODELOS = ["Naive Bayes gaussiano", "Regresión logística"]


def nuevos_modelos():
    """Instancias frescas de los dos modelos de la Parte 2 (mismos hiperparámetros)."""
    return {
        "Naive Bayes gaussiano": GaussianNB(),
        "Regresión logística": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    }


# ---------------------------------------------------------------------------
# 1) Misma división de la Parte 1 (70/30 estratificada, random_state=42)
# ---------------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
n_train_total = int(X_train.shape[0])
n_prueba = int(X_test.shape[0])

verif_t1 = None
ruta_t1 = Path("entrada/T1/resultados.json")
if ruta_t1.exists():
    try:
        t1 = json.loads(ruta_t1.read_text(encoding="utf-8"))
        verif_t1 = {
            "archivo_referencia": str(ruta_t1),
            "coincide_n_entrenamiento": n_train_total == int(t1.get("n_entrenamiento", -1)),
            "coincide_n_prueba": n_prueba == int(t1.get("n_prueba", -1)),
            "coincide_conteo_clases_entrenamiento": bool(
                int((y_train == 0).sum()) == int(t1["conteo_clases_entrenamiento"]["malignant"])
                and int((y_train == 1).sum()) == int(t1["conteo_clases_entrenamiento"]["benign"])
            ),
            "coincide_conteo_clases_prueba": bool(
                int((y_test == 0).sum()) == int(t1["conteo_clases_prueba"]["malignant"])
                and int((y_test == 1).sum()) == int(t1["conteo_clases_prueba"]["benign"])
            ),
        }
    except Exception:
        verif_t1 = {"archivo_referencia": str(ruta_t1), "error_lectura": True}

# ---------------------------------------------------------------------------
# 2) Estandarización: el escalador se ajusta SOLO con el entrenamiento (Parte 2)
# ---------------------------------------------------------------------------
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)  # misma prueba fija para TODOS los puntos

# ---------------------------------------------------------------------------
# 3) Curva de aprendizaje: submuestras estratificadas + entrenamiento + prueba fija
# ---------------------------------------------------------------------------
filas = []
for frac in FRACCIONES:
    if frac >= 1.0:
        idx = np.arange(n_train_total)
    else:
        idx, _ = train_test_split(
            np.arange(n_train_total),
            train_size=frac,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )
        idx = np.sort(idx)
    X_sub, y_sub = X_train_s[idx], y_train[idx]
    n_sub = int(len(idx))
    composicion = {"malignant": int((y_sub == 0).sum()), "benign": int((y_sub == 1).sum())}
    for nombre in NOMBRES_MODELOS:
        modelo = nuevos_modelos()[nombre]
        modelo.fit(X_sub, y_sub)
        y_pred = modelo.predict(X_test_s)
        acc = float(accuracy_score(y_test, y_pred))
        f1m = float(f1_score(y_test, y_pred, average="macro"))
        filas.append({
            "modelo": nombre,
            "fraccion": float(frac),
            "n_ejemplos_entrenamiento": n_sub,
            "accuracy": acc,
            "f1_macro": f1m,
            "composicion_submuestra": dict(composicion),
        })
        print(f"  fraccion={frac:5.2f}  n_train={n_sub:3d}  {nombre:22s} "
              f"accuracy={acc:.4f}  f1_macro={f1m:.4f}")

df = pd.DataFrame(filas)
df.to_csv(CSV_SALIDA, index=False, encoding="utf-8")
tamanos = sorted(int(t) for t in df["n_ejemplos_entrenamiento"].unique())

# ---------------------------------------------------------------------------
# 4) Figura PNG: exactitud vs número de ejemplos de entrenamiento (una curva por modelo)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5))
estilos = {
    "Naive Bayes gaussiano": {"color": "tab:blue", "marker": "o"},
    "Regresión logística": {"color": "tab:red", "marker": "s"},
}
for nombre in NOMBRES_MODELOS:
    sub = df[df["modelo"] == nombre].sort_values("n_ejemplos_entrenamiento")
    ax.plot(sub["n_ejemplos_entrenamiento"], sub["accuracy"],
            linestyle="-", linewidth=1.8, markersize=6, label=nombre, **estilos[nombre])
    for _, r in sub.iterrows():
        ax.annotate(f"{r['accuracy']:.3f}",
                    (r["n_ejemplos_entrenamiento"], r["accuracy"]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)

ax.set_xlabel("Número de ejemplos de entrenamiento")
ax.set_ylabel("Exactitud (accuracy) en el conjunto de prueba")
ax.set_title("Curva de aprendizaje — Breast Cancer (división 70/30 de la Parte 1, prueba fija)")
ax.set_xticks(tamanos)
ax.set_ylim(max(0.5, float(df["accuracy"].min()) - 0.06), 1.02)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG_PNG, dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
curva = {}
for nombre in NOMBRES_MODELOS:
    sub = df[df["modelo"] == nombre].sort_values("n_ejemplos_entrenamiento")
    curva[nombre] = {
        "fracciones": [float(v) for v in sub["fraccion"]],
        "n_ejemplos_entrenamiento": [int(v) for v in sub["n_ejemplos_entrenamiento"]],
        "accuracy": [float(v) for v in sub["accuracy"]],
        "f1_macro": [float(v) for v in sub["f1_macro"]],
    }

ns = curva["Naive Bayes gaussiano"]["n_ejemplos_entrenamiento"]
acc_nb = curva["Naive Bayes gaussiano"]["accuracy"]
acc_rl = curva["Regresión logística"]["accuracy"]
ganador_por_n = {}
for n, a_nb, a_rl in zip(ns, acc_nb, acc_rl):
    if a_nb > a_rl:
        ganador_por_n[str(n)] = "Naive Bayes gaussiano"
    elif a_rl > a_nb:
        ganador_por_n[str(n)] = "Regresión logística"
    else:
        ganador_por_n[str(n)] = "empate"

resultados = {
    "subtarea": "T3_curva_de_aprendizaje",
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "division_usada": {
        "origen": "misma división de la Parte 1, reproducida con test_size=0.3, random_state=42, estratificada por clase",
        "random_state": RANDOM_STATE,
        "test_size": 0.3,
        "estratificada": True,
        "n_entrenamiento": n_train_total,
        "n_prueba": n_prueba,
        "conteo_clases_prueba": {"malignant": int((y_test == 0).sum()),
                                 "benign": int((y_test == 1).sum())},
        "verificacion_con_T1": verif_t1,
    },
    "preprocesamiento": {
        "escalador": "StandardScaler",
        "ajustado_solo_con": f"entrenamiento completo de la Parte 1 ({n_train_total} ejemplos); nunca con la prueba",
        "nota": ("El escalador se ajusta una sola vez con el entrenamiento completo (igual que en la "
                 "Parte 2) y las submuestras se toman sobre los atributos ya estandarizados; el "
                 "conjunto de prueba transformado es exactamente el mismo para todos los puntos."),
    },
    "submuestreo": {
        "metodo": ("train_test_split(train_size=fraccion, stratify=y_train, random_state=42); "
                   "para 100% se usa el entrenamiento completo"),
        "estratificado": True,
        "random_state": RANDOM_STATE,
        "fracciones": [float(f) for f in FRACCIONES],
    },
    "modelos": {
        "Naive Bayes gaussiano": "sklearn.naive_bayes.GaussianNB",
        "Regresión logística": "sklearn.linear_model.LogisticRegression(max_iter=1000, random_state=42)",
    },
    "metrica": "accuracy (y f1_macro como referencia), evaluada siempre sobre el mismo conjunto de prueba de la Parte 1",
    "tamanos_entrenamiento": tamanos,
    "curva_aprendizaje": curva,
    "accuracy_nb_gaussiano_por_n": {str(n): a for n, a in zip(ns, acc_nb)},
    "accuracy_regresion_logistica_por_n": {str(n): a for n, a in zip(ns, acc_rl)},
    "modelo_ganador_por_n": ganador_por_n,
    "tabla_puntos": filas,
    "figura_png": FIG_PNG,
    "archivos_generados": [RUTA_RESULTADOS, CSV_SALIDA, FIG_PNG],
}

Path(RUTA_RESULTADOS).write_text(json.dumps(resultados, ensure_ascii=False, indent=2),
                                 encoding="utf-8")

# ---------------------------------------------------------------------------
# 6) Resumen
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("T3 — Curva de aprendizaje (evaluación sobre la prueba fija de la Parte 1)")
print("=" * 72)
print(f"División: {n_train_total} entrenamiento / {n_prueba} prueba "
      f"(random_state={RANDOM_STATE}, estratificada)")
print(f"Tamaños de entrenamiento evaluados: {tamanos}")
tabla = df.pivot(index="n_ejemplos_entrenamiento", columns="modelo",
                 values="accuracy").sort_index()
print("\nAccuracy en el conjunto de prueba:")
print(tabla.to_string(float_format=lambda v: f"{v:.4f}"))
print(f"\nFigura guardada en : {FIG_PNG}")
print(f"Tabla CSV          : {CSV_SALIDA}")
print(f"Resultados escritos: {RUTA_RESULTADOS}")
