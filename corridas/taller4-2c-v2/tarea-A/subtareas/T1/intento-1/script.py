from pathlib import Path
import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target
class_names = list(data.target_names)

n_total = int(X.shape[0])
n_features = int(X.shape[1])
class_counts = {class_names[int(c)]: int(np.sum(y == c)) for c in np.unique(y)}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

train_counts = {class_names[int(c)]: int(np.sum(y_train == c)) for c in np.unique(y_train)}
test_counts = {class_names[int(c)]: int(np.sum(y_test == c)) for c in np.unique(y_test)}

# Verificación de estratificación: proporciones de clase en train y test
props_train = {k: v / len(y_train) for k, v in train_counts.items()}
props_test = {k: v / len(y_test) for k, v in test_counts.items()}
props_total = {k: v / n_total for k, v in class_counts.items()}

resultados = {
    "subtarea": "T1",
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "total_ejemplos": n_total,
    "numero_atributos": n_features,
    "conteo_por_clase": class_counts,
    "proporcion_por_clase_total": props_total,
    "tamano_entrenamiento": int(len(y_train)),
    "tamano_prueba": int(len(y_test)),
    "fraccion_entrenamiento": len(y_train) / n_total,
    "fraccion_prueba": len(y_test) / n_total,
    "conteo_por_clase_entrenamiento": train_counts,
    "conteo_por_clase_prueba": test_counts,
    "proporcion_por_clase_entrenamiento": props_train,
    "proporcion_por_clase_prueba": props_test,
    "estratificacion_verificada": bool(
        all(abs(props_train[k] - props_total[k]) < 0.02 for k in class_counts)
        and all(abs(props_test[k] - props_total[k]) < 0.02 for k in class_counts)
    ),
    "random_state": 42,
    "test_size": 0.30,
    "feature_names": list(data.feature_names),
}

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

# División única disponible para subtareas siguientes
np.savez_compressed(
    "split_t1.npz",
    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    feature_names=np.array(data.feature_names, dtype=object),
    target_names=np.array(class_names, dtype=object),
)

print("Total ejemplos:", n_total)
print("Atributos:", n_features)
print("Casos por clase:", class_counts)
print("Entrenamiento (70%):", len(y_train), train_counts)
print("Prueba (30%):", len(y_test), test_counts)
print("Estratificación verificada:", resultados["estratificacion_verificada"])
