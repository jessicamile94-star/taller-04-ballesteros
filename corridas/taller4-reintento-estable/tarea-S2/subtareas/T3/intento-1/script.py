import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- rutas
DATA_PATH = Path("data/model_errors.csv")
OUT_DIR = Path("output")
PNG_PATH = OUT_DIR / "store_differences.png"
CSV_PATH = OUT_DIR / "store_differences.csv"
RESULTS_PATH = Path("resultados.json")
T2_RESULTS = Path("entrada/T2/resultados.json")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------- carga y validaciones mínimas
EXPECTED_COLS = ["store_id", "error_a", "error_b"]
df = pd.read_csv(DATA_PATH)

if len(df.columns) != 3 or sorted(map(str, df.columns)) != sorted(EXPECTED_COLS):
    raise ValueError(
        f"El CSV debe tener exactamente las columnas {EXPECTED_COLS}; encontró {list(df.columns)}"
    )

if df["store_id"].isna().any():
    raise ValueError("store_id contiene valores nulos: no se puede validar el pareo por tienda")
if df["store_id"].duplicated().any():
    dups = sorted(set(df.loc[df["store_id"].duplicated(), "store_id"].astype(str)))
    raise ValueError(f"store_id duplicado (identificador de tienda repetido): {dups}")

for col in ("error_a", "error_b"):
    v = pd.to_numeric(df[col], errors="coerce")
    if v.isna().any():
        raise ValueError(f"La columna {col} contiene valores no numéricos o nulos")
    v = v.astype(float)
    if not np.isfinite(v.to_numpy()).all():
        raise ValueError(f"La columna {col} contiene valores no finitos")
    if (v < 0).any():
        raise ValueError(f"La columna {col} contiene errores negativos")
    df[col] = v

df["store_id"] = df["store_id"].astype(str)

# --------------------------------------------------------- tabla pareada
df["difference"] = df["error_a"] - df["error_b"]
df["favors"] = np.where(df["difference"] > 0, "B",
                        np.where(df["difference"] < 0, "A", "empate"))

# --------------------------------------------------------- conteos y extremos
n_stores = int(len(df))
n_A = int((df["favors"] == "A").sum())
n_B = int((df["favors"] == "B").sum())
n_ties = int((df["favors"] == "empate").sum())

ids_A = df.loc[df["favors"] == "A", "store_id"].tolist()
ids_B = df.loc[df["favors"] == "B", "store_id"].tolist()
ids_ties = df.loc[df["favors"] == "empate", "store_id"].tolist()

i_maxB = int(df["difference"].idxmax())   # diferencia más positiva -> favorece a B
i_maxA = int(df["difference"].idxmin())   # diferencia más negativa -> favorece a A
max_B = {"store_id": df.loc[i_maxB, "store_id"],
         "difference": float(df.loc[i_maxB, "difference"]),
         "magnitud": float(abs(df.loc[i_maxB, "difference"]))}
max_A = {"store_id": df.loc[i_maxA, "store_id"],
         "difference": float(df.loc[i_maxA, "difference"]),
         "magnitud": float(abs(df.loc[i_maxA, "difference"]))}

# --------------------------------------------------------- figura
COLOR_B, COLOR_A, COLOR_TIE = "#d62728", "#1f77b4", "#7f7f7f"
colors = [COLOR_B if f == "B" else (COLOR_A if f == "A" else COLOR_TIE) for f in df["favors"]]

y_pos = np.arange(n_stores)
diffs = df["difference"].to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(9.5, 7.5))
ax.barh(y_pos, diffs, color=colors, edgecolor="black", linewidth=0.6, height=0.62)
ax.set_yticks(y_pos)

ytick_labels = []
for sid in df["store_id"]:
    lbl = sid
    if sid == max_B["store_id"] and max_B["difference"] > 0:
        lbl += " ★"
    if sid == max_A["store_id"] and max_A["difference"] < 0:
        lbl += " ★"
    ytick_labels.append(lbl)
ax.set_yticklabels(ytick_labels)
ax.invert_yaxis()  # S01 arriba, S16 abajo

# referencia visible en cero
ax.axvline(0, color="black", linewidth=1.6, zorder=3)

# etiquetas con signo y magnitud de cada diferencia
for yi, d in zip(y_pos, diffs):
    if d > 0:
        ax.text(d + 0.05, yi, f"+{d:.2f}", va="center", ha="left", fontsize=8)
    elif d < 0:
        ax.text(d - 0.05, yi, f"{d:.2f}", va="center", ha="right", fontsize=8)
    else:
        ax.text(0.06, yi, "0.00 (empate)", va="center", ha="left", fontsize=8)

pad = 1.0
ax.set_xlim(float(min(-1.0, diffs.min() - pad)), float(max(1.0, diffs.max() + pad)))
ax.set_xlabel("difference = error_a − error_b (signo y magnitud)")
ax.set_ylabel("store_id (una observación por tienda)")
ax.set_title(
    "Diferencia de error por tienda: signo y magnitud de difference\n"
    f"Favorece a A: {n_A} tiendas | Favorece a B: {n_B} tiendas | Empates: {n_ties}",
    fontsize=11,
)
ax.grid(axis="x", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

handles = [
    Patch(facecolor=COLOR_B, edgecolor="black", label="difference > 0 → favorece a B"),
    Patch(facecolor=COLOR_A, edgecolor="black", label="difference < 0 → favorece a A"),
    Patch(facecolor=COLOR_TIE, edgecolor="black", label="difference = 0 → empate"),
    Line2D([0], [0], linestyle="none", marker="*", color="black", markersize=10,
           label="★ mayor |difference| en su dirección"),
]
ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.95)

fig.tight_layout()
fig.savefig(PNG_PATH, dpi=120)
plt.close(fig)

if not PNG_PATH.exists():
    raise RuntimeError(f"No se pudo guardar la figura en {PNG_PATH}")

# --------------------------------------------------------- CSV exigido por el enunciado
df.to_csv(CSV_PATH, index=False, encoding="utf-8")

# --------------------------------------------------------- resultados.json
tabla = [
    {"store_id": str(r.store_id),
     "error_a": float(r.error_a),
     "error_b": float(r.error_b),
     "difference": float(r.difference),
     "favors": str(r.favors)}
    for r in df.itertuples(index=False)
]

results = {
    "subtarea": "T3_figura_diferencias_por_tienda",
    "figura_png": str(PNG_PATH),
    "csv_salida": str(CSV_PATH),
    "n_tiendas": n_stores,
    "tiendas_favorecen_A": n_A,
    "tiendas_favorecen_B": n_B,
    "empates": n_ties,
    "hay_empates": bool(n_ties > 0),
    "ids_favorecen_A": ids_A,
    "ids_favorecen_B": ids_B,
    "ids_empates": ids_ties,
    "mayor_diferencia_favors_B": max_B,
    "mayor_diferencia_favors_A": max_A,
    "suma_diferencias": float(df["difference"].sum()),
    "media_diferencias": float(df["difference"].mean()),
    "figura_descripcion": {
        "tipo": "barras horizontales, una por tienda",
        "muestra_16_tiendas": bool(n_stores == 16),
        "linea_cero_visible": True,
        "signo_y_magnitud": "longitud de la barra desde el cero; etiqueta numérica con signo en cada barra",
        "convencion": "difference > 0 (derecha, rojo) favorece a B; difference < 0 (izquierda, azul) favorece a A",
    },
    "tabla_pareada": tabla,
}

# verificación suave de consistencia con T2 (si está disponible)
if T2_RESULTS.exists():
    try:
        t2 = json.loads(T2_RESULTS.read_text(encoding="utf-8"))
        results["consistencia_con_T2"] = {
            "tiendas_favorecen_A": t2.get("tiendas_favorecen_A") == n_A,
            "tiendas_favorecen_B": t2.get("tiendas_favorecen_B") == n_B,
            "empates": t2.get("empates") == n_ties,
            "mayor_favors_B_store": (t2.get("mayor_diferencia_favors_B") or {}).get("store_id") == max_B["store_id"],
            "mayor_favors_A_store": (t2.get("mayor_diferencia_favors_A") or {}).get("store_id") == max_A["store_id"],
        }
    except Exception:
        results["consistencia_con_T2"] = None

RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

# --------------------------------------------------------- resumen
print("T3 completada.")
print(f"Figura guardada en: {PNG_PATH} (dpi=120, {n_stores} tiendas, línea de cero visible)")
print(f"CSV pareado guardado en: {CSV_PATH}")
print(f"Tiendas que favorecen a A: {n_A} -> {', '.join(ids_A) if ids_A else '-'}")
print(f"Tiendas que favorecen a B: {n_B} -> {', '.join(ids_B) if ids_B else '-'}")
print(f"Empates (difference = 0): {n_ties}" + (f" -> {', '.join(ids_ties)}" if ids_ties else ""))
print(f"Mayor diferencia a favor de B: {max_B['store_id']} ({max_B['difference']:+.4f})")
print(f"Mayor diferencia a favor de A: {max_A['store_id']} ({max_A['difference']:+.4f})")
