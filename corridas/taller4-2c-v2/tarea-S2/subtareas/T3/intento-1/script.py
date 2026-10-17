import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- Cargar y validar datos (data/model_errors.csv) ---
df = pd.read_csv("data/model_errors.csv")
expected = ["store_id", "error_a", "error_b"]
if list(df.columns) != expected:
    raise ValueError(f"Columnas esperadas {expected}, encontradas {list(df.columns)}")
if df["store_id"].isna().any() or df["store_id"].duplicated().any():
    raise ValueError("store_id debe ser único y no nulo (identificador de tienda)")
for c in ["error_a", "error_b"]:
    v = pd.to_numeric(df[c], errors="coerce")
    if v.isna().any() or not np.isfinite(v.to_numpy()).all() or (v < 0).any():
        raise ValueError(f"Valores de error inválidos en la columna {c}")

# --- Tabla pareada: difference y favors ---
df["difference"] = df["error_a"] - df["error_b"]
df["favors"] = np.where(df["difference"] > 0, "B",
                np.where(df["difference"] < 0, "A", "tie"))

n = len(df)
favors_A = int((df["favors"] == "A").sum())
favors_B = int((df["favors"] == "B").sum())
ties = int((df["favors"] == "tie").sum())

pos = df[df["difference"] > 0]
neg = df[df["difference"] < 0]
max_B = pos.loc[pos["difference"].idxmax()] if len(pos) else None
max_A = neg.loc[neg["difference"].idxmin()] if len(neg) else None

# --- Figura: una observación por tienda, signo y magnitud, referencia en cero ---
Path("output").mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 7))
colors = np.where(df["difference"] > 0, "#1f77b4",
          np.where(df["difference"] < 0, "#d62728", "#7f7f7f"))
ax.barh(df["store_id"], df["difference"], color=colors, edgecolor="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=1.8, linestyle="--", label="Referencia en cero")
ax.set_xlabel("difference = error_a − error_b")
ax.set_ylabel("store_id")
ax.set_title("Diferencia de error por tienda (signo y magnitud), n=16 tiendas")
ax.legend(handles=[
    Patch(facecolor="#1f77b4", edgecolor="black", label="Favorece a B (difference > 0)"),
    Patch(facecolor="#d62728", edgecolor="black", label="Favorece a A (difference < 0)"),
    Patch(facecolor="#7f7f7f", edgecolor="black", label="Empate (difference = 0)"),
    plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="Referencia en cero"),
], loc="lower right", fontsize=8)
for y, d in enumerate(df["difference"]):
    ax.text(d + (0.05 if d >= 0 else -0.05), y, f"{d:+.2f}",
            va="center", ha="left" if d >= 0 else "right", fontsize=7)
plt.tight_layout()
plt.savefig("output/store_differences.png", dpi=120)
plt.close(fig)

# --- CSV exigido ---
df.to_csv("output/store_differences.csv", index=False)

# --- resultados.json (contrato) ---
results = {
    "n_stores": int(n),
    "figure": "output/store_differences.png",
    "csv": "output/store_differences.csv",
    "zero_reference_included": True,
    "observations_in_figure": int(n),
    "favors_A_count": favors_A,
    "favors_B_count": favors_B,
    "ties_count": ties,
    "max_difference_favoring_B": (
        {"store_id": str(max_B["store_id"]), "difference": float(max_B["difference"])}
        if max_B is not None else None),
    "max_difference_favoring_A": (
        {"store_id": str(max_A["store_id"]), "difference": float(max_A["difference"])}
        if max_A is not None else None),
    "table": df.to_dict(orient="records"),
}
with open("resultados.json", "w") as f:
    json.dump(results, f, indent=2)

print("T3 completada.")
print(f"Figura: output/store_differences.png ({n} observaciones, referencia en cero incluida)")
print(f"CSV: output/store_differences.csv")
print(f"Favorece a A: {favors_A} | Favorece a B: {favors_B} | Empates: {ties}")
if max_B is not None:
    print(f"Mayor magnitud a favor de B: {max_B['store_id']} (difference={max_B['difference']:+.4f})")
if max_A is not None:
    print(f"Mayor magnitud a favor de A: {max_A['store_id']} (difference={max_A['difference']:+.4f})")
