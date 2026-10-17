# -*- coding: utf-8 -*-
"""
T3 — Hacer visible la evidencia por tienda.

- Carga la tabla pareada producida por T2 (entrada/T2/) y verifica su
  consistencia con data/model_errors.csv; si no estuviera disponible, la
  reconstruye desde el CSV original con las mismas validaciones de T2.
- Genera output/store_differences.png: una barra horizontal por tienda con el
  signo y la magnitud de difference y una línea de referencia en cero.
- Genera output/store_differences.csv con la tabla pareada.
- Cuenta cuántas tiendas favorecen a A, cuántas a B y si hay empates, e
  identifica las diferencias de mayor magnitud en cada dirección.
- Escribe todas las cifras en resultados.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

TOL = 1e-12  # |difference| <= TOL se considera empate (difference = 0)
RUTA_FIGURA = Path("output/store_differences.png")
RUTA_CSV = Path("output/store_differences.csv")
COLUMNAS = ["store_id", "error_a", "error_b", "difference", "favors"]


# ----------------------------------------------------------------------
# 1) Obtención de la tabla pareada (producto de T2) con respaldos
# ----------------------------------------------------------------------
def _validar_y_reconstruir(raw: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye la tabla pareada desde data/model_errors.csv con las validaciones de T2."""
    columnas_orig = ["store_id", "error_a", "error_b"]
    if list(raw.columns) != columnas_orig:
        raise ValueError(
            f"Se esperaban exactamente las columnas {columnas_orig}; se obtuvo {list(raw.columns)}"
        )
    if raw["store_id"].isna().any() or raw["store_id"].duplicated().any():
        raise ValueError("store_id debe ser único y no nulo (identificador de la tienda)")
    tabla = raw[columnas_orig].copy()
    for col in ("error_a", "error_b"):
        valores = pd.to_numeric(tabla[col], errors="coerce").astype(float)
        if not np.isfinite(valores).all() or (valores < 0).any():
            raise ValueError(f"Los valores de {col} deben ser numéricos, finitos y no negativos")
        tabla[col] = valores
    tabla["difference"] = tabla["error_a"] - tabla["error_b"]
    tabla["favors"] = np.where(tabla["difference"] > TOL, "B",
                               np.where(tabla["difference"] < -TOL, "A", "tie"))
    return tabla.sort_values("store_id").reset_index(drop=True)


def _favors_con_tolerancia(tabla: pd.DataFrame) -> pd.DataFrame:
    """Recalcula favors de forma consistente con la definición de T2 (tolerancia numérica)."""
    tabla = tabla.copy()
    tabla["favors"] = np.where(tabla["difference"] > TOL, "B",
                               np.where(tabla["difference"] < -TOL, "A", "tie"))
    return tabla.sort_values("store_id").reset_index(drop=True)


def cargar_tabla_pareada():
    """Prioridad: CSV de T2 -> resultados.json de T2 -> reconstrucción desde el CSV original."""
    # a) CSV de la tabla pareada generado por T2 (búsqueda recursiva en entrada/T2)
    for candidato in sorted(Path("entrada/T2").rglob("store_differences.csv")):
        try:
            df = pd.read_csv(candidato)
            if set(COLUMNAS) <= set(df.columns):
                return (_favors_con_tolerancia(df[COLUMNAS]),
                        f"tabla pareada de T2: {candidato}")
        except Exception:
            continue
    # b) resultados.json de T2 (clave tabla_pareada.filas)
    try:
        rj = Path("entrada/T2/resultados.json")
        if rj.is_file():
            with rj.open("r", encoding="utf-8") as f:
                res = json.load(f)
            filas = res["tabla_pareada"]["filas"]
            df = pd.DataFrame(filas)[COLUMNAS]
            return (_favors_con_tolerancia(df),
                    "tabla pareada de T2: entrada/T2/resultados.json (tabla_pareada.filas)")
    except Exception:
        pass
    # c) Reconstrucción desde el CSV original
    return (_validar_y_reconstruir(pd.read_csv("data/model_errors.csv")),
            "reconstruida desde data/model_errors.csv con las validaciones de T2")


# ----------------------------------------------------------------------
# 2) Carga y verificación de consistencia
# ----------------------------------------------------------------------
tabla, fuente = cargar_tabla_pareada()

verificacion = {
    "csv_original_presente": Path("data/model_errors.csv").is_file(),
    "consistente_con_csv_original": None,
}
ruta_src = Path("data/model_errors.csv")
if ruta_src.is_file():
    try:
        raw = pd.read_csv(ruta_src)
        if {"store_id", "error_a", "error_b"} <= set(raw.columns):
            m = tabla.merge(raw[["store_id", "error_a", "error_b"]], on="store_id",
                            how="outer", suffixes=("", "_src"), indicator=True)
            ok = bool((m["_merge"] == "both").all()) and len(m) == len(tabla) == len(raw)
            if ok:
                ok = bool(
                    np.allclose(m["error_a"].astype(float), m["error_a_src"].astype(float), atol=1e-9)
                    and np.allclose(m["error_b"].astype(float), m["error_b_src"].astype(float), atol=1e-9)
                )
            verificacion["consistente_con_csv_original"] = ok
            if not ok:
                tabla = _validar_y_reconstruir(raw)
                fuente = "reconstruida desde data/model_errors.csv (inconsistencia con artefactos de T2)"
                verificacion["consistente_con_csv_original"] = True
    except Exception as exc:  # verificación informativa, no bloquea
        verificacion["consistente_con_csv_original"] = f"no verificable: {exc}"

n = len(tabla)
if n != 16:
    raise ValueError(f"La tabla pareada debe contener 16 tiendas; se obtuvieron {n}")

tabla = _favors_con_tolerancia(tabla)
diff = tabla["difference"].to_numpy(dtype=float)
stores = tabla["store_id"].astype(str).tolist()

# ----------------------------------------------------------------------
# 3) Conteos por preferencia, empates y diferencias extremas
# ----------------------------------------------------------------------
fav_A = tabla.loc[tabla["favors"] == "A", "store_id"].tolist()
fav_B = tabla.loc[tabla["favors"] == "B", "store_id"].tolist()
empates = tabla.loc[tabla["favors"] == "tie", "store_id"].tolist()

pos = tabla.loc[tabla["difference"] > TOL, "difference"]
neg = tabla.loc[tabla["difference"] < -TOL, "difference"]

max_B = None
if len(pos):
    i_B = pos.idxmax()
    max_B = {"store_id": str(tabla.loc[i_B, "store_id"]),
             "difference": float(tabla.loc[i_B, "difference"])}

max_A = None
if len(neg):
    i_A = neg.idxmin()
    max_A = {"store_id": str(tabla.loc[i_A, "store_id"]),
             "difference": float(tabla.loc[i_A, "difference"])}

abs_diff = np.abs(diff)
i_abs = int(np.argmax(abs_diff))
max_abs = {
    "store_id": stores[i_abs],
    "difference": float(diff[i_abs]),
    "magnitud": float(abs_diff[i_abs]),
    "direccion": "favorece_B" if diff[i_abs] > TOL else ("favorece_A" if diff[i_abs] < -TOL else "empate"),
}

# ----------------------------------------------------------------------
# 4) Figura: una observación por tienda, signo/magnitud y referencia en cero
# ----------------------------------------------------------------------
RUTA_FIGURA.parent.mkdir(parents=True, exist_ok=True)

col_B, col_A, col_T = "#2166ac", "#b2182b", "#8c8c8c"
favors_arr = tabla["favors"].to_numpy()
mB, mA = favors_arr == "B", favors_arr == "A"
mT = ~(mB | mA)

fig, ax = plt.subplots(figsize=(9.5, 7.0))
ypos = np.arange(n)
ax.barh(ypos[mB], diff[mB], color=col_B, edgecolor="black", linewidth=0.6)
ax.barh(ypos[mA], diff[mA], color=col_A, edgecolor="black", linewidth=0.6)
if mT.any():
    ax.barh(ypos[mT], diff[mT], color=col_T, edgecolor="black", linewidth=0.6)

ax.axvline(0.0, color="black", linewidth=2.0, zorder=5)  # referencia visible en cero
ax.set_yticks(ypos)
ax.set_yticklabels(stores)
ax.tick_params(axis="y", labelsize=9)
ax.invert_yaxis()
ax.set_xlim(-2.5, 2.75)
ax.set_xlabel("difference = error_a − error_b")
ax.set_ylabel("store_id")
ax.set_title("Signo y magnitud de difference por tienda (una observación por tienda, n = 16)\n"
             "Línea negra vertical: referencia en cero")
ax.grid(axis="x", linestyle=":", alpha=0.5)
ax.set_axisbelow(True)

# Etiquetas de valor y marcas de las diferencias extremas
sufijos = {}
if max_B:
    sufijos[max_B["store_id"]] = "  ← máx B"
if max_A:
    sufijos[max_A["store_id"]] = "  ← máx A"
ids_extremos = set(sufijos)
for yi, d, sid in zip(ypos, diff, stores):
    ax.text(d + (0.06 if d >= 0 else -0.06), yi, f"{d:+.2f}" + sufijos.get(sid, ""),
            va="center", ha="left" if d >= 0 else "right", fontsize=8)
for etiqueta, sid in zip(ax.get_yticklabels(), stores):
    if sid in ids_extremos:
        etiqueta.set_fontweight("bold")

manejadores = [
    Patch(facecolor=col_B, edgecolor="black", label="Favorece a B (difference > 0)"),
    Patch(facecolor=col_A, edgecolor="black", label="Favorece a A (difference < 0)"),
]
if mT.any():
    manejadores.append(Patch(facecolor=col_T, edgecolor="black", label="Empate (difference = 0)"))
ax.legend(handles=manejadores, loc="center left", bbox_to_anchor=(1.01, 0.5),
          title="Interpretación", fontsize=9, frameon=False)

fig.tight_layout()
plt.savefig("output/store_differences.png", dpi=120, bbox_inches="tight")
plt.close(fig)
if not RUTA_FIGURA.is_file():
    raise RuntimeError("No se pudo guardar output/store_differences.png")

# ----------------------------------------------------------------------
# 5) CSV de la tabla pareada
# ----------------------------------------------------------------------
tabla_out = tabla[COLUMNAS].copy()
tabla_out.to_csv(RUTA_CSV, index=False)

# ----------------------------------------------------------------------
# 6) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj

resultados = {
    "subtarea": "T3_figura_y_conteos_diferencias_por_tienda",
    "fuente_tabla_pareada": fuente,
    "n_tiendas": int(n),
    "archivos_generados": ["output/store_differences.png", "output/store_differences.csv"],
    "figura": {
        "ruta": "output/store_differences.png",
        "descripcion": ("Barras horizontales divergentes: una barra por tienda con el signo y la "
                        "magnitud de difference; línea negra vertical en 0 como referencia; "
                        "azul = favorece a B, rojo = favorece a A, gris = empate."),
        "muestra_las_16_tiendas": True,
        "referencia_cero_visible": True,
    },
    "conteo_favors": {"A": len(fav_A), "B": len(fav_B), "tie": len(empates)},
    "n_favorecen_A": len(fav_A),
    "n_favorecen_B": len(fav_B),
    "n_empates": len(empates),
    "empates_presentes": len(empates) > 0,
    "tiendas_favorecen_A": fav_A,
    "tiendas_favorecen_B": fav_B,
    "tiendas_en_empate": empates,
    "mayor_diferencia_favorece_B": max_B,
    "mayor_diferencia_favorece_A": max_A,
    "mayor_magnitud_absoluta": max_abs,
    "estadisticas_difference": {
        "media": float(np.mean(diff)),
        "mediana": float(np.median(diff)),
        "desviacion_estandar_muestral": float(np.std(diff, ddof=1)),
        "minimo": float(np.min(diff)),
        "maximo": float(np.max(diff)),
    },
    "verificacion_contra_csv_original": verificacion,
    "tabla_pareada": tabla_out.to_dict(orient="records"),
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(_jsonable(resultados), f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 7) Resumen
# ----------------------------------------------------------------------
print("=" * 64)
print("T3 — Evidencia por tienda (difference = error_a − error_b)")
print(f"Fuente de la tabla pareada: {fuente}")
print(f"Tiendas analizadas: {n}")
print(f"Favorecen a A (difference < 0): {len(fav_A)} -> {', '.join(fav_A)}")
print(f"Favorecen a B (difference > 0): {len(fav_B)} -> {', '.join(fav_B)}")
print(f"Empates (difference = 0): {len(empates)}"
      + (f" -> {', '.join(empates)}" if empates else " (no hay empates)"))
if max_B:
    print(f"Mayor diferencia a favor de B: {max_B['store_id']} ({max_B['difference']:+.4f})")
if max_A:
    print(f"Mayor diferencia a favor de A: {max_A['store_id']} ({max_A['difference']:+.4f})")
print(f"Mayor magnitud absoluta: {max_abs['store_id']} "
      f"(|difference| = {max_abs['magnitud']:.4f}, {max_abs['direccion']})")
print(f"Verificación contra data/model_errors.csv: {verificacion['consistente_con_csv_original']}")
print(f"Figura guardada en: {RUTA_FIGURA}")
print(f"CSV guardado en: {RUTA_CSV}")
print("Cifras escritas en: resultados.json")
