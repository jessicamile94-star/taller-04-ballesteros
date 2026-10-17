# -*- coding: utf-8 -*-
"""
T4 — Efecto e incertidumbre sobre el vector de diferencias pareadas.

Sobre las diferencias pareadas difference = error_a - error_b (emparejadas por store_id):
  * reporta exactamente n, mean, sample_sd (desviación MUESTRAL, ddof=1) y
    standard_error = sample_sd / sqrt(n);
  * con scipy.stats calcula la prueba t de una muestra bilateral sobre las
    diferencias y el intervalo t bilateral del 95 % para la media pareada,
    conservando grados de libertad, estadístico, p-value e intervalo.

Salidas (contrato):
  - resultados.json
  - output/store_differences.csv  (ruta exigida por el enunciado)
  - output/store_differences.png  (ruta exigida por el enunciado)
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1) Vector de diferencias: resultados previos (entrada/T2/) -> datos -> enunciado
# ---------------------------------------------------------------------------
CANDIDATOS_PREVIOS = [
    Path("entrada/T2/output/store_differences.csv"),
    Path("entrada/T2/store_differences.csv"),
]
RUTA_DATOS = Path("data/model_errors.csv")

# Respaldo de última instancia: tabla del enunciado (store_id, error_a, error_b)
TABLA_ENUNCIADO = [
    ("S01", 12.4, 10.6), ("S02", 9.8, 9.1), ("S03", 15.1, 13.9), ("S04", 11.0, 12.4),
    ("S05", 8.7, 7.2), ("S06", 13.6, 13.9), ("S07", 10.2, 9.3), ("S08", 14.9, 12.8),
    ("S09", 7.5, 9.3), ("S10", 16.0, 15.0), ("S11", 9.3, 9.8), ("S12", 12.8, 11.1),
    ("S13", 11.7, 12.9), ("S14", 8.9, 8.3), ("S15", 13.2, 11.9), ("S16", 10.6, 11.5),
]


def _con_favors(diff: np.ndarray) -> np.ndarray:
    return np.where(diff > 0, "B", np.where(diff < 0, "A", "empate"))


def tabla_desde_csv_pareado(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    req = {"store_id", "error_a", "error_b", "difference"}
    if not req.issubset(df.columns):
        raise ValueError(f"{ruta} no contiene las columnas {sorted(req)}")
    out = pd.DataFrame({
        "store_id": df["store_id"].astype(str),
        "error_a": pd.to_numeric(df["error_a"], errors="raise").astype(float),
        "error_b": pd.to_numeric(df["error_b"], errors="raise").astype(float),
        "difference": pd.to_numeric(df["difference"], errors="raise").astype(float),
    })
    out["favors"] = (df["favors"].astype(str) if "favors" in df.columns
                     else _con_favors(out["difference"].to_numpy(dtype=float)))
    return out


def tabla_desde_datos(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    faltan = {"store_id", "error_a", "error_b"} - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas {sorted(faltan)} en {ruta}")
    a = pd.to_numeric(df["error_a"], errors="raise").to_numpy(dtype=float)
    b = pd.to_numeric(df["error_b"], errors="raise").to_numpy(dtype=float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Los errores deben ser numéricos y finitos")
    diff = a - b
    return pd.DataFrame({
        "store_id": df["store_id"].astype(str),
        "error_a": a, "error_b": b, "difference": diff, "favors": _con_favors(diff),
    })


def tabla_desde_enunciado() -> pd.DataFrame:
    filas = [(s, float(a), float(b), float(a) - float(b)) for s, a, b in TABLA_ENUNCIADO]
    df = pd.DataFrame(filas, columns=["store_id", "error_a", "error_b", "difference"])
    df["favors"] = _con_favors(df["difference"].to_numpy(dtype=float))
    return df


tabla, fuente = None, None
for cand in CANDIDATOS_PREVIOS:
    if cand.is_file():
        try:
            tabla, fuente = tabla_desde_csv_pareado(cand), str(cand)
            break
        except Exception:
            tabla = None
if tabla is None and RUTA_DATOS.is_file():
    tabla = tabla_desde_datos(RUTA_DATOS)
    fuente = f"{RUTA_DATOS} (difference = error_a - error_b recalculada)"
if tabla is None:
    tabla, fuente = tabla_desde_enunciado(), "tabla del enunciado (respaldo)"

diffs = tabla["difference"].to_numpy(dtype=float)
if not np.isfinite(diffs).all():
    raise ValueError("El vector de diferencias contiene valores no finitos")
if tabla["store_id"].duplicated().any():
    raise ValueError("store_id duplicado en la tabla pareada")
n = int(diffs.size)
if n < 2:
    raise ValueError("Se necesitan al menos 2 diferencias para la prueba t")

# ---------------------------------------------------------------------------
# 2) Efecto: n, mean, sample_sd (muestral) y SE = sd / sqrt(n)
# ---------------------------------------------------------------------------
mean = float(np.mean(diffs))
sample_sd = float(np.std(diffs, ddof=1))            # desviación MUESTRAL (ddof=1)
standard_error = float(sample_sd / math.sqrt(n))    # SE = sd / sqrt(n)
suma_diferencias = float(np.sum(diffs))
gl = n - 1

# ---------------------------------------------------------------------------
# 3) Prueba t de una muestra bilateral (scipy.stats) sobre las diferencias
# ---------------------------------------------------------------------------
try:
    t_res = stats.ttest_1samp(diffs, popmean=0.0, alternative="two-sided")
except TypeError:  # compatibilidad scipy antiguo: bilateral es el valor por defecto
    t_res = stats.ttest_1samp(diffs, popmean=0.0)
t_stat = float(t_res.statistic)
p_value = float(t_res.pvalue)

# ---------------------------------------------------------------------------
# 4) Intervalo t bilateral del 95 % para la media pareada
# ---------------------------------------------------------------------------
NIVEL = 0.95
t_critico = float(stats.t.ppf(1.0 - (1.0 - NIVEL) / 2.0, gl))  # t_{0.975, gl}
try:
    ci_low, ci_high = stats.t.interval(confidence=NIVEL, df=gl, loc=mean, scale=standard_error)
except TypeError:
    ci_low, ci_high = stats.t.interval(NIVEL, df=gl, loc=mean, scale=standard_error)
ci_low, ci_high = float(ci_low), float(ci_high)
margen = float(t_critico * standard_error)

# --- verificaciones internas -------------------------------------------------
se_scipy = float(stats.sem(diffs))                  # ddof=1 por defecto
p_manual = float(2.0 * stats.t.sf(abs(t_stat), gl))
t_rel = None
if {"error_a", "error_b"}.issubset(tabla.columns):
    t_rel = float(stats.ttest_rel(tabla["error_a"].to_numpy(dtype=float),
                                  tabla["error_b"].to_numpy(dtype=float)).statistic)

verificaciones = {
    "n_es_16": bool(n == 16),
    "se_igual_sd_sobre_sqrt_n": bool(abs(standard_error - sample_sd / math.sqrt(n)) <= 1e-12),
    "se_coincide_con_scipy_sem": bool(abs(standard_error - se_scipy) <= 1e-12),
    "p_value_coincide_con_2_sf": bool(abs(p_value - p_manual) <= 1e-12),
    "ic_coincide_con_mean_pm_tcrit_se": bool(
        abs(ci_low - (mean - margen)) <= 1e-10 and abs(ci_high - (mean + margen)) <= 1e-10),
    "ttest_rel_mismo_estadistico": (None if t_rel is None else bool(abs(t_rel - t_stat) <= 1e-9)),
    "ic_contiene_cero": bool(ci_low <= 0.0 <= ci_high),
}

# ---------------------------------------------------------------------------
# 5) Interpretación (unidades; dispersión entre tiendas vs error estándar)
# ---------------------------------------------------------------------------
n_pos = int((diffs > 0).sum())
n_neg = int((diffs < 0).sum())
decision = ("no se rechaza H0: media de las diferencias = 0 (nivel 0.05)" if p_value > 0.05
            else "se rechaza H0: media de las diferencias = 0 (nivel 0.05)")
interpretacion = {
    "media": (
        f"La diferencia media es {mean:.4f} unidades por tienda (n={n}): en promedio error_a "
        f"excede a error_b en {abs(mean):.4f} unidades; el signo positivo indica un efecto "
        f"medio que favorece a B ({n_pos} de {n} tiendas con diferencia positiva, {n_neg} negativa)."
    ),
    "dispersion_vs_se": (
        f"La desviación muestral ({sample_sd:.4f}) describe la dispersión de las diferencias "
        f"ENTRE tiendas; el error estándar ({standard_error:.4f}) = sd/√n describe la "
        f"incertidumbre de la MEDIA: la dispersión entre tiendas es √n ≈ {math.sqrt(n):.2f} "
        f"veces el error estándar."
    ),
    "ic_valores_compatibles": (
        f"El IC t del 95 % [{ci_low:.4f}, {ci_high:.4f}] unidades contiene 0: bajo el modelo t, "
        f"efectos medios entre ~{ci_low:.3f} y ~{ci_high:.3f} unidades son compatibles con los "
        f"datos. El intervalo es un rango de valores plausibles para la media tras los datos, "
        f"no una probabilidad posterior sobre el parámetro."
    ),
    "decision_prueba_t": f"t({gl}) = {t_stat:.4f}, p bilateral = {p_value:.4f} → {decision}.",
}

# ---------------------------------------------------------------------------
# 6) Archivos exigidos por el enunciado: output/store_differences.{csv,png}
# ---------------------------------------------------------------------------
out_dir = Path("output")
out_dir.mkdir(parents=True, exist_ok=True)

tabla_out = tabla[["store_id", "error_a", "error_b", "difference", "favors"]]
tabla_out.to_csv(out_dir / "store_differences.csv", index=False)

registros = [
    {"store_id": str(r.store_id), "error_a": float(r.error_a), "error_b": float(r.error_b),
     "difference": float(r.difference), "favors": str(r.favors)}
    for r in tabla_out.itertuples(index=False)
]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), gridspec_kw={"width_ratios": [2.3, 1.0]})
ax = axes[0]
colores = ["#2a9d8f" if d > 0 else "#e76f51" for d in diffs]
ax.bar(tabla_out["store_id"], diffs, color=colores, edgecolor="black", linewidth=0.5)
ax.axhline(0.0, color="black", lw=1.0)
ax.axhline(mean, color="#264653", ls="--", lw=1.6, label=f"media = {mean:.4f}")
ax.axhspan(ci_low, ci_high, color="#264653", alpha=0.12,
           label=f"IC 95% t = [{ci_low:.4f}, {ci_high:.4f}]")
ax.set_ylim(min(diffs.min(), ci_low) - 0.9, max(diffs.max(), ci_high) + 0.9)
ax.set_ylabel("Diferencia pareada (error_a − error_b)")
ax.set_title("Diferencias por tienda")
ax.tick_params(axis="x", rotation=90)
ax.grid(axis="y", alpha=0.25)
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

ax2 = axes[1]
ax2.errorbar([0], [mean], yerr=[[mean - ci_low], [ci_high - mean]], fmt="o",
             color="#264653", elinewidth=2.2, capsize=9, markersize=9)
ax2.axhline(0.0, color="black", lw=1.0)
ax2.set_xlim(-0.8, 0.8)
ax2.set_ylim(min(ci_low, diffs.min()) - 0.5, max(ci_high, diffs.max()) + 0.5)
ax2.set_xticks([0])
ax2.set_xticklabels(["media pareada"])
ax2.set_ylabel("Unidades")
ax2.set_title("Efecto e incertidumbre (IC t 95 %)")
ax2.grid(axis="y", alpha=0.25)

linea1 = (f"n = {n}    mean = {mean:.6f}    sample_sd = {sample_sd:.6f}    "
          f"SE = sd/√n = {standard_error:.6f}")
linea2 = (f"gl = {gl}    t = {t_stat:.6f}    p (bilateral) = {p_value:.6f}    "
          f"IC 95% t = [{ci_low:.6f}, {ci_high:.6f}]")
fig.text(0.5, 0.012, linea1 + "\n" + linea2, ha="center", va="bottom", fontsize=9.5,
         bbox=dict(boxstyle="round,pad=0.45", fc="#f8f9fa", ec="gray"))
fig.suptitle("T4 · Efecto pareado y su incertidumbre (verde: favorece B · rojo: favorece A)", y=0.98)
fig.tight_layout(rect=[0, 0.13, 1, 0.93])
fig.savefig(out_dir / "store_differences.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 7) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T4_efecto_e_incertidumbre",
    "fuente_diferencias": fuente,
    "definicion": "difference = error_a - error_b, pareada por store_id",
    "n": n,
    "mean": mean,
    "sample_sd": sample_sd,
    "standard_error": standard_error,
    "standard_error_formula": "SE = sample_sd / sqrt(n)",
    "suma_diferencias": suma_diferencias,
    "grados_de_libertad": gl,
    "df": gl,
    "t_statistic": t_stat,
    "p_value": p_value,
    "ci95_low": ci_low,
    "ci95_high": ci_high,
    "prueba_t": {
        "metodo": "scipy.stats.ttest_1samp (una muestra sobre las diferencias)",
        "popmean": 0.0,
        "alternativa": "two-sided",
        "estadistico_t": t_stat,
        "grados_de_libertad": gl,
        "p_value_bilateral": p_value,
        "estadistico_t_pareada_ttest_rel": t_rel,
    },
    "intervalo_confianza": {
        "nivel": NIVEL,
        "tipo": "t bilateral para la media pareada (scipy.stats.t.interval)",
        "limite_inferior": ci_low,
        "limite_superior": ci_high,
        "t_critico": t_critico,
        "margen_de_error": margen,
    },
    "conteos": {
        "diferencia_positiva_favorece_B": n_pos,
        "diferencia_negativa_favorece_A": n_neg,
        "empates": int((diffs == 0).sum()),
    },
    "verificaciones": verificaciones,
    "tabla_diferencias": registros,
    "interpretacion": interpretacion,
    "salidas": {
        "csv": "output/store_differences.csv",
        "figura": "output/store_differences.png",
    },
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 8) Resumen
# ---------------------------------------------------------------------------
print("=" * 74)
print("T4 · Efecto e incertidumbre sobre las diferencias pareadas")
print(f"Fuente de las diferencias : {fuente}")
print(f"n = {n} | mean = {mean:.6f} | sample_sd = {sample_sd:.6f} "
      f"| SE = sd/sqrt(n) = {standard_error:.6f}")
print(f"Prueba t de una muestra (bilateral, popmean=0): t({gl}) = {t_stat:.6f} | p = {p_value:.6f}")
print(f"IC t 95% media pareada: [{ci_low:.6f}, {ci_high:.6f}] "
      f"(t_crit = {t_critico:.6f}, margen = {margen:.6f})")
print(f"Verificaciones: {verificaciones}")
print("Archivos escritos: resultados.json, output/store_differences.csv, "
      "output/store_differences.png")
print("=" * 74)
