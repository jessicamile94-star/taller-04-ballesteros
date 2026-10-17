# -*- coding: utf-8 -*-
"""
T6 — Bootstrap pareado: intervalo percentil del 95 % de la media de las diferencias.

- Remuestrea TIENDAS COMPLETAS mediante el vector de diferencias
  (difference = error_a - error_b); NUNCA remuestrea error_a y error_b por separado.
- Usa np.random.default_rng(SEED=20260829) y BOOTSTRAP_RESAMPLES=10_000.
- Reporta el IC percentil del 95 % como análisis de sensibilidad aproximado y
  verifica que dos ejecuciones con la misma semilla producen el mismo intervalo.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Parámetros fijos del enunciado
# ---------------------------------------------------------------------------
SEED = 20260829
BOOTSTRAP_RESAMPLES = 10_000
NIVEL = 0.95
P_INF = (100.0 - 100.0 * NIVEL) / 2.0   # 2.5
P_SUP = 100.0 - P_INF                   # 97.5

# ---------------------------------------------------------------------------
# 1) Cargar datos y construir el vector de diferencias (una fila = una tienda)
# ---------------------------------------------------------------------------
ruta_csv = Path("data/model_errors.csv")
df = pd.read_csv(ruta_csv)

columnas = ["store_id", "error_a", "error_b"]
faltantes = [c for c in columnas if c not in df.columns]
if faltantes:
    raise ValueError(f"{ruta_csv} carece de las columnas {faltantes}")

df = df[columnas].copy()
df["store_id"] = df["store_id"].astype(str)
duplicados = df.loc[df["store_id"].duplicated(), "store_id"].tolist()
if duplicados:
    raise ValueError(f"store_id duplicado: {duplicados}")

df["error_a"] = pd.to_numeric(df["error_a"], errors="raise")
df["error_b"] = pd.to_numeric(df["error_b"], errors="raise")
if not np.isfinite(df[["error_a", "error_b"]].to_numpy(dtype=float)).all():
    raise ValueError("error_a y error_b deben ser numéricos y finitos")

# Orden determinista por identificador de tienda (la unidad de remuestreo)
df = df.sort_values("store_id", kind="mergesort").reset_index(drop=True)

df["difference"] = df["error_a"] - df["error_b"]
df["favors"] = np.where(df["difference"] > 0, "B",
                        np.where(df["difference"] < 0, "A", "tie"))

difs = df["difference"].to_numpy(dtype=float)
n_tiendas = int(difs.size)

# ---------------------------------------------------------------------------
# 2) Bootstrap pareado: remuestreo de tiendas completas (filas del vector)
# ---------------------------------------------------------------------------
def bootstrap_pareado(vector_diferencias: np.ndarray, semilla: int, n_remuestras: int):
    """Remuestrea tiendas completas con reemplazo y devuelve las medias bootstrap.

    Cada remuestra extrae n filas COMPLETAS del vector de diferencias (una fila
    = una tienda con su diferencia pareada), preservando la estructura del par.
    Nunca se remuestrean error_a y error_b por separado.
    """
    rng = np.random.default_rng(semilla)  # RNG nuevo por ejecución, misma semilla
    m = vector_diferencias.size
    indices = rng.integers(0, m, size=(n_remuestras, m))  # tiendas con reemplazo
    medias = vector_diferencias[indices].mean(axis=1)
    lo, hi = np.percentile(medias, [P_INF, P_SUP])
    return medias, float(lo), float(hi)

# Dos ejecuciones independientes con la MISMA semilla (criterio de éxito)
boot_1, lo_1, hi_1 = bootstrap_pareado(difs, SEED, BOOTSTRAP_RESAMPLES)
boot_2, lo_2, hi_2 = bootstrap_pareado(difs, SEED, BOOTSTRAP_RESAMPLES)

intervalos_identicos = bool(lo_1 == lo_2 and hi_1 == hi_2)
distribuciones_identicas = bool(np.array_equal(boot_1, boot_2))

# El intervalo reportado corresponde a la primera ejecución
boot = boot_1
lo_ic, hi_ic = lo_1, hi_1

# ---------------------------------------------------------------------------
# 3) Estadísticos reportados
# ---------------------------------------------------------------------------
media_obs = float(difs.mean())
mediana_obs = float(np.median(difs))
sd_obs = float(difs.std(ddof=1))
ee_teorico = float(sd_obs / np.sqrt(n_tiendas))
ee_boot = float(boot.std(ddof=1))
sesgo_boot = float(boot.mean() - media_obs)
p50_boot = float(np.percentile(boot, 50.0))
contiene_cero = bool(lo_ic <= 0.0 <= hi_ic)

# ---------------------------------------------------------------------------
# 4) Consistencia opcional con la subtarea previa T2
# ---------------------------------------------------------------------------
ruta_t2 = Path("entrada/T2/results.json")
coincide_t2 = None
if ruta_t2.exists():
    with ruta_t2.open(encoding="utf-8") as f:
        t2 = json.load(f)
    filas_t2 = t2.get("tabla_pareada", {}).get("filas", [])
    if filas_t2:
        difs_t2 = {str(f["store_id"]): float(f["difference"]) for f in filas_t2}
        coincide_t2 = bool(
            set(difs_t2) == set(df["store_id"])
            and all(np.isclose(difs_t2[s], d, rtol=0.0, atol=1e-9)
                    for s, d in zip(df["store_id"], difs))
        )

# ---------------------------------------------------------------------------
# 5) Figuras
# ---------------------------------------------------------------------------
Path("output").mkdir(parents=True, exist_ok=True)

# Figura exigida por el enunciado: output/store_differences.png
fig, ejes = plt.subplots(1, 2, figsize=(11.5, 4.8))

ax = ejes[0]
colores = ["#2166ac" if d > 0 else "#b2182b" for d in difs]
ax.bar(np.arange(1, n_tiendas + 1), difs, color=colores,
       edgecolor="black", linewidth=0.4)
ax.axhline(0.0, color="black", lw=0.8)
ax.axhline(media_obs, color="#1b7837", ls="--", lw=1.4)
ax.set_xticks(np.arange(1, n_tiendas + 1))
ax.set_xticklabels(df["store_id"], rotation=90, fontsize=8)
ax.set_xlabel("Tienda (store_id)")
ax.set_ylabel("difference = error_a − error_b")
ax.set_title("Diferencias pareadas por tienda\n(unidad de remuestreo: tienda completa)")
ax.legend(handles=[
    Patch(color="#2166ac", label="difference > 0 (favorece a B)"),
    Patch(color="#b2182b", label="difference < 0 (favorece a A)"),
    Line2D([0], [0], color="#1b7837", ls="--", lw=1.4,
           label=f"media = {media_obs:.4f}"),
], loc="best", fontsize=8)
ax.grid(axis="y", alpha=0.25)

ax = ejes[1]
ax.hist(boot, bins=60, color="#92c5de", edgecolor="white", linewidth=0.3)
ax.axvspan(lo_ic, hi_ic, color="#f4a582", alpha=0.35,
           label=f"IC percentil 95 % = [{lo_ic:.4f}, {hi_ic:.4f}]")
ax.axvline(lo_ic, color="black", lw=1.4)
ax.axvline(hi_ic, color="black", lw=1.4)
ax.axvline(media_obs, color="#1b7837", ls="--", lw=1.6,
           label=f"media observada = {media_obs:.4f}")
ax.set_xlabel("Media bootstrap de las diferencias")
ax.set_ylabel("Frecuencia")
ax.set_title(f"Bootstrap pareado: {BOOTSTRAP_RESAMPLES} remuestras\n"
             f"de tiendas completas (semilla = {SEED})")
ax.legend(fontsize=8, loc="upper left")
ax.grid(axis="y", alpha=0.25)

fig.suptitle("T6 · Bootstrap pareado mediante el vector de diferencias", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("output/store_differences.png", dpi=120)
plt.close(fig)

# Figura adicional de la subtarea (carpeta actual): ECDF de las medias bootstrap
fig2, ax2 = plt.subplots(figsize=(7.0, 4.2))
x_ord = np.sort(boot)
ecdf = np.arange(1, x_ord.size + 1) / x_ord.size
ax2.plot(x_ord, ecdf, color="#2166ac", lw=1.6)
ax2.axvspan(lo_ic, hi_ic, color="#f4a582", alpha=0.30,
            label=f"IC percentil 95 % = [{lo_ic:.4f}, {hi_ic:.4f}]")
ax2.axvline(lo_ic, color="black", lw=1.3)
ax2.axvline(hi_ic, color="black", lw=1.3)
ax2.axvline(0.0, color="gray", ls=":", lw=1.2, label="referencia nula: media = 0")
ax2.set_xlabel("Media bootstrap de las diferencias")
ax2.set_ylabel("Probabilidad acumulada empírica")
ax2.set_title(f"ECDF de {BOOTSTRAP_RESAMPLES} medias bootstrap pareadas\n"
              f"(semilla = {SEED}; análisis de sensibilidad aproximado)")
ax2.legend(fontsize=8, loc="lower right")
ax2.grid(alpha=0.25)
fig2.tight_layout()
fig2.savefig("bootstrap_percentil95.png", dpi=120)
plt.close(fig2)

# ---------------------------------------------------------------------------
# 6) CSV exigido por el enunciado: output/store_differences.csv
# ---------------------------------------------------------------------------
df.to_csv(Path("output/store_differences.csv"), index=False)

# ---------------------------------------------------------------------------
# 7) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T6_bootstrap_pareado_intervalo_percentil_95",
    "metodo": {
        "descripcion": (
            "Bootstrap pareado: se remuestrean tiendas completas (filas del vector "
            "de diferencias, difference = error_a - error_b) con reemplazo y se "
            "calcula la media de cada remuestra; nunca se remuestrean error_a y "
            "error_b por separado."
        ),
        "unidad_de_remuestreo": "tienda completa (par pareado)",
        "estadistico": "media de las diferencias",
        "tipo_de_intervalo": "percentil",
        "generador_aleatorio": "np.random.default_rng",
    },
    "semilla": SEED,
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "n_tiendas": n_tiendas,
    "ic95_percentil_limite_inferior": lo_ic,
    "ic95_percentil_limite_superior": hi_ic,
    "estadisticos_observados": {
        "media_diferencias": media_obs,
        "mediana_diferencias": mediana_obs,
        "desviacion_estandar_diferencias_ddof1": sd_obs,
        "error_estandar_teorico_de_la_media": ee_teorico,
    },
    "intervalo_percentil_95": {
        "nivel": NIVEL,
        "percentiles_usados": [P_INF, P_SUP],
        "limite_inferior": lo_ic,
        "limite_superior": hi_ic,
        "longitud": hi_ic - lo_ic,
        "contiene_cero": contiene_cero,
        "interpretacion": (
            f"IC percentil del 95 % de la media de las diferencias = [{lo_ic}, {hi_ic}]; "
            + ("incluye 0, por lo que este análisis de sensibilidad aproximado no "
               "descarta una media de diferencias nula."
               if contiene_cero else
               "no incluye 0, lo que sugiere una media de diferencias distinta de cero.")
        ),
    },
    "distribucion_bootstrap": {
        "media": float(boot.mean()),
        "error_estandar_bootstrap": ee_boot,
        "sesgo_vs_media_observada": sesgo_boot,
        "p2_5": lo_ic,
        "p50": p50_boot,
        "p97_5": hi_ic,
        "minimo": float(boot.min()),
        "maximo": float(boot.max()),
    },
    "reproducibilidad": {
        "criterio": "dos ejecuciones con la misma semilla producen el mismo intervalo",
        "verificado": intervalos_identicos,
        "semilla": SEED,
        "remuestras_por_ejecucion": BOOTSTRAP_RESAMPLES,
        "intervalo_ejecucion_1": [lo_1, hi_1],
        "intervalo_ejecucion_2": [lo_2, hi_2],
        "intervalos_identicos": intervalos_identicos,
        "distribuciones_bootstrap_identicas": distribuciones_identicas,
        "nota": (
            "Cada ejecución crea su propio np.random.default_rng(SEED); con la misma "
            "semilla y los mismos datos, las 10 000 remuestras y el intervalo son "
            "idénticos entre ejecuciones."
        ),
    },
    "consistencia_con_T2": {
        "archivo_consultado": "entrada/T2/results.json",
        "disponible": bool(ruta_t2.exists()),
        "diferencias_coinciden": coincide_t2,
    },
    "diferencias_por_tienda": [
        {
            "store_id": str(r.store_id),
            "error_a": float(r.error_a),
            "error_b": float(r.error_b),
            "difference": float(r.difference),
            "favors": str(r.favors),
        }
        for r in df.itertuples()
    ],
    "explicaciones": {
        "por_que_se_remuestrean_tiendas_completas": (
            "El diseño es pareado: cada tienda aporta un par (error_a, error_b) medido "
            "sobre los mismos datos y la unidad independiente de análisis es la tienda. "
            "El estadístico de interés es la media de las diferencias dentro del par, y "
            "la correlación entre error_a y error_b solo se preserva si la fila completa "
            "se remuestrea como una unidad. Remuestrear error_a y error_b por separado "
            "rompería el emparejamiento: las diferencias reconstruidas mezclarían errores "
            "de tiendas distintas, se destruiría la correlación dentro del par y el "
            "intervalo respondería a la pregunta de dos muestras independientes, no a la "
            "comparación pareada."
        ),
        "que_problema_no_puede_corregir_el_bootstrap": (
            "El bootstrap solo genera variabilidad remuestreando la muestra observada, por "
            "lo que hereda sus limitaciones: no corrige un sesgo de selección ni la falta "
            "de representatividad (p. ej., que las 16 tiendas no sean una muestra aleatoria "
            "de la población de interés), no corrige dependencias entre tiendas (estructura "
            "jerárquica, espacial o temporal) ni errores de medición sistemáticos compartidos "
            "por ambos modelos, y no crea información que no esté en los datos. Además, con "
            "n = 16 los extremos de la distribución remuestreada dependen de pocos puntos y "
            "los valores atípicos entran en el intervalo con la misma frecuencia con que "
            "aparecen en la muestra original."
        ),
        "por_que_es_un_analisis_de_sensibilidad_aproximado": (
            "El método percentil aproxima la distribución muestral de la media sin suponer "
            "normalidad, pero su cobertura real puede desviarse del 95 % nominal en muestras "
            "pequeñas; por eso el intervalo se reporta como análisis de sensibilidad "
            "aproximado que complementa (no sustituye) las referencias nulas exactas y "
            "paramétricas."
        ),
    },
    "archivos_generados": [
        "output/store_differences.csv",
        "output/store_differences.png",
        "bootstrap_percentil95.png",
        "resultados.json",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 8) Resumen
# ---------------------------------------------------------------------------
print("=" * 66)
print("T6 · Bootstrap pareado (remuestreo de tiendas completas)")
print("=" * 66)
print(f"Datos: {ruta_csv} | tiendas (pares pareados): {n_tiendas}")
print(f"Media observada de las diferencias : {media_obs:.6f}")
print(f"Semilla: {SEED} | remuestras: {BOOTSTRAP_RESAMPLES}")
print(f"IC percentil 95 %: [{lo_ic:.6f}, {hi_ic:.6f}] (contiene 0: {contiene_cero})")
print(f"EE bootstrap: {ee_boot:.6f} | sesgo bootstrap: {sesgo_boot:.6f}")
print(f"Reproducibilidad (2 ejecuciones, misma semilla): idénticos = {intervalos_identicos}")
print("Archivos: output/store_differences.csv, output/store_differences.png,")
print("          bootstrap_percentil95.png, resultados.json")
