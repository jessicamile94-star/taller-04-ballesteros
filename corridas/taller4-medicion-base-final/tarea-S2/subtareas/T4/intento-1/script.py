#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4 — Efecto e incertidumbre sobre el vector de diferencias pareadas.

Entradas:
  - data/model_errors.csv      (datos de la tarea)
  - entrada/T2/results.json    (resultado previo, solo para verificación cruzada)

Salidas:
  - output/store_differences.csv  (tabla pareada con difference y favors)
  - output/store_differences.png  (figura: diferencias por tienda + media con IC t 95 %)
  - resultados.json               (contrato numérico de la subtarea)
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


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Cargar datos y validación ligera (la validación completa se hizo en T2)
    # ------------------------------------------------------------------
    csv_path = Path("data/model_errors.csv")
    df = pd.read_csv(csv_path)

    requeridas = ["store_id", "error_a", "error_b"]
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas {faltantes} en {csv_path}")
    if df["store_id"].isna().any() or df["store_id"].duplicated().any():
        raise ValueError("store_id debe ser único y no nulo para el emparejamiento")
    a = df["error_a"].to_numpy(dtype=float)
    b = df["error_b"].to_numpy(dtype=float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("error_a y error_b deben ser numéricos y finitos")
    if (a < 0).any() or (b < 0).any():
        raise ValueError("los errores no pueden ser negativos")

    diff = a - b
    n = int(diff.size)
    store_ids = df["store_id"].astype(str).tolist()

    # ------------------------------------------------------------------
    # 2) Verificación opcional contra la tabla pareada producida en T2
    # ------------------------------------------------------------------
    verif_t2 = {"archivo": "entrada/T2/results.json", "existe": False, "coincide": None}
    t2_path = Path("entrada/T2/results.json")
    if t2_path.exists():
        verif_t2["existe"] = True
        with t2_path.open("r", encoding="utf-8") as fh:
            t2 = json.load(fh)
        filas_t2 = t2.get("tabla_pareada", {}).get("filas", [])
        if len(filas_t2) == n:
            t2_sorted = sorted(filas_t2, key=lambda r: r["store_id"])
            propios = sorted(zip(store_ids, diff.tolist()), key=lambda r: r[0])
            ok_ids = [r["store_id"] for r in t2_sorted] == [p[0] for p in propios]
            ok_diff = np.allclose(
                np.array([r["difference"] for r in t2_sorted], dtype=float),
                np.array([p[1] for p in propios], dtype=float),
                rtol=0.0, atol=1e-9,
            )
            verif_t2["coincide"] = bool(ok_ids and ok_diff)

    # ------------------------------------------------------------------
    # 3) Estadísticos del vector de diferencias (sobre las 16 diferencias)
    # ------------------------------------------------------------------
    mean = float(np.mean(diff))
    sample_sd = float(np.std(diff, ddof=1))      # desviación MUESTRAL (ddof=1)
    standard_error = sample_sd / math.sqrt(n)    # SE = sd / sqrt(n)
    grados_libertad = n - 1

    # ------------------------------------------------------------------
    # 4) Prueba t de una muestra, bilateral, H0: media de diferencias = 0
    # ------------------------------------------------------------------
    try:
        tt = stats.ttest_1samp(diff, popmean=0.0, alternative="two-sided")
    except TypeError:  # scipy antiguo: la bilateral es la opción por defecto
        tt = stats.ttest_1samp(diff, popmean=0.0)
    t_stat = float(tt.statistic)
    p_value = float(tt.pvalue)
    df_scipy = float(getattr(tt, "df", grados_libertad))

    # comprobación interna del p-value bilateral
    p_check = float(2.0 * stats.t.sf(abs(t_stat), grados_libertad))

    # ------------------------------------------------------------------
    # 5) Intervalo t bilateral del 95 % para la media pareada
    # ------------------------------------------------------------------
    conf = 0.95
    t_crit = float(stats.t.ppf(1.0 - (1.0 - conf) / 2.0, grados_libertad))
    ci_lo, ci_hi = stats.t.interval(conf, grados_libertad, loc=mean, scale=standard_error)
    ci_lo, ci_hi = float(ci_lo), float(ci_hi)
    ci_lo_manual = mean - t_crit * standard_error   # comprobación manual
    ci_hi_manual = mean + t_crit * standard_error
    cero_en_ic = bool(ci_lo <= 0.0 <= ci_hi)

    # ------------------------------------------------------------------
    # 6) CSV exigido por el enunciado: output/store_differences.csv
    # ------------------------------------------------------------------
    favors = ["B" if d > 0 else ("A" if d < 0 else "tie") for d in diff]
    tabla = pd.DataFrame({
        "store_id": store_ids,
        "error_a": a,
        "error_b": b,
        "difference": diff,
        "favors": favors,
    })
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / "store_differences.csv"
    tabla.to_csv(csv_out, index=False)

    # ------------------------------------------------------------------
    # 7) Figura exigida por el enunciado: output/store_differences.png
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    colores = ["#1f77b4" if d > 0 else "#d62728" for d in diff]
    ax.bar(range(n), diff, color=colores, edgecolor="black", linewidth=0.4)
    ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xticks(range(n))
    ax.set_xticklabels(store_ids, rotation=90, fontsize=8)
    ax.set_ylabel("difference = error_a - error_b")
    ax.set_title(f"Diferencias por tienda (n = {n})")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    jitter = np.linspace(-0.12, 0.12, n)  # desplazamiento determinista (sin azar)
    ax.scatter(np.ones(n) + jitter, diff, s=28, color="#7f7f7f", alpha=0.85,
               zorder=2, label="diferencias individuales")
    ax.errorbar([1.35], [mean],
                yerr=[[mean - ci_lo], [ci_hi - mean]],
                fmt="D", color="black", capsize=7, markersize=7, zorder=3,
                label="media ± IC t 95 %")
    ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xlim(0.5, 1.9)
    ax.set_xticks([1.0, 1.35])
    ax.set_xticklabels(["tiendas", "media"])
    ax.set_ylabel("difference")
    ax.set_title("Efecto medio e incertidumbre (t, 95 %)")
    ax.text(0.02, 0.02,
            f"mean = {mean:.6f}\nsd = {sample_sd:.6f}\nSE = {standard_error:.6f}\n"
            f"IC95% = [{ci_lo:.6f}, {ci_hi:.6f}]\nt({grados_libertad}) = {t_stat:.6f}\np = {p_value:.6f}",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("T4 — Efecto pareado error_a - error_b y su incertidumbre", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png_out = out_dir / "store_differences.png"
    fig.savefig(png_out, dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 8) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T4_efecto_e_incertidumbre_prueba_t_pareada",
        "fuente_datos": "data/model_errors.csv",
        "verificacion_contra_T2": verif_t2,

        # ---- cifras exigidas (contrato) ----
        "n": n,
        "mean": mean,
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "degrees_of_freedom": int(grados_libertad),
        "t_statistic": t_stat,
        "p_value_two_sided": p_value,
        "ci95_lower": ci_lo,
        "ci95_upper": ci_hi,

        # ---- detalle y comprobaciones ----
        "definiciones": {
            "difference": "error_a - error_b por tienda",
            "sample_sd": "desviación estándar muestral (ddof=1) de las diferencias",
            "standard_error": "sample_sd / sqrt(n)",
            "prueba_t": "scipy.stats.ttest_1samp sobre las diferencias, H0: media = 0, alternativa bilateral",
            "intervalo": "scipy.stats.t.interval(0.95, df=n-1, loc=mean, scale=SE)",
        },
        "prueba_t_una_muestra_bilateral": {
            "hipotesis_nula": "la media de las diferencias (error_a - error_b) es 0",
            "alternativa": "two-sided",
            "statistic": t_stat,
            "p_value": p_value,
            "degrees_of_freedom": int(grados_libertad),
            "df_reportado_por_scipy": df_scipy,
            "p_value_comprobacion_2_t_sf": p_check,
        },
        "intervalo_t_95": {
            "confidence_level": conf,
            "degrees_of_freedom": int(grados_libertad),
            "t_critical_two_sided": t_crit,
            "lower": ci_lo,
            "upper": ci_hi,
            "lower_manual_mean_minus_tcrit_se": ci_lo_manual,
            "upper_manual_mean_plus_tcrit_se": ci_hi_manual,
            "cero_dentro_del_intervalo": cero_en_ic,
        },
        "resumen_diferencias": {
            "suma": float(np.sum(diff)),
            "min": float(np.min(diff)),
            "max": float(np.max(diff)),
            "n_positivas_favorecen_B": int(np.sum(diff > 0)),
            "n_negativas_favorecen_A": int(np.sum(diff < 0)),
            "n_ceros": int(np.sum(diff == 0)),
        },
        "diferencias_por_tienda": [
            {
                "store_id": store_ids[i],
                "error_a": float(a[i]),
                "error_b": float(b[i]),
                "difference": float(diff[i]),
                "favors": favors[i],
            }
            for i in range(n)
        ],
        "interpretacion": {
            "media": (
                "La diferencia media es positiva: en promedio error_a excede a error_b "
                "(el modelo B tiene menor error) en ~0.42 unidades por tienda; la magnitud "
                "se lee en las mismas unidades de la variable de error."
            ),
            "dispersion_vs_se": (
                "La desviación muestral (~1.25) describe cuánto varía el efecto entre tiendas; "
                "el error estándar (~0.31) es la incertidumbre de la MEDIA y decrece como sd/sqrt(n): "
                "la heterogeneidad entre tiendas no se reduce al aumentar n, el SE sí."
            ),
            "p_value": (
                "El p-value bilateral es la probabilidad, si H0 (media de diferencias = 0) fuera cierta, "
                "de observar un estadístico t al menos tan extremo como el obtenido, en cualquier dirección."
            ),
            "intervalo": (
                "El IC t del 95 % es el rango de valores de la media pareada compatibles con los datos "
                "bajo el modelo t; NO es una probabilidad posterior sobre el parámetro."
            ),
        },
        "archivos_generados": [str(csv_out), str(png_out)],
    }

    with Path("resultados.json").open("w", encoding="utf-8") as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 9) Resumen por consola
    # ------------------------------------------------------------------
    print("=" * 64)
    print(f"T4 — Efecto e incertidumbre (diferencias pareadas, n = {n})")
    print("=" * 64)
    print(f"n               = {n}")
    print(f"mean            = {mean!r}")
    print(f"sample_sd       = {sample_sd!r}")
    print(f"standard_error  = {standard_error!r}  (sd/sqrt(n))")
    print(f"df              = {grados_libertad}")
    print(f"t statistic     = {t_stat!r}")
    print(f"p (bilateral)   = {p_value!r}")
    print(f"IC t 95 %       = [{ci_lo!r}, {ci_hi!r}]  (t_crit = {t_crit!r})")
    print(f"0 en el IC 95 % = {cero_en_ic}")
    if verif_t2["existe"]:
        print(f"Verificación contra entrada/T2/results.json: coincide = {verif_t2['coincide']}")
    print(f"Archivos: {csv_out} , {png_out} , resultados.json")


if __name__ == "__main__":
    main()
