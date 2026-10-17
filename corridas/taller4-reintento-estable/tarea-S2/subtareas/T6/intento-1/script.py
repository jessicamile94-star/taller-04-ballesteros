# -*- coding: utf-8 -*-
"""
T6 — Bootstrap pareado de la diferencia de errores (modelo A vs modelo B) por tienda.

- Remuestrea TIENDAS COMPLETAS: se muestrean con reemplazo filas del vector de
  diferencias pareadas (difference = error_a - error_b), nunca los errores de A
  y B por separado (romper el emparejamiento destruiría la correlación intra-tienda).
- Semilla fija SEED = 20260829, BOOTSTRAP_RESAMPLES = 10_000.
- Reporta el intervalo percentil del 95% (análisis de sensibilidad aproximado) y
  verifica que dos ejecuciones con la misma semilla dan EXACTAMENTE el mismo intervalo.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20260829
BOOTSTRAP_RESAMPLES = 10_000
ALFA = 0.05
P_INF, P_SUP = 100 * ALFA / 2, 100 * (1 - ALFA / 2)  # 2.5 y 97.5


def cargar_datos():
    """Lee data/model_errors.csv (respaldo: resultados de T2) y valida el formato."""
    ruta = Path("data/model_errors.csv")
    if ruta.exists():
        df = pd.read_csv(ruta)
        columnas = ["store_id", "error_a", "error_b"]
        if list(df.columns) != columnas:
            raise ValueError(
                f"Se exigen exactamente las columnas {columnas}; se encontraron {list(df.columns)}"
            )
    else:
        alt = Path("entrada/T2/resultados.json")
        if not alt.exists():
            raise FileNotFoundError(
                "No se encontró data/model_errors.csv ni entrada/T2/resultados.json"
            )
        previo = json.loads(alt.read_text(encoding="utf-8"))
        df = pd.DataFrame(previo["tabla_pareada"])[["store_id", "error_a", "error_b"]]

    if df["store_id"].isna().any():
        raise ValueError("store_id contiene valores nulos")
    if df["store_id"].duplicated().any():
        dup = df.loc[df["store_id"].duplicated(), "store_id"].tolist()
        raise ValueError(f"store_id duplicado (identificador repetido): {dup}")
    for c in ("error_a", "error_b"):
        valores = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(valores).all():
            raise ValueError(f"{c} debe ser numérico y finito (se encontraron nulos/no finitos)")
        if (valores < 0).any():
            raise ValueError(f"{c} contiene valores negativos")
        df[c] = valores
    return df


def bootstrap_pareado(diffs, semilla, n_boot):
    """Remuestrea tiendas completas (índices del vector de diferencias) y devuelve
    la distribución bootstrap de la media y su intervalo percentil del 95%."""
    rng = np.random.default_rng(semilla)
    n = diffs.size
    # Cada fila de índices es una "muestra de tiendas completas" con reemplazo:
    # al muestrear índices del vector de diferencias se preserva el emparejamiento.
    idx = rng.integers(0, n, size=(n_boot, n))
    medias = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(medias, [P_INF, P_SUP])
    return medias, float(lo), float(hi)


def verificar_contra_t2(df):
    """Comprueba que las diferencias coinciden con las de T2 (si está disponible)."""
    ruta = Path("entrada/T2/resultados.json")
    if not ruta.exists():
        return None
    previo = json.loads(ruta.read_text(encoding="utf-8"))
    tabla = {r["store_id"]: r["difference"] for r in previo.get("tabla_pareada", [])}
    return all(
        s in tabla and np.isclose(tabla[s], d, rtol=0.0, atol=1e-9)
        for s, d in zip(df["store_id"], df["difference"])
    )


def main():
    # ---------- 1) Datos pareados ----------
    df = cargar_datos()
    df["difference"] = df["error_a"] - df["error_b"]
    df["favors"] = np.where(df["difference"] > 0, "B", np.where(df["difference"] < 0, "A", "Tie"))
    diffs = df["difference"].to_numpy(dtype=float)
    n = diffs.size

    # Archivos exigidos por el enunciado (mismas rutas que T2)
    Path("output").mkdir(parents=True, exist_ok=True)
    df.to_csv("output/store_differences.csv", index=False)
    coincide_t2 = verificar_contra_t2(df)

    # ---------- 2) Estadísticos observados ----------
    media_obs = float(diffs.mean())
    mediana_obs = float(np.median(diffs))
    sd_obs = float(diffs.std(ddof=1))
    se_t = float(sd_obs / np.sqrt(n))

    # ---------- 3) Bootstrap pareado: dos corridas con la MISMA semilla ----------
    m1, lo1, hi1 = bootstrap_pareado(diffs, SEED, BOOTSTRAP_RESAMPLES)
    m2, lo2, hi2 = bootstrap_pareado(diffs, SEED, BOOTSTRAP_RESAMPLES)
    intervalos_identicos = bool(lo1 == lo2 and hi1 == hi2)
    distribuciones_identicas = bool(np.array_equal(m1, m2))

    # Contraste: una semilla distinta debe dar un intervalo distinto (sanidad)
    m3, lo3, hi3 = bootstrap_pareado(diffs, SEED + 1, BOOTSTRAP_RESAMPLES)
    distinto_otra_semilla = bool(not (lo1 == lo3 and hi1 == hi3))

    # ---------- 4) Resumen de la distribución bootstrap ----------
    boot = {
        "media": float(m1.mean()),
        "mediana": float(np.median(m1)),
        "desv_est": float(m1.std(ddof=1)),
        "min": float(m1.min()),
        "max": float(m1.max()),
        "sesgo_vs_observado": float(m1.mean() - media_obs),
        "prop_remuestras_menores_iguales_cero": float((m1 <= 0).mean()),
        "cuantiles": {
            "p0": float(np.percentile(m1, 0)),
            "p5": float(np.percentile(m1, 5)),
            "p25": float(np.percentile(m1, 25)),
            "p50": float(np.percentile(m1, 50)),
            "p75": float(np.percentile(m1, 75)),
            "p95": float(np.percentile(m1, 95)),
            "p100": float(np.percentile(m1, 100)),
        },
    }

    # ---------- 5) Figuras ----------
    # Figura exigida: diferencias por tienda
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colores = np.where(diffs > 0, "#1f77b4", "#d62728")
    ax.bar(df["store_id"], diffs, color=colores)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(media_obs, color="green", ls="--", lw=1.2,
               label=f"media de diferencias = {media_obs:.4f}")
    ax.set_ylabel("difference = error_a - error_b")
    ax.set_title("Diferencia de error por tienda (pareado A vs B)")
    ax.tick_params(axis="x", rotation=90)
    ax.legend()
    fig.tight_layout()
    fig.savefig("output/store_differences.png", dpi=120)
    plt.close(fig)

    # Figura de la subtarea: distribución bootstrap de la media
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(m1, bins=60, color="#4c72b0", alpha=0.85, edgecolor="white")
    ax.axvline(media_obs, color="black", lw=1.5, label=f"media observada = {media_obs:.4f}")
    ax.axvline(lo1, color="crimson", ls="--", lw=1.5, label=f"IC95% inf = {lo1:.4f}")
    ax.axvline(hi1, color="crimson", ls="--", lw=1.5, label=f"IC95% sup = {hi1:.4f}")
    ax.set_xlabel("media bootstrap de la diferencia pareada")
    ax.set_ylabel("frecuencia")
    ax.set_title(f"Bootstrap pareado: {BOOTSTRAP_RESAMPLES} remuestras de tiendas (semilla {SEED})")
    ax.legend()
    fig.tight_layout()
    fig.savefig("bootstrap_media_diferencias.png", dpi=120)
    plt.close(fig)

    # ---------- 6) resultados.json (contrato) ----------
    resultados = {
        "subtarea": "T6_bootstrap_pareado",
        "semilla": SEED,
        "remuestras_bootstrap": BOOTSTRAP_RESAMPLES,
        "n_tiendas": int(n),
        "vector_diferencias": diffs.tolist(),
        "estadisticas_observadas": {
            "media_diferencias": media_obs,
            "mediana_diferencias": mediana_obs,
            "desv_est_muestral_ddof1": sd_obs,
            "error_estandar_media_t": se_t,
        },
        "intervalo_percentil_95": {
            "limite_inferior": lo1,
            "limite_superior": hi1,
            "metodo": "percentil",
            "nivel": 0.95,
            "percentiles_usados": [P_INF, P_SUP],
        },
        "ic95_percentil": [lo1, hi1],
        "bootstrap_distribucion_media": boot,
        "reproducibilidad_misma_semilla": {
            "semilla": SEED,
            "corrida_1_intervalo": [lo1, hi1],
            "corrida_2_intervalo": [lo2, hi2],
            "intervalos_exactamente_iguales": intervalos_identicos,
            "distribuciones_bootstrap_exactamente_iguales": distribuciones_identicas,
        },
        "contraste_semilla_distinta": {
            "semilla": SEED + 1,
            "intervalo": [lo3, hi3],
            "intervalo_distinto_al_de_semilla_20260829": distinto_otra_semilla,
        },
        "verificacion_contra_T2": {
            "archivo": "entrada/T2/resultados.json",
            "diferencias_coinciden": coincide_t2,
        },
        "archivos_generados": {
            "csv": "output/store_differences.csv",
            "png_diferencias": "output/store_differences.png",
            "png_bootstrap": "bootstrap_media_diferencias.png",
            "resultados": "resultados.json",
        },
        "explicaciones": {
            "por_que_remuestrear_tiendas_completas": (
                "La unidad experimental es la tienda: error_a y error_b de una misma tienda "
                "comparten condiciones (clientela, inventario, temporada) y por eso están "
                "correlacionados. Remuestrear tiendas completas —muestrear con reemplazo filas "
                "del vector de diferencias— preserva el emparejamiento y esa correlación "
                "intra-par, de modo que la distribución bootstrap aproxima la ley muestral del "
                "estadístico correcto (la media de las diferencias pareadas). Si se remuestrearan "
                "por separado los errores de A y de B se rompería el pareo: las diferencias "
                "remuestreadas ya no corresponderían a pares reales, se destruiría la "
                "correlación entre modelos y la variabilidad del estimador quedaría mal "
                "representada (típicamente inflada), produciendo un intervalo inválido para el "
                "diseño pareado."
            ),
            "que_no_puede_corregir_el_bootstrap": (
                "El bootstrap asume que la muestra empírica es representativa y que las "
                "observaciones son intercambiables (i.i.d.). No corrige automáticamente: "
                "(1) sesgo de selección o tiendas no elegidas al azar; (2) dependencia entre "
                "tiendas (estructura jerárquica, correlación espacial/temporal, efectos de "
                "región o campaña); (3) el tamaño muestral pequeño (n=16 explora mal las colas "
                "de la distribución poblacional); (4) errores de medición o parejas mal "
                "construidas; (5) violaciones del diseño (p. ej., modelos evaluados en periodos "
                "o condiciones distintas). Solo refleja la variabilidad presente en los datos "
                "observados."
            ),
            "papel_analisis_sensibilidad": (
                "El intervalo percentil del 95% es una referencia aproximada de sensibilidad: "
                "indica cuánto podría variar la media de la diferencia bajo el mecanismo de "
                "remuestreo de tiendas. Como el 0 sí cae dentro del intervalo, la ventaja "
                "observada de B no es robusta a la variabilidad muestral a este nivel de "
                "confianza."
            ),
        },
    }
    Path("resultados.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---------- 7) Resumen ----------
    print("=== T6: Bootstrap pareado ===")
    print(f"Semilla: {SEED} | Remuestras: {BOOTSTRAP_RESAMPLES} | Tiendas: {n}")
    print(f"Media observada de diferencias: {media_obs:.6f} (DE muestral = {sd_obs:.6f})")
    print(f"IC95% percentil: [{lo1:.6f}, {hi1:.6f}]")
    print(f"Reproducibilidad (misma semilla): corrida 2 = [{lo2:.6f}, {hi2:.6f}] "
          f"-> intervalos idénticos: {intervalos_identicos}; "
          f"distribuciones idénticas: {distribuciones_identicas}")
    print(f"Contraste semilla {SEED + 1}: intervalo = [{lo3:.6f}, {hi3:.6f}] "
          f"(distinto: {distinto_otra_semilla})")
    print(f"Proporción de remuestras <= 0: {boot['prop_remuestras_menores_iguales_cero']:.4f}")
    print("Archivos: output/store_differences.csv, output/store_differences.png, "
          "bootstrap_media_diferencias.png, resultados.json")


if __name__ == "__main__":
    main()
