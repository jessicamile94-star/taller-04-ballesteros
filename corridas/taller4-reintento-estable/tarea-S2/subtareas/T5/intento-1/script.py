#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5 — Referencia nula exacta por cambios de signo (enumeración 2^16 = 65536)
y contraste con el procedimiento principal de T4 (prueba t bilateral).

Pasos:
  1) Recupera el vector de diferencias pareadas d_i = error_a - error_b (n = 16),
     pareadas por store_id (data/model_errors.csv; respaldo: entrada/T2 | entrada/T4).
  2) Enumera las 2^16 configuraciones de signos s_i en {-1,+1} y calcula la media de
     cada configuración: m(s) = (1/n) * sum_i s_i * d_i.
  3) Cuenta las configuraciones con |m(s)| >= |media observada| y reporta
     total = 65536, conteo de extremas y p-value exacto = conteo / 65536.
  4) Contrasta con el estadístico t y el p-value bilateral de T4 (procedimiento principal).
  5) Verificación independiente exacta: programación dinámica en décimas (enteros),
     que reproduce la enumeración sin aritmética flotante.

Salidas:
  - resultados.json
  - t5_distribucion_nula_cambios_signo.png
  - t5_comparacion_p_values.png
  - output/store_differences.csv, output/store_differences.png (exigidos por el enunciado)
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

RUTA_CSV = Path("data/model_errors.csv")
RUTA_T4 = Path("entrada/T4/resultados.json")
TOL = 1e-9  # tolerancia para empates exactos en punto flotante


# ----------------------------------------------------------------------
# Carga de datos
# ----------------------------------------------------------------------
def cargar_diferencias():
    """Devuelve (DataFrame con 'difference', descripción de la fuente)."""
    if RUTA_CSV.exists():
        df = pd.read_csv(RUTA_CSV)
        requeridas = ["store_id", "error_a", "error_b"]
        faltantes = [c for c in requeridas if c not in df.columns]
        if faltantes:
            raise ValueError(f"{RUTA_CSV}: faltan las columnas {faltantes}")
        df = df.copy()
        df["error_a"] = pd.to_numeric(df["error_a"], errors="raise").astype(float)
        df["error_b"] = pd.to_numeric(df["error_b"], errors="raise").astype(float)
        if df["store_id"].isna().any() or df["store_id"].duplicated().any():
            raise ValueError(f"{RUTA_CSV}: store_id con valores nulos o duplicados")
        df["difference"] = df["error_a"] - df["error_b"]
        df = df.sort_values("store_id", kind="mergesort").reset_index(drop=True)
        desc = "data/model_errors.csv (difference = error_a - error_b, pareada por store_id)"
        return df, desc
    for cand, clave in (
        (Path("entrada/T2/resultados.json"), "tabla_pareada"),
        (Path("entrada/T4/resultados.json"), "tabla_diferencias"),
    ):
        if cand.exists():
            with open(cand, encoding="utf-8") as f:
                previo = json.load(f)
            df = pd.DataFrame(previo[clave])
            desc = f"{cand} (tabla de diferencias de la subtarea previa)"
            return df, desc
    raise FileNotFoundError(
        "No se encontró data/model_errors.csv ni tablas de diferencias previas en entrada/."
    )


def cargar_prueba_t(n, d):
    """Estadístico t, gl y p-value bilateral del procedimiento principal (T4)."""
    if RUTA_T4.exists():
        with open(RUTA_T4, encoding="utf-8") as f:
            t4 = json.load(f)
        pt = t4.get("prueba_t") or {}
        t_stat = t4.get("t_statistic", pt.get("estadistico_t"))
        p_t = t4.get("p_value", pt.get("p_value_bilateral"))
        gl = t4.get("grados_de_libertad", t4.get("df", pt.get("grados_de_libertad", n - 1)))
        if t_stat is not None and p_t is not None:
            ic = t4.get("intervalo_confianza") or {}
            lo = ic.get("limite_inferior", t4.get("ci95_low"))
            hi = ic.get("limite_superior", t4.get("ci95_high"))
            return {
                "fuente": "entrada/T4/resultados.json (procedimiento principal de T4)",
                "metodo": "prueba t de una muestra bilateral sobre las diferencias (popmean=0)",
                "estadistico_t": float(t_stat),
                "grados_de_libertad": int(gl),
                "p_value_bilateral": float(p_t),
                "media": float(t4.get("mean", float(np.mean(d)))),
                "sample_sd": float(t4.get("sample_sd", float(np.std(d, ddof=1)))),
                "standard_error": float(
                    t4.get("standard_error", float(np.std(d, ddof=1) / np.sqrt(n)))
                ),
                "ic95_bilateral": [
                    None if lo is None else float(lo),
                    None if hi is None else float(hi),
                ],
                "recalculado": False,
            }
    # Respaldo: recalcular con los mismos parámetros que T4
    res = stats.ttest_1samp(d, popmean=0.0, alternative="two-sided")
    media = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = float(stats.sem(d))
    gl = n - 1
    lo, hi = stats.t.interval(0.95, df=gl, loc=media, scale=se)
    return {
        "fuente": "recalculado con scipy.stats.ttest_1samp (entrada/T4/resultados.json no disponible)",
        "metodo": "prueba t de una muestra bilateral sobre las diferencias (popmean=0)",
        "estadistico_t": float(res.statistic),
        "grados_de_libertad": int(gl),
        "p_value_bilateral": float(res.pvalue),
        "media": media,
        "sample_sd": sd,
        "standard_error": se,
        "ic95_bilateral": [float(lo), float(hi)],
        "recalculado": True,
    }


# ----------------------------------------------------------------------
# Enumeración exacta de las 2^16 configuraciones de signos
# ----------------------------------------------------------------------
def enumerar_cambios_de_signo(d):
    """Devuelve (total, medias nulas, sumas nulas) para las 2^n configuraciones."""
    n = int(d.size)
    total = 2 ** n
    idx = np.arange(total, dtype=np.int64)
    bits = (idx[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1  # (65536, 16)
    signos = 1.0 - 2.0 * bits.astype(np.float64)  # bit 0 -> +1, bit 1 -> -1
    sumas = signos @ d
    medias = sumas / n
    return total, medias, sumas


# ----------------------------------------------------------------------
# Verificación independiente: DP exacta en décimas (enteros, sin flotantes)
# ----------------------------------------------------------------------
def dp_entera_decimas(d, umbral_decimas):
    v = np.rint(np.asarray(d, dtype=float) * 10.0).astype(np.int64)
    if not np.all(np.abs(v / 10.0 - np.asarray(d, dtype=float)) < 1e-9):
        return {"aplicable": False, "motivo": "las diferencias no son múltiplos de 0.1"}
    lim = int(np.abs(v).sum())
    size = 2 * lim + 1
    poly = np.zeros(size, dtype=np.int64)
    poly[lim] = 1
    for vi in v:
        a = int(abs(vi))
        nuevo = np.zeros(size, dtype=np.int64)
        nuevo[a:] += poly[: size - a]      # s -> s + a
        nuevo[: size - a] += poly[a:]      # s -> s - a
        poly = nuevo
    sumas = np.arange(size, dtype=np.int64) - lim
    umbral = int(umbral_decimas)
    distribucion = [[int(s), int(c)] for s, c in zip(sumas, poly) if c > 0]
    return {
        "aplicable": True,
        "metodo": (
            "programación dinámica exacta sobre sumas en décimas (enteros); "
            "equivalente a enumerar los 2^16 signos sin aritmética flotante"
        ),
        "suma_cuentas": int(poly.sum()),
        "conteo_extremas": int(poly[np.abs(sumas) >= umbral].sum()),
        "cola_derecha": int(poly[sumas >= umbral].sum()),
        "cola_izquierda": int(poly[sumas <= -umbral].sum()),
        "numero_sumas_distintas": int((poly > 0).sum()),
        "suma_max_abs_decimas": int(np.abs(sumas).max()),
        "umbral_decimas": umbral,
        "distribucion_suma_decimas_cuenta": distribucion,
    }


# ----------------------------------------------------------------------
def main():
    # 1) Vector de diferencias pareadas ----------------------------------
    df, fuente = cargar_diferencias()
    d = df["difference"].to_numpy(dtype=float)
    n = int(d.size)
    if n != 16:
        raise ValueError(f"El enunciado exige 16 diferencias pareadas; se obtuvieron {n}.")
    if 2 ** n != 65536:
        raise ValueError("El enunciado exige 2^16 = 65536 configuraciones de signos.")

    obs_media = float(np.mean(d))
    obs_suma = float(np.sum(d))
    abs_obs = abs(obs_media)
    umbral_dec = int(round(abs_obs * n * 10.0))  # |suma observada| en décimas (67)

    # 2) Enumeración exhaustiva de las 2^16 configuraciones --------------
    total, medias_nulas, sumas_nulas = enumerar_cambios_de_signo(d)

    mascara_ext = np.abs(medias_nulas) >= (abs_obs - TOL)
    conteo_extremas = int(np.count_nonzero(mascara_ext))
    p_exacto = conteo_extremas / total
    cola_der = int(np.count_nonzero(medias_nulas >= (abs_obs - TOL)))
    cola_izq = int(np.count_nonzero(medias_nulas <= (-abs_obs + TOL)))

    obs_incluida = bool(abs(medias_nulas[0] - obs_media) <= 1e-12 and bool(mascara_ext[0]))
    espejo_incluido = bool(abs(medias_nulas[-1] + obs_media) <= 1e-12 and bool(mascara_ext[-1]))
    ordenadas = np.sort(medias_nulas)
    simetrica = bool(np.allclose(ordenadas, -ordenadas[::-1], atol=1e-12))
    media_nulas = float(np.mean(medias_nulas))
    sd_nulas = float(np.std(medias_nulas))  # ddof=0: enumeración completa
    sd_nulas_teor = float(np.sqrt(float(np.sum(d ** 2))) / n)

    # 3) Verificación independiente (DP entera) ---------------------------
    dp = dp_entera_decimas(d, umbral_dec)

    # 4) Procedimiento principal de T4 ------------------------------------
    t4 = cargar_prueba_t(n, d)
    p_t = float(t4["p_value_bilateral"])
    t_stat = float(t4["estadistico_t"])
    gl_t = int(t4["grados_de_libertad"])

    dif_abs = abs(p_exacto - p_t)
    razon = (p_exacto / p_t) if p_t > 0 else None
    misma5 = bool((p_exacto >= 0.05) == (p_t >= 0.05))
    misma10 = bool((p_exacto >= 0.10) == (p_t >= 0.10))
    mcd = math.gcd(conteo_extremas, total)
    fraccion = f"{conteo_extremas // mcd}/{total // mcd}"
    cociente_z = (obs_media / sd_nulas) if sd_nulas > 0 else None
    cociente_txt = f"{cociente_z:.4f}" if cociente_z is not None else "n/a"
    razon_txt = f"{razon:.4f}" if razon is not None else "n/a"

    ic_lo, ic_hi = (t4.get("ic95_bilateral") or [None, None])
    if ic_lo is not None and ic_hi is not None:
        ic_txt = f"[{ic_lo:.4f}, {ic_hi:.4f}]" + (" (contiene 0)" if ic_lo <= 0 <= ic_hi else "")
    else:
        ic_txt = "no disponible"

    if misma5:
        decision_txt = (
            f"Ambos p-values ({p_exacto:.6f} exacto por cambios de signo; {p_t:.6f} t bilateral) superan "
            "0.05: ninguna de las dos referencias nulas rechaza H0 al 5 % (ni al 10 %). Coinciden "
            f"cualitativamente, pero provienen de referencias distintas (t continua con {gl_t} gl frente "
            "a enumeración discreta de 65536 configuraciones), por lo que no son procedimientos intercambiables."
        )
    else:
        decision_txt = (
            f"Los p-values ({p_exacto:.6f} cambios de signo; {p_t:.6f} t bilateral) divergen en la "
            "decisión al 5 %, ilustrando que las referencias nulas no son intercambiables."
        )

    # 5) Figuras -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.hist(medias_nulas, bins=80, color="#9ecae1", edgecolor="white", linewidth=0.3,
            label="65536 medias nulas (cambios de signo)")
    ax.axvspan(abs_obs, float(medias_nulas.max()), color="firebrick", alpha=0.20,
               label="región extrema (|media| >= media observada)")
    ax.axvspan(float(medias_nulas.min()), -abs_obs, color="firebrick", alpha=0.20)
    ax.axvline(obs_media, color="firebrick", lw=2.0, label=f"media observada = {obs_media:.4f}")
    ax.axvline(-obs_media, color="firebrick", lw=1.5, ls="--")
    ax.axvline(0.0, color="0.3", lw=0.8)
    texto = (
        f"Configuraciones extremas: {conteo_extremas} / {total}\n"
        f"p-value exacto = {conteo_extremas}/65536 = {p_exacto:.6f}\n"
        f"p-value t bilateral (T4) = {p_t:.6f}"
    )
    ax.text(0.02, 0.97, texto, transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.9))
    ax.set_xlabel("Media de la configuración de signos: (1/n) * sum_i s_i * d_i")
    ax.set_ylabel("Número de configuraciones")
    ax.set_title("T5 · Referencia nula exacta por cambios de signo (2^16 = 65536 configuraciones)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig("t5_distribucion_nula_cambios_signo.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    etiquetas = ["t bilateral\n(T4, procedimiento principal)", "Cambios de signo\n(exacta, 2^16 configuraciones)"]
    barras = ax.bar(etiquetas, [p_t, p_exacto], color=["#4c72b0", "#dd8452"], width=0.5)
    ax.axhline(0.05, color="black", ls="--", lw=1.0, label="alpha = 0.05")
    for b, v in zip(barras, [p_t, p_exacto]):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.004, f"{v:.6f}", ha="center", fontsize=9)
    ax.set_ylim(0.0, max(p_t, p_exacto) * 1.3)
    ax.set_ylabel("p-value bilateral")
    ax.set_title("T5 · Contraste de referencias nulas: prueba t vs cambios de signo")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("t5_comparacion_p_values.png", dpi=120)
    plt.close(fig)

    # 6) Archivos exigidos por el enunciado --------------------------------
    Path("output").mkdir(parents=True, exist_ok=True)
    df_out = df.copy()
    if "favors" not in df_out.columns:
        df_out["favors"] = np.where(df_out["difference"].to_numpy(dtype=float) > 0, "B", "A")
    preferidas = ["store_id", "error_a", "error_b", "difference", "favors"]
    cols = [c for c in preferidas if c in df_out.columns]
    cols += [c for c in df_out.columns if c not in cols]
    df_out[cols].to_csv("output/store_differences.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colores = np.where(d >= 0, "#2e7d32", "#c62828")
    ax.bar(df_out["store_id"].astype(str), d, color=colores)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axhline(obs_media, color="#1565c0", ls="--", lw=1.2, label=f"media = {obs_media:.4f}")
    ax.set_xlabel("Tienda (store_id)")
    ax.set_ylabel("difference = error_a - error_b")
    ax.set_title("Diferencias pareadas por tienda (error_a - error_b)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("output/store_differences.png", dpi=120)
    plt.close(fig)

    # 7) Discusión ----------------------------------------------------------
    discusion = {
        "p_value_condicional_a_h0": (
            "El p-value se interpreta SOLO de forma condicional a la hipótesis nula: SI H0 fuera cierta "
            "(tendencia central 0 en las diferencias pareadas y signos intercambiables, es decir, cada "
            "d_i con la misma probabilidad de observarse como +d_i o como -d_i), la probabilidad de que "
            "una configuración de signos produzca una media con magnitud >= |media observada| = "
            f"{abs_obs:.5f} sería {conteo_extremas}/{total} = {p_exacto:.6f}. El p-value NO es la "
            "probabilidad de que H0 sea cierta ni la probabilidad de que el resultado se deba al azar: "
            "es una frecuencia relativa dentro de la referencia nula frente al resultado observado."
        ),
        "diseno": (
            "Diseño pareado por tienda: cada una de las 16 tiendas aporta un par (error_a, error_b) "
            "verificado por store_id; la unidad de análisis es la diferencia d_i = error_a - error_b y "
            "el estadístico es su media. La referencia por cambios de signo NO remuestrea tiendas ni "
            "reordena pares: solo reasigna signos a las 16 magnitudes observadas (2^16 configuraciones), "
            "por lo que es condicional a los datos observados."
        ),
        "supuestos_prueba_t": (
            "La prueba t de una muestra sobre las diferencias (T4) asume que estas son una muestra "
            "aleatoria de una población normal, o apela al TCL si n es grande; estima la dispersión con "
            f"la varianza muestral respecto a la media (sd = {t4['sample_sd']:.4f}, "
            f"SE = sd/sqrt(n) = {t4['standard_error']:.4f}) y compara t = media/SE = {t_stat:.4f} con una "
            f"distribución t continua con {gl_t} grados de libertad (p = {p_t:.6f}; IC95 {ic_txt}). "
            "Su p-value solo es exacto si la población de diferencias es normal; en otro caso es una "
            "aproximación asintótica."
        ),
        "supuestos_cambios_de_signo": (
            "La prueba exacta por cambios de signo solo asume que, bajo H0, el signo de cada diferencia "
            "es intercambiable (simetría de cada d_i respecto de 0, p. ej. por emparejamiento o "
            "aleatorización equilibrada). No requiere normalidad ni TCL: la referencia es la distribución "
            "discreta y finita de las 65536 medias obtenidas reasignando signos, y el p-value es exacto "
            "(múltiplo de 1/65536) bajo ese supuesto. A cambio, la inferencia es condicional a las 16 "
            "magnitudes observadas y no modela el muestreo de otras tiendas de una población."
        ),
        "por_que_no_son_intercambiables": (
            "Usan los mismos datos pero referencias nulas, supuestos y estadísticos distintos: "
            "(1) Referencia: la t compara con una ley continua t(15) derivada del modelo de muestreo "
            "normal; los cambios de signo comparan con una ley discreta de 65536 valores generada al "
            "reasignar signos a las diferencias observadas. "
            f"(2) Dispersión: la t estandariza la media por SE = {t4['standard_error']:.4f} (varianza "
            "respecto a la media, con corrección n-1); la referencia de signos deja la media sin "
            f"estandarizar y su dispersión natural es sqrt(sum(d_i^2))/n = {sd_nulas:.4f} (segundo "
            f"momento respecto a 0): por eso t = {t_stat:.4f} mientras que el cociente análogo bajo la "
            f"referencia de signos es {obs_media:.5f}/{sd_nulas:.4f} = {cociente_txt}. "
            f"(3) Naturaleza del p-value: el de cambios de signo es exacto bajo su nula (fracción "
            f"{conteo_extremas}/65536, sin aproximación asintótica); el de la t es aproximado salvo "
            "normalidad exacta de las diferencias. "
            "(4) Hipótesis nula distinta: 'la media de las diferencias proviene de una población normal "
            "con media 0' no es lo mismo que 'los signos de las diferencias observadas son "
            "intercambiables'. Con diferencias asimétricas, valores atípicos o n pequeño, ambos "
            "procedimientos pueden divergir sustancialmente; aquí, con n = 16 y diferencias "
            f"razonablemente simétricas, resultan numéricamente cercanos (t = {p_t:.6f} frente a "
            f"signos = {p_exacto:.6f}) y coinciden en la decisión al 5 %, pero la cercanía numérica no "
            "los vuelve intercambiables: cada p-value responde a la pregunta de su propia referencia nula."
        ),
        "lectura_conjunta": (
            f"Ambos procedimientos coinciden en la decisión cualitativa (p >= 0.05: no rechazar H0; "
            f"IC t del 95 % {ic_txt}), pero el soporte es de naturaleza distinta: la t cuantifica la "
            "evidencia bajo un modelo poblacional normal (aproximado), y los cambios de signo ofrecen "
            "una garantía exacta bajo un supuesto de intercambiabilidad de signos que no usa "
            "distribuciones paramétricas. Reportar los dos p-values hace explícito qué parte de la "
            "conclusión depende de los supuestos."
        ),
    }

    # 8) resultados.json (contrato de la subtarea) ---------------------------
    resultados = {
        "subtarea": "T5_referencia_nula_exacta_cambios_de_signo",
        "fuente_diferencias": fuente,
        "definicion_diferencia": "difference = error_a - error_b, pareada por store_id",
        "n": n,
        "diferencias": [float(x) for x in d],
        "tabla_diferencias": [
            {"store_id": str(sid), "difference": float(dd)}
            for sid, dd in zip(df["store_id"], df["difference"])
        ],
        "suma_observada": obs_suma,
        "media_observada": obs_media,
        "suma_observada_en_decimas": umbral_dec,
        "media_observada_exacta": umbral_dec / (10.0 * n),
        "total_configuraciones": int(total),
        "conteo_configuraciones_extremas": conteo_extremas,
        "p_value_exacto_cambios_de_signo": p_exacto,
        "estadistico_t_T4": t_stat,
        "grados_de_libertad_t_T4": gl_t,
        "p_value_t_T4": p_t,
        "referencia_nula_cambios_de_signo": {
            "descripcion": (
                "Enumeración exhaustiva de las 2^16 configuraciones de signos s_i en {-1,+1} aplicadas "
                "al vector de diferencias observado; estadístico de cada configuración: "
                "m(s) = (1/n) * sum_i s_i * d_i."
            ),
            "estadistico_observado": obs_media,
            "total_configuraciones": int(total),
            "media_de_las_medias_nulas": media_nulas,
            "sd_de_las_medias_nulas": sd_nulas,
            "sd_teorica_sqrt_suma_d2_sobre_n": sd_nulas_teor,
            "min_media_nula": float(medias_nulas.min()),
            "max_media_nula": float(medias_nulas.max()),
            "numero_de_sumas_nulas_distintas": (
                dp.get("numero_sumas_distintas") if dp.get("aplicable") else None
            ),
            "distribucion_suma_decimas_cuenta": (
                dp.get("distribucion_suma_decimas_cuenta") if dp.get("aplicable") else None
            ),
        },
        "prueba_exacta_cambios_de_signo": {
            "criterio": "|media nula| >= |media observada| (bilateral, incluye la igualdad)",
            "media_observada": obs_media,
            "total_configuraciones": int(total),
            "conteo_extremas": conteo_extremas,
            "p_value_exacto": p_exacto,
            "p_value_fraccion": f"{conteo_extremas}/{total}",
            "p_value_fraccion_simplificada": fraccion,
            "cola_derecha_media_ge_observada": cola_der,
            "cola_izquierda_media_le_menos_observada": cola_izq,
            "configuracion_observada_incluida_en_conteo": obs_incluida,
            "configuracion_espejo_incluida_en_conteo": espejo_incluido,
        },
        "prueba_t_T4_procedimiento_principal": t4,
        "comparacion": {
            "p_value_t": p_t,
            "p_value_cambios_de_signo": p_exacto,
            "diferencia_absoluta": dif_abs,
            "razon_p_signos_sobre_p_t": razon,
            "estadistico_t": t_stat,
            "cociente_media_sobre_sd_referencia_signos": cociente_z,
            "misma_decision_cualitativa_al_5pct": misma5,
            "misma_decision_cualitativa_al_10pct": misma10,
            "decision": decision_txt,
        },
        "verificacion_independiente_dp_entera": (
            {k: v for k, v in dp.items() if k != "distribucion_suma_decimas_cuenta"}
            if dp.get("aplicable") else dp
        ),
        "verificaciones": {
            "n_diferencias_16": n == 16,
            "total_configuraciones_65536": total == 65536,
            "enumeracion_cubre_2_n_configuraciones": int(medias_nulas.size) == 65536,
            "colas_suman_conteo_extremas": (cola_der + cola_izq) == conteo_extremas,
            "distribucion_nula_simetrica_respecto_de_0": simetrica,
            "media_de_las_medias_nulas_cero": abs(media_nulas) < 1e-12,
            "configuracion_observada_contada": obs_incluida,
            "configuracion_espejo_contada": espejo_incluido,
            "dp_aplicable": bool(dp.get("aplicable", False)),
            "dp_suma_cuentas_65536": dp.get("suma_cuentas") == 65536,
            "dp_coincide_con_enumeracion": dp.get("conteo_extremas") == conteo_extremas,
            "p_value_exacto_en_rango_0_1": 0.0 <= p_exacto <= 1.0,
            "procedimiento_determinista_sin_semilla": True,
        },
        "discusion": discusion,
        "figuras": [
            "t5_distribucion_nula_cambios_signo.png",
            "t5_comparacion_p_values.png",
            "output/store_differences.png",
        ],
        "archivos_exigidos_enunciado": {
            "output/store_differences.csv": "tabla pareada con difference y favors (regenerada)",
            "output/store_differences.png": "figura de diferencias por tienda (regenerada)",
        },
        "nota_aleatoriedad": (
            "Procedimiento determinista: enumeración exhaustiva de las 2^16 configuraciones, sin "
            "remuestreo ni simulación; no requiere semilla."
        ),
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # 9) Resumen -------------------------------------------------------------
    print("=" * 70)
    print("T5 · Referencia nula exacta por cambios de signo (2^16 = 65536)")
    print("=" * 70)
    print(f"Fuente de diferencias   : {fuente}")
    print(f"n diferencias pareadas  : {n}")
    print(f"Media observada         : {obs_media:.6f}  (suma = {obs_suma:.4f})")
    print(f"Configuraciones totales : {total}")
    print(f"Extremas (|media| >= {abs_obs:.5f}): {conteo_extremas}")
    print(f"  cola derecha  (>= +obs): {cola_der}")
    print(f"  cola izquierda (<= -obs): {cola_izq}")
    print(f"p-value exacto          : {conteo_extremas}/65536 = {p_exacto:.6f}")
    print(f"Prueba t (T4)           : t({gl_t}) = {t_stat:.4f} | p bilateral = {p_t:.6f}")
    print(f"|p_signos - p_t| = {dif_abs:.6f} | razon signos/t = {razon_txt}")
    print(
        "Decision al 5%          : "
        + ("coincidente (no se rechaza H0 en ambas)" if misma5 else "divergente")
    )
    if dp.get("aplicable"):
        print(
            f"Verificacion DP entera  : conteo = {dp.get('conteo_extremas')} | "
            f"coincide = {dp.get('conteo_extremas') == conteo_extremas}"
        )
    print("Archivos escritos: resultados.json, t5_distribucion_nula_cambios_signo.png,")
    print("  t5_comparacion_p_values.png, output/store_differences.csv, output/store_differences.png")


if __name__ == "__main__":
    main()
