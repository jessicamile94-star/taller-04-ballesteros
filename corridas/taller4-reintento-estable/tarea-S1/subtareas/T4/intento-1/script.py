#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4 — Construir un resumen por servidor.

Agrupa el DataFrame de T3 (con load_score y requires_review) por la columna
`server` y construye una tabla con una fila por servidor y exactamente estas
columnas, en este orden:

    server, observations, mean_power_w, max_temperature_c, mean_load, review_count

Significado de las métricas:
    observations      : número de filas del servidor
    mean_power_w      : media de power_w (W)
    max_temperature_c : máximo de temperature_c (°C)
    mean_load         : media de load_score
    review_count      : conteo de filas con requires_review == True

Salidas:
    - output/server_analysis.parquet : tabla resumen (exigida por el enunciado)
    - resultados.json                : contrato de la subtarea (cifras completas)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rutas y constantes
# ---------------------------------------------------------------------------
RUTA_DATOS = Path("data/server_measurements.csv")
RUTA_PARQUET = Path("output/server_analysis.parquet")
RUTA_RESULTADOS = Path("resultados.json")

COLUMNAS_TABLA = ["server", "observations", "mean_power_w",
                  "max_temperature_c", "mean_load", "review_count"]
COL_NECESARIAS = ["server", "power_w", "temperature_c", "load_score", "requires_review"]

# Fila de referencia del enunciado (AI-SRV-01)
ESPERADO_AI01 = {"observations": 50, "mean_power_w": 363.4598,
                 "max_temperature_c": 86.50, "mean_load": -0.614527,
                 "review_count": 2}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def leer_tabla(path):
    """Lee un archivo parquet/csv; devuelve DataFrame o None."""
    try:
        suf = Path(path).suffix.lower()
        if suf == ".parquet":
            return pd.read_parquet(path)
        if suf in (".csv", ".txt"):
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def normalizar_columnas(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def a_bool(serie):
    """Convierte una serie a bool de forma robusta."""
    if serie.dtype == bool:
        return serie
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce").fillna(0) > 0.5
    return serie.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def cargar_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def fnum(x):
    """float JSON-seguro (NaN/inf -> None)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(v) or np.isinf(v)) else v


def sanitizar(o):
    """Convierte escalares de numpy a tipos JSON."""
    if isinstance(o, dict):
        return {k: sanitizar(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitizar(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return fnum(float(o))
    if isinstance(o, float):
        return fnum(o)
    return o


# ---------------------------------------------------------------------------
# Búsqueda de artefactos de subtareas previas (entrada/)
# ---------------------------------------------------------------------------
def buscar_resultados_t3():
    raiz = Path("entrada")
    if not raiz.exists():
        return None, None
    for p in sorted(raiz.rglob("*.json")):
        if "t3" not in str(p).lower():
            continue
        j = cargar_json(p)
        if isinstance(j, dict) and ("indices_requires_review" in j or
                                    "filas_requires_review" in j):
            return j, str(p)
    return None, None


def _archivos_datos_previos():
    raiz = Path("entrada")
    if not raiz.exists():
        return []
    return [p for p in raiz.rglob("*")
            if p.is_file() and p.suffix.lower() in (".parquet", ".csv")]


def _prioridad_ruta(p, token):
    s = str(p).lower()
    return (0 if token in s else 1, 0 if s.endswith(".parquet") else 1, s)


def buscar_df_completo(n_esperado):
    """Busca el DataFrame de T3 (todas las columnas necesarias y longitud correcta)."""
    for p in sorted(_archivos_datos_previos(), key=lambda q: _prioridad_ruta(q, "t3")):
        df = leer_tabla(p)
        if df is None:
            continue
        df = normalizar_columnas(df)
        if not all(c in df.columns for c in COL_NECESARIAS):
            continue
        if n_esperado is not None and len(df) != n_esperado:
            continue
        return df.reset_index(drop=True), str(p)
    return None, None


def buscar_lista_load_score(n):
    """Busca una lista numérica de longitud n (load_score) en resultados.json previos."""
    raiz = Path("entrada")
    if not raiz.exists():
        return None, None, None
    hallazgos = []

    def recorrer(obj, ruta, origen):
        if isinstance(obj, list):
            if len(obj) == n and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj):
                hallazgos.append((origen, ruta, [float(v) for v in obj]))
            for i, v in enumerate(obj):
                recorrer(v, f"{ruta}[{i}]", origen)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                recorrer(v, f"{ruta}.{k}", origen)

    for p in sorted(raiz.rglob("*.json")):
        j = cargar_json(p)
        if j is not None:
            recorrer(j, "$", str(p))
    if not hallazgos:
        return None, None, None

    def puntualidad(h):
        s = (h[0] + " " + h[1]).lower()
        return (0 if "load" in s else 1, 0 if "t2" in s else 1, s)

    hallazgos.sort(key=puntualidad)
    return hallazgos[0][2], hallazgos[0][0], hallazgos[0][1]


def buscar_archivo_load_score(n):
    """Busca un archivo previo con columna load_score (aunque falte requires_review)."""
    for p in sorted(_archivos_datos_previos(), key=lambda q: _prioridad_ruta(q, "t2")):
        df = leer_tabla(p)
        if df is None:
            continue
        df = normalizar_columnas(df)
        if "load_score" in df.columns and len(df) == n:
            return df.reset_index(drop=True), str(p)
    return None, None


# ---------------------------------------------------------------------------
# Validación contra la referencia de T3 y recuperación de load_score
# ---------------------------------------------------------------------------
def diferencias_con_referencia(df_ref, valores, filas_ref):
    """Compara `valores` (alineados con df_ref) con los load_score conocidos que
    T3 reportó para las filas marcadas (emparejamiento por server+timestamp).
    Devuelve (n_comparadas, diferencia_maxima)."""
    if valores is None or not filas_ref:
        return 0, float("inf")
    if not isinstance(df_ref, pd.DataFrame) or \
            not {"server", "timestamp"}.issubset(df_ref.columns):
        return 0, float("inf")
    ts = pd.to_datetime(df_ref["timestamp"], errors="coerce")
    pos = {}
    for i in range(len(df_ref)):
        pos.setdefault((str(df_ref["server"].iloc[i]).strip(), ts.iloc[i]), i)
    diffs = []
    for f in filas_ref:
        if not isinstance(f, dict) or "load_score" not in f:
            continue
        try:
            clave = (str(f.get("server")).strip(),
                     pd.to_datetime(f.get("timestamp"), errors="coerce"))
        except Exception:
            continue
        if pd.isna(clave[1]):
            continue
        i = pos.get(clave)
        if i is None or i >= len(valores):
            continue
        try:
            diffs.append(abs(float(valores[i]) - float(f["load_score"])))
        except (TypeError, ValueError):
            continue
    if not diffs:
        return 0, float("inf")
    return len(diffs), float(max(diffs))


def decidir_aceptacion(n_comp, dmax):
    if n_comp == 0:
        return True, "sin_coincidencia_directa"
    if dmax < 1e-9:
        return True, "exacta"
    if dmax < 1e-6:
        return True, "precision_limitada"
    return False, "no_coincide"


def z_matriz(df, ddof):
    cols = ["gpu_utilization", "cpu_utilization", "memory_gb"]
    if not all(c in df.columns for c in cols):
        return None
    zs = []
    for c in cols:
        s = df[c].astype(float)
        sd = s.std(ddof=ddof)
        sd = 1.0 if (pd.isna(sd) or sd == 0) else float(sd)
        zs.append(((s - s.mean()) / sd).to_numpy(dtype=float))
    return np.column_stack(zs)


def recuperar_load_score(df, filas_ref, idx_pos=None):
    """Deduce load_score ajustando modelos candidatos a los valores conocidos de T3."""
    info = {"metodo": None, "exacto": False, "residuo_max": None, "n_puntos": 0}
    if not filas_ref:
        info["nota"] = "sin_valores_de_referencia"
        return None, info
    if not {"server", "timestamp"}.issubset(df.columns):
        info["nota"] = "sin_claves_para_emparejar"
        return None, info
    try:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        pos = {}
        for i in range(len(df)):
            pos.setdefault((str(df["server"].iloc[i]).strip(), ts.iloc[i]), i)
        idx, val = [], []
        for f in filas_ref:
            if not isinstance(f, dict) or "load_score" not in f:
                continue
            clave = (str(f.get("server")).strip(),
                     pd.to_datetime(f.get("timestamp"), errors="coerce"))
            if pd.isna(clave[1]):
                continue
            i = pos.get(clave)
            if i is not None:
                idx.append(i)
                val.append(float(f["load_score"]))
        info["emparejamiento"] = "por server+timestamp"
        if len(idx) < 5 and idx_pos is not None and len(idx_pos) == len(filas_ref):
            idx, val = [], []
            for i, f in zip(idx_pos, filas_ref):
                if isinstance(f, dict) and "load_score" in f and 0 <= int(i) < len(df):
                    idx.append(int(i))
                    val.append(float(f["load_score"]))
            info["emparejamiento"] = "posicional (índices de T3)"
        if len(idx) < 5:
            info["nota"] = "referencia_insuficiente"
            return None, info
        idx = np.asarray(idx, dtype=int)
        val = np.asarray(val, dtype=float)
        info["n_puntos"] = int(len(idx))

        candidatos = []
        familias = [
            ["gpu_utilization", "cpu_utilization", "memory_gb"],
            ["gpu_utilization", "cpu_utilization", "memory_gb", "power_w"],
            ["gpu_utilization", "cpu_utilization", "memory_gb", "power_w", "temperature_c"],
        ]
        for feats in familias:
            if not all(c in df.columns for c in feats):
                continue
            Xk = df.iloc[idx][feats].to_numpy(dtype=float)
            A = np.column_stack([np.ones(len(idx)), Xk])
            coef, *_ = np.linalg.lstsq(A, val, rcond=None)
            resid = float(np.max(np.abs(A @ coef - val)))
            candidatos.append((resid, "afin", feats, coef))

        for ddof in (0, 1):
            Z = z_matriz(df, ddof)
            if Z is None:
                continue
            resid = float(np.max(np.abs(Z[idx].max(axis=1) - val)))
            candidatos.append((resid, "maxz", ddof, None))

        if not candidatos:
            info["nota"] = "sin_atributos_para_ajustar"
            return None, info

        candidatos.sort(key=lambda t: t[0])
        resid, tipo, arg, coef = candidatos[0]
        info["residuo_max"] = resid
        info["exacto"] = bool(resid < 1e-6)
        if tipo == "afin":
            X = np.column_stack([np.ones(len(df)), df[arg].to_numpy(dtype=float)])
            info["metodo"] = "combinacion_afin_de[" + "+".join(arg) + "]"
            return X @ coef, info
        Z = z_matriz(df, arg)
        info["metodo"] = f"maximo_de_zscores(ddof={arg})"
        return Z.max(axis=1), info
    except Exception as exc:
        info["nota"] = f"error_en_ajuste: {exc}"
        return None, info


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    print("=" * 76)
    print("T4 — Construir un resumen por servidor (agrupación por `server`)")
    print("=" * 76)

    info = {}

    # 0) Referencias de T3 ---------------------------------------------------
    t3_json, ruta_t3json = buscar_resultados_t3()
    idx_t3, filas_ref = None, []
    if t3_json is not None:
        idx_t3 = t3_json.get("indices_requires_review")
        filas_ref = t3_json.get("filas_requires_review") or []
        print(f"[i] Referencia de T3: {ruta_t3json}")

    base = None
    if RUTA_DATOS.exists():
        base = pd.read_csv(RUTA_DATOS)
        cols_norm = [str(c).strip().lower().replace(" ", "_") for c in base.columns]
        if "server" not in cols_norm:
            for sep in (";", "\t"):
                try:
                    alt = pd.read_csv(RUTA_DATOS, sep=sep)
                    if any(str(c).strip().lower().replace(" ", "_") == "server"
                           for c in alt.columns):
                        base = alt
                        break
                except Exception:
                    continue
        base = normalizar_columnas(base)
        if "timestamp" in base.columns:
            base["timestamp"] = pd.to_datetime(base["timestamp"], errors="coerce")
        base = base.reset_index(drop=True)
        print(f"[i] Datos base: {RUTA_DATOS} ({len(base)} filas)")

    n_esperado = None
    if t3_json is not None and isinstance(t3_json.get("n_observaciones"), int):
        n_esperado = int(t3_json["n_observaciones"])
    elif base is not None:
        n_esperado = len(base)

    # 1) DataFrame completo de T3 (camino preferente) ------------------------
    work = None
    df_t3, ruta_t3 = buscar_df_completo(n_esperado)
    if df_t3 is not None:
        cand_ls = pd.to_numeric(df_t3["load_score"], errors="coerce").to_numpy(dtype=float)
        n_comp, dmax = diferencias_con_referencia(df_t3, cand_ls, filas_ref)
        ok, modo = decidir_aceptacion(n_comp, dmax)
        if ok:
            work = df_t3
            info["origen_dataframe"] = f"DataFrame de T3: {ruta_t3}"
            info["origen_load_score"] = f"columna load_score de {ruta_t3} (coincidencia: {modo})"
            info["origen_requires_review"] = f"columna requires_review de {ruta_t3}"
            print(f"[OK] DataFrame de T3 cargado desde {ruta_t3} ({len(work)} filas)")
        else:
            print(f"[!] {ruta_t3} no coincide con la referencia de T3; se descarta.")

    # 2) Reconstrucción si no hay DataFrame completo --------------------------
    if work is None:
        if base is None:
            raise SystemExit("ERROR: sin DataFrame de T3 ni data/server_measurements.csv.")
        work = base.copy()
        info["origen_dataframe"] = f"reconstruido desde {RUTA_DATOS}"

        valores, fuente_ls = None, None

        # 2a) lista numérica de load_score en resultados.json previos
        lista, origen_l, ruta_l = buscar_lista_load_score(len(work))
        if lista is not None:
            n_comp, dmax = diferencias_con_referencia(
                work, np.asarray(lista, dtype=float), filas_ref)
            ok, modo = decidir_aceptacion(n_comp, dmax)
            if ok:
                valores = np.asarray(lista, dtype=float)
                fuente_ls = f"lista de load_score en {origen_l} ({ruta_l}); coincidencia: {modo}"
                print(f"[OK] load_score tomado de la lista en {origen_l}")

        # 2b) archivo previo con columna load_score
        if valores is None:
            df_ls, ruta_ls = buscar_archivo_load_score(len(work))
            if df_ls is not None:
                cand = pd.to_numeric(df_ls["load_score"], errors="coerce").to_numpy(dtype=float)
                if {"server", "timestamp"}.issubset(df_ls.columns):
                    n_comp, dmax = diferencias_con_referencia(df_ls, cand, filas_ref)
                else:
                    n_comp, dmax = diferencias_con_referencia(work, cand, filas_ref)
                ok, modo = decidir_aceptacion(n_comp, dmax)
                if ok:
                    valores = cand
                    fuente_ls = f"columna load_score de {ruta_ls}; coincidencia: {modo}"
                    print(f"[OK] load_score tomado de {ruta_ls}")

        # 2c) deducir la fórmula a partir de los valores conocidos de T3
        if valores is None:
            valores, info_rec = recuperar_load_score(work, filas_ref, idx_pos=idx_t3)
            if valores is not None:
                fuente_ls = (f"load_score deducido por ajuste ({info_rec['metodo']}; "
                             f"exacto={info_rec['exacto']}, "
                             f"residuo_max={info_rec['residuo_max']})")
                info["recuperacion_load_score"] = info_rec
                print(f"[~] load_score deducido por ajuste (exacto={info_rec['exacto']})")

        if valores is None:
            raise SystemExit("ERROR: no se pudo obtener load_score para todas las filas.")
        work["load_score"] = valores
        info["origen_load_score"] = fuente_ls

        # requires_review: índices autoritativos de T3 o regla del enunciado
        if idx_t3 is not None and len(idx_t3) <= len(work):
            req = np.zeros(len(work), dtype=bool)
            req[np.asarray(idx_t3, dtype=int)] = True
            work["requires_review"] = req
            info["origen_requires_review"] = f"índices_requires_review de {ruta_t3json}"
        elif {"load_score", "temperature_c"}.issubset(work.columns):
            work["requires_review"] = (work["load_score"] > 1.5) | (work["temperature_c"] > 80)
            info["origen_requires_review"] = "regla: load_score>1.5 o temperature_c>80"
        else:
            raise SystemExit("ERROR: no se puede reconstruir requires_review.")

    # 3) Normalización de tipos ----------------------------------------------
    work["server"] = work["server"].astype(str)
    work["requires_review"] = a_bool(work["requires_review"]).astype(bool)
    work["load_score"] = pd.to_numeric(work["load_score"], errors="coerce")
    for c in ("power_w", "temperature_c"):
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    if {"load_score", "temperature_c"}.issubset(work.columns):
        regla = (work["load_score"] > 1.5) | (work["temperature_c"] > 80)
        info["coherencia_regla_t3"] = bool(np.array_equal(
            regla.to_numpy(dtype=bool), work["requires_review"].to_numpy(dtype=bool)))

    n_review_total = int(work["requires_review"].sum())
    n_nulos_load = int(work["load_score"].isna().sum())

    # 4) Agrupación por server y tabla resumen --------------------------------
    g = work.groupby("server", sort=True)
    resumen = pd.DataFrame({
        "observations": g.size(),
        "mean_power_w": g["power_w"].mean(),
        "max_temperature_c": g["temperature_c"].max(),
        "mean_load": g["load_score"].mean(),
        "review_count": g["requires_review"].sum().astype("int64"),
    }).reset_index()
    resumen = resumen[COLUMNAS_TABLA]
    resumen["observations"] = resumen["observations"].astype("int64")
    resumen["review_count"] = resumen["review_count"].astype("int64")

    n_serv = len(resumen)
    ok_6filas = (n_serv == 6)
    ok_orden = list(resumen.columns) == COLUMNAS_TABLA

    # 5) Verificación contra el resultado esperado (AI-SRV-01) ----------------
    verif = {}
    f01 = resumen.loc[resumen["server"] == "AI-SRV-01"]
    if len(f01) == 1:
        r = f01.iloc[0]
        pares = [
            ("observations", int(r["observations"]), ESPERADO_AI01["observations"],
             lambda a, e: a == e),
            ("mean_power_w", float(r["mean_power_w"]), ESPERADO_AI01["mean_power_w"],
             lambda a, e: abs(a - e) < 5e-5),
            ("max_temperature_c", float(r["max_temperature_c"]),
             ESPERADO_AI01["max_temperature_c"], lambda a, e: abs(a - e) < 1e-6),
            ("mean_load", float(r["mean_load"]), ESPERADO_AI01["mean_load"],
             lambda a, e: abs(a - e) < 5e-7),
            ("review_count", int(r["review_count"]), ESPERADO_AI01["review_count"],
             lambda a, e: a == e),
        ]
        for nombre, obtenido, esperado, cmp_ in pares:
            verif[nombre] = {"obtenido": obtenido, "esperado": esperado,
                             "ok": bool(cmp_(obtenido, esperado))}
    exito = bool(ok_6filas and ok_orden and verif and all(v["ok"] for v in verif.values()))

    # 6) Tabla completa por pantalla ------------------------------------------
    print("\nTabla resumen por servidor:")
    disp = resumen.copy()
    disp["mean_power_w"] = disp["mean_power_w"].map(lambda v: f"{v:.4f}")
    disp["max_temperature_c"] = disp["max_temperature_c"].map(lambda v: f"{v:.2f}")
    disp["mean_load"] = disp["mean_load"].map(lambda v: f"{v:.6f}")
    print(disp.to_string(index=False))

    # 7) Parquet exigido por el enunciado -------------------------------------
    RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    resumen.to_parquet(str(RUTA_PARQUET), index=False, engine="pyarrow")
    chk = pd.read_parquet(str(RUTA_PARQUET))
    print(f"\n[OK] Parquet escrito: {RUTA_PARQUET} "
          f"({chk.shape[0]} filas x {chk.shape[1]} columnas)")

    # 8) resultados.json (contrato) -------------------------------------------
    tabla, tabla_red = [], []
    for _, r in resumen.iterrows():
        tabla.append({
            "server": str(r["server"]),
            "observations": int(r["observations"]),
            "mean_power_w": fnum(r["mean_power_w"]),
            "max_temperature_c": fnum(r["max_temperature_c"]),
            "mean_load": fnum(r["mean_load"]),
            "review_count": int(r["review_count"]),
        })
        tabla_red.append({
            "server": str(r["server"]),
            "observations": int(r["observations"]),
            "mean_power_w": round(float(r["mean_power_w"]), 4),
            "max_temperature_c": round(float(r["max_temperature_c"]), 2),
            "mean_load": round(float(r["mean_load"]), 6),
            "review_count": int(r["review_count"]),
        })

    resultados = {
        "subtarea": "T4_resumen_por_servidor",
        "descripcion": ("Tabla con una fila por servidor y columnas: server, observations, "
                        "mean_power_w, max_temperature_c, mean_load, review_count"),
        "agrupacion": "server",
        "fuente_datos": info.get("origen_dataframe", str(RUTA_DATOS)),
        "origen_load_score": info.get("origen_load_score",
                                      "columna load_score del DataFrame de T3"),
        "origen_requires_review": info.get("origen_requires_review",
                                           "columna requires_review del DataFrame de T3"),
        "n_observaciones_total": int(len(work)),
        "n_servidores": int(n_serv),
        "columnas_tabla": COLUMNAS_TABLA,
        "orden_columnas_correcto": bool(ok_orden),
        "tabla_con_6_filas": bool(ok_6filas),
        "tabla_resumen": tabla,
        "tabla_resumen_redondeada": tabla_red,
        "ai_srv_01": tabla[0] if tabla else None,
        "verificacion_ai_srv_01": verif,
        "criterio_exitoso": exito,
        "n_requires_review_total": n_review_total,
        "load_score_nulos": n_nulos_load,
        "archivo_parquet": str(RUTA_PARQUET),
    }
    for clave in ("coherencia_regla_t3", "recuperacion_load_score"):
        if clave in info:
            resultados[clave] = info[clave]

    RUTA_RESULTADOS.write_text(json.dumps(sanitizar(resultados), indent=2,
                                          ensure_ascii=False), encoding="utf-8")

    # 9) Resumen breve ---------------------------------------------------------
    print(f"\nResumen: {n_serv} servidores | {len(work)} observaciones | "
          f"{n_review_total} filas marcadas para revisión")
    print(f"Columnas en orden exigido: {ok_orden} | 6 filas: {ok_6filas}")
    for k, v in verif.items():
        estado = "OK" if v["ok"] else "FALLA"
        print(f"  AI-SRV-01 · {k}: obtenido={v['obtenido']} | esperado={v['esperado']} [{estado}]")
    print(f"Criterio de éxito: {'CUMPLIDO' if exito else 'NO CUMPLIDO — revisar'}")
    print(f"Archivos escritos: {RUTA_PARQUET} , {RUTA_RESULTADOS}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(f"[ERROR] {e}")
        Path("resultados.json").write_text(
            json.dumps({"subtarea": "T4_resumen_por_servidor", "error": str(e),
                        "criterio_exitoso": False}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        raise
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        Path("resultados.json").write_text(
            json.dumps({"subtarea": "T4_resumen_por_servidor",
                        "error": f"{type(e).__name__}: {e}",
                        "criterio_exitoso": False}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        raise
