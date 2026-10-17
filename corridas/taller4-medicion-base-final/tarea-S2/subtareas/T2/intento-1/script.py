#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subtarea T2 — Cargar y validar data/model_errors.csv, verificar el emparejamiento
mediante store_id (no por longitud de vectores) y construir la tabla pareada de 16
filas con columnas store_id, error_a, error_b, difference, favors.

Salidas:
- output/store_differences.csv : tabla pareada (contrato del enunciado)
- output/store_differences.png : figura de evidencia por tienda
- resultados.json              : todas las cifras que pide la subtarea
El CSV original NUNCA se modifica (se evidencia con sha256 antes/después).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DATA_PATH = Path("data") / "model_errors.csv"
OUT_DIR = Path("output")
CSV_OUT = OUT_DIR / "store_differences.csv"
PNG_OUT = OUT_DIR / "store_differences.png"
RESULTS_PATH = Path("resultados.json")

EXPECTED_COLUMNS = ["store_id", "error_a", "error_b"]
EXPECTED_N_PAIRS = 16


def sha256_archivo(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validar_columnas(df: pd.DataFrame) -> None:
    """El CSV debe tener EXACTAMENTE las columnas store_id, error_a, error_b."""
    if len(df.columns) != len(EXPECTED_COLUMNS) or set(df.columns) != set(EXPECTED_COLUMNS):
        raise ValueError(
            f"El CSV debe tener exactamente las columnas {EXPECTED_COLUMNS}; "
            f"se encontraron {list(df.columns)}."
        )


def validar_store_id(df: pd.DataFrame) -> None:
    """store_id único y no nulo; el ValueError menciona el/los identificadores."""
    ids = df["store_id"]
    mask_nulos = ids.isna() | ids.astype(str).str.strip().eq("")
    if mask_nulos.any():
        filas = df.index[mask_nulos.to_numpy()].tolist()
        raise ValueError(
            f"store_id contiene valores nulos o vacíos en las filas {filas}; "
            "cada identificador debe estar presente."
        )
    mask_dup = ids.duplicated(keep=False)
    if mask_dup.any():
        dup = sorted({str(v) for v in ids[mask_dup].unique()})
        raise ValueError(f"store_id duplicado: {dup}; cada identificador debe ser único.")


def validar_errores(df: pd.DataFrame) -> pd.DataFrame:
    """Errores numéricos, finitos y no negativos; ValueError menciona store_id."""
    valido = df.copy()
    for col in ("error_a", "error_b"):
        serie = df[col]
        if pd.api.types.is_numeric_dtype(serie):
            valores = serie.astype(float)
        else:
            convertidos = pd.to_numeric(serie, errors="coerce")
            malos = df.loc[convertidos.isna() & serie.notna(), "store_id"].tolist()
            if malos:
                raise ValueError(
                    f"Columna '{col}': valores no numéricos para store_id={malos}."
                )
            valores = convertidos.astype(float)
        valido[col] = valores

        arr = valores.to_numpy()
        no_finitos = df.loc[~np.isfinite(arr), "store_id"].tolist()
        if no_finitos:
            raise ValueError(
                f"Columna '{col}': valores no finitos (NaN/inf) para store_id={no_finitos}."
            )
        negativos = df.loc[arr < 0, "store_id"].tolist()
        if negativos:
            detalle = {
                str(s): float(valido.loc[valido["store_id"] == s, col].iloc[0])
                for s in negativos
            }
            raise ValueError(f"Columna '{col}': valores negativos para {detalle}.")
    return valido


def verificar_emparejamiento_por_store_id(df: pd.DataFrame) -> bool:
    """
    Verificación del emparejamiento MEDIANTE store_id (no por longitud):
    un merge uno-a-uno sobre store_id debe reconstruir exactamente las mismas filas.
    """
    izq = df[["store_id", "error_a"]]
    der = df[["store_id", "error_b"]]
    try:
        merged = izq.merge(der, on="store_id", how="inner", validate="one_to_one")
    except pd.errors.MergeError as exc:
        raise ValueError(
            f"El emparejamiento por store_id no es uno-a-uno: {exc}"
        ) from exc
    if len(merged) != len(df) or set(merged["store_id"]) != set(df["store_id"]):
        raise ValueError(
            "Emparejamiento por store_id fallido: no todas las tiendas tienen "
            "error_a y error_b emparejados."
        )
    a = merged.sort_values("store_id").reset_index(drop=True)
    b = df.sort_values("store_id").reset_index(drop=True)
    if not a.equals(b):
        raise ValueError(
            "Emparejamiento por store_id fallido: los pares reconstruidos no "
            "coinciden con las filas originales."
        )
    return True


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {DATA_PATH}")

    sha_antes = sha256_archivo(DATA_PATH)

    # ---------- Carga y validaciones (el CSV original no se modifica) ----------
    df = pd.read_csv(DATA_PATH)
    validar_columnas(df)
    df = df[EXPECTED_COLUMNS]  # orden canónico; el conjunto exacto ya fue validado

    if len(df) != EXPECTED_N_PAIRS:
        raise ValueError(
            f"Se esperaban {EXPECTED_N_PAIRS} pares y el CSV tiene {len(df)} filas."
        )
    validar_store_id(df)
    df_valido = validar_errores(df)
    emparejado = verificar_emparejamiento_por_store_id(df_valido)

    # ---------- Tabla pareada ----------
    pareada = df_valido[EXPECTED_COLUMNS].copy()
    pareada["difference"] = pareada["error_a"] - pareada["error_b"]
    pareada["favors"] = np.select(
        [pareada["difference"] > 0, pareada["difference"] < 0],
        ["B", "A"],
        default="tie",
    )

    fav_A = pareada.loc[pareada["favors"] == "A", "store_id"].astype(str).tolist()
    fav_B = pareada.loc[pareada["favors"] == "B", "store_id"].astype(str).tolist()
    empates = pareada.loc[pareada["favors"] == "tie", "store_id"].astype(str).tolist()

    diffs = pareada["difference"].to_numpy()
    i_max = int(np.argmax(diffs))            # mayor diferencia a favor de B
    i_min = int(np.argmin(diffs))            # mayor magnitud a favor de A
    i_absmax = int(np.argmax(np.abs(diffs))) # mayor magnitud global
    fila_maxB = pareada.iloc[i_max]
    fila_maxA = pareada.iloc[i_min]
    fila_absmax = pareada.iloc[i_absmax]

    # Demostración de por qué la longitud no basta: permutar error_b mantiene
    # las longitudes pero rompe la correspondencia por tienda.
    error_b_perm = np.roll(pareada["error_b"].to_numpy(), 1)
    diff_perm = pareada["error_a"].to_numpy() - error_b_perm
    longitudes_iguales = len(pareada["error_a"]) == len(pareada["error_b"])
    diferencias_cambian = not np.allclose(diffs, diff_perm)

    # ---------- Archivos exigidos ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pareada.to_csv(CSV_OUT, index=False)

    x = np.arange(len(pareada))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    w = 0.38
    ax1.bar(x - w / 2, pareada["error_a"], width=w, color="#4C72B0", label="error_a (modelo A)")
    ax1.bar(x + w / 2, pareada["error_b"], width=w, color="#DD8452", label="error_b (modelo B)")
    ax1.set_ylabel("Error")
    ax1.set_title("T2 · Errores por tienda y diferencia pareada (error_a − error_b)")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(axis="y", alpha=0.3)

    colores = {"B": "#4C72B0", "A": "#DD8452", "tie": "#7F7F7F"}
    ax2.bar(x, pareada["difference"], color=[colores[f] for f in pareada["favors"]])
    ax2.axhline(0.0, color="black", linewidth=0.8)
    for xi, d in zip(x, diffs):
        ax2.text(xi, d + (0.05 if d >= 0 else -0.05), f"{d:+.2f}",
                 ha="center", va="bottom" if d >= 0 else "top", fontsize=8)
    ax2.text(i_max, diffs[i_max] + 0.38, "máx a favor de B", ha="center", va="bottom",
             fontsize=8, fontweight="bold")
    ax2.text(i_min, diffs[i_min] - 0.38, "máx a favor de A", ha="center", va="top",
             fontsize=8, fontweight="bold")
    ax2.set_ylabel("difference = error_a − error_b")
    ax2.set_xlabel("store_id")
    ax2.set_xticks(x)
    ax2.set_xticklabels(pareada["store_id"].astype(str))
    handles = [Patch(color=colores["B"], label="favors = 'B' (difference > 0)"),
               Patch(color=colores["A"], label="favors = 'A' (difference < 0)")]
    ax2.legend(handles=handles, loc="lower right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=120)
    plt.close(fig)

    sha_despues = sha256_archivo(DATA_PATH)

    # ---------- resultados.json (contrato) ----------
    filas = [
        {
            "store_id": str(r.store_id),
            "error_a": float(r.error_a),
            "error_b": float(r.error_b),
            "difference": float(r.difference),
            "favors": str(r.favors),
        }
        for r in pareada.itertuples(index=False)
    ]
    resultados = {
        "subtarea": "T2_cargar_validar_y_emparejar_model_errors",
        "archivo_entrada": str(DATA_PATH),
        "validaciones": {
            "columnas_exactas_store_id_error_a_error_b": True,
            "store_id_unico_y_no_nulo": True,
            "errores_numericos_finitos_no_negativos": True,
            "n_pares_esperados": EXPECTED_N_PAIRS,
            "csv_original_no_modificado": bool(sha_antes == sha_despues),
        },
        "csv_sha256_antes": sha_antes,
        "csv_sha256_despues": sha_despues,
        "emparejamiento_por_store_id": {
            "verificado": bool(emparejado),
            "metodo": (
                "merge uno-a-uno sobre store_id y comparación de las filas "
                "reconstruidas con las originales (no por longitud de vectores)"
            ),
            "n_pares": int(len(pareada)),
        },
        "por_que_la_comprobacion_de_longitud_es_insuficiente": (
            "Que len(error_a) == len(error_b) solo confirma el número de valores, no su "
            "correspondencia: cualquier permutación de una columna mantiene la longitud "
            "pero cambia qué error_b pertenece a cada tienda. La correspondencia se "
            "garantiza verificando la clave store_id (única y no nula) mediante un "
            "merge uno-a-uno sobre store_id."
        ),
        "demostracion_longitud_insuficiente": {
            "longitudes_iguales_tras_permutar_error_b": bool(longitudes_iguales),
            "diferencias_cambian_tras_permutar_error_b": bool(diferencias_cambian),
        },
        "tabla_pareada": {
            "n_filas": int(len(pareada)),
            "columnas": list(pareada.columns),
            "filas": filas,
        },
        "conteo_favors": {"A": len(fav_A), "B": len(fav_B), "tie": len(empates)},
        "tiendas_favorecen_A": fav_A,
        "tiendas_favorecen_B": fav_B,
        "tiendas_en_empate": empates,
        "mayor_diferencia_favorece_B": {
            "store_id": str(fila_maxB["store_id"]),
            "difference": float(fila_maxB["difference"]),
        },
        "mayor_diferencia_favorece_A": {
            "store_id": str(fila_maxA["store_id"]),
            "difference": float(fila_maxA["difference"]),
        },
        "mayor_magnitud_absoluta": {
            "store_id": str(fila_absmax["store_id"]),
            "difference": float(fila_absmax["difference"]),
            "magnitud": float(abs(fila_absmax["difference"])),
        },
        "archivos_generados": [str(CSV_OUT), str(PNG_OUT)],
    }
    RESULTS_PATH.write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------- Resumen ----------
    print("T2 · Resumen")
    print(f"  CSV validado: {DATA_PATH} (columnas exactas, store_id único/no nulo, "
          "errores numéricos, finitos y no negativos)")
    print(f"  Emparejamiento verificado por store_id: {emparejado} · pares: {len(pareada)}")
    print(f"  Favorecen a A ({len(fav_A)}): {fav_A}")
    print(f"  Favorecen a B ({len(fav_B)}): {fav_B}")
    print(f"  Empates: {len(empates)}")
    print(f"  Mayor diferencia a favor de B: {fila_maxB['store_id']} "
          f"({fila_maxB['difference']:+.4f})")
    print(f"  Mayor diferencia a favor de A: {fila_maxA['store_id']} "
          f"({fila_maxA['difference']:+.4f})")
    print(f"  CSV original sin modificar (sha256 antes = después): {sha_antes == sha_despues}")
    print(f"  Archivos generados: {CSV_OUT}, {PNG_OUT}, {RESULTS_PATH}")


if __name__ == "__main__":
    main()
