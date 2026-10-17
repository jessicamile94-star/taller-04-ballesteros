#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subtarea T2: Auditar los datos y construir los pares.

- Carga data/model_errors.csv validando:
    * exactamente las columnas store_id, error_a, error_b
    * store_id único y no nulo (ValueError que menciona el identificador)
    * errores numéricos, finitos y no negativos (ValueError)
- Construye la tabla pareada (store_id, error_a, error_b, difference, favors)
  verificando el emparejamiento mediante store_id (no por longitud de vectores).
- Guarda output/store_differences.csv y output/store_differences.png.
- Escribe resultados.json con todas las cifras pedidas.
- El CSV original no se modifica (verificado por hash antes/después).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPECTED_COLS = ["store_id", "error_a", "error_b"]
CSV_PATH = Path("data/model_errors.csv")
OUT_DIR = Path("output")
OUT_CSV = OUT_DIR / "store_differences.csv"
OUT_PNG = OUT_DIR / "store_differences.png"
RESULTADOS_PATH = Path("resultados.json")


def hash_archivo(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Deben existir exactamente las columnas store_id, error_a, error_b."""
    cols = list(df.columns)
    if len(cols) != len(EXPECTED_COLS) or set(cols) != set(EXPECTED_COLS):
        faltantes = [c for c in EXPECTED_COLS if c not in cols]
        extra = [c for c in cols if c not in EXPECTED_COLS]
        raise ValueError(
            f"El CSV debe tener exactamente las columnas {EXPECTED_COLS}. "
            f"Faltantes: {faltantes}; inesperadas: {extra}; encontradas: {cols}."
        )
    return df[EXPECTED_COLS].copy()


def validar_store_id(df: pd.DataFrame) -> None:
    """Cada identificador debe ser único y no nulo; el ValueError lo menciona."""
    ids = df["store_id"]
    if ids.isna().any():
        filas = df.index[ids.isna()].tolist()
        raise ValueError(f"store_id: identificador nulo en filas {filas}.")
    if pd.api.types.is_object_dtype(ids) or pd.api.types.is_string_dtype(ids):
        vacios = ids.astype(str).str.strip().eq("")
        if vacios.any():
            raise ValueError(
                f"store_id: identificador nulo/vacío en filas {df.index[vacios].tolist()}."
            )
    dup_mask = ids.duplicated(keep=False)
    if dup_mask.any():
        dups = sorted({str(v) for v in ids[dup_mask].unique()})
        raise ValueError(f"store_id: identificadores duplicados: {dups}.")


def validar_error(df: pd.DataFrame, col: str) -> np.ndarray:
    """Los errores deben ser numéricos, finitos y no negativos."""
    serie = df[col]
    if pd.api.types.is_numeric_dtype(serie):
        nums = serie.astype(float)
    else:
        nums = pd.to_numeric(serie, errors="coerce").astype(float)
    no_validos = ~np.isfinite(nums)  # cubre NaN (nulo o no numérico) y ±inf
    if no_validos.any():
        detalle = [
            (str(df.at[i, "store_id"]), repr(serie.at[i])) for i in df.index[no_validos]
        ]
        raise ValueError(
            f"{col}: valores no numéricos, nulos o no finitos en "
            f"pares (store_id, valor): {detalle}."
        )
    negativos = nums < 0
    if negativos.any():
        detalle = [
            (str(df.at[i, "store_id"]), float(nums.at[i])) for i in df.index[negativos]
        ]
        raise ValueError(f"{col}: valores negativos en pares (store_id, valor): {detalle}.")
    return nums.to_numpy()


def verificar_pares_por_store_id(tabla: pd.DataFrame) -> None:
    """
    El emparejamiento se confirma por identificador: cada store_id aparece
    exactamente una vez y difference/favors se recalculan buscando cada id
    en la tabla (comprobar longitudes iguales no demuestra correspondencia).
    """
    if tabla["store_id"].isna().any() or tabla["store_id"].duplicated().any():
        raise ValueError("Emparejamiento inválido: store_id debe ser único y no nulo.")
    pares = {
        str(sid): (float(a), float(b))
        for sid, a, b in zip(tabla["store_id"], tabla["error_a"], tabla["error_b"])
    }
    for sid, (a, b) in pares.items():
        fila = tabla.loc[tabla["store_id"] == sid]
        if len(fila) != 1:
            raise ValueError(f"El identificador {sid} no aparece exactamente una vez.")
        fila = fila.iloc[0]
        if not (np.isclose(fila["error_a"], a) and np.isclose(fila["error_b"], b)):
            raise ValueError(f"Par inconsistente para {sid}.")
        if not np.isclose(fila["difference"], a - b):
            raise ValueError(f"Difference mal emparejada para {sid}.")
        esperado = "B" if (a - b) > 0 else ("A" if (a - b) < 0 else "")
        if str(fila["favors"]) != esperado:
            raise ValueError(f"Favors mal emparejado para {sid}.")


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {CSV_PATH}")

    hash_antes = hash_archivo(CSV_PATH)

    df = pd.read_csv(CSV_PATH)
    df = validar_columnas(df)
    validar_store_id(df)
    error_a = validar_error(df, "error_a")
    error_b = validar_error(df, "error_b")

    # Tabla pareada: se conservan las tres columnas originales y se añaden
    # difference y favors, alineadas fila a fila con store_id.
    tabla = pd.DataFrame(
        {
            "store_id": df["store_id"].astype(str),
            "error_a": error_a,
            "error_b": error_b,
        }
    )
    tabla["difference"] = tabla["error_a"] - tabla["error_b"]
    tabla["favors"] = np.where(
        tabla["difference"] > 0, "B", np.where(tabla["difference"] < 0, "A", "")
    )

    verificar_pares_por_store_id(tabla)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(OUT_CSV, index=False)

    # Figura: diferencia por tienda, coloreada según a quién favorece.
    colores = [
        "#2a9d8f" if f == "B" else ("#e76f51" if f == "A" else "#9e9e9e")
        for f in tabla["favors"]
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(tabla["store_id"], tabla["difference"], color=colores)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("store_id")
    ax.set_ylabel("difference = error_a - error_b")
    ax.set_title("Diferencia de error por tienda (verde: favorece a B, rojo: favorece a A)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120)
    plt.close(fig)

    # Verificación del archivo escrito y de que el CSV original quedó intacto.
    chequeo = pd.read_csv(OUT_CSV)
    filas, columnas = chequeo.shape
    hash_despues = hash_archivo(CSV_PATH)
    csv_intacto = bool(hash_antes == hash_despues)

    n_A = int((tabla["favors"] == "A").sum())
    n_B = int((tabla["favors"] == "B").sum())
    n_empates = int((tabla["difference"] == 0).sum())

    mayor_B = None
    if n_B > 0:
        i = tabla["difference"].idxmax()
        mayor_B = {
            "store_id": str(tabla.at[i, "store_id"]),
            "difference": float(tabla.at[i, "difference"]),
        }
    mayor_A = None
    if n_A > 0:
        i = tabla["difference"].idxmin()
        mayor_A = {
            "store_id": str(tabla.at[i, "store_id"]),
            "difference": float(tabla.at[i, "difference"]),
        }

    tabla_records = [
        {
            "store_id": str(r["store_id"]),
            "error_a": float(r["error_a"]),
            "error_b": float(r["error_b"]),
            "difference": float(r["difference"]),
            "favors": str(r["favors"]),
        }
        for r in tabla.to_dict(orient="records")
    ]

    resultados = {
        "subtarea": "T2_auditar_datos_y_construir_pares",
        "archivo_entrada": str(CSV_PATH),
        "columnas_exigidas": EXPECTED_COLS,
        "validaciones": {
            "columnas_exactas_ok": True,
            "store_id_unico_y_no_nulo_ok": True,
            "errores_numericos_finitos_no_negativos_ok": True,
            "pares_verificados_por_store_id": True,
            "csv_original_intacto": csv_intacto,
        },
        "salida_csv": {
            "ruta": str(OUT_CSV),
            "n_filas": int(filas),
            "n_columnas": int(columnas),
            "columnas": list(chequeo.columns),
        },
        "salida_png": str(OUT_PNG),
        "tabla_pareada": tabla_records,
        "tiendas_favorecen_A": n_A,
        "tiendas_favorecen_B": n_B,
        "empates": n_empates,
        "ids_favorecen_A": tabla.loc[tabla["favors"] == "A", "store_id"].tolist(),
        "ids_favorecen_B": tabla.loc[tabla["favors"] == "B", "store_id"].tolist(),
        "mayor_diferencia_favors_B": mayor_B,
        "mayor_diferencia_favors_A": mayor_A,
        "suma_diferencias": float(tabla["difference"].sum()),
        "media_diferencias": float(tabla["difference"].mean()),
        "nota": (
            "Comprobar solo que error_a y error_b tienen la misma longitud no garantiza "
            "que las filas correspondan a las mismas tiendas: un reordenamiento, un "
            "desplazamiento o un id faltante/duplicado emparejaría valores de tiendas "
            "distintas. La correspondencia se valida clave a clave mediante store_id."
        ),
    }
    with RESULTADOS_PATH.open("w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print("T2 completada.")
    print(f"  Pares validados por store_id: {len(tabla)}")
    if (filas, columnas) != (16, 5):
        print(f"  ADVERTENCIA: se esperaban 16 filas x 5 columnas; obtenido {filas}x{columnas}")
    print(f"  Favorecen a A: {n_A} | Favorecen a B: {n_B} | Empates: {n_empates}")
    if mayor_B:
        print(f"  Mayor diferencia a favor de B: {mayor_B['store_id']} ({mayor_B['difference']:.4f})")
    if mayor_A:
        print(f"  Mayor diferencia a favor de A: {mayor_A['store_id']} ({mayor_A['difference']:.4f})")
    print(f"  Escrito: {OUT_CSV} ({filas} filas x {columnas} columnas)")
    print(f"  Figura: {OUT_PNG}")
    print(f"  CSV original intacto: {csv_intacto}")


if __name__ == "__main__":
    main()
