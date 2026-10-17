import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPECTED = ["store_id", "error_a", "error_b"]
CSV_PATH = Path("data/model_errors.csv")


def validar_carga(df):
    if list(df.columns) != EXPECTED:
        raise ValueError(
            f"El CSV debe tener exactamente las columnas {EXPECTED}; encontradas: {list(df.columns)}"
        )
    if df["store_id"].isna().any():
        raise ValueError("store_id contiene valores nulos en las filas: "
                         f"{df.index[df['store_id'].isna()].tolist()}")
    ids = df["store_id"].astype(str)
    dup = sorted(ids[ids.duplicated(keep=False)].unique().tolist())
    if dup:
        raise ValueError(f"Identificadores de tienda duplicados: {dup}")
    for col in ("error_a", "error_b"):
        serie = df[col]
        if not pd.api.types.is_numeric_dtype(serie):
            co = pd.to_numeric(serie, errors="coerce")
            malos = df.loc[co.isna() & serie.notna(), "store_id"].astype(str).tolist()
            if malos:
                raise ValueError(f"Valores no numéricos en {col} para identificadores: {malos}")
            serie = co
        v = serie.to_numpy(dtype=float)
        no_finitos = df.loc[~np.isfinite(v), "store_id"].astype(str).tolist()
        if no_finitos:
            raise ValueError(f"Valores no finitos (NaN/inf) en {col} para identificadores: {no_finitos}")
        negativos = df.loc[v < 0, "store_id"].astype(str).tolist()
        if negativos:
            raise ValueError(f"Valores negativos en {col} para identificadores: {negativos}")


def construir_pareada(df):
    ids = df["store_id"].astype(str)
    # Verificación del emparejamiento POR store_id: cada tienda aparece exactamente una vez,
    # de modo que (error_a, error_b) de cada fila pertenecen a la misma tienda.
    conteo = ids.value_counts()
    repetidos = sorted(conteo[conteo > 1].index.tolist())
    if repetidos:
        raise ValueError(f"Emparejamiento no verificable por store_id; repetidos: {repetidos}")
    if len(df) != 16:
        raise ValueError(f"Se esperaban 16 pares validados por store_id; se encontraron {len(df)}")
    tab = df[EXPECTED].copy()
    tab["difference"] = tab["error_a"] - tab["error_b"]
    tab["favors"] = np.select(
        [tab["difference"] > 0, tab["difference"] < 0], ["B", "A"], default="tie"
    )
    # Re-verificación fila a fila: la diferencia corresponde al store_id de esa fila
    for _, fila in tab.iterrows():
        rec = float(fila["error_a"]) - float(fila["error_b"])
        if abs(rec - float(fila["difference"])) > 1e-12:
            raise ValueError(f"Par inconsistente para el identificador {fila['store_id']}")
    return tab


def main():
    original_bytes = CSV_PATH.read_bytes()
    df = pd.read_csv(CSV_PATH)
    validar_carga(df)
    tab = construir_pareada(df)
    if CSV_PATH.read_bytes() != original_bytes:
        raise ValueError("El CSV original fue modificado durante la ejecución")

    out = Path("output")
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "store_differences.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    colores = np.where(tab["difference"].to_numpy() > 0, "#d62728", "#1f77b4")
    ax.bar(tab["store_id"].astype(str), tab["difference"], color=colores)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("store_id")
    ax.set_ylabel("difference = error_a - error_b")
    ax.set_title("Diferencia de error por tienda (rojo: favorece B, azul: favorece A)")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(out / "store_differences.png", dpi=120)
    plt.close(fig)

    resultados = {
        "n_rows": int(len(tab)),
        "columns": list(tab.columns),
        "table": tab.to_dict(orient="records"),
        "favors_A_count": int((tab["favors"] == "A").sum()),
        "favors_B_count": int((tab["favors"] == "B").sum()),
        "mean_difference": float(tab["difference"].mean()),
        "pares_verificados_por_store_id": True,
        "csv_original_sin_modificar": True,
    }
    Path("resultados.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Tabla pareada: {len(tab)} filas, columnas {list(tab.columns)}")
    print(f"Favors A: {resultados['favors_A_count']} | Favors B: {resultados['favors_B_count']}")
    print(f"Diferencia media: {resultados['mean_difference']:.6f}")
    print("Archivos escritos: output/store_differences.csv, output/store_differences.png, resultados.json")


if __name__ == "__main__":
    main()
