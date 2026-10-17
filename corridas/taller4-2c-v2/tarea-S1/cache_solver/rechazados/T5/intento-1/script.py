```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RUTA_PARQUET = Path("output/server_analysis.parquet")

# --- Referencia de T3 ---
t3 = {}
_ref = Path("entrada/T3/resultados.json")
if _ref.exists():
    t3 = json.loads(_ref.read_text())
fila23_ref = t3.get("fila_23", {})
cols_esperadas = t3.get("columnas_df_nuevo")
n_obs = int(t3["n_observaciones"]) if "n_observaciones" in t3 else None
target_ls = fila23_ref.get("load_score")
resumen_ls = t3.get("load_score_resumen", {})

# --- Obtener el DataFrame analizado de T3 ---
df_analizado, origen = None, None

# a) archivo dejado por T3, si existe
_d = Path("entrada/T3")
if _d.exists():
    for f in sorted(list(_d.glob("*.parquet")) + list(_d.glob("*.csv"))):
        try:
            cand = (pd.read_parquet(f) if f.suffix == ".parquet"
                    else pd.read_csv(f, parse_dates=["timestamp"]))
        except Exception:
            continue
        if not {"load_score", "requires_review"}.issubset(cand.columns):
            continue
        cand = cand.reset_index(drop=True)
        if n_obs is not None and len(cand) != n_obs:
            continue
        if target_ls is not None:
            try:
                r = cand.iloc[23]
                ok23 = (str(r["server"]) == str(fila23_ref.get("server"))
                        and abs(float(r["load_score"]) - float(target_ls)) <= 1e-9)
            except Exception:
                ok23 = False
            if not ok23:
                continue
        df_analizado, origen = cand, f"
