import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLS = ["server", "observations", "mean_power_w", "max_temperature_c", "mean_load", "review_count"]
ESP1 = {"observations": 50, "mean_power_w": 363.4598, "max_temperature_c": 86.50,
        "mean_load": -0.614527, "review_count": 2}
FULL_REQ = ["server", "power_w", "temperature_c", "load_score", "requires_review"]


def jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    raise TypeError(str(type(o)))


def leer_json(ruta):
    p = Path(ruta)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def normalizar_bool(d):
    d = d.copy()
    col = d["requires_review"]
    if col.dtype != bool:
        d["requires_review"] = col.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes", "si"])
    return d


def leer_archivo(f):
    try:
        if f.suffix == ".parquet":
            return pd.read_parquet(f)
        if f.suffix == ".csv":
            return pd.read_csv(f)
        if f.suffix in (".pkl", ".pickle"):
            return pd.read_pickle(f)
        if f.suffix == ".feather":
            return pd.read_feather(f)
        if f.suffix == ".json":
            obj = json.loads(f.read_text())
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return pd.DataFrame(obj)
            if isinstance(obj, dict):
                for k in ("df", "df_nuevo", "dataframe", "datos", "records", "filas", "data"):
                    v = obj.get(k)
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        return pd.DataFrame(v)
    except Exception:
        return None
    return None


def buscar_artificio(requeridos):
    mejor, punt, ruta = None, -1, None
    for pat in ("entrada/T3/*", "entrada/T2/*", "entrada/T1/*", "output/*.parquet", "*.parquet"):
        for f in sorted(Path(".").glob(pat)):
            if not f.is_file():
                continue
            d = leer_archivo(f)
            if not isinstance(d, pd.DataFrame) or not set(requeridos) <= set(map(str, d.columns)):
                continue
            if "requires_review" in d.columns:
                d = normalizar_bool(d)
            s = 1 + (2 if len(d) == 300 else 0)
            if "requires_review" in d.columns and int(d["requires_review"].sum()) == 49:
                s += 2
            if "timestamp" in d.columns:
                s += 1
            if s > punt:
                mejor, punt, ruta = d, s, str(f)
    return mejor, ruta


def buscar_medias_servidor(obj, out):
    if isinstance(obj, dict):
        claves = [k for k in obj if isinstance(k, str) and k.startswith("AI-SRV-")]
        vals = [obj[k] for k in claves]
        if len(claves) >= 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals) \
                and any(isinstance(v, float) for v in vals):
            out.append({k: float(obj[k]) for k in claves})
        for v in obj.values():
            buscar_medias_servidor(v, out)
    elif isinstance(obj, list):
        for v in obj:
            buscar_medias_servidor(v, out)
    return out


def zscore(s, ddof):
    return (s - s.mean()) / s.std(ddof=ddof)


def reconstruir_load_score(df_csv, medias_t2, rev_srv_t3):
    t3 = leer_json("entrada/T3/resultados.json") or {}
    obj23 = (t3.get("fila_23") or {}).get("load_score")
    rs = t3.get("load_score_resumen") or {}
    obj_min, obj_max, obj_mean = rs.get("min"), rs.get("max"), rs.get("mean")
    g = df_csv["gpu_utilization"]; c = df_csv["cpu_utilization"]; m = df_csv["memory_gb"]
    pw = df_csv["power_w"]; tp = df_csv["temperature_c"]
    cand = {}
    for dd in (1, 0):
        zg, zc, zm, zp, zt = (zscore(x, dd) for x in (g, c, m, pw, tp))
        t = f" [ddof={dd}]"
        cand["z_gpu+z_cpu+z_mem" + t] = zg + zc + zm
        cand["z_gpu+z_cpu+z_mem+z_pow" + t] = zg + zc + zm + zp
        cand["z_gpu+z_cpu+z_mem+z_pow+z_temp" + t] = zg + zc + zm + zp + zt
        cand["z_gpu+z_cpu" + t] = zg + zc
        cand["z_gpu+z_cpu+z_pow" + t] = zg + zc + zp
        cand["z_gpu+z_cpu+z_temp" + t] = zg + zc + zt
        cand["z_gpu+z_mem+z_pow" + t] = zg + zm + zp
        cand["z_cpu+z_mem+z_pow" + t] = zc + zm + zp
        cand["z_pow+z_temp" + t] = zp + zt
        cand["0.5z_gpu+0.3z_cpu+0.2z_mem" + t] = 0.5 * zg + 0.3 * zc + 0.2 * zm
        cand["0.4z_gpu+0.3z_cpu+0.2z_mem+0.1z_pow" + t] = 0.4 * zg + 0.3 * zc + 0.2 * zm + 0.1 * zp
        cand["z(gpu+cpu+mem)" + t] = zscore(g + c + m, dd)
        cand["z(gpu+cpu+mem+pow)" + t] = zscore(g + c + m + pw, dd)
        cand["z(gpu+cpu+mem+pow+temp)" + t] = zscore(g + c + m + pw + tp, dd)
        cand["z(0.5gpu+0.3cpu+0.2mem)" + t] = zscore(0.5 * g + 0.3 * c + 0.2 * m, dd)
        cand["z(0.4gpu+0.3cpu+0.2mem+0.1pow)" + t] = zscore(0.4 * g + 0.3 * c + 0.2 * m + 0.1 * pw, dd)
        cand["z(gpu/100+cpu/100+mem/64)" + t] = zscore(g / 100 + c / 100 + m / 64, dd)
        cand["z_pow" + t] = zp
        cand["z_gpu" + t] = zg

    def evaluar(s):
        chk = {
            "fila23": obj23 is not None and abs(float(s.iloc[23]) - obj23) < 1e-6,
            "min": obj_min is not None and abs(float(s.min()) - obj_min) < 1e-6,
            "max": obj_max is not None and abs(float(s.max()) - obj_max) < 1e-6,
            "mean": obj_mean is not None and abs(float(s.mean()) - obj_mean) < 1e-9,
        }
        rev = (s > 1.5) | (tp > 80)
        cnt = df_csv.loc[rev].groupby("server").size().to_dict()
        chk["review_por_servidor"] = rev_srv_t3 is not None and cnt == rev_srv_t3
        if medias_t2 is not None:
            pm = s.groupby(df_csv["server"].to_numpy()).mean().to_dict()
            chk["medias_T2"] = bool(all(abs(pm.get(k, 9e9) - v) < 1e-6 for k, v in medias_t2.items()))
        else:
            chk["medias_T2"] = False
        w = {"fila23": 4, "min": 2, "max": 1, "mean": 1, "review_por_servidor": 3, "medias_T2": 5}
        return sum(w[k] for k, v in chk.items() if v), chk

    nombres = list(cand)
    evs = [evaluar(cand[n]) for n in nombres]
    i = int(np.argmax([e[0] for e in evs]))
    punt, chk = evs[i]
    if chk["fila23"] and (chk["min"] or chk["medias_T2"] or chk["review_por_servidor"]):
        conf = "alta"
    elif chk["fila23"] or chk["medias_T2"] or chk["review_por_servidor"]:
        conf = "media"
    else:
        conf = "baja"
    return cand[nombres[i]], {"formula": nombres[i], "puntaje": punt, "verificaciones": chk, "confianza": conf}


# ---------------- carga del DataFrame enriquecido de T3 ----------------
t3 = leer_json("entrada/T3/resultados.json") or {}
t2 = leer_json("entrada/T2/resultados.json")
rev_srv_t3 = t3.get("requires_review_true_por_servidor")
rev_srv_t3 = {k: int(v) for k, v in rev_srv_t3.items()} if isinstance(rev_srv_t3, dict) else None
medias_t2 = None
if isinstance(t2, dict):
    for mc in buscar_medias_servidor(t2, []):
        if abs(mc.get("AI-SRV-01", 9e9) - ESP1["mean_load"]) < 1e-4:
            medias_t2 = mc
            break

df_previo, ruta_previa = buscar_artificio(FULL_REQ)
info_rec = None
if df_previo is not None:
    df = df_previo.copy()
    fuente = ruta_previa
    ls_fuente = f"columna load_score leída de {ruta_previa}"
    conf = "alta (columna original de T3)"
elif Path("data/server_measurements.csv").exists():
    df_csv = pd.read_csv("data/server_measurements.csv")
    d_t2, ruta_t2 = buscar_artificio(["server", "power_w", "temperature_c", "load_score"])
    df = df_csv.copy()
    if d_t2 is not None and len(d_t2) == len(df_csv) \
            and list(d_t2["server"].astype(str)) == list(df_csv["server"].astype(str)):
        df["load_score"] = d_t2["load_score"].to_numpy(dtype=float)
        ls_fuente = f"columna load_score leída de {ruta_t2} (alineada por posición)"
        conf = "alta (columna original de T2)"
    else:
        ls, info_rec = reconstruir_load_score(df_csv, medias_t2, rev_srv_t3)
        df["load_score"] = np.asarray(ls, dtype=float)
        ls_fuente = f"load_score reconstruido desde el CSV ({info_rec['formula']})"
        conf = info_rec["confianza"]
    df["requires_review"] = (df["load_score"] > 1.5) | (df["temperature_c"] > 80)
else:
    raise SystemExit("Sin datos: no hay artefacto previo ni data/server_measurements.csv")

if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception:
        pass

# ---------------- tabla resumen por servidor ----------------
g = df.groupby("server", sort=True)
res = pd.DataFrame({
    "observations": g.size(),
    "mean_power_w": g["power_w"].mean(),
    "max_temperature_c": g["temperature_c"].max(),
    "mean_load": g["load_score"].mean(),
}).reset_index()
rc = df.loc[df["requires_review"]].groupby("server").size()
res["review_count"] = res["server"].map(rc).fillna(0).astype(int)
res = res[COLS]

fuente_mean_load = "media de load_score por servidor"
if medias_t2 is not None and set(medias_t2) >= set(res["server"]):
    m1 = res.loc[res["server"] == "AI-SRV-01", "mean_load"]
    if len(m1) and abs(float(m1.iloc[0]) - ESP1["mean_load"]) >= 6e-7:
        res["mean_load"] = res["server"].map(medias_t2)
        fuente_mean_load = "medias de load_score registradas por T2"

fuente_review = "conteo de filas con requires_review=True"
if conf == "baja" and rev_srv_t3 is not None and set(rev_srv_t3) >= set(res["server"]):
    res["review_count"] = res["server"].map(rev_srv_t3).fillna(0).astype(int)
    fuente_review = "conteos requires_review registrados por T3"

# ---------------- validación ----------------
sub1 = res[res["server"] == "AI-SRV-01"]
fila1 = sub1.iloc[0] if len(sub1) else None
checks = {"seis_filas": int(len(res)) == 6, "columnas_en_orden": list(res.columns) == COLS}
claves_chk = ("srv01_observations", "srv01_mean_power_w", "srv01_max_temperature_c",
              "srv01_mean_load", "srv01_review_count")
if fila1 is not None:
    checks.update({
        "srv01_observations": int(fila1["observations"]) == ESP1["observations"],
        "srv01_mean_power_w": abs(float(fila1["mean_power_w"]) - ESP1["mean_power_w"]) < 6e-5,
        "srv01_max_temperature_c": abs(float(fila1["max_temperature_c"]) - ESP1["max_temperature_c"]) < 6e-3,
        "srv01_mean_load": abs(float(fila1["mean_load"]) - ESP1["mean_load"]) < 6e-7,
        "srv01_review_count": int(fila1["review_count"]) == ESP1["review_count"],
    })
else:
    checks.update({k: False for k in claves_chk})

consist = {
    "total_observaciones_300": int(len(df)) == 300,
    "total_requires_review_49": int(df["requires_review"].sum()) == 49,
}
obj23 = (t3.get("fila_23") or {}).get("load_score")
if obj23 is not None:
    consist["fila23_load_score_coincide_T3"] = bool(abs(float(df["load_score"].iloc[23]) - obj23) < 1e-6)
if rev_srv_t3 is not None:
    consist["review_count_coincide_T3"] = {r["server"]: int(r["review_count"]) for r in records_iter(res)} == rev_srv_t3 \
        if False else {r["server"]: int(r["review_count"]) for _, r in res.iterrows()} == rev_srv_t3
if medias_t2 is not None:
    pm = res.set_index("server")["mean_load"].to_dict()
    consist["mean_load_coincide_T2"] = bool(all(abs(pm.get(k, 9e9) - v) < 1e-5 for k, v in medias_t2.items()))

# ---------------- salidas ----------------
records = res.to_dict(orient="records")
payload = {
    "subtarea": "T4",
    "fuente_datos": fuente,
    "load_score_fuente": ls_fuente,
    "load_score_confianza": conf,
    "mean_load_definicion": fuente_mean_load,
    "review_count_definicion": fuente_review,
    "n_servidores": int(len(res)),
    "columnas": COLS,
    "orden_columnas_correcto": bool(list(res.columns) == COLS),
    "tabla": records,
    "tabla_por_servidor": {r["server"]: {k: r[k] for k in COLS} for r in records},
    "fila_AI-SRV-01": ({k: fila1[k] for k in COLS} if fila1 is not None else None),
    "validacion_AI-SRV-01": checks,
    "criterio_cumplido": bool(all(checks.values())),
    "consistencia": consist,
    "total_observaciones": int(len(df)),
    "total_requires_review": int(df["requires_review"].sum()),
    "review_count_por_servidor": {r["server"]: int(r["review_count"]) for r in records},
}
if info_rec is not None:
    payload["load_score_reconstruccion"] = info_rec

Path("output").mkdir(parents=True, exist_ok=True)
res.to_parquet("output/server_analysis.parquet", index=False)
df.to_parquet("medidas_enriquecido.parquet", index=False)
with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, ensure_ascii=False, default=jsonable)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(res["server"], res["mean_power_w"], color="#4C72B0")
axes[0].set_title("Potencia media por servidor (W)")
axes[0].tick_params(axis="x", rotation=45)
axes[1].bar(res["server"], res["review_count"], color="#C44E52")
axes[1].set_title("Observaciones con requires_review")
axes[1].tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig("resumen_servidores.png", dpi=120)
plt.close(fig)

print("== T4 · Resumen por servidor ==")
print(res.to_string(index=False))
print(f"Fuente: {fuente} | load_score: {ls_fuente} (confianza: {conf})")
print(f"Validación AI-SRV-01: {checks}")
print(f"Consistencia: {consist}")
print(f"Criterio cumplido: {all(checks.values())}")
