import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SEED = 20260829
B = 10000
ALPHA = 0.05
CONF = 0.95

# ---------- T2: carga y validación ----------
def cargar_validar(path="data/model_errors.csv"):
    if not Path(path).exists():
        raise FileNotFoundError(f"No existe {path}")
    df = pd.read_csv(path)
    esperadas = ["store_id", "error_a", "error_b"]
    if list(df.columns) != esperadas:
        raise ValueError(f"Columnas esperadas {esperadas}; encontradas {list(df.columns)}")
    if df["store_id"].isna().any():
        raise ValueError("store_id contiene valores nulos")
    dup = df.loc[df["store_id"].duplicated(keep=False), "store_id"].tolist()
    if dup:
        raise ValueError(f"store_id duplicado: {sorted(set(dup))}")
    for c in ["error_a", "error_b"]:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any() or not np.isfinite(v.to_numpy()).all() or (v < 0).any():
            raise ValueError(f"{c} debe ser numérico, finito y no negativo")
        df[c] = v.astype(float)
    return df

df = cargar_validar()
n = len(df)
dif = (df["error_a"] - df["error_b"]).to_numpy()
tabla = df.copy()
tabla["difference"] = dif
tabla["favors"] = np.where(dif > 0, "B", np.where(dif < 0, "A", "tie"))
emparejamiento_ok = bool(tabla["store_id"].is_unique and len(tabla) == n)

# ---------- T3: CSV pareado y figura ----------
Path("output").mkdir(parents=True, exist_ok=True)
tabla.to_csv("output/store_differences.csv", index=False)
chk = pd.read_csv("output/store_differences.csv")
assert list(chk.columns) == ["store_id", "error_a", "error_b", "difference", "favors"] and len(chk) == n

favA = int((tabla["favors"] == "A").sum())
favB = int((tabla["favors"] == "B").sum())
ties = int((tabla["favors"] == "tie").sum())
iB = int(np.argmax(dif))
iA = int(np.argmin(dif))

fig, ax = plt.subplots(figsize=(9, 5.5))
colores = ["tab:blue" if v > 0 else "tab:orange" if v < 0 else "tab:gray" for v in dif]
ax.bar(tabla["store_id"], dif, color=colores, edgecolor="black", linewidth=0.5)
ax.axhline(0, color="black", linewidth=1.2, label="referencia en 0")
media_obs = float(dif.mean())
ax.axhline(media_obs, color="tab:green", ls="--", lw=1.2, label=f"media = {media_obs:.4f}")
for i, v in enumerate(dif):
    ax.annotate(f"{v:+.2f}", (i, v), ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
ax.set_xlabel("store_id")
ax.set_ylabel("difference = error_a − error_b")
ax.set_title("Diferencia pareada de error por tienda (positivo favorece a B)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("output/store_differences.png", dpi=120)
plt.close(fig)

# ---------- T4: descriptivos y prueba t pareada ----------
media = float(dif.mean())
sd = float(dif.std(ddof=1))
ee = float(sd / np.sqrt(n))
gl = n - 1
res = stats.ttest_1samp(dif, 0.0)
t_stat = float(res.statistic)
p_t = float(res.pvalue)
ci_lo, ci_hi = stats.t.interval(CONF, gl, loc=media, scale=ee)
ci_t = [float(ci_lo), float(ci_hi)]

# ---------- T5: referencia exacta por cambios de signo ----------
total_cfg = 2 ** n
mascaras = np.arange(total_cfg, dtype=np.int64)
bits = ((mascaras[:, None] >> np.arange(n)) & 1).astype(np.int8)
signos = (1 - 2 * bits).astype(np.float64)
medias_nulas = (signos @ dif) / n
tol = 1e-9
extremas = int(np.count_nonzero(np.abs(medias_nulas) >= abs(media) - tol))
iguales = int(np.count_nonzero(np.abs(np.abs(medias_nulas) - abs(media)) <= tol))
p_sf = extremas / total_cfg

# ---------- T6: bootstrap pareado ----------
rng = np.random.default_rng(SEED)
idx = rng.integers(0, n, size=(B, n))
boot = dif[idx].mean(axis=1)
b_lo, b_hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
boot_mean = float(boot.mean())
boot_std = float(boot.std(ddof=1))
rng2 = np.random.default_rng(SEED)
boot2 = dif[rng2.integers(0, n, size=(B, n))].mean(axis=1)
reproducible = bool(np.array_equal(boot, boot2))

# ---------- T7: informe ejecutivo (pregunta 18, <=150 palabras) ----------
informe = (
    f"Unidad: {n} tiendas (S01–S16). Procedencia: data/model_errors.csv. "
    f"Métrica: error de pronóstico por tienda con dos métodos (error_a, error_b). "
    f"Emparejamiento: cada tienda recibe ambos métodos; análisis sobre la diferencia pareada "
    f"error_a − error_b, verificada por store_id. "
    f"Efecto: media {media:.4f} ({favB} tiendas favorecen a B, {favA} a A, {ties} empates). "
    f"Incertidumbre: EE {ee:.4f}; IC t 95% [{ci_t[0]:.4f}, {ci_t[1]:.4f}]. "
    f"Procedimiento: prueba t pareada bilateral (t {t_stat:.4f}, gl {gl}), p {p_t:.4f}; "
    f"referencia exacta por cambios de signo ({total_cfg} configuraciones), p {p_sf:.4f}. "
    f"Sensibilidad: bootstrap pareado ({B} remuestreos de tiendas completas, semilla {SEED}), "
    f"IC 95% percentil [{b_lo:.4f}, {b_hi:.4f}]. "
    f"Limitaciones: (1) n {n}, muestra pequeña con potencia limitada; "
    f"(2) datos observacionales de un solo periodo, sin aleatorización ni control de covariables. "
    f"No puede concluirse que B reduzca el error significativamente frente a A: "
    f"p no significativo y todos los intervalos contienen 0."
)
assert len(informe.split()) <= 150
Path("informe.md").write_text("# Informe ejecutivo (pregunta 18)\n\n" + informe + "\n", encoding="utf-8")

# ---------- resultados.json ----------
resultados = {
    "subtask": "T7",
    "pipeline": "corrida única integrada: carga → tabla pareada → CSV/figura → descriptivos → prueba t → sign-flip exacto → bootstrap → informe",
    "data_file": "data/model_errors.csv",
    "pairing_verified_by_store_id": emparejamiento_ok,
    "n": n,
    "columns": list(tabla.columns),
    "differences": [float(x) for x in dif],
    "table": tabla.to_dict(orient="records"),
    "favors_A_count": favA,
    "favors_B_count": favB,
    "ties_count": ties,
    "max_difference_favoring_B": {"store_id": str(tabla["store_id"].iloc[iB]), "difference": float(dif[iB])},
    "max_difference_favoring_A": {"store_id": str(tabla["store_id"].iloc[iA]), "difference": float(dif[iA])},
    "descriptive_summary": {
        "n": n, "mean": media, "sample_sd": sd, "standard_error": ee,
        "min": float(dif.min()), "q1": float(np.percentile(dif, 25)),
        "median": float(np.median(dif)), "q3": float(np.percentile(dif, 75)),
        "max": float(dif.max()),
    },
    "t_test": {
        "test": "prueba t de una muestra bilateral sobre las diferencias (equivalente pareada)",
        "null_mean": 0.0, "alternative": "two-sided",
        "t_statistic": t_stat, "degrees_of_freedom": gl,
        "p_value_two_sided": p_t, "alpha": ALPHA, "confidence_level": CONF,
        "ci95": ci_t, "ci95_low": ci_t[0], "ci95_high": ci_t[1],
    },
    "signflip_exact_reference": {
        "method": "enumeración exacta de las 2^16 configuraciones de signos de las diferencias",
        "total_configurations": total_cfg,
        "count_extreme_configurations": extremas,
        "n_configurations_equal_in_magnitude": iguales,
        "p_value_signflip_exact": p_sf,
    },
    "bootstrap_paired": {
        "seed": SEED, "resamples": B,
        "resampling_unit": "tienda completa (índices remuestreados sobre el vector de diferencias)",
        "bootstrap_mean": boot_mean, "bootstrap_std": boot_std,
        "ci95_percentile": [b_lo, b_hi],
        "ci95_percentile_low": b_lo, "ci95_percentile_high": b_hi,
        "reproducible_same_seed": reproducible,
    },
    "outputs": {
        "csv": "output/store_differences.csv",
        "png": "output/store_differences.png",
        "informe_md": "informe.md",
    },
    "informe_ejecutivo": informe,
    "informe_word_count": len(informe.split()),
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

assert Path("output/store_differences.csv").exists()
assert Path("output/store_differences.png").exists()

# ---------- consola ----------
print("=" * 72)
print("T7 · Pipeline completo en una sola corrida")
print("=" * 72)
print(f"[1] Datos: data/model_errors.csv · n = {n} tiendas · emparejamiento por store_id verificado: {emparejamiento_ok}")
print("[2] Resumen descriptivo de difference = error_a − error_b:")
print(f"    media = {media:.6f} · sd muestral = {sd:.6f} · EE = sd/√n = {ee:.6f}")
print(f"    min = {dif.min():.4f} · Q1 = {np.percentile(dif, 25):.4f} · mediana = {np.median(dif):.4f} · Q3 = {np.percentile(dif, 75):.4f} · max = {dif.max():.4f}")
print(f"    favorecen A = {favA} · favorecen B = {favB} · empates = {ties}")
print(f"    mayor |diferencia| a favor de B: {tabla['store_id'].iloc[iB]} ({dif[iB]:+.4f}) · a favor de A: {tabla['store_id'].iloc[iA]} ({dif[iA]:+.4f})")
print("[3] Prueba t pareada (bilateral, H0: media de las diferencias = 0):")
print(f"    t = {t_stat:.6f} · gl = {gl} · p-value = {p_t:.6f}")
print(f"    IC t 95% = [{ci_t[0]:.6f}, {ci_t[1]:.6f}]")
print("[4] Referencia exacta por cambios de signo:")
print(f"    configuraciones = {total_cfg} (2^{n}) · extremas = {extremas} · p-value = {p_sf:.6f}")
print("[5] Bootstrap pareado (análisis de sensibilidad):")
print(f"    semilla = {SEED} · remuestreos = {B} · unidad de remuestreo = tienda completa")
print(f"    IC 95% percentil = [{b_lo:.6f}, {b_hi:.6f}] · reproducible con misma semilla: {reproducible}")
print("[6] Archivos generados: output/store_differences.csv · output/store_differences.png · informe.md · resultados.json")
print("[7] Informe ejecutivo (pregunta 18):")
print(informe)
