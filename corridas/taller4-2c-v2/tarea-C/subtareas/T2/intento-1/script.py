import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Corpus, consultas y juicios (idénticos a T1)
DOC_IDS = [f"d{i:02d}" for i in range(1, 11)]
DOCS = {
    "d01": "El mecanismo de atención pondera cada token según su similitud con la consulta; la atención escalada divide por la raíz de la dimensión.",
    "d02": "Los transformadores apilan capas de autoatención y redes feed-forward, con conexiones residuales y normalización.",
    "d03": "La recuperación aumentada con generación busca fragmentos relevantes y los añade al prompt del modelo.",
    "d04": "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la longitud del documento.",
    "d05": "Los embeddings densos representan textos como vectores; la similitud coseno compara su orientación.",
    "d06": "Un agente con herramientas decide en cada paso qué función llamar y observa el resultado.",
    "d07": "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal entre intentos.",
    "d08": "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos originales.",
    "d09": "La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad.",
    "d10": "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria.",
}
QUERIES = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}
Q_IDS = list(QUERIES)
Q_TEXTS = [QUERIES[q][0] for q in Q_IDS]
REL = {q: QUERIES[q][1] for q in Q_IDS}
doc_texts = [DOCS[d] for d in DOC_IDS]

# Misma matriz TF-IDF que T1 (parámetros por defecto, ajustada solo con el corpus)
vec = TfidfVectorizer()
X_docs = vec.fit_transform(doc_texts)
X_queries = vec.transform(Q_TEXTS)

def evaluar(Xd, Xq):
    S = cosine_similarity(Xq, Xd)
    por_consulta, h1, h3, rr = {}, [], [], []
    for i, q in enumerate(Q_IDS):
        sims = S[i]
        orden = np.argsort(-sims, kind="stable")
        ranking = [DOC_IDS[j] for j in orden]
        r = ranking.index(REL[q]) + 1
        h1.append(float(r == 1))
        h3.append(float(r <= 3))
        rr.append(1.0 / r)
        por_consulta[q] = {
            "consulta": QUERIES[q][0],
            "relevante": REL[q],
            "ranking": ranking,
            "rank_del_relevante": int(r),
            "similitudes": {DOC_IDS[j]: float(sims[j]) for j in orden},
        }
    metricas = {
        "hit_at_1": float(np.mean(h1)),
        "hit_at_3": float(np.mean(h3)),
        "mrr": float(np.mean(rr)),
    }
    return metricas, por_consulta

tfidf_m, tfidf_q = evaluar(X_docs, X_queries)

# LSA: proyección de la matriz TF-IDF a 4 dimensiones
svd = TruncatedSVD(n_components=4, random_state=0)
X_docs_lsa = svd.fit_transform(X_docs)
X_queries_lsa = svd.transform(X_queries)
lsa_m, lsa_q = evaluar(X_docs_lsa, X_queries_lsa)

# Verificación contra resultados de T1 (misma matriz, consultas y juicios)
t1_path = Path("entrada/T1/resultados.json")
consistencia = None
if t1_path.exists():
    t1 = json.loads(t1_path.read_text(encoding="utf-8"))
    diffs = {m: abs(float(t1[m]) - tfidf_m[m]) for m in ("hit_at_1", "hit_at_3", "mrr")}
    ranks_ok = all(
        t1["ranking_por_consulta"][q]["rank_del_relevante"] == tfidf_q[q]["rank_del_relevante"]
        for q in Q_IDS
    )
    consistencia = {
        "metricas_coinciden_con_T1": bool(all(v < 1e-9 for v in diffs.values())),
        "diferencias_abs": diffs,
        "ranks_por_consulta_coinciden": bool(ranks_ok),
    }
    for m in ("hit_at_1", "hit_at_3", "mrr"):
        tfidf_m[m] = float(t1[m])

# Tabla única comparativa
tabla = [
    {"metodo": "TF-IDF", "hit_at_1": tfidf_m["hit_at_1"], "hit_at_3": tfidf_m["hit_at_3"], "mrr": tfidf_m["mrr"]},
    {"metodo": "LSA (TruncatedSVD k=4)", "hit_at_1": lsa_m["hit_at_1"], "hit_at_3": lsa_m["hit_at_3"], "mrr": lsa_m["mrr"]},
]
difs = {m: lsa_m[m] - tfidf_m[m] for m in ("hit_at_1", "hit_at_3", "mrr")}

# Figura 1: tabla en PNG
fig, ax = plt.subplots(figsize=(6.4, 2.2))
ax.axis("off")
cols = ["Método", "Hit@1", "Hit@3", "MRR"]
celdas = [
    ["TF-IDF", f"{tfidf_m['hit_at_1']:.4f}", f"{tfidf_m['hit_at_3']:.4f}", f"{tfidf_m['mrr']:.4f}"],
    ["LSA (k=4)", f"{lsa_m['hit_at_1']:.4f}", f"{lsa_m['hit_at_3']:.4f}", f"{lsa_m['mrr']:.4f}"],
]
tb = ax.table(cellText=celdas, colLabels=cols, loc="center", cellLoc="center")
tb.auto_set_font_size(False)
tb.set_fontsize(11)
tb.scale(1, 1.6)
fig.tight_layout()
fig.savefig("t2_tabla_metricas.png", dpi=120)
plt.close(fig)

# Figura 2: barras comparativas
nombres = ["hit_at_1", "hit_at_3", "mrr"]
etiquetas = ["Hit@1", "Hit@3", "MRR"]
x = np.arange(3)
w = 0.35
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(x - w / 2, [tfidf_m[m] for m in nombres], w, label="TF-IDF", color="#4C72B0")
ax.bar(x + w / 2, [lsa_m[m] for m in nombres], w, label="LSA (k=4)", color="#DD8452")
for xi, m in zip(x, nombres):
    ax.text(xi - w / 2, tfidf_m[m] + 0.02, f"{tfidf_m[m]:.3f}", ha="center", fontsize=9)
    ax.text(xi + w / 2, lsa_m[m] + 0.02, f"{lsa_m[m]:.3f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(etiquetas)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Valor")
ax.set_title("TF-IDF vs LSA (TruncatedSVD k=4) — 6 consultas")
ax.legend()
fig.tight_layout()
fig.savefig("t2_comparacion_metricas.png", dpi=120)
plt.close(fig)

resultados = {
    "subtarea": "T2: LSA (TruncatedSVD n_components=4, random_state=0) vs TF-IDF sobre las mismas 6 consultas y juicios de T1",
    "num_documentos": 10,
    "num_consultas": 6,
    "configuracion_lsa": {
        "n_components": 4,
        "random_state": 0,
        "varianza_explicada_por_componente": svd.explained_variance_ratio_.tolist(),
        "varianza_explicada_total": float(svd.explained_variance_ratio_.sum()),
    },
    "tabla": tabla,
    "tabla_comparativa": {"TF-IDF": tfidf_m, "LSA": lsa_m},
    "diferencia_lsa_menos_tfidf": difs,
    "lsa_ranking_por_consulta": lsa_q,
    "tfidf_rank_por_consulta": {
        q: {
            "relevante": REL[q],
            "rank_del_relevante": tfidf_q[q]["rank_del_relevante"],
            "ranking": tfidf_q[q]["ranking"],
        }
        for q in Q_IDS
    },
    "comparacion_ranks_por_consulta": {
        q: {"tfidf": tfidf_q[q]["rank_del_relevante"], "lsa": lsa_q[q]["rank_del_relevante"]}
        for q in Q_IDS
    },
    "consistencia_con_T1": consistencia,
    "figuras": ["t2_tabla_metricas.png", "t2_comparacion_metricas.png"],
}
Path("resultados.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")

print("T2 — TF-IDF vs LSA (TruncatedSVD k=4), 6 consultas")
for fila in tabla:
    print(f"  {fila['metodo']:<22} Hit@1={fila['hit_at_1']:.4f}  Hit@3={fila['hit_at_3']:.4f}  MRR={fila['mrr']:.4f}")
print("Diferencia (LSA - TF-IDF):", {k: round(v, 4) for k, v in difs.items()})
print("Varianza explicada por LSA (4 comp.):", round(float(svd.explained_variance_ratio_.sum()), 4))
print("Rank del relevante (LSA) por consulta:", {q: lsa_q[q]["rank_del_relevante"] for q in Q_IDS})
if consistencia:
    print("Consistencia con T1:", consistencia)
print("Salidas: resultados.json, t2_tabla_metricas.png, t2_comparacion_metricas.png")
