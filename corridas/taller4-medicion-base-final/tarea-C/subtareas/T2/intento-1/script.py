# -*- coding: utf-8 -*-
"""
T2 — Semántica latente (LSA) sobre la misma matriz TF-IDF y los mismos juicios de T1.

Pipeline:
  1) Reconstruye el corpus (10 documentos) y las 6 consultas con sus juicios (datos del enunciado).
  2) Vectoriza con TfidfVectorizer (parámetros por defecto) ajustado SOLO con los documentos
     (misma matriz TF-IDF que T1; vocabulario esperado: 105 términos).
  3) Proyecta a 4 dimensiones con TruncatedSVD(n_components=4, random_state=0) (LSA),
     ajustado SOLO con los documentos; las consultas se proyectan con transform (fold-in).
  4) Recalcula similitud coseno y rankings (10 documentos ordenados por consulta).
  5) Evalúa Hit@1, Hit@3 y MRR y construye UNA tabla comparativa TF-IDF vs LSA (6 cifras).

Salidas: resultados.json, tabla_comparativa.csv, tabla_comparativa.png,
         comparacion_metricas.png, rank_relevante_por_consulta.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------------
# Datos del enunciado (idénticos a T1)
# ----------------------------------------------------------------------------------
DOCUMENTOS = {
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

CONSULTAS = [
    ("q1", "¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    ("q2", "¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    ("q3", "¿Qué técnica entrena matrices de bajo rango?", "d08"),
    ("q4", "¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    ("q5", "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    ("q6", "¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
]

doc_ids = list(DOCUMENTOS.keys())
doc_texts = [DOCUMENTOS[d] for d in doc_ids]
q_ids = [c[0] for c in CONSULTAS]
q_texts = [c[1] for c in CONSULTAS]
relevantes = [c[2] for c in CONSULTAS]

# ----------------------------------------------------------------------------------
# 1) Misma matriz TF-IDF que T1 (vectorizador ajustado únicamente con el corpus)
# ----------------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto, como en la Parte 1
X_docs = vectorizer.fit_transform(doc_texts)
X_queries = vectorizer.transform(q_texts)
S_tfidf = cosine_similarity(X_queries, X_docs)  # (6, 10)

# ----------------------------------------------------------------------------------
# 2) LSA: proyección de la matriz TF-IDF a 4 dimensiones
#    (SVD ajustado solo con documentos; consultas proyectadas al mismo espacio)
# ----------------------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
X_docs_lsa = svd.fit_transform(X_docs)
X_queries_lsa = svd.transform(X_queries)
S_lsa = cosine_similarity(X_queries_lsa, X_docs_lsa)  # (6, 10)

# ----------------------------------------------------------------------------------
# 3) Rankings (orden estable: empates resueltos por orden de documento, como en T1)
# ----------------------------------------------------------------------------------
def calcular_rankings(S):
    rankings, sims_orden = [], []
    for fila in S:
        orden = np.argsort(-fila, kind="stable")
        rankings.append([doc_ids[i] for i in orden])
        sims_orden.append([float(fila[i]) for i in orden])
    return rankings, sims_orden


def evaluar(rankings):
    n = len(rankings)
    hit1 = hit3 = 0
    rr_sum = 0.0
    ranks_rel = []
    for rank, rel in zip(rankings, relevantes):
        pos = rank.index(rel) + 1
        ranks_rel.append(pos)
        hit1 += int(pos == 1)
        hit3 += int(pos <= 3)
        rr_sum += 1.0 / pos
    metricas = {"hit_at_1": hit1 / n, "hit_at_3": hit3 / n, "mrr": rr_sum / n}
    return metricas, ranks_rel


rank_tfidf, sims_tfidf = calcular_rankings(S_tfidf)
rank_lsa, sims_lsa = calcular_rankings(S_lsa)
m_tfidf, ranks_rel_tfidf = evaluar(rank_tfidf)
m_lsa, ranks_rel_lsa = evaluar(rank_lsa)

# ----------------------------------------------------------------------------------
# 4) Verificación contra los resultados oficiales de T1 (misma matriz, mismos juicios)
# ----------------------------------------------------------------------------------
t1 = None
t1_path = Path("entrada") / "T1" / "resultados.json"
if t1_path.exists():
    try:
        t1 = json.loads(t1_path.read_text(encoding="utf-8"))
    except Exception:
        t1 = None

tfidf_oficial = None
if isinstance(t1, dict) and all(k in t1 for k in ("hit_at_1", "hit_at_3", "mrr")):
    tfidf_oficial = {k: float(t1[k]) for k in ("hit_at_1", "hit_at_3", "mrr")}

verif = {"t1_disponible": t1 is not None}
if tfidf_oficial is not None:
    difs = [abs(m_tfidf[k] - tfidf_oficial[k]) for k in ("hit_at_1", "hit_at_3", "mrr")]
    verif["max_dif_metricas_recalculadas_vs_T1"] = float(max(difs))
    verif["metricas_T1"] = tfidf_oficial
    if isinstance(t1, dict) and "por_consulta" in t1 and len(t1["por_consulta"]) == len(q_ids):
        verif["rankings_tfidf_identicos_por_consulta"] = [
            rank_tfidf[i] == t1["por_consulta"][i]["ranking_10_documentos"]
            for i in range(len(q_ids))
        ]

# Fila TF-IDF de la tabla: valores oficiales de T1 si están disponibles
fila_tfidf_tabla = tfidf_oficial if tfidf_oficial is not None else m_tfidf

# ----------------------------------------------------------------------------------
# 5) Rankings LSA por consulta (con comparación del rank del relevante vs TF-IDF)
# ----------------------------------------------------------------------------------
por_consulta_lsa = []
for i, (qid, qtext, rel) in enumerate(zip(q_ids, q_texts, relevantes)):
    pos_lsa = ranks_rel_lsa[i]
    pos_tfidf = ranks_rel_tfidf[i]
    por_consulta_lsa.append({
        "consulta_id": qid,
        "consulta": qtext,
        "documento_relevante": rel,
        "ranking_10_documentos": rank_lsa[i],
        "similitudes_en_orden_ranking": sims_lsa[i],
        "rank_del_relevante": pos_lsa,
        "reciprocal_rank": 1.0 / pos_lsa,
        "rank_del_relevante_tfidf": pos_tfidf,
        "delta_rank_vs_tfidf": pos_tfidf - pos_lsa,
    })

# ----------------------------------------------------------------------------------
# 6) Tabla comparativa única: 3 métricas × 2 métodos = 6 cifras
# ----------------------------------------------------------------------------------
tabla = pd.DataFrame({
    "metodo": ["TF-IDF (linea base lexica, Parte 1)", "LSA (TruncatedSVD, 4 dimensiones)"],
    "hit_at_1": [fila_tfidf_tabla["hit_at_1"], m_lsa["hit_at_1"]],
    "hit_at_3": [fila_tfidf_tabla["hit_at_3"], m_lsa["hit_at_3"]],
    "mrr": [fila_tfidf_tabla["mrr"], m_lsa["mrr"]],
})
tabla.to_csv("tabla_comparativa.csv", index=False, encoding="utf-8")

# ------------------------------ Figura 1: la tabla comparativa como PNG ------------
fig, ax = plt.subplots(figsize=(7.8, 2.4))
ax.axis("off")
col_labels = ["Método", "Hit@1", "Hit@3", "MRR"]
cell_text = [
    [fila["metodo"]] + [f"{fila[m]:.4f}" for m in ("hit_at_1", "hit_at_3", "mrr")]
    for _, fila in tabla.iterrows()
]
tb = ax.table(cellText=cell_text, colLabels=col_labels, loc="center",
              cellLoc="center", colWidths=[0.46, 0.18, 0.18, 0.18])
tb.auto_set_font_size(False)
tb.set_fontsize(11)
tb.scale(1, 1.9)
for j in range(4):
    tb[0, j].set_facecolor("#D9E2F3")
    tb[0, j].set_text_props(weight="bold")
ax.set_title("Tarea C · Parte 2 — TF-IDF vs LSA (6 consultas, 10 documentos)",
             fontsize=11, pad=14)
plt.savefig("tabla_comparativa.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ------------------------------ Figura 2: barras de las 3 métricas -----------------
vals_tfidf = [fila_tfidf_tabla["hit_at_1"], fila_tfidf_tabla["hit_at_3"], fila_tfidf_tabla["mrr"]]
vals_lsa = [m_lsa["hit_at_1"], m_lsa["hit_at_3"], m_lsa["mrr"]]
fig, ax = plt.subplots(figsize=(7.0, 4.2))
x = np.arange(3)
w = 0.35
b1 = ax.bar(x - w / 2, vals_tfidf, w, label="TF-IDF (léxico)", color="#4C72B0")
b2 = ax.bar(x + w / 2, vals_lsa, w, label="LSA (TruncatedSVD, 4 dim)", color="#DD8452")
for barras in (b1, b2):
    for b in barras:
        ax.annotate(f"{b.get_height():.3f}",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(["Hit@1", "Hit@3", "MRR"])
ax.set_ylim(0, 1.12)
ax.set_ylabel("Valor de la métrica")
ax.set_title("Métricas de recuperación: TF-IDF vs LSA (6 consultas)")
ax.legend(loc="upper right", fontsize=9)
plt.savefig("comparacion_metricas.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ------------------------------ Figura 3: rank del relevante por consulta ----------
fig, ax = plt.subplots(figsize=(7.0, 4.2))
xq = np.arange(len(q_ids))
ax.bar(xq - w / 2, ranks_rel_tfidf, w, label="TF-IDF", color="#4C72B0")
ax.bar(xq + w / 2, ranks_rel_lsa, w, label="LSA (4 dim)", color="#DD8452")
for xi in range(len(q_ids)):
    ax.annotate(str(ranks_rel_tfidf[xi]), (xq[xi] - w / 2, ranks_rel_tfidf[xi]),
                ha="center", va="center", fontsize=9)
    ax.annotate(str(ranks_rel_lsa[xi]), (xq[xi] + w / 2, ranks_rel_lsa[xi]),
                ha="center", va="center", fontsize=9)
ax.set_xticks(xq)
ax.set_xticklabels(q_ids)
ax.set_yticks(range(1, 11))
ax.set_ylim(10.5, 0.5)  # rank 1 arriba (mejor)
ax.set_ylabel("Posición del documento relevante")
ax.set_title("Rank del documento relevante por consulta: TF-IDF vs LSA")
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.savefig("rank_relevante_por_consulta.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------------
# 7) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------------------
resultados = {
    "subtarea": "T2_semantica_latente_lsa",
    "descripcion": "Proyección LSA de la matriz TF-IDF a 4 dimensiones, recálculo de similitud "
                   "coseno y rankings, evaluación Hit@1/Hit@3/MRR y tabla comparativa única "
                   "TF-IDF vs LSA sobre las mismas 6 consultas y juicios de T1.",
    "metodo_lsa": "TruncatedSVD(n_components=4, random_state=0) ajustado sobre la matriz TF-IDF "
                  "de los 10 documentos; consultas proyectadas con transform (fold-in)",
    "num_documentos": 10,
    "num_consultas": 6,
    "dimensiones_lsa": 4,
    "tam_vocabulario": int(len(vectorizer.vocabulary_)),
    "varianza_explicada_por_componente": [float(v) for v in svd.explained_variance_ratio_],
    "varianza_explicada_acumulada": float(svd.explained_variance_ratio_.sum()),
    "rankings_lsa_por_consulta": por_consulta_lsa,
    "metricas": {
        "tfidf": m_tfidf,
        "tfidf_T1_oficial": tfidf_oficial,
        "lsa": m_lsa,
    },
    "hit_at_1_tfidf": fila_tfidf_tabla["hit_at_1"],
    "hit_at_3_tfidf": fila_tfidf_tabla["hit_at_3"],
    "mrr_tfidf": fila_tfidf_tabla["mrr"],
    "hit_at_1_lsa": m_lsa["hit_at_1"],
    "hit_at_3_lsa": m_lsa["hit_at_3"],
    "mrr_lsa": m_lsa["mrr"],
    "tabla_comparativa": tabla.to_dict(orient="records"),
    "verificacion_contra_T1": verif,
    "archivos_generados": [
        "resultados.json",
        "tabla_comparativa.csv",
        "tabla_comparativa.png",
        "comparacion_metricas.png",
        "rank_relevante_por_consulta.png",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------------
# Resumen
# ----------------------------------------------------------------------------------
print("=" * 72)
print("T2 — Semántica latente (LSA) sobre la matriz TF-IDF de la Parte 1")
print("=" * 72)
print(f"Vocabulario TF-IDF: {len(vectorizer.vocabulary_)} términos")
print(f"Varianza explicada por los 4 componentes LSA: "
      f"{svd.explained_variance_ratio_.sum():.4f} "
      f"({', '.join(f'{v:.3f}' for v in svd.explained_variance_ratio_)})")
print("\nRankings LSA por consulta (relevante entre corchetes):")
for r in por_consulta_lsa:
    seq = " > ".join(
        f"[{d}]" if d == r["documento_relevante"] else d
        for d in r["ranking_10_documentos"]
    )
    print(f"  {r['consulta_id']}: {seq}")
    print(f"      rank del relevante: LSA={r['rank_del_relevante']} "
          f"| TF-IDF={r['rank_del_relevante_tfidf']}")
print("\nTabla comparativa (3 métricas × 2 métodos = 6 cifras):")
print(tabla.to_string(index=False))
if verif.get("t1_disponible"):
    print(f"\nVerificación vs T1: máx. diferencia en métricas TF-IDF = "
          f"{verif.get('max_dif_metricas_recalculadas_vs_T1', float('nan'))}")
print("\nArchivos generados: resultados.json, tabla_comparativa.csv, "
      "tabla_comparativa.png, comparacion_metricas.png, rank_relevante_por_consulta.png")
