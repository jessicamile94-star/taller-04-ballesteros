# T1 — Línea base léxica: TF-IDF + similitud coseno sobre corpus mínimo (10 docs, 6 consultas)
# MMIA 6013 · Taller 03 v2 · Tarea C
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------
# 1) Corpus (10 documentos) y consultas (6) con juicios de relevancia
#    (datos tomados textualmente del enunciado)
# ------------------------------------------------------------------
DOC_IDS = ["d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09", "d10"]

DOCS = [
    "El mecanismo de atención pondera cada token según su similitud con la consulta; "
    "la atención escalada divide por la raíz de la dimensión.",
    "Los transformadores apilan capas de autoatención y redes feed-forward, con "
    "conexiones residuales y normalización.",
    "La recuperación aumentada con generación busca fragmentos relevantes y los añade "
    "al prompt del modelo.",
    "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la "
    "longitud del documento.",
    "Los embeddings densos representan textos como vectores; la similitud coseno "
    "compara su orientación.",
    "Un agente con herramientas decide en cada paso qué función llamar y observa el "
    "resultado.",
    "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal "
    "entre intentos.",
    "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos "
    "originales.",
    "La temperatura reescala los logits antes del softmax; valores bajos concentran "
    "la probabilidad.",
    "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria.",
]

QUERIES = [
    ("q1", "¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    ("q2", "¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    ("q3", "¿Qué técnica entrena matrices de bajo rango?", "d08"),
    ("q4", "¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    ("q5", "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    ("q6", "¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
]

# ------------------------------------------------------------------
# 2) Vectorización TF-IDF (parámetros por defecto) y similitud coseno
#    El vectorizador se ajusta SOLO con los documentos del corpus;
#    las consultas solo se transforman.
# ------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto, tal como pide el enunciado
X_docs = vectorizer.fit_transform(DOCS)
X_queries = vectorizer.transform([texto for _, texto, _ in QUERIES])

S = cosine_similarity(X_queries, X_docs)  # forma (6, 10)

# ------------------------------------------------------------------
# 3) Ranking por consulta, rango del relevante y métricas
# ------------------------------------------------------------------
por_consulta = []
reciprocal_ranks = []
hits1 = []
hits3 = []

for i, (qid, qtext, relevante) in enumerate(QUERIES):
    scores = S[i]
    # Orden descendente por similitud; desempates estables por orden original del corpus
    orden = np.argsort(-scores, kind="stable")
    ranking = [DOC_IDS[j] for j in orden]
    ranking_scores = [float(scores[j]) for j in orden]

    rank_relevante = ranking.index(relevante) + 1  # 1-based
    rr = 1.0 / rank_relevante

    reciprocal_ranks.append(rr)
    hits1.append(1.0 if rank_relevante == 1 else 0.0)
    hits3.append(1.0 if rank_relevante <= 3 else 0.0)

    por_consulta.append({
        "consulta_id": qid,
        "consulta": qtext,
        "documento_relevante": relevante,
        "ranking_10_documentos": ranking,
        "similitudes_en_orden_ranking": ranking_scores,
        "rank_del_relevante": int(rank_relevante),
        "reciprocal_rank": float(rr),
    })

hit_at_1 = float(np.mean(hits1))
hit_at_3 = float(np.mean(hits3))
mrr = float(np.mean(reciprocal_ranks))

# ------------------------------------------------------------------
# 4) results.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T1_linea_base_lexica_tfidf",
    "vectorizador": "TfidfVectorizer (scikit-learn, parametros por defecto)",
    "similitud": "coseno",
    "num_documentos": len(DOC_IDS),
    "num_consultas": len(QUERIES),
    "tam_vocabulario": int(len(vectorizer.vocabulary_)),
    "por_consulta": por_consulta,
    "hit_at_1": hit_at_1,
    "hit_at_3": hit_at_3,
    "mrr": mrr,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 5) Figura: reciprocal rank por consulta + métricas agregadas
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
qids = [c["consulta_id"] for c in por_consulta]
rrs = [c["reciprocal_rank"] for c in por_consulta]
barras = ax.bar(qids, rrs, color="#4C72B0", edgecolor="black")
ax.axhline(1.0, color="green", linestyle="--", linewidth=1, label="RR = 1 (rank 1)")
ax.axhline(1 / 3, color="orange", linestyle="--", linewidth=1, label="RR = 1/3 (rank 3)")
for barra, c in zip(barras, por_consulta):
    ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.02,
            f"rank {c['rank_del_relevante']}", ha="center", fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Reciprocal Rank (1/rank del relevante)")
ax.set_title("T1 — Línea base TF-IDF + coseno (por consulta)")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
plt.savefig("t1_linea_base_tfidf.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 6) Resumen por consola
# ------------------------------------------------------------------
print("=" * 70)
print("T1 — Línea base léxica: TF-IDF (default) + similitud coseno")
print("=" * 70)
print(f"Vocabulario TF-IDF: {resultados['tam_vocabulario']} términos")
print("-" * 70)
for c in por_consulta:
    print(f"{c['consulta_id']} | relevante={c['documento_relevante']} | "
          f"rank={c['rank_del_relevante']} | RR={c['reciprocal_rank']:.4f}")
    print(f"    ranking: {' > '.join(c['ranking_10_documentos'])}")
print("-" * 70)
print(f"Hit@1 = {hit_at_1:.4f}  ({int(sum(hits1))}/6 consultas)")
print(f"Hit@3 = {hit_at_3:.4f}  ({int(sum(hits3))}/6 consultas)")
print(f"MRR   = {mrr:.4f}")
print("-" * 70)
print("Salidas: resultados.json, t1_linea_base_tfidf.png")
