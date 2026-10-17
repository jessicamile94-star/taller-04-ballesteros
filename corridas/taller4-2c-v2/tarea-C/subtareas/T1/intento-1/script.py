import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Corpus de 10 documentos (enunciado, Parte 1)
docs = {
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

# 6 consultas con su juicio de relevancia (enunciado, Parte 1)
queries = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}

doc_ids = list(docs.keys())
doc_texts = [docs[d] for d in doc_ids]
query_ids = list(queries.keys())
query_texts = [queries[q][0] for q in query_ids]

# TF-IDF con parámetros por defecto; el vectorizador se ajusta SOLO con el corpus
vectorizer = TfidfVectorizer()
X_docs = vectorizer.fit_transform(doc_texts)
X_queries = vectorizer.transform(query_texts)

# Similitud coseno consulta-documento
S = cosine_similarity(X_queries, X_docs)

rankings = {}
hits1, hits3, rr = [], [], []
for i, qid in enumerate(query_ids):
    qtext, rel = queries[qid]
    order = np.argsort(-S[i])  # orden descendente por similitud
    ranked = [doc_ids[j] for j in order]
    rank_of_rel = ranked.index(rel) + 1
    hits1.append(1 if rank_of_rel == 1 else 0)
    hits3.append(1 if rank_of_rel <= 3 else 0)
    rr.append(1.0 / rank_of_rel)
    rankings[qid] = {
        "consulta": qtext,
        "relevante": rel,
        "ranking": ranked,
        "rank_del_relevante": rank_of_rel,
        "similitudes": {doc_ids[j]: float(S[i, j]) for j in order},
    }

results = {
    "metodo": "TF-IDF (TfidfVectorizer, parámetros por defecto) + similitud coseno",
    "num_documentos": len(doc_ids),
    "num_consultas": len(query_ids),
    "hit_at_1": float(np.mean(hits1)),
    "hit_at_3": float(np.mean(hits3)),
    "mrr": float(np.mean(rr)),
    "ranking_por_consulta": rankings,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("=== T1: Línea base léxica TF-IDF ===")
print(f"Hit@1: {results['hit_at_1']:.4f}")
print(f"Hit@3: {results['hit_at_3']:.4f}")
print(f"MRR:   {results['mrr']:.4f}")
for qid, r in rankings.items():
    print(f"{qid} (rel={r['relevante']}, rank={r['rank_del_relevante']}): {r['ranking']}")
print("Resultados guardados en resultados.json")
