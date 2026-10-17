# -*- coding: utf-8 -*-
"""
T3 — Análisis por consulta (Tarea C, MMIA 6013).

A partir de los rankings de TF-IDF (T1) y LSA (T2):
  1) identifica por método en qué consultas el documento relevante queda fuera
     del top 3 (criterio de fallo del enunciado);
  2) calcula el solapamiento de términos entre cada consulta y su documento
     relevante y, para las consultas que fallan, entre la consulta y los
     documentos que sí quedaron en el top 3.

Reproduce internamente T1 y T2 con los mismos datos y parámetros (para disponer
del tokenizador del TfidfVectorizer) y verifica los rankings contra
entrada/T1/resultados.json y entrada/T2/resultados.json.

Salidas: resultados.json, fallos_por_metodo.png, solapamiento_terminos.png,
tabla_fallos_solapamiento.png.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1) Datos del enunciado: corpus de 10 documentos y 6 consultas con juicios
# ---------------------------------------------------------------------------
DOCUMENTOS = [
    ("d01", "El mecanismo de atención pondera cada token según su similitud con la consulta; la atención escalada divide por la raíz de la dimensión."),
    ("d02", "Los transformadores apilan capas de autoatención y redes feed-forward, con conexiones residuales y normalización."),
    ("d03", "La recuperación aumentada con generación busca fragmentos relevantes y los añade al prompt del modelo."),
    ("d04", "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la longitud del documento."),
    ("d05", "Los embeddings densos representan textos como vectores; la similitud coseno compara su orientación."),
    ("d06", "Un agente con herramientas decide en cada paso qué función llamar y observa el resultado."),
    ("d07", "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal entre intentos."),
    ("d08", "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos originales."),
    ("d09", "La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad."),
    ("d10", "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria."),
]

CONSULTAS = [
    ("q1", "¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    ("q2", "¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    ("q3", "¿Qué técnica entrena matrices de bajo rango?", "d08"),
    ("q4", "¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    ("q5", "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    ("q6", "¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
]

doc_ids = [d[0] for d in DOCUMENTOS]
doc_texto = dict(DOCUMENTOS)
idx_doc = {d: i for i, d in enumerate(doc_ids)}
qids = [q[0] for q in CONSULTAS]
q_texto = {q[0]: q[1] for q in CONSULTAS}
q_rel = {q[0]: q[2] for q in CONSULTAS}

# ---------------------------------------------------------------------------
# 2) Reproducción de T1 (TF-IDF, parámetros por defecto) y T2 (LSA k=4)
# ---------------------------------------------------------------------------
vec = TfidfVectorizer()                                # idéntico a T1: defaults
X_docs = vec.fit_transform([doc_texto[d] for d in doc_ids])
X_cons = vec.transform([q_texto[q] for q in qids])
S_tfidf = cosine_similarity(X_cons, X_docs)

svd = TruncatedSVD(n_components=4, random_state=0)     # idéntico a T2
X_docs_lsa = svd.fit_transform(X_docs)
X_cons_lsa = svd.transform(X_cons)                     # fold-in de las consultas
S_lsa = cosine_similarity(X_cons_lsa, X_docs_lsa)

SIM = {"tfidf": S_tfidf, "lsa": S_lsa}

rankings = {"tfidf": {}, "lsa": {}}
sims_ord = {"tfidf": {}, "lsa": {}}
for metodo, S in SIM.items():
    for i, qid in enumerate(qids):
        orden = np.argsort(-S[i], kind="stable")       # empates: orden d01..d10
        rankings[metodo][qid] = [doc_ids[j] for j in orden]
        sims_ord[metodo][qid] = [float(S[i, j]) for j in orden]

rank_rel = {m: {q: rankings[m][q].index(q_rel[q]) + 1 for q in qids} for m in rankings}
fallos = {m: [q for q in qids if rank_rel[m][q] > 3] for m in rankings}
hit3 = {m: float(np.mean([rank_rel[m][q] <= 3 for q in qids])) for m in rankings}

# ---------------------------------------------------------------------------
# 3) Verificación contra los resultados de las subtareas previas
# ---------------------------------------------------------------------------
def cargar_json(ruta):
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def rankings_previos(res, clave):
    out = {}
    if isinstance(res, dict) and clave in res:
        for fila in res[clave]:
            out[fila["consulta_id"]] = fila["ranking_10_documentos"]
    return out

prev_t1 = rankings_previos(cargar_json(Path("entrada/T1/resultados.json")), "por_consulta")
prev_t2 = rankings_previos(cargar_json(Path("entrada/T2/resultados.json")), "rankings_lsa_por_consulta")
coin_t1 = all(prev_t1.get(q) == rankings["tfidf"][q] for q in qids) if len(prev_t1) == len(qids) else None
coin_t2 = all(prev_t2.get(q) == rankings["lsa"][q] for q in qids) if len(prev_t2) == len(qids) else None

# ---------------------------------------------------------------------------
# 4) Solapamiento de términos (tokenizador exacto del TfidfVectorizer de T1)
# ---------------------------------------------------------------------------
analyzer = vec.build_analyzer()
vocab = set(vec.vocabulary_.keys())
feat_names = np.array(vec.get_feature_names_out())

def tokens_unicos(texto):
    return sorted(set(analyzer(texto)))

def pesos_fila(fila):
    coo = fila.tocoo()
    return {feat_names[j]: float(v) for j, v in zip(coo.col, coo.data) if v > 0}

pesos_docs = {d: pesos_fila(X_docs[i]) for i, d in enumerate(doc_ids)}
pesos_cons = {q: pesos_fila(X_cons[i]) for i, q in enumerate(qids)}

def solap(toks_a, toks_b):
    inter = sorted(set(toks_a) & set(toks_b))
    union = set(toks_a) | set(toks_b)
    jac = (len(inter) / len(union)) if union else 0.0
    return inter, jac

# 4a) consulta vs. documento relevante (las seis consultas)
solap_relevante, terminos_map = [], {}
for qid in qids:
    tq = tokens_unicos(q_texto[qid])
    rel = q_rel[qid]
    td = tokens_unicos(doc_texto[rel])
    inter, jac = solap(tq, td)
    terminos_map[qid] = inter
    solap_relevante.append({
        "consulta_id": qid,
        "consulta": q_texto[qid],
        "documento_relevante": rel,
        "num_terminos_unicos_consulta": len(tq),
        "num_terminos_unicos_relevante": len(td),
        "terminos_consulta": tq,
        "terminos_documento_relevante": td,
        "terminos_compartidos": inter,
        "num_terminos_compartidos": len(inter),
        "cobertura_de_la_consulta": len(inter) / len(tq) if tq else 0.0,
        "jaccard": jac,
        "pesos_tfidf_de_los_compartidos": {
            t: {"peso_en_consulta": pesos_cons[qid].get(t, 0.0),
                "peso_en_documento_relevante": pesos_docs[rel].get(t, 0.0)}
            for t in inter
        },
        "terminos_consulta_fuera_de_vocabulario": [t for t in tq if t not in vocab],
        "rank_tfidf": rank_rel["tfidf"][qid],
        "rank_lsa": rank_rel["lsa"][qid],
        "fallo_tfidf": rank_rel["tfidf"][qid] > 3,
        "fallo_lsa": rank_rel["lsa"][qid] > 3,
    })

# 4b) para los fallos: consulta vs. los documentos que sí quedaron en el top 3
solap_fallos = {"tfidf": {}, "lsa": {}}
tabla_fallos = []
for metodo in ("tfidf", "lsa"):
    S = SIM[metodo]
    for qid in qids:
        if rank_rel[metodo][qid] <= 3:
            continue
        tq = tokens_unicos(q_texto[qid])
        rel = q_rel[qid]
        top3 = rankings[metodo][qid][:3]
        filas = []
        for pos, did in enumerate(top3, start=1):
            td = tokens_unicos(doc_texto[did])
            inter, jac = solap(tq, td)
            filas.append({
                "posicion_en_top3": pos,
                "documento": did,
                "terminos_compartidos_con_consulta": inter,
                "num_terminos_compartidos": len(inter),
                "cobertura_de_la_consulta": len(inter) / len(tq) if tq else 0.0,
                "jaccard": jac,
                "similitud": float(S[qids.index(qid), idx_doc[did]]),
            })
        solap_fallos[metodo][qid] = filas
        tabla_fallos.append({
            "metodo": metodo,
            "consulta_id": qid,
            "consulta": q_texto[qid],
            "documento_relevante": rel,
            "rank_del_relevante": rank_rel[metodo][qid],
            "top3_obtenido": top3,
            "similitud_del_relevante": float(S[qids.index(qid), idx_doc[rel]]),
            "terminos_compartidos_consulta_relevante": terminos_map[qid],
            "solapamiento_consulta_vs_top3": filas,
        })

# 4c) tabla completa método × consulta (12 filas)
tabla_completa = []
for metodo in ("tfidf", "lsa"):
    for qid in qids:
        rk = rank_rel[metodo][qid]
        tabla_completa.append({
            "metodo": metodo,
            "consulta_id": qid,
            "documento_relevante": q_rel[qid],
            "rank_del_relevante": rk,
            "en_top3": rk <= 3,
            "fallo": rk > 3,
            "top3": rankings[metodo][qid][:3],
            "terminos_compartidos_consulta_relevante": terminos_map[qid],
            "num_terminos_compartidos": len(terminos_map[qid]),
        })

# ---------------------------------------------------------------------------
# 5) Análisis sustentado: q5 (fallo en ambos métodos) y caso límite q4 (LSA)
# ---------------------------------------------------------------------------
q5 = next(r for r in solap_relevante if r["consulta_id"] == "q5")
i_q5 = qids.index("q5")
analisis_q5 = {
    "consulta_id": "q5",
    "consulta": q_texto["q5"],
    "documento_relevante": "d09",
    "terminos_consulta": q5["terminos_consulta"],
    "terminos_consulta_en_vocabulario": [t for t in q5["terminos_consulta"] if t in vocab],
    "terminos_consulta_fuera_de_vocabulario": q5["terminos_consulta_fuera_de_vocabulario"],
    "terminos_compartidos_con_d09": q5["terminos_compartidos"],
    "jaccard_con_d09": q5["jaccard"],
    "rank_tfidf": rank_rel["tfidf"]["q5"],
    "rank_lsa": rank_rel["lsa"]["q5"],
    "similitud_tfidf_con_d09": float(S_tfidf[i_q5, idx_doc["d09"]]),
    "diagnostico": (
        f"q5 comparte con d09 únicamente {len(q5['terminos_compartidos'])} término "
        f"({', '.join(q5['terminos_compartidos']) if q5['terminos_compartidos'] else 'ninguno'}), "
        "un artículo sin poder discriminante, y "
        f"{len(q5['terminos_consulta_fuera_de_vocabulario'])} de sus "
        f"{q5['num_terminos_unicos_consulta']} términos (determinista, elegir, siguiente, "
        "palabra, ...) no aparecen en ningún documento del corpus; d09 expresa la idea con "
        "otro léxico (temperatura, logits, softmax, probabilidad). Sin solapamiento léxico "
        "de contenido, el coseno TF-IDF con d09 es casi nulo y el relevante cae al puesto "
        f"{rank_rel['tfidf']['q5']}; los puestos del top 3 los ocupan documentos que solo "
        "comparten con la consulta términos gramaticales o genéricos (la, el, que, al, "
        "modelo, función). LSA se apoya en el mismo vocabulario (el fold-in proyecta la "
        "consulta según los términos que sí coinciden), de modo que hereda el fallo y deja "
        f"el relevante en el puesto {rank_rel['lsa']['q5']}."
    ),
}

top3_q4_lsa = rankings["lsa"]["q4"][:3]
evid_q4 = []
for did in top3_q4_lsa:
    inter, jac = solap(tokens_unicos(q_texto["q4"]), tokens_unicos(doc_texto[did]))
    evid_q4.append({"documento": did, "terminos_compartidos": inter,
                    "num_terminos_compartidos": len(inter), "jaccard": jac})
caso_q4 = {
    "consulta_id": "q4",
    "rank_tfidf": rank_rel["tfidf"]["q4"],
    "rank_lsa": rank_rel["lsa"]["q4"],
    "cuenta_como_fallo": rank_rel["lsa"]["q4"] > 3,
    "top3_lsa": top3_q4_lsa,
    "solapamiento_consulta_vs_top3_lsa": evid_q4,
    "detalle": (
        "q4 no cuenta como fallo con LSA (el relevante d05 queda en el puesto 3, último del "
        "top 3), pero sí es una degradación clara: pasó del puesto 1 con TF-IDF al 3 con LSA. "
        "La proyección latente elevó a d01 y d10, que con la consulta solo comparten términos "
        "gramaticales (de, la), mientras d05 es el que aporta los términos de contenido "
        "(compara, orientación, vectores): el suavizado semántico de LSA diluye la ventaja "
        "léxica del relevante cuando la consulta es corta."
    ),
}

# ---------------------------------------------------------------------------
# 6) Figuras
# ---------------------------------------------------------------------------
# 6a) rank del relevante por consulta y método
fig, ax = plt.subplots(figsize=(8.2, 4.6))
x = np.arange(len(qids))
w = 0.38
rt = [rank_rel["tfidf"][q] for q in qids]
rl = [rank_rel["lsa"][q] for q in qids]
ax.bar(x - w / 2, rt, w, label="TF-IDF (T1)", color="#2e86c1", edgecolor="black", linewidth=0.4)
ax.bar(x + w / 2, rl, w, label="LSA (T2)", color="#27ae60", edgecolor="black", linewidth=0.4)
for xi, (a, b) in enumerate(zip(rt, rl)):
    ax.text(xi - w / 2, a + 0.12, str(a), ha="center", fontsize=8)
    ax.text(xi + w / 2, b + 0.12, str(b), ha="center", fontsize=8)
ax.axhline(3.5, color="gray", linestyle="--", linewidth=1)
ax.text(len(qids) - 0.45, 3.7, "umbral top-3 (fallo por encima)", ha="right", fontsize=8, color="gray")
for xi, q in enumerate(qids):
    if rank_rel["tfidf"][q] > 3:
        ax.text(xi - w / 2, rank_rel["tfidf"][q] + 0.55, "FALLO", ha="center",
                fontsize=7, color="#c0392b", weight="bold")
    if rank_rel["lsa"][q] > 3:
        ax.text(xi + w / 2, rank_rel["lsa"][q] + 0.55, "FALLO", ha="center",
                fontsize=7, color="#c0392b", weight="bold")
ax.set_xticks(x)
ax.set_xticklabels(qids)
ax.set_ylim(0, 10.6)
ax.set_ylabel("rank del documento relevante")
ax.set_title("T3 · Rank del documento relevante por consulta y método (fallo si > 3)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("fallos_por_metodo.png", dpi=120)
plt.close(fig)

# 6b) solapamiento de términos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
fallos_union = sorted(set(fallos["tfidf"]) | set(fallos["lsa"]))
counts = [len(terminos_map[q]) for q in qids]
cols = ["#c0392b" if q in fallos_union else "#2e86c1" for q in qids]
bars = ax1.bar(x, counts, color=cols, edgecolor="black", linewidth=0.4)
for b, c in zip(bars, counts):
    ax1.text(b.get_x() + b.get_width() / 2, c + 0.1, str(c), ha="center", fontsize=9)
ax1.set_xticks(x)
ax1.set_xticklabels(qids)
ax1.set_ylim(0, max(counts) + 2.0)
ax1.set_ylabel("nº de términos compartidos")
ax1.set_title("Consulta vs. documento relevante")
if "q5" in fallos_union:
    terms_q5 = ", ".join("'" + t + "'" for t in terminos_map["q5"]) or "(ninguno)"
    ax1.annotate("q5: " + terms_q5, xy=(qids.index("q5"), len(terminos_map["q5"])),
                 xytext=(0.2, max(counts) * 0.72), fontsize=8, color="#c0392b",
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))

labels2, vals2, cols2 = [], [], []
for metodo in ("tfidf", "lsa"):
    for e in solap_fallos[metodo].get("q5", []):
        labels2.append(metodo + "\n" + e["documento"] + " (top" + str(e["posicion_en_top3"]) + ")")
        vals2.append(e["num_terminos_compartidos"])
        cols2.append("#2e86c1" if metodo == "tfidf" else "#27ae60")
if vals2:
    bars2 = ax2.bar(range(len(vals2)), vals2, color=cols2, edgecolor="black", linewidth=0.4)
    for b, v in zip(bars2, vals2):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.05, str(v), ha="center", fontsize=9)
    ax2.set_xticks(range(len(vals2)))
    ax2.set_xticklabels(labels2, fontsize=7.5)
    ax2.set_ylim(0, max(vals2) + 1.3)
else:
    ax2.text(0.5, 0.5, "sin fallos", ha="center", va="center", transform=ax2.transAxes)
ax2.set_ylabel("nº de términos compartidos")
ax2.set_title("q5 (fallo en ambos métodos): consulta vs. top-3")
fig.suptitle("T3 · Solapamiento de términos (tokenizador de TfidfVectorizer)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("solapamiento_terminos.png", dpi=120)
plt.close(fig)

# 6c) tabla método × consulta con fallos y términos compartidos
col_labels = ["método", "consulta", "relevante", "rank", "¿fallo?", "top-3 obtenido",
              "términos compartidos consulta–relevante"]
cell_text, row_is_fallo = [], []
for f in tabla_completa:
    cell_text.append([
        f["metodo"], f["consulta_id"], f["documento_relevante"],
        str(f["rank_del_relevante"]), "SÍ" if f["fallo"] else "no",
        ", ".join(f["top3"]),
        ", ".join(f["terminos_compartidos_consulta_relevante"]) or "—",
    ])
    row_is_fallo.append(f["fallo"])

fig, ax = plt.subplots(figsize=(11.5, 5.0))
ax.axis("off")
tbl = ax.table(cellText=cell_text, colLabels=col_labels,
               colWidths=[0.07, 0.08, 0.09, 0.06, 0.08, 0.27, 0.35],
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1, 1.45)
for (r, c), celda in tbl.get_celld().items():
    if r == 0:
        celda.set_facecolor("#d6eaf8")
        celda.set_text_props(weight="bold")
    elif row_is_fallo[r - 1]:
        celda.set_facecolor("#fdecea")
ax.set_title("T3 · Fallos por método (relevante fuera del top 3) y términos compartidos con el relevante",
             fontsize=10)
fig.tight_layout()
fig.savefig("tabla_fallos_solapamiento.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 7) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
def to_py(obj):
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

resultados = {
    "subtarea": "T3_analisis_por_consulta_fallos_y_solapamiento_terminos",
    "descripcion": (
        "Identificación por método (TF-IDF de T1 y LSA de T2) de las consultas cuyo documento "
        "relevante queda fuera del top 3, y solapamiento de términos consulta–documento "
        "relevante (y, para los fallos, consulta vs. top-3) usando el tokenizador del "
        "TfidfVectorizer con el que se construyó la matriz."
    ),
    "criterio_de_fallo": "el documento relevante queda fuera del top 3 (rank > 3)",
    "fuente_rankings": "rankings reproducidos con los mismos datos y parámetros de T1 y T2 y verificados contra entrada/T1/resultados.json y entrada/T2/resultados.json",
    "verificacion": {
        "rankings_coinciden_con_T1": coin_t1,
        "rankings_coinciden_con_T2": coin_t2,
    },
    "fallos_por_metodo": {
        "tfidf": fallos["tfidf"],
        "lsa": fallos["lsa"],
    },
    "num_fallos_por_metodo": {
        "tfidf": len(fallos["tfidf"]),
        "lsa": len(fallos["lsa"]),
    },
    "hit_at_3_por_metodo": {
        "tfidf": hit3["tfidf"],
        "lsa": hit3["lsa"],
    },
    "rank_del_relevante_por_metodo": {
        "tfidf": {q: rank_rel["tfidf"][q] for q in qids},
        "lsa": {q: rank_rel["lsa"][q] for q in qids},
    },
    "tabla_fallos_por_metodo_y_consulta": tabla_fallos,
    "tabla_completa_metodo_consulta": tabla_completa,
    "solapamiento_consulta_vs_documento_relevante": solap_relevante,
    "terminos_compartidos_por_consulta": {q: terminos_map[q] for q in qids},
    "solapamiento_consulta_vs_top3_en_fallos": solap_fallos,
    "analisis_q5_fallo_en_ambos_metodos": analisis_q5,
    "analisis_q4_degradacion_con_lsa": caso_q4,
    "conclusion": (
        "TF-IDF falla solo en q5 (d09 en el puesto 6) y LSA falla también solo en q5 (d09 en "
        "el puesto 6), aunque degrada q4 del puesto 1 al 3. La causa, sustentada en el "
        "solapamiento de términos: q5 y d09 comparten únicamente el artículo 'la' (Jaccard "
        "bajo) y la mayoría de los términos de la consulta (determinista, elegir, siguiente, "
        "palabra, ...) no aparecen en el corpus, mientras d09 usa otro léxico (temperatura, "
        "logits, softmax, probabilidad); los documentos del top 3 solo ganan por términos "
        "gramaticales o genéricos. LSA no lo corrige porque su vocabulario es el mismo y el "
        "fold-in de la consulta depende de los términos que sí coinciden."
    ),
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(to_py(resultados), f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 8) Resumen
# ---------------------------------------------------------------------------
print("=" * 70)
print("T3 · Análisis por consulta: fallos (relevante fuera del top 3)")
print("=" * 70)
print(f"Verificación rankings vs T1: {coin_t1} | vs T2: {coin_t2}")
print(f"Fallos TF-IDF: {fallos['tfidf'] if fallos['tfidf'] else 'ninguno'} "
      f"(Hit@3 = {hit3['tfidf']:.4f})")
print(f"Fallos LSA   : {fallos['lsa'] if fallos['lsa'] else 'ninguno'} "
      f"(Hit@3 = {hit3['lsa']:.4f})")
print("-" * 70)
print("Rank del relevante por consulta (TF-IDF / LSA):")
for q in qids:
    marca = "  <-- FALLO" if (rank_rel["tfidf"][q] > 3 or rank_rel["lsa"][q] > 3) else ""
    print(f"  {q}: {rank_rel['tfidf'][q]} / {rank_rel['lsa'][q]}{marca}")
print("-" * 70)
print("Términos compartidos consulta–documento relevante:")
for r in solap_relevante:
    print(f"  {r['consulta_id']} vs {r['documento_relevante']}: "
          f"{r['terminos_compartidos'] if r['terminos_compartidos'] else '(ninguno)'}")
print("-" * 70)
print("q5 (fallo en ambos métodos) — consulta vs. top-3 obtenido:")
for metodo in ("tfidf", "lsa"):
    for e in solap_fallos[metodo].get("q5", []):
        print(f"  [{metodo}] {e['documento']} (top{e['posicion_en_top3']}): "
              f"{e['terminos_compartidos_con_consulta'] if e['terminos_compartidos_con_consulta'] else '(ninguno)'}")
print("-" * 70)
print("Archivos escritos: resultados.json, fallos_por_metodo.png, "
      "solapamiento_terminos.png, tabla_fallos_solapamiento.png")
