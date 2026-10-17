import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

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
DOC_IDS, Q_IDS = list(DOCS), list(QUERIES)
STOP = {"el","la","los","las","un","una","unos","unas","de","del","al","y","o","en","con","por","para",
        "que","qué","se","su","sus","es","son","como","más","cada","antes","según","entre","sin","sobre"}
TOK = re.compile(r"(?u)\b\w\w+\b")
toks = lambda t: TOK.findall(t.lower())

doc_tokens = {d: set(toks(DOCS[d])) for d in DOC_IDS}
DF = {t: sum(1 for d in DOC_IDS if t in doc_tokens[d]) for t in set().union(*doc_tokens.values())}
es_contenido = lambda t: t not in STOP and DF[t] <= 3

vec = TfidfVectorizer()  # mismos parámetros por defecto que T1
X = vec.fit_transform([DOCS[d] for d in DOC_IDS])
Qm = vec.transform([QUERIES[q][0] for q in Q_IDS])
vidx = {t: i for i, t in enumerate(vec.get_feature_names_out())}
Xa, Qa = X.toarray(), Qm.toarray()  # filas L2 (coseno = producto punto)

svd = TruncatedSVD(n_components=4, random_state=0)  # misma configuración que T2
Xl = normalize(svd.fit_transform(X))
Ql = normalize(svd.transform(Qm))
S_lsa = Ql @ Xl.T

def ranks_from_S(S):
    return {q: [DOC_IDS[i] for i in np.argsort(-S[j], kind="stable")] for j, q in enumerate(Q_IDS)}

def cargar(ruta, clave):
    data = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return {q: list(v["ranking"]) for q, v in data[clave].items()}

rankings, fuentes = {}, {}
for met, (ruta, clave) in {"TF-IDF": ("entrada/T1/resultados.json", "ranking_por_consulta"),
                           "LSA": ("entrada/T2/resultados.json", "lsa_ranking_por_consulta")}.items():
    try:
        rankings[met] = cargar(ruta, clave); fuentes[met] = ruta
    except Exception as e:
        print(f"[aviso] {ruta} ilegible ({e}); se recalcula {met} con la configuración previa.")
        rankings[met] = ranks_from_S(Qa @ Xa.T if met == "TF-IDF" else S_lsa)
        fuentes[met] = "recalculado localmente"
    for q in Q_IDS:
        assert len(rankings[met][q]) == 10 and set(rankings[met][q]) == set(DOC_IDS)

for met, own in [("TF-IDF", ranks_from_S(Qa @ Xa.T)), ("LSA", ranks_from_S(S_lsa))]:
    ok = all(rankings[met][q] == own[q] for q in Q_IDS)
    print(f"Verificación {met}: rankings {'coinciden' if ok else 'NO coinciden'} con el recálculo local.")

rank_rel = {met: {q: rankings[met][q].index(QUERIES[q][1]) + 1 for q in Q_IDS} for met in rankings}
fallos = {met: [q for q in Q_IDS if rank_rel[met][q] > 3] for met in rankings}
limites = {met: [q for q in Q_IDS if 1 < rank_rel[met][q] <= 3] for met in rankings}
problematicas = sorted(set(sum(fallos.values(), [])) | set(sum(limites.values(), [])))

def detalle(qid, doc_id):
    out = []
    for t in sorted(doc_tokens[doc_id] & set(toks(QUERIES[qid][0]))):
        out.append({"termino": t, "df": DF[t], "tipo": "contenido" if es_contenido(t) else "funcional",
                    "peso_consulta": float(Qa[Q_IDS.index(qid), vidx[t]]),
                    "peso_documento": float(Xa[DOC_IDS.index(doc_id), vidx[t]])})
    out.sort(key=lambda r: (r["tipo"] != "contenido", r["df"], r["termino"]))
    return out

lista_ts = lambda det: ", ".join(f"{t['termino']}({t['tipo'][0]})" for t in det) or "ninguno"

analisis = {}
for q in problematicas:
    texto_q, rel = QUERIES[q]
    det_rel = detalle(q, rel)
    por_metodo = {}
    for met in rankings:
        rk = rankings[met][q]
        intrusos = []
        for d in rk[:3]:
            if d == rel:
                continue
            det = detalle(q, d)
            intrusos.append({"doc": d, "rank": rk.index(d) + 1, "num": len(det),
                             "num_contenido": sum(1 for t in det if t["tipo"] == "contenido"),
                             "detalle": det,
                             "contenido": [t["termino"] for t in det if t["tipo"] == "contenido"],
                             "compartidos_doc_vs_relevante": sorted(doc_tokens[d] & doc_tokens[rel])})
        por_metodo[met] = {"rank_relevante": rank_rel[met][q], "falla_top3": rank_rel[met][q] > 3,
                           "top3": rk[:3], "intrusos_top3": intrusos}
    analisis[q] = {"consulta": texto_q, "relevante": rel,
                   "terminos_consulta": sorted(set(toks(texto_q))),
                   "terminos_documento_relevante": sorted(doc_tokens[rel]),
                   "compartidos_con_relevante": det_rel,
                   "num_compartidos_con_relevante": len(det_rel),
                   "num_contenido_con_relevante": sum(1 for t in det_rel if t["tipo"] == "contenido"),
                   "contenido_con_relevante": [t["termino"] for t in det_rel if t["tipo"] == "contenido"],
                   "por_metodo": por_metodo}

explicaciones = {}
for q in problematicas:
    a = analisis[q]
    for met in rankings:
        pm = a["por_metodo"][met]
        if pm["falla_top3"]:
            conts = sorted({t for it in pm["intrusos_top3"] for t in it["contenido"]})
            trozos = [f"{q} falla en {met}: el relevante {a['relevante']} queda en el rank {pm['rank_relevante']} (fuera del top 3).",
                      f"Solape consulta↔relevante: {a['num_compartidos_con_relevante']} término(s) [{lista_ts(a['compartidos_con_relevante'])}], de contenido: {a['num_contenido_con_relevante']}."]
            trozos += [f"Documento del top 3 {it['doc']} (rank {it['rank']}): {it['num']} término(s) con la consulta [{lista_ts(it['detalle'])}]." for it in pm["intrusos_top3"]]
            if met == "TF-IDF":
                trozos.append("Causa: la consulta no comparte términos de contenido con su relevante; el coseno TF-IDF premia a los documentos del top"
                              + (f" que sí comparten términos de contenido ({', '.join(conts)})" if conts else " que comparten más términos (funcionales) con la consulta") + ".")
            else:
                trozos.append("Causa: con solape léxico de contenido casi nulo, la proyección LSA a k=4 sustituye las coincidencias exactas por la estructura latente del corpus y acerca la consulta a documentos generales, dejando el relevante fuera del top 3.")
            explicaciones[f"{met}::{q}"] = " ".join(trozos)
        elif pm["rank_relevante"] > 1:
            sup = [it for it in pm["intrusos_top3"] if it["rank"] < pm["rank_relevante"]]
            trozos = [f"{q} no falla el top 3 en {met} (rank {pm['rank_relevante']}) pero pierde Hit@1: el relevante {a['relevante']} comparte {a['num_compartidos_con_relevante']} término(s) [{lista_ts(a['compartidos_con_relevante'])}] y aun así es superado por "
                      + "; ".join(f"{it['doc']} (rank {it['rank']}, {it['num']} término(s): {lista_ts(it['detalle'])})" for it in sup) + ".",
                      "Causa: los documentos que lo superan solo comparten términos funcionales; con k=4 la reducción difumina las coincidencias exactas de contenido y mezcla los documentos en la dirección general del corpus."]
            explicaciones[f"{met}::{q}"] = " ".join(trozos)

tabla_todas = []
for q in Q_IDS:
    texto_q, rel = QUERIES[q]
    det = detalle(q, rel)
    tabla_todas.append({"consulta": q, "texto": texto_q, "relevante": rel,
                        "terminos_compartidos_con_relevante": [t["termino"] for t in det],
                        "num_terminos_compartidos": len(det),
                        "terminos_de_contenido": [t["termino"] for t in det if t["tipo"] == "contenido"],
                        "rank_relevante_tfidf": rank_rel["TF-IDF"][q], "rank_relevante_lsa": rank_rel["LSA"][q],
                        "falla_tfidf": rank_rel["TF-IDF"][q] > 3, "falla_lsa": rank_rel["LSA"][q] > 3})

filas = []
for met in rankings:
    for q in Q_IDS:
        rk, rel = rankings[met][q], QUERIES[q][1]
        mostrar = list(rk[:3]) + ([rel] if rel not in rk[:3] else [])
        for d in mostrar:
            det = detalle(q, d)
            filas.append({"metodo": met, "consulta": q, "doc": d, "rank": rk.index(d) + 1,
                          "rol": "relevante" if d == rel else "top3_no_relevante",
                          "num_terminos_compartidos_con_consulta": len(det),
                          "num_de_contenido": sum(1 for t in det if t["tipo"] == "contenido"),
                          "terminos_compartidos": "; ".join(t["termino"] for t in det),
                          "terminos_de_contenido": "; ".join(t["termino"] for t in det if t["tipo"] == "contenido")})
pd.DataFrame(filas).to_csv("tabla_terminos_compartidos.csv", index=False, encoding="utf-8-sig")

casos = [(met, q) for met in rankings for q in (fallos[met] + limites[met])]
if casos:
    fig, axes = plt.subplots(1, len(casos), figsize=(4.8 * len(casos), 4.4), squeeze=False)
    for ax, (met, q) in zip(axes[0], casos):
        rk, rel = rankings[met][q], QUERIES[q][1]
        mostrar = list(rk[:3]) + ([rel] if rank_rel[met][q] > 3 else [])
        counts = [len(doc_tokens[d] & set(toks(QUERIES[q][0]))) for d in mostrar]
        ax.bar(range(len(mostrar)), counts, color=["#2a9d8f" if d == rel else "#e76f51" for d in mostrar])
        ax.set_xticks(range(len(mostrar)))
        ax.set_xticklabels([f"{d}\n(rank {rk.index(d)+1})" for d in mostrar], fontsize=8)
        for i, c in enumerate(counts):
            ax.text(i, c + 0.05, str(c), ha="center", fontsize=9)
        ax.set_ylim(0, max(counts) + 1.2)
        ax.set_ylabel("Términos compartidos con la consulta")
        ax.set_title(f"{met} · {q} — relevante {rel} en rank {rank_rel[met][q]}", fontsize=9)
    fig.suptitle("T3: solape léxico consulta–documento en consultas problemáticas (verde = relevante, rojo = resto del top 3)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig("fallos_terminos_compartidos.png", dpi=120)
    plt.close(fig)

res = {
    "subtarea": "T3: análisis por consulta — fallos (relevante fuera del top 3) y términos compartidos consulta-documento para TF-IDF y LSA",
    "criterio_fallo": "rank_del_relevante > 3 (documento relevante fuera del top 3)",
    "definiciones": {
        "termino": "token de TfidfVectorizer por defecto: minúsculas y regex (?u)\\b\\w\\w+\\b (sin stopwords ni stemming), igual que T1/T2",
        "tipo_de_termino": "'contenido' si no está en la lista funcional mínima y DF <= 3 en el corpus de 10 documentos; 'funcional' en caso contrario",
        "intrusos_top3": "documentos del top 3 distintos del relevante: en consultas que fallan son los mal rankeados; en casos límite, los que superan al relevante",
    },
    "fuentes_rankings": fuentes,
    "fallos_por_metodo": {met: {"consultas": fallos[met], "num_fallos": len(fallos[met])} for met in rankings},
    "rank_del_relevante_por_metodo": {met: {q: rank_rel[met][q] for q in Q_IDS} for met in rankings},
    "casos_limite_hit1": {met: [{"consulta": q, "rank": rank_rel[met][q]} for q in limites[met]] for met in rankings},
    "consultas_problematicas": problematicas,
    "analisis_consultas_problematicas": analisis,
    "tabla_todas_las_consultas": tabla_todas,
    "explicaciones": explicaciones,
    "archivos_generados": ["resultados.json", "tabla_terminos_compartidos.csv", "fallos_terminos_compartidos.png"],
}
Path("resultados.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== T3 resumen ===")
for met in rankings:
    print(f"{met}: fallos (relevante fuera del top 3): {fallos[met] or 'ninguno'} | casos límite Hit@1 (rank 2-3): {limites[met] or 'ninguno'}")
for q in problematicas:
    a = analisis[q]
    print(f"{q} (relevante {a['relevante']}): solape con relevante = {[t['termino'] for t in a['compartidos_con_relevante']]} (contenido: {a['contenido_con_relevante'] or 'ninguno'})")
    for met in rankings:
        pm = a["por_metodo"][met]
        print(f"  {met}: rank relevante = {pm['rank_relevante']}, top3 = {pm['top3']}, solapes top3 = {[(it['doc'], it['num']) for it in pm['intrusos_top3']]}")
print("Archivos escritos: resultados.json, tabla_terminos_compartidos.csv, fallos_terminos_compartidos.png")
