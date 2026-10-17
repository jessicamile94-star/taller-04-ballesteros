## Objetivo

Comparar un recuperador léxico (TF-IDF + similitud coseno) con semántica latente (LSA, TruncatedSVD a 4 dimensiones) sobre un corpus de 10 documentos y 6 consultas con un documento relevante cada una, medir Hit@1, Hit@3 y MRR, analizar en qué consultas falla cada método según el solapamiento de términos, y discutir qué aportaría un recuperador denso.

## Método

- **Datos:** 10 documentos (d01–d10) y 6 consultas (q1–q6) con juicios q1→d04, q2→d03, q3→d08, q4→d05, q5→d09, q6→d02.
- **Línea base léxica (Parte 1):** TfidfVectorizer de scikit-learn con parámetros por defecto (vocabulario de 105 términos), similitud coseno consulta–documento y ordenamiento de los 10 documentos por consulta.
- **Semántica latente (Parte 2):** TruncatedSVD(n_components=4, random_state=0) ajustado sobre la matriz TF-IDF; consultas proyectadas con transform (fold-in) y coseno en el espacio latente. Varianza explicada acumulada: 0.3953.
- **Evaluación:** Hit@1, Hit@3 y MRR sobre las 6 consultas; fallo = relevante fuera del top 3. El solapamiento de términos usa el tokenizador del vectorizador. Los rankings TF-IDF de la Parte 2 reproducen los de la Parte 1 (diferencia máxima 0.0).

## Resultados

**Tabla comparativa (Parte 2):**

| Método | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| TF-IDF (línea base léxica) | 0.8333 | 0.8333 | 0.8611 |
| LSA (4 dimensiones) | 0.6667 | 0.8333 | 0.75 |

![Comparación de métricas TF-IDF vs LSA](figuras/T2_comparacion_metricas.png)

**Rank del relevante por consulta:**

| Consulta | Relevante | Rank TF-IDF | Rank LSA |
|---|---|---|---|
| q1 | d04 | 1 | 1 |
| q2 | d03 | 1 | 1 |
| q3 | d08 | 1 | 1 |
| q4 | d05 | 1 | 3 |
| q5 | d09 | 6 | 6 |
| q6 | d02 | 1 | 1 |

**Fallos (Parte 3):** ambos métodos fallan solo en q5 (rank 6); LSA además degrada q4 de 1 a 3 (d05: 0.8984, frente a d01: 0.9220 y d10: 0.9206). En q5, la consulta comparte con d09 un único término ("la") y la similitud TF-IDF del relevante es 0.0790. El top 3 de TF-IDF (d03, d06, d04) cubre más términos de la consulta: d03 comparte "al", "la", "modelo" (cobertura 0.2308; similitud 0.3107), d06 comparte "el", "qué" (0.2009) y d04 comparte "la", "que" (0.1896). Con LSA el relevante alcanza 0.4999, pero d04 (0.9427), d01 (0.8066) y d03 (0.6141) lo superan. En las consultas sin fallo el solapamiento es alto: q1 comparte 8 términos con d04 (p. ej. "ranking", "léxica", "frecuencia", "pondera"), q3 y q6 comparten 5, q4 comparte 4 y q2 comparte 3.

![Fallos por método y rank del relevante](figuras/T3_fallos_por_metodo.png)

## Discusión

**Por qué falla cada método en q5.** TF-IDF exige coincidencia léxica: la consulta usa "determinista", "elegir" y "palabra", que no aparecen en d09 ("temperatura", "logits", "softmax", "probabilidad"); el único puente es "la", así que documentos que comparten palabras más frecuentes de la consulta ("modelo" en d03, "qué"/"el" en d06) superan al relevante. LSA no lo corrige: con 4 componentes sobre solo 10 documentos explica 0.3953 de la varianza y el espacio latente capta co-ocurrencias generales, no sinonimia; d09 sube a 0.4999, pero d04 y d01 suben más. En q4, ese mismo suavizado hace que d01 y d10, vecinos latentes de la consulta, desplacen a d05 al rank 3.

**Recuperador denso para q5.** Un modelo de embeddings entrenado aprendería, a partir de grandes corpus, que "más determinista" se relaciona con "temperatura baja concentra la probabilidad"; colocaría cerca los vectores de q5 y d09 pese a no compartir términos de contenido, resolviendo el desajuste de vocabulario que ni TF-IDF ni una LSA entrenada con 10 documentos pueden captar.

**Costos frente a TF-IDF.** Requiere datos y cómputo de entrenamiento (o un modelo preentrenado), mayor costo de inferencia para codificar consultas y documentos, un índice denso de vectores más costoso en memoria que la matriz dispersa de 105 términos, y pierde interpretabilidad: TF-IDF permite explicar cada puntuación por términos compartidos (como el 0.3107 de d03 en q5 por "modelo"), mientras que las dimensiones latentes y los embeddings no son directamente legibles.
