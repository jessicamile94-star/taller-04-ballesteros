## Objetivo

Comparar la recuperación léxica (TF-IDF + similitud coseno) contra la semántica latente (LSA mediante TruncatedSVD a 4 dimensiones) sobre un corpus mínimo de 10 documentos y 6 consultas con un documento relevante cada una. Se miden Hit@1, Hit@3 y MRR, se analizan los fallos por consulta examinando los términos compartidos entre consulta y documento relevante, y se discute por qué un recuperador denso resolvería mejor el caso fallido (q5) y a qué costo.

## Método

- **Datos:** corpus d01–d10; consultas q1–q6 con juicios q1→d04, q2→d03, q3→d08, q4→d05, q5→d09, q6→d02.
- **Línea base (Parte 1):** `TfidfVectorizer` con parámetros por defecto (minúsculas, token regex `\b\w\w+\b`, sin stopwords ni stemming) y similitud coseno; se ordenan los 10 documentos por consulta.
- **LSA (Parte 2):** `TruncatedSVD(n_components=4, random_state=0)` sobre la misma matriz TF-IDF; misma evaluación. Los rankings de TF-IDF coinciden entre ambas ejecuciones (diferencias 0.0).
- **Métricas:** Hit@1, Hit@3 y MRR agregados sobre las 6 consultas. **Fallo:** relevante fuera del top 3 (rank > 3). En el análisis léxico, un término es "de contenido" si no es funcional y su DF ≤ 3 en el corpus.

## Resultados

**Tabla Parte 2 — ambos métodos:**

| Método | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| TF-IDF | 0.8333 | 0.8333 | 0.8611 |
| LSA (k=4) | 0.6667 | 0.8333 | 0.75 |

LSA retiene 0.3953 de la varianza total. Diferencia LSA−TF-IDF: Hit@1 −0.1667, Hit@3 0.0, MRR −0.1111.

**Tabla Parte 3 — rank del relevante por consulta:**

| Consulta | Relevante | Rank TF-IDF | Rank LSA |
|---|---|---|---|
| q1 | d04 | 1 | 1 |
| q2 | d03 | 1 | 1 |
| q3 | d08 | 1 | 1 |
| q4 | d05 | 1 | 3 |
| q5 | d09 | 6 | 6 |
| q6 | d02 | 1 | 1 |

Fallos (rank > 3): TF-IDF falla solo en q5; LSA falla solo en q5 (1 fallo cada uno). Caso límite: q4 en LSA queda rank 3, dentro del top 3.

![Comparación de métricas TF-IDF vs LSA](figuras/T2_t2_comparacion_metricas.png)

**q5 (fallo común):** la consulta y d09 comparten un único término, "la" (funcional, DF=6); **0 términos de contenido** ("determinista", "elegir", "palabra" no aparecen en d09, que habla de "temperatura", "logits", "softmax", "probabilidad"). En TF-IDF, d03 encabeza con 0.3107 frente a 0.0790 de d09 porque comparte "modelo" (contenido, DF=1, peso en consulta 0.4545); d06 y d04 suben solo por funcionales ("qué"/"el", "que"/"la"). En LSA, d09 queda 6.º (0.4999) tras d04 (0.9427), d01, d03, d06 y d10.

**q4 (caso límite):** la consulta comparte con d05 tres términos de contenido con DF=1: "compara", "orientación" y "vectores"; TF-IDF lo coloca 1.º (0.4569). En LSA los intrusos d01 (0.9220) y d10 (0.9206) solo comparten funcionales ("de"/"la" y "la"/"los"), pero la proyección comprime las diferencias y d05 baja al puesto 3 con 0.8984.

![Fallos por consulta y términos compartidos](figuras/T3_fallos_terminos_compartidos.png)

## Discusión

En este corpus mínimo TF-IDF supera a LSA: Hit@1 0.8333 frente a 0.6667 y MRR 0.8611 frente a 0.75, con Hit@3 empatado en 0.8333. Proyectar a 4 componentes que capturan solo 0.3953 de la varianza mezcla señales en 10 documentos: en q4 el relevante cae del puesto 1 al 3 sin que los intrusos aporten contenido, y en q5 el fallo persiste.

El único fallo de ambos métodos es q5, y la causa es léxica: no existe solape de contenido entre la consulta y d09 (0 términos compartidos de contenido; el único común es "la", DF=6). TF-IDF se deja llevar por "modelo" (d03, DF=1) y LSA, al derivar de la misma matriz y comprimir a 4 dimensiones, hereda el problema y añade ruido de la proyección.

Un recuperador denso con embeddings de un modelo entrenado resolvería q5 mejor porque aprende, en un corpus masivo, que "más determinista al elegir la siguiente palabra" es paráfrasis de "temperatura baja concentra la probabilidad": mapea sinónimos y relaciones semánticas sin requerir términos compartidos, que aquí son exactamente 0. TF-IDF no puede establecer esa equivalencia por construcción, y LSA entrenado sobre solo 10 documentos tampoco tiene señal para aprenderla.

Los costos frente a TF-IDF son: (1) un modelo preentrenado, con su memoria y posible GPU para inferencia; (2) cómputo de codificación del corpus y de cada consulta en tiempo de búsqueda; (3) indexación y almacenamiento de vectores densos (típicamente con índices ANN); (4) menor interpretabilidad —en TF-IDF cada similitud se traza a términos ponderados (p. ej., "modelo" con peso 0.4545 explica el falso positivo d03 en q5), mientras que las dimensiones densas no son inspeccionables—; y (5) riesgo de desajuste de dominio, que obliga a reentrenar o afinar el modelo.
