# Taller 4 — LLMOps y solver multiagente

En este repositorio trabajé con la ruta **B — solver multiagente**. Aquí resumo los resultados que pude reproducir y las partes de las corridas que quedaron incompletas. Los datos y archivos mencionados se pueden revisar en el propio repositorio.

## Evidencia comprobable en el repositorio

| Requisito | Estado observado |
| --- | --- |
| Golden set de 20 o más comprobaciones | 55 checks distribuidos en 7 tareas (A, B, C, S1, S2, E, P). |
| Checks de juez | 7: A10, A11, C09, C10, E02, P01 y P02. Todos incluyen criterio. |
| Etiquetas humanas previas al juez | 22 filas etiquetadas en `etiquetas.csv`; contiene los siete checks de tipo juez. |
| Banco de ataques | 21 casos: 17 de plantilla y 4 propios del solver (B01–B04). |
| Guardrails | La medición local de los 21 casos obtuvo 21 aciertos, 0 falsos negativos y 0 falsos positivos. CSV: `guardrails.csv`. |
| Versión de guardrails | `solver-guardrails-v1`; los nuevos eventos locales y spans de guardrail incluyen esta versión para facilitar comparaciones reproducibles. Las corridas A/E/P ya exportadas son evidencia anterior a este campo. |
| Guardrails de ejemplo | Sobre el mismo banco: 7 falsos negativos (53.8 %) y 3 falsos positivos (37.5 %). CSV: `guardrails_ejemplo.csv`. |
| Resultados por checks | `resultados_solver.csv` tiene 55 checks; `juez_solver.csv`, 36 filas de checks juzgables. |
| Spans locales | `spans.jsonl` tiene 157 filas de 7 corridas: 7 corrida, 14 guardrail, 108 LLM y 28 herramienta. |
| Corrida base reproducida | `corridas/taller4-medicion-base-final/tarea-A`: completada con GLM-5.3-Flash, 35,764 tokens de entrada y 47,971 de salida; 9/11 checks automáticos. |
| Corridas de las siete tareas | A, B y C completadas; E/P completadas; S1/S2 parciales. Todos los artefactos y trazas están en `corridas/taller4-medicion-base-final/`. |
| Evaluación determinística del golden | 47/48 checks decidibles aprobados. S208 falla; los siete checks de juez se excluyen de esta tasa. CSV: `resultados_solver_base_completo.csv`. |
| Enlace de resultados con Langfuse | `resultados_solver_base_completo_con_traza.csv` agrega a cada check el `traza_id`, `prompt_version`, `session_id` y etiquetas de la corrida de su tarea. Los siete ids y sus metadatos se comprobaron contra `spans_langfuse_completo.jsonl`. |
| Exportación Langfuse de las siete | 318 spans de 7 corridas filtradas en `spans_langfuse_completo.jsonl`; p50 175.787 s, p95 8,352.170 s, 339,368 tokens de entrada, 885,813 de salida, costo simulado USD 0.582393. |
| Comparación con resultados históricos | Sobre 47 checks determinísticos de A/B/C/S1/S2: 46/47 aprobados antes y 46/47 ahora. S103 pasó y S208 regresó. Detalle: `comparacion_resultados.csv`. |
| Juez final | Los siete criterios conceptuales (A10, A11, C09, C10, E02, P01, P02) fueron aprobados; acuerdo 100 % con las siete etiquetas humanas (kappa indefinido por una sola clase). CSV: `juez_solver_base_completo_reintento.csv`; golden filtrado: `golden_juez_pendientes.json`. |
| Reintento tras intermitencia H200 | S1 y S2 se repitieron en `corridas/taller4-reintento-estable/`; el endpoint volvió a responder, pero ambas corridas terminaron parciales por agotamiento de presupuesto. S1 aprobó T1–T4 y dejó T5 pendiente; S2 aprobó T1–T6, dejó T7 pendiente y no ejecutó T8. |
| Juez del control A/E/P | 11 veredictos: acuerdo 100 % con 6 checks automáticos y 9 etiquetas humanas. Kappa indefinido en ambas comparaciones porque hay una sola clase. |
| Exportación Langfuse de control A | 43 spans de una traza en `spans_langfuse.jsonl`; p50/p95 157.266 s; 34,109 tokens de entrada y 38,764 de salida; precio de referencia gpt-4o-mini: USD 0.028375. El mayor tiempo fue `programador` (87.293 s de generations). |
| Guardrail sandbox en el control A | Tres spans, uno por ejecución; total medido 12.7 s. La comprobación 21/21 del banco es aparte y no sustituye pruebas de ataque dentro del solver. |
| Corridas adversarial y PII | E y P ejecutadas en `corridas/taller4-medicion-base-final/`; E se completó tras eliminar la línea inyectada. P se completó con correo/cédula redactados. `spans_guardrails.jsonl` conserva 42 spans; p50 42.984 s, p95 56.025 s, costo simulado USD 0.012052. |
| Evaluación del subconjunto A/E/P | 10/15 en el verificador automático: A aprueba 9/11; E aprueba E01. Los cinco checks de tipo juez se marcan como desconocidos automáticamente y se revisan con el juez. |

## Brechas que siguen abiertas

- `spans_langfuse_completo.jsonl` es la exportación filtrada de Langfuse para las siete corridas actuales. `spans_langfuse.jsonl` y `spans_guardrails.jsonl` conservan las exportaciones previas de control A/E/P.
- E y P ya tienen enunciados sintéticos y corridas reales. E eliminó la directiva escondida; P eliminó la PII antes de enviarla al LLM. Se comprobaron reporte, traza local, caché del enunciado y spans exportados.
- El banco ya cumple los cuatro casos propios. Su CSV fue comprobado localmente; el CSV guardado puede regenerarse con el comando de abajo.
- La comparación histórica/nueva usa el mismo golden y evaluador sobre A/B/C/S1/S2; hubo una mejora (S103) y una regresión (S208), con tasa neta sin cambio. No demuestra una mejora general.
- La versión `solver-guardrails-v1` queda registrada desde ahora; falta ejecutar de nuevo el golden con una versión posterior para completar una comparación antes/después.
- Los reintentos más recientes de S1 y S2 alcanzaron la H200 y generaron entregables, pero quedaron parciales: S1 aprobó T1–T4 y T5 agotó tres intentos; S2 aprobó T1–T6 y T7 agotó su intento, mientras T8 quedó sin ejecutar por presupuesto. Los reportes nuevos, artefactos y trazas están en `corridas/taller4-reintento-estable/`.
- El juez conceptual ya tiene siete veredictos y coincide con las etiquetas humanas. Aún falta completar S1/T5 y S2/T7–T8 antes de afirmar cumplimiento total.
- Las siete trazas tienen `session_id=taller4-ballesteros-base` y las etiquetas `taller-04` y `solver-b`. La sesión ya estaba guardada; añadí las etiquetas a las trazas existentes y volví a exportar los campos de Langfuse. No repetí el solver ni cambié sus métricas.
- La comparación histórica/instrumentada deja una corrección y una regresión, sin mejora neta. Falta una corrida del golden completo con otra versión; la Parte 2.c sigue pendiente.
- El Python global es 3.14, pero `uv sync` ya preparó el entorno requerido de Python 3.12. En esta sesión respondieron los endpoints de la H200 y `auth_check()` de Langfuse.

## Comandos reproducibles

Desde la raíz del repositorio, con Python 3.12 y la VPN de USFQ activa:

```bash
USE_LANGFUSE=true uv run python scripts/correr.py \
  solver-v2/enunciados/tarea-a-generativo-discriminativo.pdf \
  --salida corridas/taller4-medicion-base-final/tarea-A

USE_LANGFUSE=true uv run python scripts/correr.py \
  taller4-ballesteros/enunciados/tarea-E-inyeccion.pdf \
  --salida corridas/taller4-medicion-base-final/tarea-E

USE_LANGFUSE=true uv run python scripts/correr.py \
  taller4-ballesteros/enunciados/tarea-P-privacidad.pdf \
  --salida corridas/taller4-medicion-base-final/tarea-P

(
  cd taller4-ballesteros
  python3 ../Lab-04-LLMOps/probar_guardrails.py \
    --entrada guardrails:validar_entrada \
    --salida guardrails:validar_salida \
    --casos ataques.json \
    --csv guardrails.csv
)

uv run --directory Lab-04-LLMOps python exportar_trazas.py \
  --sesion taller4-ballesteros-base -o ../taller4-ballesteros/spans_exportados.jsonl
python3 taller4-ballesteros/filtrar_spans.py \
  corridas/taller4-medicion-base-final/tarea-A/traza.jsonl \
  taller4-ballesteros/spans_exportados.jsonl \
  taller4-ballesteros/spans_langfuse.jsonl
python3 taller4-ballesteros/filtrar_spans.py \
  corridas/taller4-medicion-base-final/tarea-E/traza.jsonl \
  corridas/taller4-medicion-base-final/tarea-P/traza.jsonl \
  taller4-ballesteros/spans_exportados.jsonl \
  taller4-ballesteros/spans_guardrails.jsonl
uv run --directory Lab-04-LLMOps python resumir_trazas.py \
  ../taller4-ballesteros/spans_langfuse.jsonl --precio propietario_economico

uv run python Lab-03-Agentes/solver-v2/evaluar_solver.py \
  --golden taller4-ballesteros/golden_base_control.json \
  --solo-evaluar <carpeta-corridas-base> \
  --salida taller4-ballesteros/resultados_solver_base.csv

uv run --directory Lab-04-LLMOps python juez_solver.py \
  --golden ../taller4-ballesteros/golden_base_control.json \
  --corridas <carpeta-corridas-base> \
  --resultados ../taller4-ballesteros/resultados_solver_base.csv \
  --etiquetas ../taller4-ballesteros/etiquetas.csv \
  --salida ../taller4-ballesteros/juez_solver_base.csv
```

Los comandos de exportación y filtrado de arriba corresponden a las exportaciones anteriores de control A/E/P. La exportación filtrada más reciente de las siete corridas está en `spans_langfuse_completo.jsonl`; sus checks se enlazan con `resultados_solver_base_completo_con_traza.csv`.

Las etiquetas se prepararon antes de abrir el CSV del juez. Si se modifica el golden o se cambian los modelos, hay que volver a producir los CSV correspondientes. Las credenciales se conservan en `.env` local; no copiar valores de `.env` a trazas, informes ni salidas de comandos.
