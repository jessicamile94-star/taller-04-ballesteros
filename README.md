# Taller 04 — LLMOps

**Estudiante:** Jessica Ballesteros

**Materia:** MMIA 6013 — IA Generativa y Agentes

**Ruta:** B — Solver multiagente del Taller 03

Esta entrega evalúa el solver con un golden set, observa sus ejecuciones con trazas de Langfuse y revisa guardrails para inyección de instrucciones y datos personales.

## Entrega

- [Informe completo en PDF](taller4-ballesteros/output/pdf/Informe_Taller4_Ballesteros.pdf)
- [Informe editable en Markdown](taller4-ballesteros/informe_taller4.md)
- [Resultados del golden set](taller4-ballesteros/resultados_solver_base_completo.csv)
- [Resultados de evaluación del juez](taller4-ballesteros/juez_solver_base_completo_reintento.csv)
- [Comparación de resultados históricos](taller4-ballesteros/comparacion_resultados.csv)
- [Trazas Langfuse exportadas](taller4-ballesteros/spans_langfuse_completo.jsonl)
- [Medición de guardrails](taller4-ballesteros/guardrails.csv)

## Estado de cumplimiento

- **Golden set:** 54 de 55 comprobaciones aprobaron en la evaluación automática; S208 falló.
- **Juez:** aprobó los siete criterios conceptuales (A10, A11, C09, C10, E02, P01 y P02) y coincidió con las siete etiquetas humanas.
- **Guardrails propios:** 21/21 casos del banco local, sin falsos positivos ni falsos negativos.
- **Corridas:** A, B, C, E y P completaron sus tareas. Los reintentos de S1 y S2 produjeron entregables, pero quedaron parciales: S1/T5 pendiente; S2/T7 pendiente y T8 no ejecutada por agotamiento del presupuesto.

Por tanto, la entrega documenta evidencia de evaluación y observabilidad, pero las tareas S1 y S2 no están completamente resueltas. Los resultados, limitaciones y pasos de reproducción están detallados en el [informe](taller4-ballesteros/informe_taller4.md).

## Reproducir y revisar resultados

Los comandos se ejecutan desde la raíz del repositorio, con Python 3.12 y la VPN GlobalProtect de USFQ conectada para acceder a los endpoints de la H200.

```bash
# Sin volver a llamar al modelo: verificar el golden contra las corridas guardadas
uv run python Lab-03-Agentes/solver-v2/evaluar_solver.py \
  --golden taller4-ballesteros/golden_set.json \
  --solo-evaluar corridas/taller4-medicion-base-final \
  --salida taller4-ballesteros/resultados_solver_reproducido.csv

# Ejecutar una tarea con el solver
USE_LANGFUSE=true uv run python scripts/correr.py \
  tareas/reales/semana1/enunciado.pdf \
  --salida corridas/reproduccion/tarea-S1

# Revisar una corrida y sus eventos
uv run python scripts/evidencia.py corridas/reproduccion/tarea-S1
```

Las corridas nuevas de S1 y S2, incluidos sus entregables y trazas, están en `corridas/taller4-reintento-estable/`. No reemplazan las corridas base usadas para las métricas agregadas del informe.

## Estructura de la entrega

| Ruta | Contenido |
|---|---|
| `taller4-ballesteros/` | Golden set, resultados, guardrails, informe y exportaciones de trazas del Taller 04 |
| `corridas/taller4-medicion-base-final/` | Corridas base utilizadas en la evaluación y métricas agregadas |
| `corridas/taller4-reintento-estable/` | Reintentos de S1 y S2 con la H200 reconectada |
| `solver/` | Solver multiagente, guardrails, trazas y procedencia |
| `Lab-04-LLMOps/` | Evaluador, juez y utilidades del Taller 04 |
| `Lab-03-Agentes/` | Material de referencia del solver desarrollado para el Taller 03 |

## Nota de integridad académica

El solver evaluado es un sistema multiagente basado en LLM. En la elaboración y documentación de esta entrega también se utilizó asistencia de IA; el informe describe su alcance y la verificación realizada.

## Anexo: Taller 03

El Taller 03 corresponde al solver utilizado como sistema bajo evaluación en esta entrega. Su guía de instalación, arquitectura y reproducción queda como material complementario en [docs/anexo_taller03.md](docs/anexo_taller03.md).
