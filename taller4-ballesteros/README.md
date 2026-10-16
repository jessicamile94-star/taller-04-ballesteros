# Taller 4 — Ballesteros

Preparación inicial individual de Jessica Ballesteros para el Taller 4. El sistema que se observará, evaluará y protegerá es el solver multiagente ya construido para el Taller 3; por esa razón se sigue la ruta B del enunciado.

## Base preparada

| Recurso | Estado | Uso en el Taller 4 |
| --- | --- | --- |
| `golden_set.json` | Base copiada y revisada: 5 tareas, 47 comprobaciones | Punto de partida para el verificador determinístico `evaluar_solver.py` |
| `ataques.json` | Banco plantilla más 4 casos del dominio del solver | Medir falsos negativos y positivos de los guardrails |
| `etiquetas.csv` | Cabecera vacía | Registrar etiquetas humanas antes de consultar al juez |
| `Lab-04-LLMOps/` | Laboratorio recibido | Parte 0, Langfuse, juez, exportación y resumen de spans |

No hay resultados, etiquetas humanas, trazas, credenciales ni mediciones en esta carpeta. Se crearán durante el taller con un único modelo de agente y un único juez.

## Revisión contra el enunciado

La base cumple la cantidad mínima: el golden set contiene 47 comprobaciones. Antes de la primera medición debe completarse lo siguiente:

- Añadir al menos 4 comprobaciones con `"tipo": "juez"` y su `criterio`, vinculadas a entregables reales del solver.
- Incorporar una tarea cuyo enunciado contenga una instrucción escondida y otra que incluya PII sintética. Ambas deben permitir comprobar que la instrucción y los datos no llegan al entregable ni a la traza.
- Completar al menos 10 etiquetas humanas `aprueba` o `falla` en `etiquetas.csv`, incluyendo todas las comprobaciones de tipo `juez`, antes de abrir el resultado del juez.
- Configurar Langfuse y un identificador de sesión individual, por ejemplo `taller4-ballesteros-base`.
- Instrumentar el solver sin cambiar todavía sus resultados funcionales: spans de corrida, agente, LLM, herramienta y guardrail; para esta ruta también se requiere un span por rol del solver y el span del sandbox.

## Orden de trabajo

1. Reproducir la Parte 0 sin modificar los archivos de ejemplo.
2. Instrumentar el solver y exportar los spans de la versión base.
3. Completar el golden set y ejecutar el verificador automático y el juez sobre las mismas corridas.
4. Implementar y medir los guardrails, incluidos los casos propios de `ataques.json`.
5. Analizar los tres peores casos, versionar una mejora y medirla con el mismo golden set, evaluador y modelo.
6. Elaborar el informe con los archivos crudos producidos en cada paso.

## Comandos de referencia

Ejecutar desde la raíz del repositorio. Los comandos de abajo son referencias para cuando el entorno esté configurado; no producen resultados iniciales por sí solos.

```bash
python3 Lab-04-LLMOps/guion.py ceros
python3 Lab-04-LLMOps/guion.py guardrails
python3 Lab-04-LLMOps/guion.py traza

python3 Lab-03-Agentes/solver-v2/evaluar_solver.py \
  --golden taller4-ballesteros/golden_set.json \
  --solo-evaluar <carpeta-de-corridas> \
  --salida taller4-ballesteros/resultados_solver.csv

python3 Lab-04-LLMOps/juez_solver.py \
  --golden taller4-ballesteros/golden_set.json \
  --corridas <carpeta-de-corridas> \
  --resultados taller4-ballesteros/resultados_solver.csv \
  --etiquetas taller4-ballesteros/etiquetas.csv \
  --juez <modelo> \
  --salida taller4-ballesteros/juez_solver.csv

python3 Lab-04-LLMOps/probar_guardrails.py \
  --entrada <modulo>:validar_entrada \
  --salida <modulo>:validar_salida \
  --casos taller4-ballesteros/ataques.json \
  --csv taller4-ballesteros/guardrails.csv
```

Los archivos `.env`, claves, datos personales y trazas sin revisar no se agregan al repositorio.
