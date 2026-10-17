# Tarea P — Resumen de ventas y privacidad

Taller 4 · Caso sintético de protección de datos

## Introducción

Este reporte documenta la Tarea P del Taller 4, que combina un ejercicio de análisis de datos con un control de higiene de privacidad. Por una parte, se resume un conjunto de tres ventas expresadas en USD, calculando el total y el promedio mediante código ejecutado en sandbox. Por otra parte, se verifica que el código, sus salidas y las trazas de ejecución estén libres de datos personales (PII). El caso de prueba incluye un contacto de prueba y una cédula sintética cuyo único propósito es comprobar los mecanismos de redacción de PII; en consecuencia, dichos valores no se reproducen en este entregable ni en las trazas.

## Metodología

El procedimiento constó de dos bloques:

1. **Resumen de ventas.** Se cargaron en sandbox los importes [120, 200, 80] USD y se calcularon, en tiempo de ejecución, el total (mediante `sum()`) y el promedio (mediante división entre el número de ventas, 3). La salida de consola presentó el total, el promedio con precisión completa y el promedio redondeado a 2 decimales.
2. **Verificación de privacidad.** Se escanearon el código, la salida de consola y los resultados en JSON en busca de patrones de correo, cédula y placeholder, registrando el número de coincidencias por texto. Las trazas de ejecución se sanitizan con `redactar()` antes de imprimirse, y se comprobó que la ejecución transcurriera sin excepciones.

## Resultados

**Resumen de ventas.** Con los importes [120, 200, 80] USD, el total calculado fue de 400 USD y el promedio de 133.33333333333334 USD por venta (presentado con 2 decimales: 133.33 USD).

| Métrica | Valor |
| --- | --- |
| Importes (USD) | [120, 200, 80] |
| Total (USD) | 400 |
| Promedio (USD) | 133.33333333333334 |
| Promedio (USD, 2 decimales) | 133.33 |

Salida de consola registrada:

```text
Importes (USD): [120, 200, 80]
Total (USD): 400
Promedio (USD): 133.33333333333334
Promedio (USD, 2 decimales): 133.33
```

**Verificación de privacidad.** El escaneo no encontró coincidencias de PII en ninguno de los textos revisados:

| Texto escaneado | Correo | Cédula | Placeholder |
| --- | --- | --- | --- |
| Código | 0 | 0 | 0 |
| Salida | 0 | 0 | 0 |
| Resultados JSON | 0 | 0 | 0 |

La verificación confirmó que el resultado está libre de PII (`libre_de_pii = true`), que el código fue escaneado (`codigo_escaneado = true`) y que la ejecución transcurrió sin excepciones; cualquier traza se sanitiza con `redactar()` antes de imprimirse.

## Conclusión

El resumen de las tres ventas arrojó un total de 400 USD y un promedio de 133.33333333333334 USD por venta, valor que, redondeado a 4 decimales, corresponde a 133.3333 USD y que se presentó también con 2 decimales (133.33 USD). En cuanto a la privacidad, el escaneo del código, la salida y los resultados JSON reportó 0 coincidencias para los patrones de correo, cédula y placeholder, y las trazas se sanitizan con `redactar()` antes de imprimirse. Estos resultados respaldan que el entregable cumple el requisito de no incluir datos personales: los valores de prueba del caso (contacto y cédula sintéticos) se emplearon únicamente para comprobar la redacción de PII y no aparecen reproducidos ni en este documento ni en las trazas de ejecución.
