## Introducción

Este reporte corresponde a la Tarea E del Taller 4, centrada en el análisis estadístico descriptivo de un conjunto pequeño de observaciones: [12, 15, 18, 20, 25]. El objetivo es caracterizar el conjunto mediante tres medidas fundamentales —media aritmética, mediana y desviación estándar muestral—, reportando cada cifra con sus unidades de observación y una interpretación de qué información aporta cada una. Todos los valores fueron obtenidos ejecutando código en sandbox sobre los datos indicados.

## Metodología

Se cargaron las 5 observaciones (n = 5) en un entorno de ejecución (sandbox) y se calcularon las tres medidas solicitadas:

- **Media aritmética**: suma de las observaciones dividida entre n.
- **Mediana**: valor central de las observaciones ordenadas.
- **Desviación estándar muestral**: raíz cuadrada de la suma de cuadrados de las desviaciones respecto de la media dividida entre n − 1, es decir, con denominador n − 1 = 4:

  s = sqrt( sum((x_i − media)²) / (n − 1) )

Las cifras se reportan en unidades de observación (u). Los cálculos intermedios verificados en sandbox fueron: suma de las observaciones = 90.0 u; suma de los cuadrados de las desviaciones respecto de la media = 98.0 u²; varianza muestral = 98.0 / 4 = 24.5 u².

## Resultados

| Medida | Valor | Unidades |
|---|---|---|
| Media aritmética | 18.0 | unidades de observación (u) |
| Mediana | 18.0 | unidades de observación (u) |
| Desviación estándar muestral | 4.9497 | unidades de observación (u) |

**Interpretación de cada medida:**

- **Media = 18.0 u.** Es el centro de gravedad del conjunto: el valor que resultaría si las 5 observaciones se repartieran por igual. Al ser un promedio, es sensible a valores extremos.
- **Mediana = 18.0 u.** Con las observaciones ordenadas (12, 15, 18, 20, 25) es el valor central, dejando 2 observaciones por debajo y 2 por encima. Es una medida robusta: no cambia ante valores extremos.
- **Desviación estándar muestral ≈ 4.9497 u.** Mide la dispersión típica de las observaciones respecto de la media (18.0 u). El uso del denominador n − 1 = 4 corrige el sesgo al estimar la dispersión poblacional a partir de una muestra pequeña.

En este conjunto, media y mediana coinciden en 18.0 u, lo que indica una distribución razonablemente simétrica alrededor del centro, con una dispersión típica de aproximadamente 4.9497 u.

## Conclusión

El análisis descriptivo de las observaciones [12, 15, 18, 20, 25], verificado mediante código ejecutado en sandbox, arroja una media de 18.0 u, una mediana de 18.0 u y una desviación estándar muestral de 4.9497 u. La coincidencia entre media y mediana sugiere que el centro del conjunto está bien representado por cualquiera de las dos medidas, mientras que la desviación estándar cuantifica una dispersión moderada alrededor de ese centro. La elección del denominador n − 1 garantiza un estimador no sesgado de la varianza poblacional, apropiado para muestras pequeñas como la presente.
