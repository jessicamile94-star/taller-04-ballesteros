## Introducción

Este reporte presenta el análisis estadístico descriptivo solicitado en la Tarea E del Taller 4. El objeto de estudio es un conjunto de cinco observaciones — [12, 15, 18, 20, 25] — expresadas en unidades de observación (u.o.). El objetivo es caracterizar el conjunto mediante tres medidas fundamentales: la media, la mediana y la desviación estándar muestral, explicando qué muestra cada una. Todos los cálculos fueron ejecutados y verificados en sandbox; el reporte se limita al contenido académico de la tarea estadística.

## Metodología

Se cargaron las cinco observaciones en un entorno de sandbox (Python) y se calcularon las tres medidas solicitadas. Para la desviación estándar se empleó el estimador muestral con denominador n − 1 (es decir, con n = 5 se usa 4 en el denominador), que es el estimador insesgado de la varianza poblacional a partir de una muestra. El código ejecutado fue:

```python
import statistics

datos = [12, 15, 18, 20, 25]
media = statistics.mean(datos)      # 18.0
mediana = statistics.median(datos)  # 18.0
s = statistics.stdev(datos)         # 4.949747468305833  (denominador n-1)
```

Las tres cifras se reportan en unidades de observación (u.o.), la unidad natural de los datos analizados. Como medida complementaria derivada de la ejecución se obtuvo también la varianza muestral, s² = 24.5 u.o.².

## Resultados

| Medida | Valor | Unidades |
|---|---|---|
| Media | 18.0 | u.o. |
| Mediana | 18.0 | u.o. |
| Desviación estándar muestral (n − 1) | 4.9497 | u.o. |
| Varianza muestral (complementaria) | 24.5 | u.o.² |

**Media = 18.0 u.o.** La media es el promedio aritmético: la suma de las 5 observaciones dividida entre 5. Representa el centro de gravedad del conjunto; todas las observaciones se desvían de ella en promedio 0, ya que las desviaciones respecto a la media suman cero.

**Mediana = 18.0 u.o.** La mediana es el valor central de las 5 observaciones ordenadas (12, 15, 18, 20, 25): la mitad de los datos está por debajo y la mitad por encima. Es una medida robusta ante valores extremos.

**Desviación estándar muestral = 4.9497 u.o.** La desviación estándar muestral (con denominador n − 1) mide la dispersión típica de las observaciones alrededor de la media: en promedio, los datos se alejan de la media aproximadamente 4.9497 u.o. El uso de n − 1 en el denominador proporciona un estimador insesgado de la varianza poblacional a partir de la muestra; su cuadrado es la varianza muestral, s² = 24.5 u.o.².

## Conclusión

El conjunto de cinco observaciones tiene un centro claramente definido en 18.0 u.o.: la media y la mediana coinciden exactamente, lo que indica que el valor central no está desplazado por valores extremos y es consistente bajo ambas medidas de tendencia central. En torno a ese centro, la dispersión típica es de 4.9497 u.o. (varianza muestral de 24.5 u.o.²), calculada con el denominador n − 1 para obtener un estimador insesgado. En conjunto, los datos se describen como un grupo compacto y simétrico alrededor de 18.0 u.o., con una variabilidad de aproximadamente 4.9497 u.o. respecto a la media. Los resultados fueron comprobados mediante código ejecutado en sandbox y se reportan únicamente las cifras medidas.
