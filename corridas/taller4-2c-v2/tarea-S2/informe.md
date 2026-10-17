# Informe — Comparación defendible de dos modelos sobre tiendas pareadas

**Datos:** `data/model_errors.csv` (16 tiendas; sintéticos, deterministas; archivo original sin modificar). **Salidas generadas:** `output/store_differences.csv` y `output/store_differences.png`. **Constantes:** semilla bootstrap 20260829; 10000 remuestras.

## Parte 1. Contrato de análisis previo

Este contrato se fijó antes de ejecutar cualquier prueba estadística, como exige el diseño.

**1. ¿Qué dos modelos se comparan?**
Se comparan dos modelos de regresión ya entrenados, A y B, que pronostican la demanda semanal de las mismas 16 tiendas. La comparación se realiza sobre sus errores absolutos, registrados en `data/model_errors.csv` en las columnas `error_a` y `error_b`, medidos en unidades vendidas.

**2. ¿Qué población o proceso objetivo propone para interpretar el ejercicio?**
Se propone interpretar las 16 tiendas como una muestra de un proceso de generación de errores de pronóstico de demanda semanal en tiendas pareadas, donde cada tienda produce un par de errores (A, B) y el interés inferencial recae en la diferencia media de errores Δ de ese proceso. Lo que impide afirmar que los datos representan realmente esa población es su procedencia: son sintéticos, deterministas y creados para la actividad; no provienen de un muestreo de tiendas reales. El proceso objetivo es por tanto una construcción didáctica y los resultados no constituyen evidencia sobre una empresa, mercado o población existente.

**3. ¿Cuál es la métrica principal y qué dirección representa un mejor resultado?**
La métrica principal es el error absoluto de pronóstico, en unidades vendidas; un valor menor representa un mejor pronóstico. Como la diferencia se define como error_a − error_b, una diferencia positiva favorece a B.

**4. ¿Cuál es la unidad de análisis?**
La tienda: cada una de las 16 tiendas aporta un par de errores y una diferencia d_i. El tamaño muestral del análisis pareado es n = 16.

**5. ¿Por qué las observaciones están emparejadas y cómo verificará los pares?**
Están emparejadas porque ambos modelos evalúan las mismas tiendas sobre los mismos datos: cada tienda contribuye con dos mediciones correlacionadas, y el emparejamiento concentra la comparación en la diferencia dentro de cada tienda, eliminando la variabilidad entre tiendas. Verificaré los pares mediante la clave `store_id`, con una unión uno a uno que exija identificadores únicos y no nulos; no con una simple comparación de longitudes de vectores.

**6. Defina el estimando y la diferencia por tienda.**
El estimando es Δ = E(d_i), con d_i = e_A(i) − e_B(i), la diferencia de errores absolutos de la tienda i. Una diferencia positiva significa que el modelo A tuvo mayor error absoluto que B en esa tienda, es decir, el par favorece a B; una diferencia negativa favorece a A.

**7. Escriba la hipótesis nula y la alternativa bilateral.**
H0: Δ = 0 (no hay diferencia media de errores entre los modelos). H1: Δ ≠ 0 (alternativa bilateral).

**8. ¿Cuál será el procedimiento principal y qué supuestos requiere?**
El procedimiento principal será una prueba t de una muestra sobre el vector de diferencias (equivalente a la prueba t pareada), bilateral, acompañada de un intervalo t del 95 % para la media. Requiere: emparejamiento correcto, diferencias independientes entre tiendas, diferencias aproximadamente normales y una métrica en escala de intervalo. Con n = 16, la normalidad de las diferencias es el supuesto más frágil; por eso el diseño incluye referencias alternativas (cambios de signo y bootstrap pareado).

**9. ¿Qué resultados reportará además del p-value?**
n, diferencia media, desviación estándar muestral y error estándar; grados de libertad, estadístico t e intervalo t del 95 %; conteo de tiendas que favorecen a cada modelo y diferencias de mayor magnitud en cada dirección; la referencia exacta por cambios de signo (total de configuraciones, conteo de extremas y su p-value) y el intervalo bootstrap percentil del 95 % con semilla y número de remuestras. La conclusión se redactará en unidades vendidas y no se reducirá a "significativo/no significativo".

**10. ¿Qué afirmaciones no permite realizar este diseño?**
No permite afirmar causalidad sobre por qué un modelo sería mejor; generalizar a tiendas, mercados o periodos reales (los datos son sintéticos); concluir que los modelos son equivalentes o que Δ = 0 a partir de un resultado no significativo; interpretar el p-value como la probabilidad de que H0 sea verdadera; ni hablar del desempeño de los modelos más allá de esta métrica y este conjunto de datos.

## Parte 2. Auditoría y evidencia por tienda

**11. Confirme que los 16 pares fueron validados mediante store_id.**
La auditoría confirmó que el CSV contiene exactamente las columnas `store_id`, `error_a` y `error_b`, con 16 filas; cada `store_id` es único y no nulo; los errores son numéricos, finitos y no negativos; y el archivo original no fue modificado. La tabla pareada conservó las tres columnas originales y añadió `difference` (= error_a − error_b) y `favors`, y los 16 pares fueron verificados mediante `store_id` (emparejamiento verificado: verdadero). Comprobar únicamente que ambas columnas tienen la misma longitud no es suficiente porque la igualdad de longitudes no garantiza correspondencia: los registros podrían estar en distinto orden, o una tienda podría estar duplicada o ausente en una de las dos columnas sin que se detecte. Sin una clave de unión, un desalineamiento de una sola fila mezclaría errores de tiendas distintas, contaminaría todas las diferencias d_i y, con ellas, la media, la desviación, la prueba t y todos los intervalos.

| store_id | error_a | error_b | difference | favors |
|:---------|--------:|--------:|-----------:|:------:|
| S01 | 12.4 | 10.6 | 1.8000 | B |
| S02 | 9.8 | 9.1 | 0.7000 | B |
| S03 | 15.1 | 13.9 | 1.2000 | B |
| S04 | 11.0 | 12.4 | -1.4000 | A |
| S05 | 8.7 | 7.2 | 1.5000 | B |
| S06 | 13.6 | 13.9 | -0.3000 | A |
| S07 | 10.2 | 9.3 | 0.9000 | B |
| S08 | 14.9 | 12.8 | 2.1000 | B |
| S09 | 7.5 | 9.3 | -1.8000 | A |
| S10 | 16.0 | 15.0 | 1.0000 | B |
| S11 | 9.3 | 9.8 | -0.5000 | A |
| S12 | 12.8 | 11.1 | 1.7000 | B |
| S13 | 11.7 | 12.9 | -1.2000 | A |
| S14 | 8.9 | 8.3 | 0.6000 | B |
| S15 | 13.2 | 11.9 | 1.3000 | B |
| S16 | 10.6 | 11.5 | -0.9000 | A |

**12. Tiendas favorecidas y diferencias de mayor magnitud.**
10 tiendas favorecen a B y 6 favorecen a A; no hay empates (0 diferencias exactamente nulas). La diferencia de mayor magnitud que favorece a B es la de S08, con difference = 2.1000; la de mayor magnitud que favorece a A es la de S09, con difference = -1.8000. La figura guardada en `output/store_differences.png` muestra las 16 observaciones, una por tienda, con el signo y la magnitud de cada diferencia y una línea de referencia visible en cero: en ella se aprecian 10 observaciones por encima de cero y 6 por debajo.

## Parte 3. Efecto e incertidumbre

**13. Descriptivos de las diferencias.**

| Cantidad | Valor |
|---|---:|
| n (tiendas) | 16 |
| Diferencia media (unidades vendidas) | 0.4187 |
| Desviación estándar muestral | 1.2534 |
| Error estándar (sd/√n) | 0.3133 |

El signo positivo indica que, en promedio, el modelo A registra 0.4187 unidades vendidas más de error absoluto que B por tienda; la magnitud es algo menor de medio punto por tienda-semana. Conviene distinguir dos cantidades: la dispersión entre tiendas es alta (las diferencias van de -1.8000 a 2.1000, con mediana 0.8000 y cuartiles -0.6000 y 1.3500; sd = 1.2534), mientras que el error estándar (0.3133, con la identidad EE = sd/√n verificada) mide la precisión con la que se estima la media, no la variabilidad tienda a tienda. Las tiendas individuales varían mucho más (sd = 1.2534) que la incertidumbre de la media (EE = 0.3133).

**14. Intervalo t del 95 %.**
El intervalo t bilateral del 95 % para la media pareada es [-0.2491, 1.0866]. Bajo el modelo utilizado (diferencias independientes y aproximadamente normales, varianza estimada con 15 grados de libertad), los valores del efecto compatibles con los datos van desde una ventaja de 0.2491 unidades para A hasta una ventaja de 1.0866 unidades para B, e incluyen el 0. Este intervalo no es una probabilidad posterior sobre Δ: su interpretación es de cobertura frecuencista —si el modelo es correcto y el procedimiento se repitiera muchas veces, el 95 % de los intervalos así construidos contendría al verdadero Δ—, y no que exista un 95 % de probabilidad de que Δ esté dentro de este intervalo concreto.

## Parte 4. Referencias nulas y sensibilidad

**15. Prueba t y significado del p-value.**
El estadístico t es 1.3364 con 15 grados de libertad, y el p-value bilateral es 0.2013. El p-value representa, condicional a H0 (Δ = 0), al diseño (16 tiendas pareadas, una diferencia por tienda) y a los supuestos del procedimiento (independencia y normalidad aproximada de las diferencias), la probabilidad de observar un estadístico t al menos tan extremo en magnitud como 1.3364. No es la probabilidad de que H0 sea verdadera, ni la probabilidad de que el resultado se deba al azar. Como 0.2013 > 0.05, no se rechaza H0; pero la conclusión no se reduce a eso: el intervalo [-0.2491, 1.0866] muestra que los datos son compatibles tanto con una pequeña ventaja de A como con una ventaja de B superior a una unidad, es decir, el estudio no aporta evidencia suficiente para discriminar entre los modelos en ninguna dirección.

**16. Referencia exacta por cambios de signo.**
La enumeración completa de las 2^16 = 65536 configuraciones de signos produjo 13372 configuraciones extremas (896 con magnitud igual a la observada y 12476 más extremas), de modo que p = 13372/65536 = 0.2040; bajo esta referencia nula, las medias posibles abarcan de -1.1813 a 1.1813. Esta prueba y la t no tienen supuestos idénticos: la t asume diferencias normales e independientes y obtiene la referencia nula de la distribución t teórica con varianza estimada (gl = 15); la prueba por cambios de signo solo asume que, bajo H0, el signo de cada diferencia es intercambiable (simetría respecto de 0) y construye la referencia por enumeración exacta, sin supuesto de normalidad. Los p-values resultan numéricamente cercanos (0.2013 frente a 0.2040; diferencia absoluta 0.0027, relativa 0.0134), pero los procedimientos no son intercambiables: cambian el supuesto distribucional y la referencia nula, y su coincidencia aquí es una propiedad de estos datos, no una garantía general.

| Procedimiento | Resultado |
|---|---|
| Prueba t bilateral (gl = 15) | t = 1.3364; p = 0.2013 |
| IC t del 95 % | [-0.2491, 1.0866] |
| Cambios de signo exactos | 13372/65536 configuraciones extremas; p = 0.2040 |
| Bootstrap percentil 95 % (semilla 20260829; 10000 remuestras) | [-0.1938, 1.0062] |

**17. Bootstrap pareado.**
Se usó `np.random.default_rng(20260829)` con 10000 remuestras, remuestreando tiendas completas: los índices se sortean sobre el vector de diferencias, nunca `error_a` y `error_b` por separado. El intervalo percentil del 95 % es [-0.1938, 1.0062] (media bootstrap 0.4210, desviación bootstrap 0.3056); dos ejecuciones con la misma semilla produjeron intervalos idénticos, confirmando la reproducibilidad. Se remuestrean tiendas completas porque la tienda es la unidad de muestreo y el emparejamiento vive dentro de ella: remuestrear los errores de A y B por separado rompería la correspondencia por tienda y destruiría la dependencia entre los dos errores de la misma tienda, invalidando la incertidumbre de la media pareada. El bootstrap no puede corregir automáticamente los problemas de la muestra original: si las 16 tiendas están sesgadas o no representan el proceso de interés, las remuestras heredan ese sesgo; tampoco repara un emparejamiento incorrecto ni errores de medición, y con n = 16 su precisión sigue limitada por la información disponible.

## Parte 5. Informe ejecutivo

**18. Informe ejecutivo (máximo 150 palabras).**

Unidad: 16 tiendas (S01–S16). Procedencia: data/model_errors.csv, datos sintéticos y deterministas; no representan tiendas reales. Métrica: error absoluto de pronóstico en unidades vendidas; menor es mejor. Emparejamiento: cada tienda fue evaluada con ambos modelos; pares verificados por store_id. Efecto: diferencia media error_a − error_b = 0.4187 unidades (10 tiendas favorecen a B, 6 a A, sin empates). Incertidumbre: EE 0.3133; IC t del 95 % [-0.2491, 1.0866]. Procedimiento: prueba t pareada bilateral (t = 1.3364, gl = 15), p = 0.2013. Sensibilidad: cambios de signo exactos, p = 0.2040 (13372/65536); bootstrap pareado (10000 remuestras, semilla 20260829), IC 95 % [-0.1938, 1.0062]. Limitaciones: muestra pequeña (n = 16) con potencia limitada; datos de un solo periodo sin aleatorización. No puede concluirse que B reduzca el error frente a A: p no significativo y todos los intervalos contienen 0.
