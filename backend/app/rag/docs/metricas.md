# Métricas de evaluación

## R² (coeficiente de determinación)
Mide qué proporción de la variación del objetivo explica el modelo. 1.0 es perfecto, 0 equivale a predecir siempre el promedio y puede ser negativo si el modelo es peor que el promedio. Un R² de 0.80 significa que el modelo explica el 80% de la variabilidad.

## MAE (error absoluto medio)
Promedio de |real - predicho|. Está en las mismas unidades que el objetivo, por lo que es fácil de comunicar: "en promedio nos equivocamos por 12,000 pesos". Es robusto a valores atípicos.

## RMSE (raíz del error cuadrático medio)
Raíz de la media de los errores al cuadrado. Penaliza más los errores grandes que el MAE. Si RMSE es mucho mayor que MAE, hay algunos errores muy grandes.

## Accuracy (exactitud)
Porcentaje de predicciones correctas en clasificación. Puede engañar con clases desbalanceadas: si el 95% de clientes no abandona, predecir siempre "no abandona" da 95% de accuracy sin aprender nada.

## F1-score
Media armónica entre precisión (de los que predije positivos, cuántos lo eran) y recall (de los positivos reales, cuántos detecté). F1 macro promedia por clase y es más justo con clases desbalanceadas.
