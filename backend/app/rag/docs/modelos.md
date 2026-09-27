# Modelos de machine learning

## Regresión lineal
Modela el objetivo como suma ponderada de las variables: y = w1·x1 + ... + b. Es interpretable: cada coeficiente indica cuánto cambia y cuando esa variable aumenta una unidad (o una desviación estándar si los datos están estandarizados). Asume relaciones lineales.

## Regresión logística
Modelo de clasificación: aplica la función sigmoide a una combinación lineal para obtener una probabilidad entre 0 y 1. Se entrena minimizando la entropía cruzada.

## Árbol de decisión
Divide los datos con preguntas del tipo "¿área > 120?" eligiendo en cada paso la división que más reduce la impureza (Gini o varianza). Es interpretable y captura relaciones no lineales, pero un árbol profundo sobreajusta con facilidad.

## Random Forest
Conjunto de muchos árboles entrenados con muestras aleatorias de filas y variables; promedia sus predicciones. Reduce la varianza de un árbol individual y suele funcionar muy bien sin mucho ajuste. Su importancia de variables indica cuánto contribuye cada una a reducir el error.

## Importancia de variables
Indica qué variables usa más el modelo, no causalidad. Variables correlacionadas entre sí pueden repartirse la importancia.
