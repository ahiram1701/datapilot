# Validación de modelos

## Train/test split
Se separan los datos en entrenamiento y prueba. El modelo solo ve el conjunto de entrenamiento; el de prueba simula datos nuevos. Evaluar sobre los mismos datos de entrenamiento sobreestima el desempeño.

## Overfitting (sobreajuste)
Ocurre cuando el modelo memoriza el ruido del entrenamiento. Señal típica: métrica de train muy superior a la de test (por ejemplo R² train 0.99 y test 0.70). Se combate con modelos más simples, regularización, más datos o limitando la profundidad de los árboles.

## Underfitting (subajuste)
El modelo es demasiado simple para capturar el patrón: métricas bajas tanto en train como en test. Se combate con modelos más expresivos o mejores variables.

## Validación cruzada k-fold
Divide los datos en k partes; entrena k veces dejando una parte distinta como prueba y promedia. Da una estimación más estable que un único split. La desviación estándar entre folds indica cuán sensible es el modelo a qué datos le tocan.
