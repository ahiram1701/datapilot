# Deep learning

## Redes neuronales
Capas de neuronas que aplican una transformación lineal seguida de una activación no lineal (ReLU, sigmoide). Se entrenan con backpropagation: la regla de la cadena calcula el gradiente de la pérdida respecto a cada peso, y el descenso de gradiente los actualiza.

## CNN (redes convolucionales)
Usan filtros que se deslizan sobre la entrada para detectar patrones locales (bordes, texturas). Comparten pesos, por lo que son eficientes para imágenes.

## RNN (redes recurrentes)
Procesan secuencias manteniendo un estado oculto que se actualiza en cada paso. Sufren de gradientes que se desvanecen en secuencias largas; LSTM y GRU lo mitigan con compuertas.

## Transformers
Reemplazan la recurrencia por atención: cada token calcula, con productos punto entre vectores query y key, cuánto atender a cada otro token (softmax(QKᵀ/√d)·V). Permiten paralelizar y capturar dependencias largas. Son la base de los LLMs que usan los agentes de este proyecto.

## Cuándo usar deep learning en datos tabulares
En datos tabulares pequeños o medianos, los modelos basados en árboles (Random Forest, gradient boosting) suelen igualar o superar a las redes neuronales, con menos datos y ajuste.
