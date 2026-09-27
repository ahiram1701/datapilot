# Matemáticas para IA

## Descenso de gradiente
Algoritmo de optimización: se calcula el gradiente de la función de pérdida respecto a los parámetros y se da un paso en la dirección opuesta, w ← w − α·∇L. La tasa de aprendizaje α controla el tamaño del paso: muy grande diverge, muy pequeña converge lento.

## Gradiente del error cuadrático medio
Para ŷ = Xw + b y MSE = (1/n)Σ(ŷ−y)², el gradiente es ∂L/∂w = (2/n)·Xᵀ(ŷ−y) y ∂L/∂b = (2/n)·Σ(ŷ−y). Es álgebra lineal: un producto matriz-vector calcula todas las derivadas a la vez.

## Estandarización
Restar la media y dividir por la desviación estándar en cada variable. Hace que todas tengan escala comparable, lo que acelera la convergencia del descenso de gradiente y hace comparables los coeficientes.

## Correlación de Pearson
Mide relación lineal entre dos variables, de −1 a 1. Cercana a 0 no significa independencia: puede haber relaciones no lineales. Correlación no implica causalidad.
