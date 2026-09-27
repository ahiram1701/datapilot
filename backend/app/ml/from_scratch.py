"""Regresión lineal entrenada con descenso de gradiente, solo con NumPy.

Modelo:   ŷ = X·w + b
Pérdida:  MSE = (1/n) Σ (ŷ - y)²
Gradientes:
    ∂MSE/∂w = (2/n) · Xᵀ(ŷ - y)
    ∂MSE/∂b = (2/n) · Σ(ŷ - y)
"""
from __future__ import annotations

import numpy as np


class LinearRegressionGD:
    def __init__(self, learning_rate: float = 0.05, epochs: int = 500):
        self.lr = learning_rate
        self.epochs = epochs
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.loss_history: list[float] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.loss_history = []
        for _ in range(self.epochs):
            error = self.predict(X) - y
            self.loss_history.append(float(np.mean(error ** 2)))
            grad_w = (2 / n) * (X.T @ error)  # ∂MSE/∂w
            grad_b = (2 / n) * np.sum(error)  # ∂MSE/∂b
            self.w -= self.lr * grad_w        # paso en dirección opuesta al gradiente
            self.b -= self.lr * grad_b
        return self
