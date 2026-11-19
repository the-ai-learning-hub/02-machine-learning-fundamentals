"""A tiny neural network written in pure NumPy, to open the black box.

Notation (same as the slides):

* ``X`` has shape ``(n, d)`` — one example per row.
* Hidden layer: ``z1 = X @ W1 + b1``, ``a1 = tanh(z1)``.
* Output layer: ``z2 = a1 @ W2 + b2``, ``ŷ = sigmoid(z2)``.
* Loss: binary cross-entropy ``L = −mean(y·log ŷ + (1−y)·log(1−ŷ))``.

Backpropagation is just the chain rule applied from the loss back to every weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-12


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function, numerically stable for large |z|."""
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


def step(z: np.ndarray) -> np.ndarray:
    """Heaviside step used by the classic perceptron."""
    return (z >= 0).astype(float)


def bce_loss(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Binary cross-entropy averaged over the batch."""
    y_hat = np.clip(y_hat, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))


def train_perceptron(X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 20,
                     seed: int = 0) -> tuple[np.ndarray, float, list[int]]:
    """Rosenblatt's perceptron rule. Returns weights, bias and the errors per epoch.

    It converges only if the classes are linearly separable (AND, OR — but not XOR).
    """
    rng = np.random.default_rng(seed)
    w, b = rng.normal(0, 0.1, X.shape[1]), 0.0
    errors = []
    for _ in range(epochs):
        wrong = 0
        for xi, yi in zip(X, y):
            update = lr * (yi - step(xi @ w + b))
            w, b = w + update * xi, b + update
            wrong += int(update != 0)
        errors.append(wrong)
    return w, b, errors


@dataclass
class MLP:
    """One hidden layer (tanh) + sigmoid output, trained with full-batch gradient descent."""

    n_in: int
    n_hidden: int
    seed: int = 0
    params: dict[str, np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # Xavier/Glorot-style initialisation keeps tanh away from saturation at the start.
        self.params = {
            "W1": rng.normal(0, np.sqrt(1 / self.n_in), (self.n_in, self.n_hidden)),
            "b1": np.zeros(self.n_hidden),
            "W2": rng.normal(0, np.sqrt(1 / self.n_hidden), (self.n_hidden, 1)),
            "b2": np.zeros(1),
        }

    # ------------------------------------------------------------------ forward / backward
    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return ``ŷ`` with shape ``(n,)`` and a cache of intermediate values for backprop."""
        p = self.params
        z1 = X @ p["W1"] + p["b1"]
        a1 = np.tanh(z1)
        z2 = a1 @ p["W2"] + p["b2"]
        y_hat = sigmoid(z2).ravel()
        return y_hat, {"X": X, "z1": z1, "a1": a1, "y_hat": y_hat}

    def backward(self, y: np.ndarray, cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Gradients of the mean BCE loss with respect to every parameter (chain rule)."""
        n = len(y)
        X, a1, y_hat = cache["X"], cache["a1"], cache["y_hat"]
        dz2 = (y_hat - y).reshape(-1, 1) / n          # dL/dz2 for sigmoid + BCE
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * (1 - a1**2)                         # tanh'(z) = 1 − tanh(z)²
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return bce_loss(y, self.forward(X)[0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[0]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------------ training
    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.5, epochs: int = 2000,
            snapshots: tuple[int, ...] = ()) -> dict[str, list]:
        """Full-batch gradient descent. Returns ``{"loss": [...], "snapshots": {epoch: params}}``."""
        history: dict[str, list] = {"loss": [], "snapshots": []}
        for epoch in range(1, epochs + 1):
            y_hat, cache = self.forward(X)
            history["loss"].append(bce_loss(y, y_hat))
            grads = self.backward(y, cache)
            for name in self.params:
                self.params[name] -= lr * grads[name]
            if epoch in snapshots:
                history["snapshots"].append((epoch, {k: v.copy() for k, v in self.params.items()}))
        return history


def gradient_check(model: MLP, X: np.ndarray, y: np.ndarray, eps: float = 1e-5) -> float:
    """Largest relative difference between backprop and numerical (central-difference) gradients.

    Values below ~1e-6 mean the backward pass is correct.
    """
    _, cache = model.forward(X)
    analytic = model.backward(y, cache)
    worst = 0.0
    for name, param in model.params.items():
        numeric = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            original = param[idx]
            param[idx] = original + eps
            plus = model.loss(X, y)
            param[idx] = original - eps
            minus = model.loss(X, y)
            param[idx] = original
            numeric[idx] = (plus - minus) / (2 * eps)
        denom = np.maximum(np.abs(numeric) + np.abs(analytic[name]), 1e-12)
        worst = max(worst, float(np.max(np.abs(numeric - analytic[name]) / denom)))
    return worst
