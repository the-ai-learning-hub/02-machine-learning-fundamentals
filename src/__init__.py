"""Shared helpers for Lesson 02 · Machine Learning: fundamentos (The AI Learning Hub).

Modules
-------
paths          Locate the repository root and the ``data/`` folder (local or Colab).
data           Load the lesson datasets (Windsor house prices, Titanic titanic3).
preprocessing  Feature engineering and scikit-learn preprocessing pipelines.
plotting       Shared Matplotlib style, palette and plotting helpers.
nn_numpy       A tiny multilayer perceptron written in NumPy (forward, backprop, gradient check).
rl_utils       A gridworld environment and tabular Q-learning, plus RL plots.

Importing this package has no side effects (no plotting style is applied, nothing is downloaded).
"""

__all__ = ["paths", "data", "preprocessing", "plotting", "nn_numpy", "rl_utils"]
