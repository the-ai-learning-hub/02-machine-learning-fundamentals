"""Minimal tests for the shared code. Run from the repository root: ``uv run pytest -q``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import best_threshold, confusion_at
from src.nn_numpy import MLP, gradient_check, train_perceptron
from src.preprocessing import GroupMedianImputer, extract_title
from src.rl_utils import COIN_LAYOUT, GridWorld, evaluate, moving_average, q_learning

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_Y = np.array([0, 1, 1, 0], dtype=float)


@pytest.mark.parametrize(
    ("name", "title"),
    [("Allen, Miss. Elisabeth Walton", "Miss"), ("Rothes, the Countess. of", "Rare"),
     ("Uruchurtu, Don. Manuel E", "Rare"), ("Aubart, Mme. Leontine Pauline", "Mrs")],
)
def test_extract_title(name: str, title: str) -> None:
    assert extract_title(name) == title


def test_group_median_imputer_uses_training_data_only() -> None:
    train = pd.DataFrame({"title": ["Mr", "Mr", "Miss"], "age": [30.0, 40.0, 20.0]})
    test = pd.DataFrame({"title": ["Mr", "Miss", "Rare"], "age": [np.nan, np.nan, np.nan]})
    out = GroupMedianImputer().fit(train).transform(test)
    assert out["age"].tolist() == [35.0, 20.0, 30.0]  # unseen group → overall training median


def test_confusion_and_threshold() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    c = confusion_at(y, p, 0.5)
    assert (c["TP"], c["FP"], c["FN"], c["TN"]) == (1, 1, 1, 1)
    assert best_threshold(y, p, cost_fp=10, cost_fn=1) > 0.6  # expensive FP → stricter threshold


def test_perceptron_fails_on_xor_only() -> None:
    assert train_perceptron(XOR_X, np.array([0, 0, 0, 1.0]))[2][-1] == 0  # AND
    assert train_perceptron(XOR_X, XOR_Y)[2][-1] > 0                    # XOR


def test_mlp_backprop_matches_numerical_gradient_and_learns_xor() -> None:
    model = MLP(2, 4, seed=0)
    assert gradient_check(model, XOR_X, XOR_Y) < 1e-6
    model.fit(XOR_X, XOR_Y, lr=0.5, epochs=2000)
    assert model.predict(XOR_X).tolist() == [0, 1, 1, 0]


def test_gridworld_rules() -> None:
    env = GridWorld()
    assert env.reset() == env.start
    s, r, done, _ = env.step(3)  # ← into the outer wall: stay, pay one step
    assert (s, r, done) == (env.start, -1.0, False)


def test_q_learning_solves_the_default_world_and_hacks_the_coin_world() -> None:
    Q, _ = q_learning(GridWorld(), episodes=400, seed=0)
    assert evaluate(GridWorld(), Q, episodes=1) == {"mean_return": 12.0, "success_rate": 1.0, "mean_length": 9.0}
    Q, _ = q_learning(GridWorld(layout=COIN_LAYOUT), episodes=400, seed=0)
    assert evaluate(GridWorld(layout=COIN_LAYOUT), Q, episodes=1)["success_rate"] == 0.0


def test_moving_average() -> None:
    assert moving_average([0, 2, 4, 6], window=2).tolist() == [0.0, 1.0, 3.0, 5.0]
