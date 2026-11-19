"""Shared plotting style and helpers.

Call :func:`apply_style` once at the top of a notebook. Colours follow the Okabe–Ito palette
(safe for colour-blind readers); every helper also uses markers or labels, never colour alone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#: Okabe–Ito palette (colour-blind safe), with names used across the lesson.
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#6B7A90",
    "navy": "#0E1A2E",
}
CYCLE = [PALETTE[k] for k in ("blue", "orange", "green", "red", "purple", "sky", "grey")]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def apply_style() -> None:
    """Apply the lesson's Matplotlib style (light background, readable on a projector)."""
    mpl.rcParams.update(
        {
            "figure.figsize": (8, 4.5),
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.prop_cycle": mpl.cycler(color=CYCLE),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "font.size": 11,
        }
    )


def fmt_pt(value: float, decimals: int = 0) -> str:
    """Format a number the Portuguese way: ``26825.4`` → ``"26 825"``, ``0.84`` (2) → ``"0,84"``."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", " ").replace(".", ",").replace(" ", " ")


def pt_axis(ax: plt.Axes, axis: str = "y", decimals: int = 0, max_ticks: int = 6) -> plt.Axes:
    """Use Portuguese number formatting (and at most ``max_ticks`` ticks) on ``ax``'s x or y axis."""
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: fmt_pt(v, decimals)))
    target.set_major_locator(mpl.ticker.MaxNLocator(max_ticks))
    return ax


def label_bars(ax: plt.Axes, values, errors=None, decimals: int = 0, suffix: str = "") -> plt.Axes:
    """Write each value at the end of a horizontal bar, *after* its error bar (Portuguese format)."""
    errors = [0] * len(values) if errors is None else list(errors)
    span = max(v + e for v, e in zip(values, errors))
    for i, (v, e) in enumerate(zip(values, errors)):
        ax.text(v + e + span * 0.015, i, fmt_pt(v, decimals) + suffix, va="center", fontsize=10)
    ax.set_xlim(right=span * 1.18)
    return ax


def save_fig(fig: plt.Figure, path: str | Path) -> Path:
    """Save ``fig`` as PNG (folders are created if needed) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def plot_decision_boundary(
    predict_proba: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes | None = None,
    resolution: int = 200,
    title: str | None = None,
) -> plt.Axes:
    """Shade P(class = 1) over a 2-D grid and draw the 0.5 contour plus the points.

    ``predict_proba`` receives an ``(n, 2)`` array and returns ``(n,)`` probabilities.
    Classes are drawn with different markers *and* colours.
    """
    ax = ax or plt.gca()
    pad = 0.5
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x0, x1, resolution), np.linspace(y0, y1, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = np.asarray(predict_proba(grid)).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 11), cmap="RdBu", alpha=0.35, vmin=0, vmax=1)
    ax.contour(xx, yy, zz, levels=[0.5], colors=PALETTE["navy"], linewidths=1.5)
    for cls, marker, colour in ((0, "o", PALETTE["red"]), (1, "s", PALETTE["blue"])):
        mask = np.asarray(y) == cls
        ax.scatter(X[mask, 0], X[mask, 1], marker=marker, s=22, c=colour,
                   edgecolor="white", linewidth=0.5, label=f"classe {cls}")
    ax.set(xlim=(x0, x1), ylim=(y0, y1), xlabel="$x_1$", ylabel="$x_2$")
    ax.grid(False)
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    return ax


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, unit: str = "CAD",
                   title: str = "Resíduos") -> plt.Figure:
    """Two panels: predicted vs. real (with the y = x line) and residuals vs. predicted."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    resid = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].scatter(y_pred, y_true, s=16, alpha=0.7)
    axes[0].plot([lo, hi], [lo, hi], color=PALETTE["red"], ls="--", label="previsão perfeita")
    axes[0].set(xlabel=f"previsto ({unit})", ylabel=f"real ({unit})", title="Previsto vs. real")
    axes[0].legend()
    axes[1].scatter(y_pred, resid, s=16, alpha=0.7, color=PALETTE["orange"], marker="s")
    axes[1].axhline(0, color=PALETTE["navy"], lw=1)
    axes[1].set(xlabel=f"previsto ({unit})", ylabel=f"real − previsto ({unit})", title=title)
    for ax in axes:
        pt_axis(ax, "x", max_ticks=5)
        pt_axis(ax, "y", max_ticks=6)
    return fig


def plot_history(history: dict[str, Sequence[float]], ylabel: str = "loss",
                 title: str = "Curvas de treino", ax: plt.Axes | None = None) -> plt.Axes:
    """Plot one line per key in ``history`` (e.g. ``{"treino": [...], "validação": [...]}``)."""
    ax = ax or plt.gca()
    for (name, values), marker in zip(history.items(), MARKERS):
        values = list(values)
        every = max(1, len(values) // 12)
        ax.plot(range(1, len(values) + 1), values, marker=marker, markevery=every, ms=5, label=name)
    ax.set(xlabel="época", ylabel=ylabel, title=title)
    ax.legend()
    return ax
