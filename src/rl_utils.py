"""Gridworld + tabular Q-learning, written by hand (no RL library).

The default world is the same one used in the live demo of the slides:

    . . . . . G      S start   G goal (+20, ends)   L lava (−20, ends)
    . # # . L .      # wall (bumping costs a step)   C coin (+5, optional, never disappears)
    . . . . L .      every other move costs −1
    . # . # . .
    S . . . . .

Actions: 0 = ↑, 1 = →, 2 = ↓, 3 = ←.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_LAYOUT = (
    ".....G",
    ".##.L.",
    "....L.",
    ".#.#..",
    "S.....",
)
#: Same world with a coin on the shortest path (used for the reward-hacking experiment).
COIN_LAYOUT = (
    ".....G",
    ".##.L.",
    "....L.",
    ".#.#..",
    "S.C...",
)
ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}


@dataclass
class GridWorld:
    """A small deterministic (or slippery) gridworld.

    ``slip`` is the probability that the chosen action is replaced by a random one.
    """

    layout: tuple[str, ...] = DEFAULT_LAYOUT
    step_reward: float = -1.0
    goal_reward: float = 20.0
    lava_reward: float = -20.0
    coin_reward: float = 5.0
    slip: float = 0.0
    max_steps: int = 100
    seed: int | None = None
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.grid = np.array([list(row) for row in self.layout])
        self.n_rows, self.n_cols = self.grid.shape
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = len(ACTIONS)
        self.start = self._find("S")
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    def _find(self, char: str) -> int:
        r, c = np.argwhere(self.grid == char)[0]
        return int(r * self.n_cols + c)

    def cell(self, state: int) -> str:
        return str(self.grid[state // self.n_cols, state % self.n_cols])

    def reset(self) -> int:
        self.state, self.t = self.start, 0
        return self.state

    def step(self, action: int) -> tuple[int, float, bool, str]:
        """Apply ``action``. Returns ``(next_state, reward, done, outcome)``.

        ``outcome`` is ``"goal"``, ``"lava"``, ``"timeout"`` or ``""`` (episode continues).
        """
        if self.slip and self.rng.random() < self.slip:
            action = int(self.rng.integers(self.n_actions))
        r, c = divmod(self.state, self.n_cols)
        dr, dc = ACTIONS[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols and self.grid[nr, nc] != "#":
            self.state = nr * self.n_cols + nc
        self.t += 1
        kind = self.cell(self.state)
        if kind == "G":
            return self.state, self.goal_reward, True, "goal"
        if kind == "L":
            return self.state, self.lava_reward, True, "lava"
        reward = self.coin_reward if kind == "C" else self.step_reward
        if self.t >= self.max_steps:
            return self.state, reward, True, "timeout"
        return self.state, reward, False, ""


def epsilon_schedule(episode: int, episodes: int, eps_start: float = 1.0,
                     eps_end: float = 0.05, decay_frac: float = 0.6) -> float:
    """Linear decay from ``eps_start`` to ``eps_end`` over the first ``decay_frac`` of training."""
    frac = min(1.0, episode / max(1, decay_frac * episodes))
    return eps_start + frac * (eps_end - eps_start)


def q_learning(env: GridWorld, episodes: int = 400, alpha: float = 0.5, gamma: float = 0.95,
               eps_start: float = 1.0, eps_end: float = 0.05, decay_frac: float = 0.6,
               seed: int = 0, snapshot_at: tuple[int, ...] = ()) -> tuple[np.ndarray, dict]:
    """Tabular Q-learning with an ε-greedy behaviour policy.

    Update: ``Q[s,a] ← Q[s,a] + α · (r + γ · max_a' Q[s',a'] − Q[s,a])`` (no bootstrap at the end).
    Returns the Q-table and a history with return, length, outcome and ε per episode.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    hist: dict[str, list] = {"return": [], "length": [], "outcome": [], "epsilon": [], "snapshots": {}}
    for ep in range(episodes):
        eps = epsilon_schedule(ep, episodes, eps_start, eps_end, decay_frac)
        s, done, total, outcome = env.reset(), False, 0.0, ""
        while not done:
            if rng.random() < eps:
                a = int(rng.integers(env.n_actions))
            else:  # break ties at random so an all-zero row does not always pick ↑
                a = int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
            s2, r, done, outcome = env.step(a)
            terminal = outcome in ("goal", "lava")
            target = r if terminal else r + gamma * Q[s2].max()
            Q[s, a] += alpha * (target - Q[s, a])
            s, total = s2, total + r
        hist["return"].append(total)
        hist["length"].append(env.t)
        hist["outcome"].append(outcome)
        hist["epsilon"].append(eps)
        if ep + 1 in snapshot_at:
            hist["snapshots"][ep + 1] = Q.copy()
    return Q, hist


def greedy_action(Q: np.ndarray, state: int) -> int:
    """Best action in ``state`` (lowest index on ties, so evaluation is deterministic)."""
    return int(np.argmax(Q[state]))


def evaluate(env: GridWorld, Q: np.ndarray, episodes: int = 100) -> dict[str, float]:
    """Run the greedy policy (ε = 0, no learning). Returns mean return, success rate, mean length."""
    returns, successes, lengths = [], 0, []
    for _ in range(episodes):
        s, done, total, outcome = env.reset(), False, 0.0, ""
        while not done:
            s, r, done, outcome = env.step(greedy_action(Q, s))
            total += r
        returns.append(total)
        successes += outcome == "goal"
        lengths.append(env.t)
    return {"mean_return": float(np.mean(returns)), "success_rate": successes / episodes,
            "mean_length": float(np.mean(lengths))}


def greedy_path(env: GridWorld, Q: np.ndarray, max_steps: int = 30) -> list[int]:
    """States visited by the greedy policy from the start (stops at the end or after ``max_steps``)."""
    s, path = env.reset(), [env.start]
    for _ in range(max_steps):
        s, _, done, _ = env.step(greedy_action(Q, s))
        path.append(s)
        if done:
            break
    return path


def moving_average(values, window: int = 20) -> np.ndarray:
    """Trailing moving average (the first values use the points available so far)."""
    values = np.asarray(values, dtype=float)
    cums = np.cumsum(np.insert(values, 0, 0.0))
    out = np.empty_like(values)
    for i in range(len(values)):
        lo = max(0, i + 1 - window)
        out[i] = (cums[i + 1] - cums[lo]) / (i + 1 - lo)
    return out


# ---------------------------------------------------------------------------- plots
_CELL_STYLE = {"#": ("#3b4a63", "parede"), "L": ("#D55E00", "lava"), "G": ("#009E73", "objetivo"),
               "S": (None, "início"), "C": ("#E69F00", "moeda")}


def plot_q_values(env: GridWorld, Q: np.ndarray, ax: plt.Axes | None = None,
                  title: str = "max Q(s, a) e política greedy", path: list[int] | None = None) -> plt.Axes:
    """Heatmap of ``max_a Q(s, a)`` with the greedy arrow in each free cell."""
    ax = ax or plt.gca()
    values = Q.max(axis=1).reshape(env.n_rows, env.n_cols).astype(float)
    mask = np.isin(env.grid, ["#", "L", "G"])
    shown = np.ma.masked_where(mask, values)
    im = ax.imshow(shown, cmap="viridis")
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            kind, s = env.grid[r, c], r * env.n_cols + c
            colour, label = _CELL_STYLE.get(kind, (None, ""))
            if colour and kind != "S":
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=colour))
                ax.text(c, r, {"#": "", "L": "L", "G": "G", "C": "C"}[kind], ha="center", va="center",
                        color="white", fontsize=14, fontweight="bold")
            if kind in (".", "S", "C"):
                visited = np.any(Q[s] != 0)
                arrow = ARROWS[greedy_action(Q, s)] if visited else "·"
                ax.text(c, r - 0.12, arrow, ha="center", va="center", fontsize=17, color="white",
                        fontweight="bold")
                ax.text(c, r + 0.3, f"{values[r, c]:.1f}", ha="center", va="center", fontsize=8,
                        color="white")
            if kind == "S":
                ax.text(c - 0.38, r - 0.36, "S", fontsize=9, color="white", fontweight="bold")
    if path:
        rows, cols = np.divmod(np.array(path), env.n_cols)
        ax.plot(cols, rows, color="white", lw=2, alpha=0.6, ls="--")
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.035, label="max Q")
    return ax
