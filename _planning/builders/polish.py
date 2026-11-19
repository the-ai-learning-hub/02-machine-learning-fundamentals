"""Ronda 2b: números PT-PT nos gráficos, rótulos em PT, letra >= 10 pt, layout constrained (executar uma vez)."""
import re
import shutil
from pathlib import Path

HELPER = r'''

# 🖌️ Auxiliar de gráficos: números à portuguesa (vírgula decimal, espaço nos milhares) e letra ≥ 10 pt
import re
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from src.plotting import fmt_pt


def _decimals(axis):
    """Casas decimais necessárias para os valores dos ticks (0 a 3; em escala log, as que o menor tick pede)."""
    locs = np.asarray(axis.get_majorticklocs(), dtype=float)
    if axis.get_scale() == "log":
        locs = np.abs(locs[locs != 0])
        return int(min(4, max(0, np.ceil(-np.log10(locs.min()))))) if locs.size else 0
    for d in range(4):
        if np.allclose(locs, np.round(locs, d), atol=1e-9):
            return d
    return 3


def finish(fig, path=None):
    """Formata os números à portuguesa, garante letra >= 10 pt e (opcionalmente) guarda a figura."""
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if type(axis.get_major_formatter()).__name__ in ("StrCategoryFormatter", "FixedFormatter", "NullFormatter"):
                continue
            axis.set_major_formatter(FuncFormatter(lambda v, _, a=axis: fmt_pt(v, _decimals(a))))
    for text in fig.findobj(mpl.text.Text):
        s = text.get_text()
        if re.search(r"\d\.\d", s):
            text.set_text(re.sub(r"(\d)\.(\d)", r"\1,\2", s))
        if text.get_fontsize() < 10:
            text.set_fontsize(10)
    if path is not None:
        save_fig(fig, path)


_show = plt.show


def _show_pt(*args, **kwargs):
    for num in plt.get_fignums():
        finish(plt.figure(num))
    return _show(*args, **kwargs)


plt.show = _show_pt
'''

here = Path(__file__).parent
SUBPLOTS = re.compile(r"plt\.subplots\(([^)]*figsize=\([^)]*\)[^)]*)\)")


def common(s):
    s = s.replace("save_fig(fig, ", "finish(fig, ")
    # tight_layout -> constrained
    s = s.replace("fig.tight_layout(); ", "").replace("fig.tight_layout()\n", "\n")
    s = s.replace("fig.tight_layout()", "pass")
    lines = []
    for line in s.split("\n"):
        if not line.lstrip().startswith('"') and "layout=" not in line:
            line = SUBPLOTS.sub(lambda m: f'plt.subplots({m.group(1)}, layout="constrained")', line)
        lines.append(line)
    s = "\n".join(lines)
    # fontes pequenas
    s = re.sub(r"fontsize=(?:[0-9](?:\.[0-9])?)\b", "fontsize=10", s)
    # auxiliar logo depois de FIG = ...
    anchor = 'FIG = ROOT / "assets" / "figures"'
    assert anchor in s
    s = s.replace(anchor, anchor + HELPER, 1)
    return s


def rep(s, a, b, count=1):
    if a not in s:
        print("NÃO ENCONTRADO:", a[:80])
        return s
    return s.replace(a, b) if count == 0 else s.replace(a, b, count)


# ----------------------------------------------------------------------------- 04
p = here / "build_04.py"
s = p.read_text(encoding="utf-8")
s = common(s)
for a, b in [
    ('"loss $L(w)$"', '"perda $L(w)$"'),
    ('loss {m.loss(X_gate, y_xor):.3f}', 'perda {m.loss(X_gate, y_xor):.3f}'),
    ('acc {a:.0%}, loss {m.loss(X_moons, y_m):.2f}', 'exatidão {a:.0%}, perda {m.loss(X_moons, y_m):.2f}'),
    ('ylabel="loss (BCE)", title="XOR', 'ylabel="perda (entropia cruzada binária)", title="XOR'),
    ('ylabel="loss (BCE)", title="Luas', 'ylabel="perda (entropia cruzada binária)", title="Luas'),
    ('ylabel="loss (BCE, escala log)"', 'ylabel="perda (entropia cruzada binária, escala log)"'),
    ('ylabel="loss (entropia cruzada)"', 'ylabel="perda (entropia cruzada)"'),
    ('title="A loss de validação sobe: a rede memoriza"', 'title="A perda de validação sobe: a rede memoriza"'),
    ('ylabel="accuracy (fração)"', 'ylabel="exatidão (fração de acertos)"'),
    ('title="Treino a 100 %, validação fica abaixo"', 'title="Treino a 100 %, validação fica abaixo"'),
    ('ylabel="loss de validação", title="Regularizar evita a subida da loss de validação"', 'ylabel="perda de validação", title="Regularizar trava a subida da perda de validação"'),
    ('ylabel="accuracy de validação (média de 3 seeds)", title="Accuracy de validação por técnica"', 'ylabel="exatidão de validação (média de 3 sementes)", title="Exatidão de validação por técnica"'),
    ('"weight decay 0,1"', '"decaimento de pesos 0,1"'),
    ('"early stopping"', '"paragem antecipada"'),
    ('f"$z$ = 0,5·1 + 0,5·1 + ($b$ = {b0}) = {z:.1f}"', 'f"$z$ = 0,5·1 + 0,5·1 + ($b$ = {fmt_pt(b0, 1)}) = {fmt_pt(z, 1)}"'),
    ('f"peso {name} = {wv}"', 'f"peso {name} = {fmt_pt(wv, 1)}"'),
    ('label=f"η = {lr}"', 'label=f"η = {fmt_pt(lr, 2)}"'),
    ("label=d)\n    axes[1].plot(sub.n, sub.gflops, marker=marker, color=colour, label=d)", 'label=DEV.get(d, d))\n    axes[1].plot(sub.n, sub.gflops, marker=marker, color=colour, label=DEV.get(d, d))'),
]:
    s = rep(s, a, b, 0)
s = rep(s, "fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout=\"constrained\")\nfor d, marker, colour in zip(devices", "DEV = {\"cpu\": \"CPU\", \"cuda\": \"GPU\", \"mps\": \"GPU (Apple)\"}\nfig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout=\"constrained\")\nfor d, marker, colour in zip(devices")
p.write_text(s, encoding="utf-8")

# ----------------------------------------------------------------------------- 05
p = here / "build_05.py"
s = p.read_text(encoding="utf-8")
s = common(s)
s = rep(s, 'label="ótimo greedy (12)"', 'label="ótimo (12)"')
s = rep(s, 'média de 5 seeds', 'média de 5 sementes')
p.write_text(s, encoding="utf-8")
print("ok")
