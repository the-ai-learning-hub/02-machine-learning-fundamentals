"""Ronda 2: fichas, glossário e 'Como funciona: Q-learning' no build_05.py (executar uma vez)."""
from pathlib import Path

p = Path(__file__).with_name("build_05.py")
s = p.read_text(encoding="utf-8")
Q3 = '"""'


def rep(a, b):
    global s
    assert a in s, a[:80]
    s = s.replace(a, b, 1)


# ---------------------------------------------------------------- glossário (secção 1)
rep('nb.md(disc("Sucesso: retorno ou taxa de sucesso?",', '''nb.md("""### 📖 Palavras que vão aparecer (mini-glossário)

Os termos básicos (dataset, *feature*, modelo, treinar, prever, erro) estão no [Vocabulário do notebook 01](../../01_supervised_learning/notebooks/01_CRISP-DM_House_Prices.ipynb#vocab). Aqui ficam os do reforço; cada um volta a ser explicado quando aparece:

| Termo | Em palavras simples |
|---|---|
| **agente** | quem decide e age (aqui, o robô no mapa) |
| **ambiente** | o mundo onde o agente age e que lhe responde (o mapa e as suas regras) |
| **estado** ($s$) | a situação atual do agente (aqui, a célula onde está, numerada de 0 a 29) |
| **ação** ($a$) | o que o agente pode fazer (↑ → ↓ ←) |
| **recompensa** ($r$) | o número que o ambiente devolve depois de cada ação (−1, +20, −20) |
| **política** | a regra "em cada estado, faço esta ação" (o que queremos aprender) |
| **episódio** | uma tentativa completa, do início até ao objetivo, à lava ou ao limite de passos |
| **retorno** ($G$) | a soma das recompensas (com desconto) de um episódio |
| **tabela Q** | uma tabela com uma linha por estado e uma coluna por ação, com a nota que o agente dá a cada ação |
| **ε-greedy** | com probabilidade ε o agente tenta uma ação ao acaso (explora); senão escolhe a de melhor nota (*greedy*, "gananciosa") |
| $\\\\alpha$ e $\\\\gamma$ | $\\\\alpha$ = quanto ajusta a nota em cada passo; $\\\\gamma$ = quanto valoriza o futuro (0 a 1) |
""")
nb.md(disc("Sucesso: retorno ou taxa de sucesso?",''')

# ---------------------------------------------------------------- secção 2: conhecer o mundo
a = s.index('nb.md("""Aqui não há ficheiro de dados')
b = s.index("nb.code(r'''import json")
s = s[:a] + '''nb.md("""## 🗂️ Conhecer o mundo do agente

Aqui **não há ficheiro de dados**: os "dados" são a **experiência** que o agente gera ao tentar. Por isso conhecer os dados é conhecer o **ambiente**:

| | |
|---|---|
| **O que é** | um mapa de 5 linhas × 6 colunas (30 células), inventado para a aula e implementado em `src/rl_utils.py` |
| **Estado** | o número da célula onde o agente está (0 a 29, linha a linha; o início é o 24 e o objetivo o 5) |
| **Ações** | 4: ↑ → ↓ ← (bater numa parede ou na borda não move o agente) |
| **Recompensas** | −1 por cada passo; +20 ao chegar ao objetivo (G); −20 ao cair na lava (L) |
| **Fim de um episódio** | chegar a G, cair em L, ou 100 passos |
| **Aleatoriedade** | por omissão nenhuma; com `slip=0.1`, 10 % das ações saem ao acaso |

**Seeds:** Q-learning com `seed=0` (*seed* = número que fixa o "acaso" para o resultado ser repetível); avaliação com várias seeds.""")
''' + s[b:]

# draw_map com legenda e números
rep("def draw_map(env, ax=None, title=None, path=None, agent=None):", "def draw_map(env, ax=None, title=None, path=None, agent=None, legend=False, numbers=False):")
rep('''    if path is not None:
        rows, cols = np.divmod(np.array(path), env.n_cols)
        ax.plot(cols, rows, color=PALETTE["navy"], lw=2.5, ls="--", marker="o", ms=5)''', '''            if numbers:
                ax.text(c - 0.44, r - 0.4, str(r * env.n_cols + c), ha="left", va="top", fontsize=10, color="#59647a" if kind in ".S" else "white")
    if legend:
        names = {".": "célula livre (−1 por passo)", "#": "parede (não se atravessa)", "L": "lava (−20, fim)", "G": "objetivo (+20, fim)",
                 "S": "início", "C": "moeda (+5, não desaparece)"}
        present = [k for k in names if k in set(env.grid.ravel())]
        ax.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=CELL_COLOURS[k], edgecolor="grey", label=names[k]) for k in present],
                  loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=False, title="legenda")
    if path is not None:
        rows, cols = np.divmod(np.array(path), env.n_cols)
        ax.plot(cols, rows, color=PALETTE["navy"], lw=2.5, ls="--", marker="o", ms=5)''')
rep('draw_map(env, ax, "O mundo do agente: 30 células, 4 ações")', 'draw_map(env, ax, "O mundo do agente: 30 células, 4 ações", legend=True, numbers=True)')
rep("fig, ax = plt.subplots(figsize=(6.5, 5.2), layout=\"constrained\")\ndraw_map(env, ax, \"O mundo", "fig, ax = plt.subplots(figsize=(9.5, 5.2), layout=\"constrained\")\ndraw_map(env, ax, \"O mundo")

# ---------------------------------------------------------------- Q-learning: como funciona
rep('nb.md("""$Q(s,a)$ é o **retorno esperado**', '''nb.md("""## 🧠 Como funciona: Q-learning

**Ideia simples.** O agente mantém uma **tabela de notas**: para cada célula e cada direção, "quão boa é esta jogada?". No início todas as notas são 0. Joga, vê o que aconteceu e **corrige a nota** da jogada que fez, aproximando-a do que realmente viu (a recompensa imediata mais a melhor nota da célula seguinte). Com muitas tentativas, as notas boas "propagam-se" do objetivo para trás e o agente passa a escolher, em cada célula, a jogada de nota mais alta.

**Analogia:** um turista num bairro novo que anota "da estação, ir para norte foi bom, para sul foi mau" e vai corrigindo as notas à medida que explora.

**O ciclo de cada passo:** (1) o agente vê o **estado** $s$; (2) escolhe uma **ação** $a$ (às vezes ao acaso: explorar); (3) o ambiente devolve a **recompensa** $r$ e o novo estado $s'$; (4) o agente **atualiza** $Q(s,a)$; (5) repete até o episódio acabar.

**Quando usar:** poucos estados e ações (cabem numa tabela). **Cuidados:** a recompensa tem de descrever bem o que queremos (secção 6); com milhões de estados a tabela deixa de caber (por isso o DQN).""")
nb.code(r\'\'\'# Mecanismo: o ciclo agente-ambiente
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained"); ax.axis("off"); ax.set(xlim=(0, 10), ylim=(0, 4.2))
for x, title, body, colour in ((0.3, "AGENTE", "tabela Q: nota de cada" + chr(10) + "ação em cada estado", "#cfe3f5"),
                               (6.7, "AMBIENTE", "o mapa: paredes, lava," + chr(10) + "objetivo, regras", "#e9f5e1")):
    ax.add_patch(FancyBboxPatch((x, 1.2), 3.0, 1.8, boxstyle="round,pad=0.1", fc=colour, ec="black", lw=1.5))
    ax.text(x + 1.5, 2.65, title, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(x + 1.5, 1.85, body, ha="center", va="center", fontsize=11)
ax.annotate("", xy=(6.6, 2.7), xytext=(3.4, 2.7), arrowprops=dict(arrowstyle="-|>", lw=2.5, color=PALETTE["blue"]))
ax.text(5.0, 3.1, "1. ação $a$ (↑ → ↓ ←)" + chr(10) + "(às vezes ao acaso: explorar)", ha="center", va="bottom", fontsize=11, color=PALETTE["blue"])
ax.annotate("", xy=(3.4, 1.5), xytext=(6.6, 1.5), arrowprops=dict(arrowstyle="-|>", lw=2.5, color=PALETTE["red"]))
ax.text(5.0, 1.1, "2. novo estado $s'$ e recompensa $r$", ha="center", va="top", fontsize=11, color=PALETTE["red"])
ax.text(0.3, 0.45, "3. O agente atualiza a nota:  $Q(s,a) \\\\leftarrow Q(s,a) + \\\\alpha\\\\,[\\\\,r + \\\\gamma \\\\max Q(s',\\\\cdot) - Q(s,a)\\\\,]$", fontsize=12)
ax.set_title("Um passo do ciclo; um episódio repete-o até ao objetivo, à lava ou aos 100 passos")
finish(fig, FIG / "05_agent_env_cycle.png"); plt.show()\'\'\')
nb.md("""$Q(s,a)$ é o **retorno esperado**''')

# tabela Q depois da atualização à mão
rep("assert abs(Q_demo[4, 1] - 10) < 1e-9 and abs(Q_demo[3, 1] - 4.25) < 1e-9''')", '''assert abs(Q_demo[4, 1] - 10) < 1e-9 and abs(Q_demo[3, 1] - 4.25) < 1e-9\'\'\')
nb.code(r\'\'\'# A tabela Q: uma linha por estado, uma coluna por ação (só mostramos 3 das 30 linhas, depois das 2 atualizações)
rows_shown = [3, 4, 24]
cells = [[fmt_pt(Q_demo[st, a], 2) for a in range(4)] for st in rows_shown]
fig, ax = plt.subplots(figsize=(9, 3.2), layout="constrained"); ax.axis("off")
tab = ax.table(cellText=cells, colLabels=[f"ação {ARROWS[a]}" for a in range(4)],
               rowLabels=["estado 3 (linha 0, coluna 3)", "estado 4 (linha 0, coluna 4)", "estado 24 (o início)"], loc="center", cellLoc="center")
tab.auto_set_font_size(False); tab.set_fontsize(13); tab.scale(1, 2)
tab[(2, 1)].set_facecolor("#ffe9b3"); tab[(1, 1)].set_facecolor("#ffe9b3")   # as duas células atualizadas (→)
ax.set_title("Tabela Q (parcial): só as células amarelas mudaram; o resto continua a 0")
finish(fig, FIG / "05_q_table.png"); plt.show()\'\'\')''')

s = s.replace("\\'\\'\\'", "'''")
p.write_text(s, encoding="utf-8")
print("ok")
