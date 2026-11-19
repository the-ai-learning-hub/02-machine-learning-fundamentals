"""Ronda 2: explicações, fichas de dados e 'Como funciona' no build_04.py (executar uma vez)."""
from pathlib import Path

p = Path(__file__).with_name("build_04.py")
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert a in s, a[:80]
    s = s.replace(a, b, 1)


Q3 = '"""'

# ------------------------------------------------------------------ secção 1: glossário
rep('nb.md(disc("Quando uma rede neuronal é a ferramenta errada?",', Q3.join(['nb.md(', '''### 📖 Palavras que vão aparecer (mini-glossário)

Os termos básicos (dataset, *feature*, alvo, modelo, treinar, *overfitting*) estão no [Vocabulário do notebook 01](../../01_supervised_learning/notebooks/01_CRISP-DM_House_Prices.ipynb#vocab). Aqui ficam os específicos das redes neuronais; cada um volta a ser explicado quando aparece:

| Termo | Em palavras simples |
|---|---|
| **neurónio** | uma pequena conta: multiplica cada entrada por um **peso**, soma tudo, soma um **bias** e passa o resultado por uma **ativação** |
| **peso** ($w$) e ***bias*** ($b$) | os números que a rede **aprende**; o peso diz a importância de uma entrada, o *bias* desloca o resultado |
| **ativação** | função que decide o quanto o neurónio "dispara" (degrau, *tanh*, sigmoide, ReLU) |
| **rede / MLP** | vários neurónios organizados em camadas (*multi-layer perceptron*) |
| ***forward*** | calcular a saída da rede, da entrada para a saída |
| ***loss*** | um número que mede o quão errada está a rede (quanto menor, melhor) |
| **gradiente** | para cada peso, indica para que lado mexer para a *loss* diminuir |
| ***backprop*** | o método que calcula todos os gradientes, da saída para a entrada |
| ***learning rate*** ($\\\\eta$) | o tamanho do passo que damos em cada atualização |
| **época** | uma passagem completa por todos os exemplos de treino |
| ***batch*** | um pedaço dos exemplos usado em cada atualização |
| **tensor** | o *array* do PyTorch (como o do NumPy, mas pode ir para a GPU e calcular gradientes) |
| **GPU** | placa gráfica: milhares de unidades pequenas a trabalhar em paralelo |
''', ')\nnb.md(disc("Quando uma rede neuronal é a ferramenta errada?",']))

# ------------------------------------------------------------------ secção 2: ficha dos dados
a = s.index('nb.md("""Dois problemas minúsculos e **não linearmente separáveis**:')
b = s.index("nb.code(r'''import json")
s = s[:a] + '''nb.md("""## 🗂️ Conhecer os dados

Usamos três conjuntos. Os dois primeiros são problemas minúsculos **inventados de propósito** e **não linearmente separáveis** (isto é, nenhuma reta separa as classes); no terceiro usamos imagens reais.

| Conjunto | O que é | De onde vem | Uma linha = … | Alvo |
|---|---|---|---|---|
| **XOR** | a porta lógica "ou exclusivo" | inventado (4 linhas, a tabela-verdade) | uma combinação de duas entradas $(x_1, x_2)$, cada uma 0 ou 1 | 1 se **exatamente uma** entrada for 1, senão 0 |
| **luas** (`make_moons`) | 400 pontos 2D, duas "luas" entrelaçadas com ruído | gerado pelo scikit-learn (`noise=0.2`, `random_state=0`) | um ponto $(x_1, x_2)$ | classe 0 ou 1 (qual lua) |
| **dígitos** (`load_digits`) | imagens 8×8 de números escritos à mão | dados reais do scikit-learn (UCI) | uma imagem de 64 pixéis | o dígito, 0 a 9 (10 classes) |

**Porque dados inventados?** Sabemos a resposta certa e o problema é tão pequeno que podemos **desenhar** o que a rede aprende. Os dígitos servem para ver a mesma ideia num problema real.

**Seeds:** NumPy MLP com `seed=0`; PyTorch com `torch.manual_seed(42)` (*seed* = número que fixa o "acaso" para o resultado ser repetível).""")
''' + s[b:]

# figura: tabela-verdade + XOR + luas, e grelha dos dígitos
a = s.index("fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))\nfor cls, marker, colour in ((0")
b = s.index("nb.md(disc(\"Porque é que o XOR é difícil?\"")
new_fig = r"""fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [0.9, 1, 1.3]})
# 1) tabela-verdade do XOR, desenhada
axes[0].axis("off")
table = axes[0].table(cellText=[["0", "0", "0"], ["0", "1", "1"], ["1", "0", "1"], ["1", "1", "0"]],
                      colLabels=["$x_1$", "$x_2$", "XOR"], loc="center", cellLoc="center")
table.auto_set_font_size(False); table.set_fontsize(15); table.scale(1, 2.2)
for r in range(1, 5):
    table[r, 2].set_facecolor("#cfe3f5" if r in (2, 3) else "#f5d9cc")
axes[0].set_title("Tabela-verdade: 1 só se as entradas diferem")
# 2) os mesmos 4 pontos no plano
for cls, marker, colour in ((0, "o", PALETTE["red"]), (1, "s", PALETTE["blue"])):
    m = Y_GATES["XOR"] == cls
    axes[1].scatter(X_gate[m, 0], X_gate[m, 1], marker=marker, s=200, color=colour, edgecolor="black", label=f"XOR = {cls}")
    m = y_moons == cls
    axes[2].scatter(X_moons[m, 0], X_moons[m, 1], marker=marker, s=22, color=colour, edgecolor="white", linewidth=0.4,
                    label=f"classe {cls}")
for (x1, x2), y in zip(X_gate, Y_GATES["XOR"]):
    axes[1].annotate(f"({int(x1)}, {int(x2)})", (x1, x2), xytext=(8, 10), textcoords="offset points", fontsize=11)
axes[1].set(title="XOR: nenhuma reta separa as classes", xlabel="$x_1$ (0 ou 1)", ylabel="$x_2$ (0 ou 1)", xlim=(-0.3, 1.4), ylim=(-0.3, 1.4))
axes[2].set(title="Luas: classes curvas e com ruído", xlabel="$x_1$ (u. arbitrárias)", ylabel="$x_2$ (u. arbitrárias)")
for ax in axes[1:]: ax.legend(fontsize=9)
fig.tight_layout(); save_fig(fig, FIG / "04_problems.png"); plt.show()''')
nb.code(r'''digits = load_digits()
fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
for d, ax in enumerate(axes.ravel()):
    ax.imshow(digits.images[np.flatnonzero(digits.target == d)[0]], cmap="gray_r"); ax.set_title(f"dígito {d}"); ax.axis("off")
fig.suptitle("Dígitos reais, 8×8 pixéis: cada imagem é uma linha de 64 números (0 = branco, 16 = preto)", x=0.01, ha="left")
fig.tight_layout(); save_fig(fig, FIG / "04_digits_samples.png"); plt.show()
print("imagens:", digits.images.shape, "| como tabela:", digits.data.shape, "| classes:", np.unique(digits.target))''')"""
s = s[:a] + new_fig + "\n" + s[b:]
rep("nb.code(r'''import json", "nb.code(r'''import json")

# ------------------------------------------------------------------ 3.1 perceptrão: como funciona
rep('nb.md("""## 3.1 O perceptrão (Rosenblatt, 1958)', r'''nb.md("""## 🧠 Como funciona: o perceptrão (um neurónio)

**Ideia simples.** Um neurónio é uma **votação ponderada**: cada entrada dá o seu "voto" multiplicado por um peso (a importância dessa entrada), somam-se os votos mais um *bias* (um ponto de partida), e se o total passar de zero o neurónio "dispara" (saída 1), senão fica a 0. É o que se chama **degrau** (*step*).

**Passos:** (1) $z = w_1x_1 + w_2x_2 + b$; (2) $\\hat y = 1$ se $z \\ge 0$, senão $0$; (3) **aprender:** se errou, corrige os pesos um bocadinho na direção do exemplo.

**Analogia:** decidir se vais à praia: o sol pesa muito (+), o frio pesa (−), o trânsito pouco; somas e decides.

**Limite:** é uma **reta** no plano (os pontos com $z=0$). Só separa classes que uma reta consiga separar.""")
nb.code(r\'\'\'# Mecanismo: um neurónio com valores concretos (AND com pesos escolhidos à mão)
fig, ax = plt.subplots(figsize=(11, 4.2)); ax.axis("off"); ax.set(xlim=(0, 11), ylim=(0, 4.2))
w1, w2, b0, x1v, x2v = 0.5, 0.5, -0.7, 1, 1
z = w1 * x1v + w2 * x2v + b0
for y, name, val in ((3.1, "$x_1$", x1v), (1.1, "$x_2$", x2v)):
    ax.scatter(1.0, y, s=1500, color="#cfe3f5", edgecolor="black", zorder=3)
    ax.text(1.0, y, f"{name} = {val}", ha="center", va="center", fontsize=12, zorder=4)
for y, wv, name in ((3.1, w1, "$w_1$"), (1.1, w2, "$w_2$")):
    ax.annotate("", xy=(4.4, 2.1 + (y - 2.1) * 0.35), xytext=(1.6, y), arrowprops=dict(arrowstyle="-|>", lw=2, color=PALETTE["blue"]))
    ax.text(3.0, 2.1 + (y - 2.1) * 0.75 + 0.25, f"peso {name} = {wv}", ha="center", fontsize=11, color=PALETTE["blue"])
ax.scatter(5.2, 2.1, s=3200, color="#ffe9b3", edgecolor="black", zorder=3)
ax.text(5.2, 2.1, "Σ + $b$", ha="center", va="center", fontsize=14, zorder=4)
ax.text(5.2, 0.55, f"$z$ = 0,5·1 + 0,5·1 + ($b$ = {b0}) = {z:.1f}", ha="center", fontsize=11)
ax.annotate("", xy=(7.6, 2.1), xytext=(6.1, 2.1), arrowprops=dict(arrowstyle="-|>", lw=2))
ax.add_patch(plt.Rectangle((7.6, 1.5), 1.7, 1.2, fc="#e9f5e1", ec="black", zorder=3))
ax.text(8.45, 2.1, "degrau:" + chr(10) + "$z \\geq 0$ → 1", ha="center", va="center", fontsize=11, zorder=4)
ax.annotate("", xy=(10.3, 2.1), xytext=(9.3, 2.1), arrowprops=dict(arrowstyle="-|>", lw=2))
ax.text(10.4, 2.1, f"$\\hat y$ = {int(z >= 0)}", ha="left", va="center", fontsize=14, fontweight="bold")
ax.set_title("Um neurónio: soma ponderada das entradas + bias, depois degrau (exemplo: entrada (1, 1) no AND)")
fig.tight_layout(); save_fig(fig, FIG / "04_perceptron_diagram.png"); plt.show()\'\'\')
nb.md("""## 3.1 O perceptrão (Rosenblatt, 1958)''')

# ------------------------------------------------------------------ 3.2 MLP: como funciona
rep("nb.code(r'''xor = MLP(n_in=2, n_hidden=4, seed=0)", r'''nb.md("""## 🧠 Como funciona: o MLP (várias camadas)

**Ideia simples.** Se um neurónio só traça uma reta, **vários neurónios em camadas** traçam várias retas e combinam-nas. Cada neurónio da camada escondida aprende a responder a uma pergunta simples ("estou acima desta reta?"); o neurónio de saída combina as respostas. É assim que se resolve o XOR: uma reta pergunta "algum x é 1?", outra "ambos são 1?", e a saída diz "o primeiro sim e o segundo não".

**Passos (forward):** entrada → camada escondida (soma ponderada + *tanh*, uma ativação suave entre −1 e 1) → saída (soma ponderada + sigmoide, que dá uma probabilidade entre 0 e 1).

**Cuidados:** sem a ativação não linear, várias camadas seriam equivalentes a uma só (continuaria a ser uma reta).""")
nb.code(r\'\'\'# Mecanismo: o diagrama da rede 2 → 4 → 1 que vamos usar (cada linha é um peso a aprender)
fig, ax = plt.subplots(figsize=(10, 4.6)); ax.axis("off"); ax.set(xlim=(-0.15, 1.15), ylim=(-0.12, 1.08))
sizes, titles = (2, 4, 1), ("entrada" + chr(10) + "$X$: (n, 2)", "camada escondida (4 neurónios, tanh)" + chr(10) + "$a_1$: (n, 4)", "saída (sigmoide)" + chr(10) + "$\\hat y$: (n,)")
coords = [[(x, y) for y in (np.linspace(0.85, 0.15, n) if n > 1 else [0.5])] for x, n in zip((0.0, 0.5, 1.0), sizes)]
for left, right in zip(coords[:-1], coords[1:]):
    for p in left:
        for q in right:
            ax.plot([p[0], q[0]], [p[1], q[1]], color="#9db0c9", lw=1.1, zorder=1)
labels = (["$x_1$", "$x_2$"], ["$h_1$", "$h_2$", "$h_3$", "$h_4$"], ["$\\hat y$"])
for layer, lab, colour in zip(coords, labels, ("#cfe3f5", "#ffe9b3", "#e9f5e1")):
    for (x, y), t in zip(layer, lab):
        ax.scatter(x, y, s=1300, color=colour, edgecolor="black", zorder=3)
        ax.text(x, y, t, ha="center", va="center", fontsize=12, zorder=4)
for x, t in zip((0.0, 0.5, 1.0), titles):
    ax.text(x, -0.07, t, ha="center", va="top", fontsize=10)
ax.text(0.25, 1.02, "pesos $W_1$: (2, 4) = 8 números" + chr(10) + "+ $b_1$: 4 números", ha="center", fontsize=10, color=PALETTE["blue"])
ax.text(0.75, 1.02, "pesos $W_2$: (4, 1) = 4 números" + chr(10) + "+ $b_2$: 1 número", ha="center", fontsize=10, color=PALETTE["blue"])
ax.set_title("Rede 2-4-1: 17 parâmetros no total (8 + 4 + 4 + 1)")
fig.tight_layout(); save_fig(fig, FIG / "04_mlp_diagram.png"); plt.show()\'\'\')
nb.code(r\'\'\'xor = MLP(n_in=2, n_hidden=4, seed=0)''')

# ------------------------------------------------------------------ gradiente descendente: como funciona
rep('nb.md("""## 3.4 *Gradient check*', r'''nb.md("""### 🧠 Como funciona: descida do gradiente

**Ideia simples.** Imagina que estás num vale com nevoeiro e queres chegar ao fundo: sentes a inclinação do chão (o **gradiente**) e dás um passo para baixo. Repetes. O **tamanho do passo** é o *learning rate* $\\eta$.

**Passos:** (1) calcular a *loss* $L$; (2) calcular o gradiente $\\partial L/\\partial w$ (a inclinação); (3) atualizar $w \\leftarrow w - \\eta\\,\\partial L/\\partial w$; (4) repetir. Abaixo, o caso mais simples: **um só peso** e uma *loss* em forma de taça, $L(w) = (w-3)^2 + 1$.""")
nb.code(r\'\'\'w_axis = np.linspace(-5, 11, 200)
L = lambda w: (w - 3) ** 2 + 1
dL = lambda w: 2 * (w - 3)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, (eta, title) in zip(axes, ((0.1, "η = 0,1: passos pequenos, lento"), (0.8, "η = 0,8: bom, chega depressa"),
                                    (1.05, "η = 1,05: grande demais, diverge"))):
    ax.plot(w_axis, L(w_axis), color=PALETTE["grey"], lw=2)
    ws = [0.0]
    for _ in range(8):
        ws.append(ws[-1] - eta * dL(ws[-1]))
    ws = np.array(ws)
    for a, b in zip(ws[:-1], ws[1:]):
        ax.annotate("", xy=(b, L(b)), xytext=(a, L(a)), arrowprops=dict(arrowstyle="-|>", color=PALETTE["red"], lw=1.6))
    ax.plot(ws, L(ws), "o", color=PALETTE["red"], ms=6)
    ax.plot(3, 1, marker="*", ms=16, color=PALETTE["green"], zorder=5)
    ax.set(title=title, xlabel="peso $w$ (u. arbitrárias)", ylim=(0, 45), xlim=(-5, 11))
axes[0].set_ylabel("loss $L(w)$")
fig.suptitle("8 passos de descida do gradiente a partir de w = 0 (★ = mínimo)", x=0.01, ha="left")
fig.tight_layout(); save_fig(fig, FIG / "04_gradient_descent.png"); plt.show()\'\'\')
nb.md("""## 3.4 *Gradient check*''')

# limpar escapes do raw
s = s.replace("\\'\\'\\'", "'''")
p.write_text(s, encoding="utf-8")
print("ok")
