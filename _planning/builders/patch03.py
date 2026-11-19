"""Ronda 2: aplica as alterações de explicação ao build_03.py (executar uma vez)."""
from pathlib import Path

p = Path(__file__).with_name("build_03.py")
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert a in s, a[:80]
    s = s.replace(a, b, 1)


# ---------------------------------------------------------------- secção 1: glossário
rep('''nb.md(disc("O que é um \\"bom\\" agrupamento?",''', '''nb.md("""### 📖 Palavras que vão aparecer (mini-glossário)

Se alguma te parecer nova, o **📖 Vocabulário** do [notebook 01](../../01_supervised_learning/notebooks/01_CRISP-DM_House_Prices.ipynb#vocab) explica o básico (dataset, *feature*, alvo, modelo, treinar). Aqui ficam as específicas deste notebook; cada uma é explicada de novo quando aparece:

| Termo | Em palavras simples |
|---|---|
| ***label*** (rótulo) | a "resposta certa" de cada exemplo (por exemplo, o dígito escrito na imagem). **Aqui não a damos ao algoritmo.** |
| **aprendizagem não supervisionada** | o algoritmo só recebe os dados e procura estrutura sozinho |
| ***cluster*** (grupo) | conjunto de exemplos parecidos entre si e diferentes dos outros |
| **centroide** | o "centro" de um grupo: a média das posições dos seus pontos |
| **distância** | quão longe estão dois exemplos (aqui, a distância "em linha reta", euclidiana) |
| ***outlier*** / ruído | exemplo isolado que não pertence a nenhum grupo |
| **dimensão** | cada coluna de um exemplo (as imagens têm 64 dimensões: 64 pixéis) |
""")
nb.md(disc("O que é um \\"bom\\" agrupamento?",''')

# ---------------------------------------------------------------- secção 2: conhecer os dados
a = s.index('nb.sec(2, "🔍 Data Understanding (compreensão dos dados)")')
b = s.index("nb.code(r'''fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))\nfor ax, X, y, title in")
new_intro = r'''nb.sec(2, "🔍 Data Understanding (compreensão dos dados)")
nb.md("""## 🗂️ Conhecer os dados

Antes de qualquer algoritmo, perguntamos: **o que são estes dados?** Usamos quatro conjuntos. Três são pequenos e **inventados de propósito**: assim *sabemos a resposta certa* e podemos ver se o algoritmo a descobre. O quarto é real.

| Conjunto | O que é | De onde vem | Uma linha = … | Tamanho |
|---|---|---|---|---|
| **blobs** (`make_blobs`) | pontos em 2D, agrupados em "bolas" | **gerado** pelo scikit-learn, com uma *seed* fixa | um ponto $(x_1, x_2)$ | 300 × 2 |
| **moons** (`make_moons`) | pontos em 2D, em forma de duas "luas" | **gerado** pelo scikit-learn | um ponto $(x_1, x_2)$ | 300 × 2 |
| **dígitos** (`load_digits`) | imagens de números escritos à mão | dados reais incluídos no scikit-learn (UCI, optical recognition of handwritten digits) | uma imagem 8×8 de um dígito (0–9) | 1797 × 64 |
| **casas de Windsor** | características e preço de casas vendidas | arquivo oficial do *Journal of Applied Econometrics* (Anglin & Gençay, 1996); descarregado no 1.º uso para `data/` | uma casa vendida | 546 × 12 |

**Porque inventar dados?** Num problema real não sabemos se o algoritmo acertou, porque não há "resposta certa" para comparar. Com dados sintéticos *nós* escolhemos os grupos verdadeiros e podemos medir o erro. É o mesmo que treinar com um mapa antes de ir a um sítio desconhecido.

**Seeds:** `random_state=42` em tudo o que é gerado ou ajustado (*seed* = número que fixa o "acaso" para o resultado ser repetível).""")
'''
s = s[:a] + new_intro + s[b:]

# depois da figura blobs/moons, antes da discussão "O que é um grupo?": dígitos + casas
rep('''nb.md(disc("O que é um grupo?",''', '''nb.md("""### Os dígitos: uma imagem é uma linha de 64 números

Cada imagem tem 8×8 = 64 pixéis; cada pixel é um número de 0 (branco) a 16 (preto). Para o algoritmo, uma imagem é só uma lista de 64 números, ou seja, **um ponto num espaço com 64 dimensões**. As *labels* (0 a 9) existem, mas **escondemo-las** do algoritmo e só as usamos no fim para verificar.""")
nb.code(r\'\'\'fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
for d, ax in enumerate(axes.ravel()):
    i = np.flatnonzero(digits.target == d)[0]
    ax.imshow(digits.images[i], cmap="gray_r"); ax.set_title(f"dígito {d}"); ax.axis("off")
fig.suptitle("Um exemplo real de cada dígito (8×8 pixéis)", x=0.01, ha="left")
fig.tight_layout(); save_fig(fig, FIG / "03_digits_samples.png"); plt.show()
print("A primeira imagem como números (0 = branco, 16 = preto):")
print(digits.images[0].astype(int))
print("labels do início:", digits.target[:10], "| imagens:", digits.images.shape, "-> como linhas de 64 números:", digits.data.shape)\'\'\')
nb.md("""### As casas de Windsor (o caso real)

**Ficha do dataset.** *O que é:* 546 casas vendidas em **Windsor (Ontário, Canadá)** entre julho e setembro de **1987**. *Fonte oficial:* arquivo do *Journal of Applied Econometrics* (Anglin & Gençay, 1996), descarregado no 1.º uso para `data/`. *Uma linha:* uma casa. *Preço:* dólares canadianos de 1987 (**CAD**). *Área do lote:* pés² (1 pé² ≈ 0,093 m²).

Neste notebook **não** há alvo: vamos agrupar as casas só pelas características físicas (5 colunas) e usar o preço **depois**, para perceber os grupos.""")
nb.code(r\'\'\'housing = load_housing()
meaning = {
    "price": ("preço de venda", "numérica", "CAD (1987)"), "lotsize": ("área do lote", "numérica", "pés²"),
    "bedrooms": ("quartos", "contagem", "n.º"), "bathrooms": ("casas de banho", "contagem", "n.º"),
    "stories": ("pisos", "contagem", "n.º"), "driveway": ("tem entrada para carros", "sim/não", "—"),
    "recreation": ("tem sala de jogos", "sim/não", "—"), "fullbase": ("tem cave completa", "sim/não", "—"),
    "gasheat": ("aquecimento a gás", "sim/não", "—"), "aircon": ("tem ar condicionado", "sim/não", "—"),
    "garage": ("lugares de garagem", "contagem", "n.º"), "prefer": ("bairro preferido", "sim/não", "—"),
}
columns_table = pd.DataFrame(
    [(c, *meaning[c], housing[c].iloc[0], "sim" if c in HOUSING_NUMERIC else "não (só interpretação)" if c == "price" else "não")
     for c in housing.columns],
    columns=["coluna", "significado", "tipo", "unidade", "exemplo (1.ª casa)", "usada no agrupamento?"]).set_index("coluna")
columns_table\'\'\')
nb.code(r\'\'\'housing.head()\'\'\')
nb.code(r\'\'\'fig, axes = plt.subplots(3, 4, figsize=(13, 8))
titles = {"price": "preço (CAD)", "lotsize": "área do lote (pés²)", "bedrooms": "quartos", "bathrooms": "casas de banho",
          "stories": "pisos", "garage": "lugares de garagem"}
for ax, col in zip(axes.ravel(), housing.columns):
    if col in titles and col in ("price", "lotsize"):
        ax.hist(housing[col], bins=25, color=PALETTE["blue"], edgecolor="white")
        ax.set(xlabel=titles[col], ylabel="n.º de casas")
    elif col in titles:
        counts = housing[col].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color=PALETTE["orange"], edgecolor="white")
        ax.set(xlabel=titles[col], ylabel="n.º de casas")
    else:
        counts = housing[col].value_counts().reindex(["no", "yes"])
        ax.bar(["não", "sim"], counts.values, color=[PALETTE["grey"], PALETTE["green"]], edgecolor="white")
        ax.set(xlabel=f"{meaning[col][0]}", ylabel="n.º de casas")
    ax.set_title(col, fontsize=10)
fig.suptitle("Uma coluna por painel: casas pequenas e de poucos quartos dominam; preços e lotes são assimétricos", x=0.01, ha="left")
fig.tight_layout(); save_fig(fig, FIG / "03_housing_columns.png"); plt.show()\'\'\')
nb.md("""**Leitura:** quase todas as casas têm 2 a 4 quartos e 1 casa de banho; a maioria tem 1 ou 2 pisos. A área do lote e o preço têm uma "cauda" para a direita (poucas casas muito grandes e caras). Repara nas escalas: `lotsize` anda na casa dos milhares e `bedrooms` na casa das unidades; isto vai importar na secção 3.""")
nb.md(disc("O que é um grupo?",''')

# ---------------------------------------------------------------- secção 3: não recarregar casas
rep('''nb.code(r\'\'\'housing = load_housing()
X_house = housing[HOUSING_NUMERIC].to_numpy(dtype=float)''', '''nb.code(r\'\'\'X_house = housing[HOUSING_NUMERIC].to_numpy(dtype=float)''') if False else None
a = s.index("nb.code(r'''housing = load_housing()\nX_house")
s = s[:a] + "nb.code(r'''X_house = housing[HOUSING_NUMERIC].to_numpy(dtype=float)" + s[a + len("nb.code(r'''housing = load_housing()\nX_house = housing[HOUSING_NUMERIC].to_numpy(dtype=float)"):]
rep("Algoritmos de *clustering* medem **distâncias**.", "Algoritmos de *clustering* medem **distâncias** (quão longe estão dois exemplos).")
rep("O preço **não entra** no agrupamento; usamo-lo depois, para interpretar.\"\"\")", "O preço **não entra** no agrupamento; usamo-lo depois, para interpretar. **Escalar** (*standardize*) = converter cada coluna para média 0 e desvio-padrão 1 (o desvio-padrão mede o quanto os valores se afastam da média), para que todas tenham o mesmo peso nas distâncias.\"\"\")")

# ---------------------------------------------------------------- secção 4: K-Means (como funciona)
rep('''nb.md("""O K-Means repete dois passos até nada mudar:''', '''nb.md("""## 🧠 Como funciona: K-Means

**Ideia simples.** Imagina que queres abrir 4 lojas para servir 300 casas espalhadas pelo mapa. Primeiro colocas as 4 lojas ao acaso; cada casa vai à mais próxima; depois mudas cada loja para o **centro** das casas que a escolheram; repetes. As lojas vão-se arrumando, e cada "zona de influência" é um *cluster*.

**Passos** (o algoritmo repete 1 e 2 até nada mudar):

1. **Atribuir:** cada ponto vai para o **centroide** (a "loja") mais próximo;
2. **Atualizar:** cada centroide passa a ser a **média** dos pontos que lhe foram atribuídos.

**Quando usar:** grupos compactos e mais ou menos redondos, com uma ideia de quantos são (*k*). **Cuidados:** é preciso escolher *k*; depende da escala; só vê bolas (secção 6); pode ficar preso numa solução má (vê-se abaixo). O que se minimiza é a **inércia** = soma dos quadrados das distâncias de cada ponto ao seu centroide (quanto menor, mais compactos os grupos).""")
nb.md("""O K-Means repete dois passos até nada mudar:''')

rep('''nb.code(r\'\'\'show = [0, 1, 2]''', '''nb.code(r\'\'\'# O mecanismo, desenhado: (1) cada ponto liga ao centroide mais próximo; (2) cada centroide salta para a média dos seus pontos
cents0, labels0, _ = history[0]
cents1 = history[1][0]                                     # centroides depois do passo "atualizar"
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True, sharey=True)
pick = np.random.default_rng(1).choice(len(X_blobs), 40, replace=False)
for ax in axes:
    ax.scatter(X_blobs[:, 0], X_blobs[:, 1], s=14, color=PALETTE["grey"], alpha=0.35)
for i in pick:
    j = labels0[i]
    axes[0].plot([X_blobs[i, 0], cents0[j, 0]], [X_blobs[i, 1], cents0[j, 1]], color=CYCLE[j], lw=0.9, alpha=0.8)
    axes[0].scatter(*X_blobs[i], s=26, marker=MARKERS[j], color=CYCLE[j], edgecolor="white", zorder=3)
for j in range(4):
    axes[0].scatter(*cents0[j], marker="*", s=330, color=CYCLE[j], edgecolor="black", linewidth=1.2, zorder=5)
    axes[1].scatter(*cents0[j], marker="*", s=200, color="white", edgecolor=CYCLE[j], linewidth=1.5, zorder=4)
    axes[1].scatter(*cents1[j], marker="*", s=330, color=CYCLE[j], edgecolor="black", linewidth=1.2, zorder=5)
    axes[1].annotate("", xy=cents1[j], xytext=cents0[j], arrowprops=dict(arrowstyle="->", color=CYCLE[j], lw=2))
axes[0].set(title="Passo 1 · atribuir: cada ponto liga ao centroide mais próximo", xlabel="$x_1$ (u. arbitrárias)", ylabel="$x_2$ (u. arbitrárias)")
axes[1].set(title="Passo 2 · atualizar: o centroide salta para a média do seu grupo", xlabel="$x_1$ (u. arbitrárias)")
fig.tight_layout(); save_fig(fig, FIG / "03_kmeans_mechanism.png"); plt.show()\'\'\')
nb.code(r\'\'\'show = [0, 1, 2]''')

# ---------------------------------------------------------------- secção 6: DBSCAN (como funciona)
rep('''nb.md("""O **DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*) define grupo como uma região **densa**''', '''nb.md("""## 🧠 Como funciona: DBSCAN

**Ideia simples.** Pensa numa festa: um grupo de pessoas forma-se onde há **muita gente junta**; quem está sozinho num canto não pertence a nenhum grupo. O DBSCAN procura "multidões" e deixa os isolados de fora.

Dois parâmetros definem "juntos": o **raio** `eps` (ε, a distância máxima para dois pontos serem vizinhos) e `min_samples` (quantos vizinhos, incluindo o próprio ponto, para haver multidão). Cada ponto fica numa de 3 categorias:

- **núcleo** (*core*): tem pelo menos `min_samples` pontos dentro do raio `eps`. Os núcleos vizinhos ligam-se e crescem o grupo;
- **fronteira** (*border*): está no raio de um núcleo, mas ele próprio não tem vizinhos que cheguem;
- **ruído** (*noise*): não é núcleo nem está perto de um (label −1).

**Quando usar:** grupos de forma livre, ruído a detetar, *k* desconhecido. **Cuidados:** `eps` e `min_samples` têm de ser escolhidos; densidades muito diferentes; escala.""")
nb.code(r\'\'\'# Mecanismo: luas + 3 pontos isolados que acrescentamos de propósito
demo_X = np.vstack([X_moons, [[-0.8, 1.1], [1.1, 1.1], [0.5, -0.9]]])
EPS, MIN_S = 0.2, 5
demo_db = DBSCAN(eps=EPS, min_samples=MIN_S).fit(demo_X)
is_core = np.zeros(len(demo_X), bool); is_core[demo_db.core_sample_indices_] = True
is_noise = demo_db.labels_ == -1
is_border = ~is_core & ~is_noise
print(f"núcleo: {is_core.sum()} | fronteira: {is_border.sum()} | ruído: {is_noise.sum()}")

# um exemplo de cada tipo
i_core = np.flatnonzero(is_core)[np.argmin(np.linalg.norm(demo_X[is_core] - [0.0, 0.2], axis=1))]
i_border = np.flatnonzero(is_border)[0]
i_noise = np.flatnonzero(is_noise)[0]

fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.scatter(*demo_X[is_core].T, s=18, color=PALETTE["blue"], marker="o", label="núcleo", alpha=0.6)
ax.scatter(*demo_X[is_border].T, s=45, color=PALETTE["orange"], marker="s", label="fronteira", edgecolor="black", zorder=3)
ax.scatter(*demo_X[is_noise].T, s=60, color="black", marker="x", label="ruído", zorder=3)
for i, name, colour in ((i_core, "núcleo", PALETTE["blue"]), (i_border, "fronteira", PALETTE["orange"]), (i_noise, "ruído", "black")):
    ax.add_patch(plt.Circle(demo_X[i], EPS, fill=False, ec=colour, lw=2, ls="--"))
    n_in = int((np.linalg.norm(demo_X - demo_X[i], axis=1) <= EPS).sum())
    ax.annotate(f"{name}: {n_in} pontos no raio", demo_X[i], xytext=(demo_X[i][0] + 0.15, demo_X[i][1] + 0.38), fontsize=10,
                arrowprops=dict(arrowstyle="-", color="grey"))
ax.set(title=f"Raio ε = {EPS} e min_samples = {MIN_S}: núcleo, fronteira e ruído", xlabel="$x_1$ (u. arbitrárias)", ylabel="$x_2$ (u. arbitrárias)", aspect="equal")
ax.legend(loc="lower left")
fig.tight_layout(); save_fig(fig, FIG / "03_dbscan_mechanism.png"); plt.show()\'\'\')
nb.md("""Repara: um ponto é **núcleo** se o círculo tracejado apanhar pelo menos `min_samples` pontos (ele próprio incluído). O de **fronteira** está dentro do círculo de um núcleo mas o seu próprio círculo está quase vazio. O de **ruído** tem o círculo vazio e ninguém o "adota".""")
nb.md("""O **DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*) define grupo como uma região **densa**''')

# ---------------------------------------------------------------- secção 7: PCA (como funciona) e remover grelha repetida
a = s.index('nb.md("""Cada imagem de `load_digits` é um ponto em 64 dimensões')
b = s.index("nb.code(r'''pca = PCA(random_state=SEED).fit(digits.data)")
pca_block = r'''nb.md("""## 🧠 Como funciona: PCA

**Ideia simples.** Uma nuvem de pontos 3D pode ter forma de "panqueca": quase tudo acontece em 2 direções. A PCA (*Principal Component Analysis*, análise de componentes principais) encontra **as direções em que os dados mais variam** (a **variância** é o quanto os valores se espalham). A 1.ª componente é a direção de maior espalhamento; a 2.ª, perpendicular, a seguinte; e assim por diante. Guardar só as primeiras componentes é **comprimir** com pouca perda.

**Passos:** (1) centrar os dados na média; (2) achar a direção de maior variância; (3) achar a seguinte, perpendicular às anteriores; (4) projetar cada ponto nessas direções (as novas "coordenadas"). Não precisa de *labels*.

**Quando usar:** reduzir dimensões, visualizar, limpar ruído. **Cuidados:** é linear; mede variância, não "importância para a tarefa"; depende da escala das colunas.""")
nb.code(r\'\'\'# Mecanismo: nuvem 2D alongada e as suas duas componentes principais
rng = np.random.default_rng(SEED)
cloud = rng.normal(size=(200, 2)) * [3.0, 0.8]                  # espalhada ao longo do eixo x
theta = np.deg2rad(30)
cloud = cloud @ np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
pca2 = PCA(n_components=2).fit(cloud)
centre = pca2.mean_
fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.scatter(cloud[:, 0], cloud[:, 1], s=16, color=PALETTE["grey"], alpha=0.5, label="dados")
for k, (colour, name) in enumerate(((PALETTE["red"], "1.ª componente"), (PALETTE["blue"], "2.ª componente"))):
    vec = pca2.components_[k] * 2 * np.sqrt(pca2.explained_variance_[k])
    ax.annotate("", xy=centre + vec, xytext=centre, arrowprops=dict(arrowstyle="-|>", color=colour, lw=3))
    ax.text(*(centre + vec * 1.08), f"{name}\\n({pca2.explained_variance_ratio_[k]:.0%} da variância)", color=colour, fontsize=10, fontweight="bold")
# projeção de alguns pontos na 1.ª componente
for i in range(0, 200, 25):
    proj = centre + (cloud[i] - centre) @ pca2.components_[0] * pca2.components_[0]
    ax.plot([cloud[i, 0], proj[0]], [cloud[i, 1], proj[1]], color=PALETTE["red"], lw=0.8, ls=":")
    ax.scatter(*proj, s=30, color=PALETTE["red"], marker="D", zorder=4)
ax.set(title="A 1.ª componente é a direção de maior espalhamento", xlabel="$x_1$ (u. arbitrárias)", ylabel="$x_2$ (u. arbitrárias)", aspect="equal")
ax.legend(loc="lower right"); fig.tight_layout(); save_fig(fig, FIG / "03_pca_mechanism.png"); plt.show()\'\'\')
nb.md("""Os losangos vermelhos são a **projeção** dos pontos na 1.ª componente: ficamos com **1 número por ponto em vez de 2** e perdemos só a pequena distância perpendicular (a variância da 2.ª componente). É isto que vamos fazer com os dígitos: de 64 números para 2, 10 ou 29.

Agora nos dígitos (as imagens já foram vistas na secção 2):""")
'''
s = s[:a] + pca_block.replace("\\'\\'\\'", "'''") + s[b:]
# a linha acima inseriu r\'\'\' — corrigir
s = s.replace("nb.code(r\\'\\'\\'", "nb.code(r'''").replace("\\'\\'\\')", "''')")

p.write_text(s, encoding="utf-8")
print("ok")
