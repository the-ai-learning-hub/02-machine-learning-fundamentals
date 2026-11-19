import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import NB, disc, add_exercise

nb = NB()
FILE = "03_Unsupervised_Learning.ipynb"
nb.head("03", "Aprendizagem não supervisionada", "Que estrutura existe sem labels?", FILE, "nao-supervisionada",
        "02_unsupervised_learning",
        ["implementar o K-Means à mão (atribuir → atualizar) e compará-lo com o `KMeans` do scikit-learn;",
         "escolher *k* com o método do cotovelo e com o *silhouette*, e saber porque nenhum dos dois é infalível;",
         "explicar porque é obrigatório pôr as *features* na mesma escala antes de medir distâncias;",
         "usar o DBSCAN quando os grupos não são \"bolas\" e interpretar `eps`, `min_samples` e o ruído;",
         "usar a PCA para reduzir dimensão, visualizar 64 dimensões em 2D e reconstruir imagens;",
         "avaliar *clusters* sem labels (e com labels escondidas) e discutir os limites da segmentação."],
        "≈ 25 min", "notebooks 01 e 02 (pipelines, escala) e NumPy básico",
        ["🧭 Business Understanding (compreensão do negócio)", "🔍 Data Understanding (compreensão dos dados)",
         "🧹 Data Preparation (preparação dos dados): a escala decide tudo",
         "🤖 Modeling (modelação) A — K-Means passo a passo", "🤖 Modeling B — escolher o número de grupos",
         "🤖 Modeling C — DBSCAN: grupos com forma livre", "🤖 Modeling D — PCA: comprimir sem perder o essencial",
         "📏 Evaluation (avaliação): como avaliar sem labels?", "🚀 Deployment (implementação) e limites",
         "✏️ Exercícios", "📌 Resumo", "📚 Para saber mais"])

# ----------------------------------------------------------------------------- 1
nb.sec(1, "🧭 Business Understanding (compreensão do negócio)")
nb.md("""Na aprendizagem **supervisionada** (notebooks 01 e 02) tínhamos uma coluna-alvo: o preço, a sobrevivência. Aqui **não há alvo**. Há apenas pontos e uma pergunta: *existe estrutura nestes dados que um humano não vê à primeira?*

Três pedidos típicos de "negócio" que se resolvem sem labels:

| Pedido | Técnica | Neste notebook |
|---|---|---|
| "Quero agrupar clientes/casas parecidas para campanhas ou preços de referência" | *clustering* (K-Means, DBSCAN) | secções 4 a 6 |
| "Tenho 64 (ou 6000) colunas; quero ver e comprimir" | redução de dimensão (PCA) | secção 7 |
| "Quero detetar o que não encaixa em nenhum grupo" | ruído/*outliers* (DBSCAN) | secção 6 |

**Sucesso aqui não é uma métrica única.** Um agrupamento é bom se for *estável*, *interpretável* e *acionável* (alguém consegue fazer algo diferente por grupo). Vamos medir o que se pode medir, mas sem esquecer isto.""")
nb.md("""### 📖 Palavras que vão aparecer (mini-glossário)

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
nb.md(disc("O que é um \"bom\" agrupamento?",
           "Uma imobiliária pede-te para \"segmentar as casas\". Que perguntas farias antes de abrir o Python? Como saberias, daqui a um mês, que a segmentação foi útil?",
           """Sem alvo não há "accuracy". O critério tem de vir do uso: para que decisão serve cada segmento (preço de referência, tipo de anúncio, comercial a contactar)? Perguntas úteis: quantos grupos consegue a equipa gerir? Que variáveis podem (e não podem) entrar? Como se vai usar um grupo novo? Ao longo do notebook vais ver que o mesmo conjunto de dados dá resultados muito diferentes consoante escala, *k* e algoritmo, por isso o critério de utilidade (interpretável, estável, acionável) pesa mais do que qualquer número."""))

# ----------------------------------------------------------------------------- 2
nb.sec(2, "🔍 Data Understanding (compreensão dos dados)")
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
nb.code(r'''import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from src.plotting import apply_style, PALETTE, CYCLE, MARKERS, save_fig
from src.data import load_housing, HOUSING_NUMERIC

apply_style()
SEED = 42
FIG = ROOT / "assets" / "figures"

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


X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=SEED)
X_moons, y_moons = make_moons(n_samples=300, noise=0.07, random_state=SEED)
digits = load_digits()
print("blobs :", X_blobs.shape, "| moons:", X_moons.shape, "| digits:", digits.data.shape)''')
nb.code(r'''fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
for ax, X, y, title in ((axes[0], X_blobs, y_blobs, "4 grupos em bola (blobs)"),
                        (axes[1], X_moons, y_moons, "2 luas entrelaçadas (moons)")):
    for k in np.unique(y):
        ax.scatter(X[y == k, 0], X[y == k, 1], s=26, marker=MARKERS[k], color=CYCLE[k], edgecolor="white",
                   linewidth=0.4, label=f"grupo real {k}")
    ax.set(title=title, xlabel="$x_1$ (unidades arbitrárias)", ylabel="$x_2$ (unidades arbitrárias)")
    ax.legend(fontsize=10)
fig.suptitle("A verdade existe, mas o algoritmo não a vê: só recebe os pontos", x=0.01, ha="left", fontsize=12)

finish(fig, FIG / "03_data_overview.png"); plt.show()''')
nb.md("""### Os dígitos: uma imagem é uma linha de 64 números

Cada imagem tem 8×8 = 64 pixéis; cada pixel é um número de 0 (branco) a 16 (preto). Para o algoritmo, uma imagem é só uma lista de 64 números, ou seja, **um ponto num espaço com 64 dimensões**. As *labels* (0 a 9) existem, mas **escondemo-las** do algoritmo e só as usamos no fim para verificar.""")
nb.code(r'''fig, axes = plt.subplots(2, 5, figsize=(10, 4.4), layout="constrained")
for d, ax in enumerate(axes.ravel()):
    i = np.flatnonzero(digits.target == d)[0]
    ax.imshow(digits.images[i], cmap="gray_r"); ax.set_title(f"dígito {d}"); ax.axis("off")
fig.suptitle("Um exemplo real de cada dígito (8×8 pixéis)", x=0.01, ha="left")
finish(fig, FIG / "03_digits_samples.png"); plt.show()
print("A primeira imagem como números (0 = branco, 16 = preto):")
print(digits.images[0].astype(int))
print("labels do início:", digits.target[:10], "| imagens:", digits.images.shape, "-> como linhas de 64 números:", digits.data.shape)''')
nb.md("""### As casas de Windsor (o caso real)

**Ficha do dataset.** *O que é:* 546 casas vendidas em **Windsor (Ontário, Canadá)** entre julho e setembro de **1987**. *Fonte oficial:* arquivo do *Journal of Applied Econometrics* (Anglin & Gençay, 1996), descarregado no 1.º uso para `data/`. *Uma linha:* uma casa. *Preço:* dólares canadianos de 1987 (**CAD**). *Área do lote:* pés² (1 pé² ≈ 0,093 m²).

Neste notebook **não** há alvo: vamos agrupar as casas só pelas características físicas (5 colunas) e usar o preço **depois**, para perceber os grupos.""")
nb.code(r'''housing = load_housing()
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
columns_table''')
nb.code(r'''housing.head()''')
nb.code(r'''fig, axes = plt.subplots(3, 4, figsize=(13, 8), layout="constrained")
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
    ax.set_title(f"coluna {col}", fontsize=10)
fig.suptitle("Uma coluna por painel: casas pequenas e de poucos quartos dominam; preços e lotes são assimétricos", x=0.01, ha="left")
finish(fig, FIG / "03_housing_columns.png"); plt.show()''')
nb.md("""**Leitura:** quase todas as casas têm 2 a 4 quartos e 1 casa de banho; a maioria tem 1 ou 2 pisos. A área do lote e o preço têm uma "cauda" para a direita (poucas casas muito grandes e caras). Repara nas escalas: `lotsize` anda na casa dos milhares e `bedrooms` na casa das unidades; isto vai importar na secção 3.""")
nb.md(disc("O que é um grupo?",
           "Olha para as duas figuras. Na da direita, dirias que há 2 grupos ou 1? E se um algoritmo só souber medir \"distância ao centro do grupo\"?",
           """Um humano vê duas luas porque usa **continuidade** (pontos vizinhos encadeados). Um algoritmo baseado em centros (K-Means) só sabe "distância ao centro", o que pressupõe grupos compactos e redondos. Ou seja, a definição de "grupo" está *escondida na escolha do algoritmo*. Já vais ver (secção 6) o K-Means a cortar as luas ao meio, e o DBSCAN, que define grupo como "região densa e contínua", a acertar."""))

# ----------------------------------------------------------------------------- 3
nb.sec(3, "🧹 Data Preparation (preparação dos dados): a escala decide tudo")
nb.md("""Algoritmos de *clustering* medem **distâncias** (quão longe estão dois exemplos). Se uma coluna está em pés² (milhares) e outra em número de quartos (unidades), a distância é praticamente só pés². Vamos ver isto nas casas de Windsor, com as 5 *features* numéricas (`lotsize`, `bedrooms`, `bathrooms`, `stories`, `garage`). O preço **não entra** no agrupamento; usamo-lo depois, para interpretar. **Escalar** (*standardize*) = converter cada coluna para média 0 e desvio-padrão 1 (o desvio-padrão mede o quanto os valores se afastam da média), para que todas tenham o mesmo peso nas distâncias.""")
nb.code(r'''X_house = housing[HOUSING_NUMERIC].to_numpy(dtype=float)
print(housing[HOUSING_NUMERIC].describe().loc[["mean", "std", "min", "max"]].round(1))''')
nb.code(r'''km_raw = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(X_house)                 # sem escalar
km_scaled = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(StandardScaler().fit_transform(X_house))
km_lot = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(X_house[:, [0]])         # só lotsize

print("ARI(sem escala , só lotsize) =", round(adjusted_rand_score(km_raw.labels_, km_lot.labels_), 3))
print("ARI(com escala , só lotsize) =", round(adjusted_rand_score(km_scaled.labels_, km_lot.labels_), 3))''')
nb.md("""O **ARI** (*Adjusted Rand Index*) compara duas partições: 1 = idênticas, ≈ 0 = tão parecidas como o acaso. Sem escala, o K-Means com 5 colunas dá **exatamente** a mesma partição que usar só `lotsize` (ARI = 1,0): as outras 4 colunas foram ignoradas. Com `StandardScaler` (média 0, desvio-padrão 1) passam todas a contar.""")
nb.code(r'''fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
jitter = np.random.default_rng(SEED).uniform(-0.12, 0.12, len(housing))
for ax, labels, title in ((axes[0], km_raw.labels_, "Sem escala: só conta o tamanho do lote"),
                          (axes[1], km_scaled.labels_, "Com escala: todas as colunas contam")):
    for k in range(3):
        m = labels == k
        ax.scatter(housing.loc[m, "lotsize"], housing.loc[m, "bedrooms"] + jitter[m], s=22, marker=MARKERS[k],
                   color=CYCLE[k], edgecolor="white", linewidth=0.3, label=f"grupo {k}")
    ax.set(title=title, xlabel="área do lote (pés²)")
    ax.legend(fontsize=10)
axes[0].set_ylabel("quartos (com ruído vertical só para ver)")
finish(fig, FIG / "03_scale_effect.png"); plt.show()''')
nb.md(disc("Escalar ou não escalar?",
           "Os quartos são uma variável importante para quem compra. Sem escala, o K-Means ignorou-os por completo. É um \"erro\" do algoritmo ou da preparação dos dados? Haverá casos em que **não** quererias escalar?",
           """É um erro de **preparação**: o K-Means fez o que lhe pedimos (minimizar distâncias euclidianas), e a escala do `lotsize` é milhares de vezes maior. O ARI = 1,0 face ao agrupamento só por `lotsize` prova-o. Não escalar só faz sentido quando as unidades **já são comparáveis e a escala tem significado** (por exemplo, coordenadas geográficas em km). Escalar é também uma decisão de negócio: dar o mesmo peso a todas as colunas é uma escolha, e pode usar-se pesos explícitos."""))

nb.md("""### Segmentação de casas (curta, com interpretação)

Com as colunas escaladas e *k* = 3, o preço (que **não** foi usado) serve para dar nome aos segmentos. Valores em dólares canadianos de 1987 (CAD):""")
nb.code(r'''profile = (housing.assign(cluster=km_scaled.labels_)
           .groupby("cluster")
           .agg(casas=("price", "size"), preco_mediano_CAD=("price", "median"), lote_mediano_pes2=("lotsize", "median"),
                quartos=("bedrooms", "median"), casas_de_banho=("bathrooms", "median"),
                pisos=("stories", "median"), garagem=("garage", "median"))
           .sort_values("preco_mediano_CAD"))
profile.index = ["A", "B", "C"]
profile''')
nb.md("""**Leitura (por ordem de preço mediano):** **A**, o maior grupo, são casas de preço mais baixo (≈ 52 000 CAD), com lotes pequenos e sem garagem; **B** são casas de 1 piso com garagem dupla e preço intermédio (≈ 70 000 CAD); **C** são as maiores (4 quartos, 2 casas de banho) e as mais caras (≈ 87 000 CAD). O preço **nunca** entrou no agrupamento e ainda assim os grupos separam-no: sinal de que a estrutura é real. Mas cuidado, **os nomes A/B/C somos nós a inventá-los**: o algoritmo só devolveu números 0, 1, 2.""")

# ----------------------------------------------------------------------------- 4
nb.sec(4, "🤖 Modeling (modelação) A — K-Means passo a passo")
nb.md("""## 🧠 Como funciona: K-Means

**Ideia simples.** Imagina que queres abrir 4 lojas para servir 300 casas espalhadas pelo mapa. Primeiro colocas as 4 lojas ao acaso; cada casa vai à mais próxima; depois mudas cada loja para o **centro** das casas que a escolheram; repetes. As lojas vão-se arrumando, e cada "zona de influência" é um *cluster*.

**Passos** (o algoritmo repete 1 e 2 até nada mudar):

1. **Atribuir:** cada ponto vai para o **centroide** (a "loja") mais próximo;
2. **Atualizar:** cada centroide passa a ser a **média** dos pontos que lhe foram atribuídos.

**Quando usar:** grupos compactos e mais ou menos redondos, com uma ideia de quantos são (*k*). **Cuidados:** é preciso escolher *k*; depende da escala; só vê bolas (secção 6); pode ficar preso numa solução má (vê-se abaixo). O que se minimiza é a **inércia** = soma dos quadrados das distâncias de cada ponto ao seu centroide (quanto menor, mais compactos os grupos).""")
nb.md("""O K-Means repete dois passos até nada mudar:

1. **Atribuir:** cada ponto vai para o centroide mais próximo;
2. **Atualizar:** cada centroide passa a ser a **média** dos pontos que lhe foram atribuídos.

Escrevemos isto em NumPy, com os 4 centroides iniciais escolhidos ao acaso entre os pontos (`seed=2`; mais abaixo vês o que acontece com `seed=0`), e mostramos as primeiras 3 iterações.""")
nb.code(r'''def assign(X, centroids):
    """Passo 1: índice do centroide mais próximo de cada ponto (distância euclidiana)."""
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)   # (n, k)
    return dists.argmin(axis=1)

def update(X, labels, centroids):
    """Passo 2: cada centroide passa a ser a média dos seus pontos (se ficar vazio, mantém-se)."""
    new = centroids.copy()
    for j in range(len(centroids)):
        if np.any(labels == j):
            new[j] = X[labels == j].mean(axis=0)
    return new

def inertia(X, labels, centroids):
    """Soma dos quadrados das distâncias de cada ponto ao seu centroide."""
    return float(((X - centroids[labels]) ** 2).sum())

def init_centroids(seed, k=4):
    """k pontos escolhidos ao acaso como centroides iniciais."""
    return X_blobs[np.random.default_rng(seed).choice(len(X_blobs), size=k, replace=False)].copy()

def run_kmeans(X, centroids, max_iter=30):
    """Repete atribuir/atualizar até os centroides pararem; devolve o histórico (centroides, labels, inércia)."""
    hist, cur = [], centroids.copy()
    for _ in range(max_iter):
        labels = assign(X, cur)
        hist.append((cur.copy(), labels, inertia(X, labels, cur)))
        new = update(X, labels, cur)
        if np.allclose(new, cur):
            break
        cur = new
    return hist

centroids = init_centroids(seed=2)
print("shapes: X", X_blobs.shape, "| centroides", centroids.shape)''')
nb.code(r'''history = run_kmeans(X_blobs, centroids)
print(f"convergiu em {len(history)} iterações: os centroides já não se mexem")
for i, (_, _, inert) in enumerate(history, 1):
    print(f"iteração {i}: inércia = {inert:8.1f}")''')
nb.code(r'''# O mecanismo, desenhado: (1) cada ponto liga ao centroide mais próximo; (2) cada centroide salta para a média dos seus pontos
cents0, labels0, _ = history[0]
cents1 = history[1][0]                                     # centroides depois do passo "atualizar"
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True, sharey=True, layout="constrained")
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
axes[0].set(title="Passo 1 · atribuir ao mais próximo", xlabel="$x_1$ (u. arbitrárias)", ylabel="$x_2$ (u. arbitrárias)")
axes[1].set(title="Passo 2 · centroide salta para a média", xlabel="$x_1$ (u. arbitrárias)")
finish(fig, FIG / "03_kmeans_mechanism.png"); plt.show()''')
nb.code(r'''show = [0, 1, 2]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True, layout="constrained")
for ax, i in zip(axes, show):
    cents, labels, inert = history[i]
    for j in range(4):
        m = labels == j
        ax.scatter(X_blobs[m, 0], X_blobs[m, 1], s=18, marker=MARKERS[j], color=CYCLE[j], alpha=0.75,
                   edgecolor="white", linewidth=0.3)
        ax.scatter(*cents[j], marker="*", s=330, color=CYCLE[j], edgecolor="black", linewidth=1.2, zorder=5)
    ax.set(title=f"Iteração {i + 1}: inércia {fmt_pt(inert)}", xlabel="$x_1$ (u. arbitrárias)")
axes[0].set_ylabel("$x_2$ (u. arbitrárias)")
fig.suptitle("Atribuir → atualizar: os centroides (★) vão para o centro dos grupos", x=0.01, ha="left")
finish(fig, FIG / "03_kmeans_steps.png"); plt.show()''')
nb.md("""Agora o mesmo com `sklearn.cluster.KMeans`. A diferença principal é `n_init`: corre o algoritmo **várias vezes** com inicializações diferentes (e usa `k-means++`, que espalha os centroides iniciais) e fica com a de menor inércia.""")
nb.code(r'''km = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit(X_blobs)
print(f"sklearn: inércia = {km.inertia_:.1f} | ARI vs grupos reais = {adjusted_rand_score(y_blobs, km.labels_):.3f}")
print(f"à mão  : inércia = {history[-1][2]:.1f} (seed=2, convergiu)")
stuck = run_kmeans(X_blobs, init_centroids(seed=0))
print(f"à mão  : inércia = {stuck[-1][2]:.1f} com seed=0 ({len(stuck)} iterações): mínimo local, dois grupos reais fundidos")

# Mínimos locais: com n_init=1 a inicialização aleatória pode ficar presa numa solução pior
single = [KMeans(4, init="random", n_init=1, random_state=s).fit(X_blobs).inertia_ for s in range(10)]
print("inércia com n_init=1 (10 seeds):", np.round(sorted(single)).astype(int))''')
nb.md("""> ⚠️ **Cuidado:** o K-Means só garante um **mínimo local**. Com uma única inicialização aleatória, algumas seeds podem ficar numa inércia muito pior (acima). Por isso `n_init=10` e `k-means++` são o padrão: o resultado deixa de depender da sorte.

> 🤖 **Ligação à IA:** o mesmo ciclo "atribuir → atualizar" aparece em quantização de vetores, compressão de imagem por cores e na construção de dicionários visuais para modelos de visão.""")
nb.md(disc("Modelação",
           "O K-Means converge sempre. Isso quer dizer que a solução é a *melhor possível*? Que informação extra tens de dar ao algoritmo (e que ele não consegue descobrir sozinho)?",
           """Converge sempre (a inércia nunca aumenta), mas só para um **mínimo local**: com `n_init=1` e inicialização aleatória, as 10 seeds acima dão inércias diferentes (≈ 362 nas melhores, ≈ 1 887 ou mesmo ≈ 8 256 nas piores), e à mão a `seed=0` também ficou presa em ≈ 1 887. E o algoritmo não descobre o **número de grupos** *k*: tem de ser dado por nós, o que leva à secção seguinte."""))

# ----------------------------------------------------------------------------- 5
nb.sec(5, "🤖 Modeling B — escolher o número de grupos")
nb.code(r'''ks = range(2, 9)
inertias, sils = [], []
for k in ks:
    m = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_blobs)
    inertias.append(m.inertia_)
    sils.append(silhouette_score(X_blobs, m.labels_))
pd.DataFrame({"k": list(ks), "inércia": np.round(inertias, 0), "silhouette": np.round(sils, 3)}).set_index("k").T''')
nb.code(r'''fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
axes[0].plot(list(ks), inertias, marker="o", color=PALETTE["blue"])
axes[0].axvline(4, color=PALETTE["grey"], ls="--")
axes[0].annotate("cotovelo: k = 4", (4, inertias[2]), xytext=(5, inertias[0] * 0.7),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["grey"]))
axes[0].set(title="Cotovelo: o ganho cai muito depois de k = 4", xlabel="número de grupos k",
            ylabel="inércia (u. arbitrárias²)")
axes[1].plot(list(ks), sils, marker="s", color=PALETTE["orange"])
axes[1].plot(4, sils[2], marker="*", ms=16, color=PALETTE["red"], zorder=5)
axes[1].set(title="Silhouette: máximo em k = 4", xlabel="número de grupos k", ylabel="silhouette média (−1 a 1)")
finish(fig, FIG / "03_choose_k.png"); plt.show()''')
nb.md(r"""- **Cotovelo:** a inércia desce sempre com *k* (no limite, 1 ponto por grupo dá 0). Procuramos o ponto onde o ganho passa a ser pequeno. Aqui: depois de *k* = 4.
- **Silhouette:** para cada ponto compara a distância média ao seu grupo ($a$) com a distância média ao grupo vizinho mais próximo ($b$): $s = (b - a)/\max(a, b)$. Perto de 1: bem arrumado; perto de 0: na fronteira; negativo: provavelmente no grupo errado.""")
nb.md(disc("Dois critérios, uma decisão",
           "Neste conjunto o cotovelo e o silhouette concordam. Num conjunto real do teu trabalho eles discordam (cotovelo sugere 3, silhouette 5). Como decidias?",
           """Aqui ambos apontam para *k* = 4 (a inércia passa de ≈ 1 919 em *k* = 3 para ≈ 362 em *k* = 4, e depois só desce para ≈ 329; o silhouette é máximo, ≈ 0,83, em *k* = 4), o que coincide com os 4 grupos que geramos. Em dados reais é raro ser tão limpo. O cotovelo é visual e subjetivo; o silhouette favorece grupos compactos. Decide-se com o **uso**: quantos segmentos consegue a equipa gerir, se os grupos são **estáveis** com outras seeds ou sub-amostras e se se conseguem **interpretar**."""))

# ----------------------------------------------------------------------------- 6
nb.sec(6, "🤖 Modeling C — DBSCAN: grupos com forma livre")
nb.md("""## 🧠 Como funciona: DBSCAN

**Ideia simples.** Pensa numa festa: um grupo de pessoas forma-se onde há **muita gente junta**; quem está sozinho num canto não pertence a nenhum grupo. O DBSCAN procura "multidões" e deixa os isolados de fora.

Dois parâmetros definem "juntos": o **raio** `eps` (ε, a distância máxima para dois pontos serem vizinhos) e `min_samples` (quantos vizinhos, incluindo o próprio ponto, para haver multidão). Cada ponto fica numa de 3 categorias:

- **núcleo** (*core*): tem pelo menos `min_samples` pontos dentro do raio `eps`. Os núcleos vizinhos ligam-se e crescem o grupo;
- **fronteira** (*border*): está no raio de um núcleo, mas ele próprio não tem vizinhos que cheguem;
- **ruído** (*noise*): não é núcleo nem está perto de um (label −1).

**Quando usar:** grupos de forma livre, ruído a detetar, *k* desconhecido. **Cuidados:** `eps` e `min_samples` têm de ser escolhidos; densidades muito diferentes; escala.""")
nb.code(r'''# Mecanismo: luas + 3 pontos isolados que acrescentamos de propósito
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

fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
ax.scatter(*demo_X[is_core].T, s=18, color=PALETTE["blue"], marker="o", label="núcleo", alpha=0.6)
ax.scatter(*demo_X[is_border].T, s=45, color=PALETTE["orange"], marker="s", label="fronteira", edgecolor="black", zorder=3)
ax.scatter(*demo_X[is_noise].T, s=60, color="black", marker="x", label="ruído", zorder=3)
notes = []
for i, name, colour in ((i_core, "núcleo", PALETTE["blue"]), (i_border, "fronteira", PALETTE["orange"]), (i_noise, "ruído", "black")):
    ax.add_patch(plt.Circle(demo_X[i], EPS, fill=False, ec=colour, lw=2, ls="--"))
    n_in = int((np.linalg.norm(demo_X - demo_X[i], axis=1) <= EPS).sum())
    notes.append(f"círculo {name}: {n_in} {'ponto' if n_in == 1 else 'pontos'} dentro do raio (mínimo {MIN_S})")
ax.text(0.98, 0.98, chr(10).join(notes), transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9))
ax.set(title=f"Raio ε = {EPS}, min_samples = {MIN_S}: núcleo, fronteira, ruído", xlabel="$x_1$ (u. arbitrárias)", ylabel="$x_2$ (u. arbitrárias)", aspect="equal")
ax.legend(loc="lower left"); ax.set_ylim(-1.1, 1.9)
finish(fig, FIG / "03_dbscan_mechanism.png"); plt.show()''')
nb.md("""Repara: um ponto é **núcleo** se o círculo tracejado apanhar pelo menos `min_samples` pontos (ele próprio incluído). O de **fronteira** está dentro do círculo de um núcleo mas o seu próprio círculo está quase vazio. O de **ruído** tem o círculo vazio e ninguém o "adota".""")
nb.md("""O **DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*) define grupo como uma região **densa**: um ponto é *núcleo* se tiver pelo menos `min_samples` vizinhos a uma distância `eps`; grupos são cadeias de núcleos ligados; o que sobra é **ruído** (label −1). Não precisa de *k*, mas precisa de `eps`.

Primeiro, o K-Means (que procura bolas) nas luas:""")
nb.code(r'''km_moons = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(X_moons)
db_moons = DBSCAN(eps=0.2, min_samples=5).fit(X_moons)

def summarize(name, labels, X, y):
    n_clusters = len(set(labels) - {-1})
    sil = silhouette_score(X, labels) if n_clusters > 1 else float("nan")
    print(f"{name:<18} clusters={n_clusters}  ruído={int((labels == -1).sum()):>3}  "
          f"silhouette={sil:.3f}  ARI vs verdade={adjusted_rand_score(y, labels):.3f}")

summarize("K-Means (k=2)", km_moons.labels_, X_moons, y_moons)
summarize("DBSCAN (eps=0.2)", db_moons.labels_, X_moons, y_moons)''')
nb.code(r'''fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True, layout="constrained")
for ax, labels, title in ((axes[0], km_moons.labels_, "K-Means corta as luas ao meio"),
                          (axes[1], db_moons.labels_, "DBSCAN segue a forma de cada lua")):
    for k in sorted(set(labels)):
        m = labels == k
        if k == -1:
            ax.scatter(X_moons[m, 0], X_moons[m, 1], marker="x", s=50, color="black", label="ruído")
        else:
            ax.scatter(X_moons[m, 0], X_moons[m, 1], s=26, marker=MARKERS[k], color=CYCLE[k], edgecolor="white",
                       linewidth=0.4, label=f"grupo {k}")
    ax.set(title=title, xlabel="$x_1$ (u. arbitrárias)"); ax.legend(fontsize=10)
axes[0].set_ylabel("$x_2$ (u. arbitrárias)")
finish(fig, FIG / "03_moons_kmeans_vs_dbscan.png"); plt.show()''')
nb.md("""### O parâmetro `eps`: pequeno demais fragmenta, grande demais junta""")
nb.code(r'''eps_values = [0.05, 0.1, 0.2, 0.6]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True, sharey=True, layout="constrained")
for ax, eps in zip(axes, eps_values):
    labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X_moons)
    for k in sorted(set(labels)):
        m = labels == k
        if k == -1:
            ax.scatter(X_moons[m, 0], X_moons[m, 1], marker="x", s=36, color="black")
        else:
            ax.scatter(X_moons[m, 0], X_moons[m, 1], s=18, marker=MARKERS[k % 7], color=CYCLE[k % 7],
                       edgecolor="white", linewidth=0.3)
    ax.set(title=f"ε = {eps}\n{len(set(labels) - {-1})} grupos, {(labels == -1).sum()} ruído", xlabel="$x_1$")
axes[0].set_ylabel("$x_2$ (u. arbitrárias)")
finish(fig, FIG / "03_dbscan_eps.png"); plt.show()''')
nb.md("""> 💡 **Dica:** para escolher `eps` no mundo real, desenha-se a distância de cada ponto ao seu *k*-ésimo vizinho (gráfico *k-distance*, `NearestNeighbors`) e procura-se o "cotovelo" da curva.

> ⚠️ **Cuidado:** o DBSCAN também usa distâncias, por isso a **escala** continua a importar. E tem dificuldade com grupos de densidades muito diferentes (um `eps` único não serve para todos).""")
nb.md(disc("Quando é que o DBSCAN é a escolha certa?",
           "Vê os 4 painéis de `eps`. Que erro acontece com `eps` pequeno? E com `eps` grande? E o que fazer com os pontos marcados como ruído numa aplicação real (por exemplo, transações bancárias)?",
           """Com `eps` pequeno (0,05 e 0,1) as luas estilhaçam-se em vários micro-grupos e há muito ruído (com 0,1 são 11 grupos e 47 pontos de ruído); com `eps` = 0,2 aparecem as 2 luas sem ruído; com `eps` grande as luas fundem-se num único grupo (vê o último painel). O ruído não é "lixo": numa aplicação de fraude ou manutenção, os pontos que não pertencem a nenhum grupo denso são precisamente os candidatos a **anomalias** para revisão humana."""))

# ----------------------------------------------------------------------------- 7
nb.sec(7, "🤖 Modeling D — PCA: comprimir sem perder o essencial")
nb.md("""## 🧠 Como funciona: PCA

**Ideia simples.** Uma nuvem de pontos 3D pode ter forma de "panqueca": quase tudo acontece em 2 direções. A PCA (*Principal Component Analysis*, análise de componentes principais) encontra **as direções em que os dados mais variam** (a **variância** é o quanto os valores se espalham). A 1.ª componente é a direção de maior espalhamento; a 2.ª, perpendicular, a seguinte; e assim por diante. Guardar só as primeiras componentes é **comprimir** com pouca perda.

**Passos:** (1) centrar os dados na média; (2) achar a direção de maior variância; (3) achar a seguinte, perpendicular às anteriores; (4) projetar cada ponto nessas direções (as novas "coordenadas"). Não precisa de *labels*.

**Quando usar:** reduzir dimensões, visualizar, limpar ruído. **Cuidados:** é linear; mede variância, não "importância para a tarefa"; depende da escala das colunas.""")
nb.code(r'''# Mecanismo: nuvem 2D alongada e as suas duas componentes principais
rng = np.random.default_rng(SEED)
cloud = rng.normal(size=(200, 2)) * [3.0, 0.8]                  # espalhada ao longo do eixo x
theta = np.deg2rad(30)
cloud = cloud @ np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
pca2 = PCA(n_components=2).fit(cloud)
centre = pca2.mean_
fig, ax = plt.subplots(figsize=(7.5, 5.2), layout="constrained")
ax.scatter(cloud[:, 0], cloud[:, 1], s=16, color=PALETTE["grey"], alpha=0.5, label="dados")
for k, (colour, name) in enumerate(((PALETTE["red"], "1.ª componente"), (PALETTE["blue"], "2.ª componente"))):
    vec = pca2.components_[k] * 2 * np.sqrt(pca2.explained_variance_[k])
    ax.annotate("", xy=centre + vec, xytext=centre, arrowprops=dict(arrowstyle="-|>", color=colour, lw=3))
    ax.text(*(centre + vec * (1.12 if k == 0 else 1.3)), f"{name}" + chr(10) + f"({pca2.explained_variance_ratio_[k]:.0%} da variância)", color=colour, fontsize=10, fontweight="bold", ha="center")
# projeção de alguns pontos na 1.ª componente
for i in range(0, 200, 25):
    proj = centre + (cloud[i] - centre) @ pca2.components_[0] * pca2.components_[0]
    ax.plot([cloud[i, 0], proj[0]], [cloud[i, 1], proj[1]], color=PALETTE["red"], lw=0.8, ls=":")
    ax.scatter(*proj, s=30, color=PALETTE["red"], marker="D", zorder=4)
ax.set(title="A 1.ª componente é a direção de maior espalhamento", xlabel="$x_1$ (u. arbitrárias)", ylabel="$x_2$ (u. arbitrárias)", aspect="equal")
ax.legend(loc="lower right"); ax.set_xlim(-8, 10); ax.set_ylim(-5, 6.5); finish(fig, FIG / "03_pca_mechanism.png"); plt.show()''')
nb.md("""Os losangos vermelhos são a **projeção** dos pontos na 1.ª componente: ficamos com **1 número por ponto em vez de 2** e perdemos só a pequena distância perpendicular (a variância da 2.ª componente). É isto que vamos fazer com os dígitos: de 64 números para 2, 10 ou 29.

Agora nos dígitos (as imagens já foram vistas na secção 2):""")
nb.code(r'''pca = PCA(random_state=SEED).fit(digits.data)
cum = np.cumsum(pca.explained_variance_ratio_)
k95 = int(np.searchsorted(cum, 0.95) + 1)
print(f"2 componentes explicam {cum[1]:.1%} | 10 explicam {cum[9]:.1%} | 30 explicam {cum[29]:.1%}")
print(f"Para 95 % da variância bastam {k95} componentes (de 64)")

fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
ax.plot(range(1, 65), cum * 100, marker="o", ms=4, color=PALETTE["blue"])
ax.axhline(95, color=PALETTE["grey"], ls="--"); ax.axvline(k95, color=PALETTE["grey"], ls="--")
ax.annotate(f"{k95} componentes → 95 %", (k95, 95), xytext=(k95 + 8, 80), arrowprops=dict(arrowstyle="->"))
ax.set(title=f"{k95} das 64 dimensões guardam 95 % da variância", xlabel="número de componentes",
       ylabel="variância explicada acumulada (%)")
finish(fig, FIG / "03_pca_variance.png"); plt.show()''')
nb.code(r'''Z = PCA(n_components=2, random_state=SEED).fit_transform(digits.data)
palette10 = plt.cm.tab10(np.arange(10))
markers10 = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]

fig, ax = plt.subplots(figsize=(7, 5.2), layout="constrained")
for d in range(10):
    m = digits.target == d
    ax.scatter(Z[m, 0], Z[m, 1], s=22, marker=markers10[d], color=palette10[d], alpha=0.75, edgecolor="white",
               linewidth=0.3, label=str(d))
    ax.text(*Z[m].mean(axis=0), str(d), fontsize=15, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="black", alpha=0.85))
ax.set(title="Em 2D separam-se alguns dígitos (0, 4, 6); outros misturam-se",
       xlabel="componente principal 1 (unidades de pixel)", ylabel="componente principal 2 (unidades de pixel)")
ax.legend(title="dígito", ncol=2, fontsize=10, loc="upper right")
finish(fig); fig.savefig(FIG / "03_pca_digits_2d.png", dpi=100); plt.show()''')
nb.md("""### Reconstrução: quanto se perde ao comprimir?

`inverse_transform` volta a projetar as componentes mantidas para o espaço de 64 pixéis. Quanto menos componentes, mais desfocada a imagem.""")
nb.code(r'''ks_rec = [2, 5, 10, 20, 40]
sample_idx = [0, 1, 2, 3, 4]
fig, axes = plt.subplots(len(sample_idx), len(ks_rec) + 1, figsize=(9, 7.2))
for r, idx in enumerate(sample_idx):
    axes[r, 0].imshow(digits.images[idx], cmap="gray_r"); axes[r, 0].axis("off")
    if r == 0: axes[r, 0].set_title("original (64)", fontsize=10)
    for c, k in enumerate(ks_rec, 1):
        p = PCA(n_components=k, random_state=SEED).fit(digits.data)
        rec = p.inverse_transform(p.transform(digits.data[[idx]]))
        axes[r, c].imshow(rec.reshape(8, 8), cmap="gray_r"); axes[r, c].axis("off")
        if r == 0: axes[r, c].set_title(f"k = {k}", fontsize=10)
fig.suptitle("Com 20 a 40 componentes o dígito volta a ser legível", x=0.01, ha="left", fontsize=11)
finish(fig, FIG / "03_pca_reconstruction.png"); plt.show()''')
nb.md(disc("Compressão e visualização",
           "Guardar 29 números em vez de 64 por imagem (95 % da variância) parece um bom negócio. Em que situações **não** é? E o gráfico 2D: podes confiar que dígitos próximos são realmente parecidos?",
           """A PCA é **linear** e guarda a variância, não necessariamente o que interessa para uma tarefa (um detalhe pequeno que distingue um 3 de um 8 pode ter pouca variância e ser descartado). No gráfico 2D só ≈ 29 % da variância (2 componentes) está representada, por isso a mistura de dígitos como 5, 8 e 9 pode ser só efeito de projeção: em 64 dimensões podem estar separados. Para uma visualização não linear existem o t-SNE e o UMAP, mais caros e mais difíceis de interpretar."""))

# ----------------------------------------------------------------------------- 8
nb.sec(8, "📏 Evaluation (avaliação): como avaliar sem labels?")
nb.md("""Duas famílias de métricas:

- **Internas** (só usam os dados): *silhouette*, inércia. Medem compacidade e separação.
- **Externas** (precisam de labels escondidas): ARI, que só podemos calcular porque os conjuntos de brinquedo têm verdade conhecida. **No mundo real normalmente não existe.**

O ponto delicado: uma métrica interna mede *uma noção* de bom agrupamento (bolas compactas), que pode não ser a do problema.""")
nb.code(r'''rows = []
for name, labels, X, y in (("luas · K-Means k=2", km_moons.labels_, X_moons, y_moons),
                           ("luas · DBSCAN eps=0.2", db_moons.labels_, X_moons, y_moons)):
    rows.append({"modelo": name, "silhouette (interna)": round(silhouette_score(X, labels), 3),
                 "ARI (externa)": round(adjusted_rand_score(y, labels), 3)})
pd.DataFrame(rows).set_index("modelo")''')
nb.md("""**O silhouette dá melhor nota ao K-Means errado** (≈ 0,48 contra ≈ 0,33 do DBSCAN), enquanto o ARI mostra a verdade: o K-Means acerta pouco (ARI ≈ 0,25) e o DBSCAN acerta tudo (ARI = 1,0). O silhouette premeia grupos compactos e redondos; duas luas bem separadas mas encurvadas pontuam mal.""")
nb.code(r'''# Digits: o K-Means nos pixéis vs nas componentes principais (ARI contra os dígitos reais)
rows = []
for k in (2, 5, 10, 20, 64):
    Zk = PCA(n_components=k, random_state=SEED).fit_transform(digits.data) if k < 64 else digits.data
    labels = KMeans(n_clusters=10, n_init=10, random_state=SEED).fit_predict(Zk)
    rows.append({"dimensões usadas": k, "ARI vs dígito real": round(adjusted_rand_score(digits.target, labels), 3)})
pd.DataFrame(rows).set_index("dimensões usadas").T''')
nb.md("""Sem usar nunca a label, o K-Means encontra um agrupamento com ARI ≈ 0,67 nos 64 pixéis; só com 2 componentes cai para ≈ 0,39 (informação perdida), com 5 sobe para ≈ 0,56 e a partir de 10 componentes (≈ 0,65) fica quase ao nível dos 64 pixéis, com menos custo.""")
nb.md(disc("Avaliação",
           "O silhouette escolheu o K-Means nas luas, o ARI escolheu o DBSCAN. Num problema real não tens o ARI. Como evitas ser enganado por uma métrica interna?",
           """Nunca confiar numa só métrica: combinar (1) **visualização** (PCA/2D), (2) **estabilidade** (repetir com outras seeds ou sub-amostras, e ver se os grupos persistem), (3) **validação externa indireta** (os grupos separam uma variável que **não** usámos? nas casas, o preço), e (4) **utilidade** para a decisão. As luas mostram porquê: o silhouette = 0,48 do K-Means errado é maior que o 0,33 do DBSCAN certo."""))

# ----------------------------------------------------------------------------- 9
nb.sec(9, "🚀 Deployment (implementação) e limites")
nb.md("""Pôr *clustering* em produção levanta perguntas que a supervisionada não levanta:

- **Atribuir casos novos:** `KMeans.predict` funciona; o DBSCAN **não tem** `predict` (só `fit_predict`): terias de refazer ou usar um classificador por cima.
- **Estabilidade:** com dados novos os grupos mudam; os rótulos 0, 1, 2 também (o "cluster 0" de hoje pode ser o "cluster 2" de amanhã).
- **Deriva** (*drift*): se a população muda (casas de outro bairro), os grupos deixam de fazer sentido.
- **Ética:** segmentar pessoas pode discriminar sem ninguém ter decidido isso; revê que *features* entram.""")
nb.md(disc("Do protótipo à produção",
           "A imobiliária quer usar os 3 segmentos de casas para etiquetar anúncios novos todos os meses. Que riscos vês, e que verificações automáticas porias?",
           """Riscos: (a) os rótulos mudam entre execuções, pelo que se deve **fixar o modelo** (guardar o *pipeline* com `StandardScaler` + `KMeans` e aplicar `predict`, em vez de reajustar); (b) *drift*: monitorizar a distância média ao centroide e a proporção de casas por segmento; (c) 1987 não é hoje, ou seja, os segmentos aprendidos com Windsor-1987 não se transferem; (d) o nome "segmento barato" pode estigmatizar um bairro. Verificações: alertas quando a distribuição dos segmentos ou a inércia por ponto sair de um intervalo."""))

# ----------------------------------------------------------------------------- 10
nb.sec(10, "✏️ Exercícios")
add_exercise(nb, 1, "O melhor k para as casas", 1,
             "Com as 5 colunas **escaladas** das casas, calcula o silhouette para *k* de 2 a 6 e diz qual *k* escolhias.",
             "Xs = StandardScaler().fit_transform(X_house)\n# TODO: para k em 2..6, ajusta KMeans(k, n_init=10, random_state=SEED) e imprime o silhouette\n",
             "O silhouette é em geral mais alto para *k* pequeno em dados reais sem separação clara, e o valor é baixo (< 0,4): os segmentos são contínuos, não ilhas. Escolher entre *k* = 2 e 3 deve ser feito pela utilidade e pela interpretação dos perfis.",
             "for k in range(2, 7):\n    lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(Xs)\n    print(k, round(silhouette_score(Xs, lab), 3))")
add_exercise(nb, 2, "min_samples no DBSCAN", 2,
             "Nas luas, fixa `eps=0.2` e varia `min_samples` em 3, 5, 10, 20, 40. Quantos clusters e quantos pontos de ruído aparecem em cada caso? O que acontece quando `min_samples` é muito grande?",
             "# TODO\n",
             "`min_samples` grande exige vizinhanças mais densas: mais pontos passam a ruído e, a partir de certo valor, os clusters desfazem-se ou desaparecem. Pequeno: aceita tudo e tende a juntar grupos. `eps` e `min_samples` têm de ser afinados em conjunto.",
             "for ms in (3, 5, 10, 20, 40):\n    lab = DBSCAN(eps=0.2, min_samples=ms).fit_predict(X_moons)\n    print(ms, len(set(lab) - {-1}), 'clusters', (lab == -1).sum(), 'ruído')")
add_exercise(nb, 3, "Erro de reconstrução vs componentes", 3,
             "Para k = 1, 2, 5, 10, 20, 30, 40, 63, calcula o erro quadrático médio de reconstrução de `digits.data` e desenha-o (eixo y em escala log). Em que k a curva \"achata\"? Relaciona com a variância explicada acumulada.",
             "# TODO: PCA(k).fit(...), inverse_transform(transform(...)), média de (original - reconstruído)**2\n",
             "O erro diminui monotonamente e é a variância que fica por explicar (a soma das variâncias das componentes descartadas, por pixel). A curva desce depressa até ≈ 30 componentes, quando a variância acumulada já passa os 95 %, e depois ganha-se pouco.",
             "ks = [1, 2, 5, 10, 20, 30, 40, 63]\nerrs = []\nfor k in ks:\n    p = PCA(k, random_state=SEED).fit(digits.data)\n    rec = p.inverse_transform(p.transform(digits.data))\n    errs.append(((digits.data - rec) ** 2).mean())\nfig, ax = plt.subplots()\nax.semilogy(ks, errs, marker='o')\nax.set(xlabel='componentes', ylabel='erro quadrático médio (pixel²)', title='O erro de reconstrução desce com k')\nplt.show()")

# ----------------------------------------------------------------------------- 11
nb.sec(11, "📌 Resumo")
nb.md("""| Ideia | Em uma linha |
|---|---|
| **K-Means** | atribuir ao centroide mais próximo ↔ atualizar centroides para a média; converge para um mínimo local (`n_init`, `k-means++`) |
| **Escala** | distâncias dependem das unidades; sem `StandardScaler`, o `lotsize` decidiu sozinho (ARI = 1,0) |
| **Escolher k** | cotovelo (inércia) + silhouette + estabilidade + utilidade; nenhum é infalível |
| **DBSCAN** | grupos = regiões densas; `eps` e `min_samples`; devolve ruído (−1); sem `predict` |
| **PCA** | eixos de máxima variância; 29 componentes guardam 95 % dos dígitos; boa para ver e comprimir |
| **Avaliar sem labels** | silhouette favorece bolas (0,48 K-Means errado > 0,33 DBSCAN certo nas luas); combinar métricas, visualização e utilidade |
| **CRISP-DM** | os grupos só valem se forem interpretáveis, estáveis e acionáveis |

**Checklist:** ☐ escalei? ☐ tentei vários *k* e seeds? ☐ a forma dos grupos é redonda? ☐ interpretei com variáveis que não usei? ☐ sei como atribuir casos novos?""")

# ----------------------------------------------------------------------------- 12
nb.sec(12, "📚 Para saber mais")
nb.md("""- scikit-learn User Guide, *Clustering*: <https://scikit-learn.org/stable/modules/clustering.html> (K-Means, DBSCAN, métricas de avaliação).
- scikit-learn User Guide, *PCA*: <https://scikit-learn.org/stable/modules/decomposition.html#pca>.
- Chapman et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS/NCR/DaimlerChrysler.
- Dados das casas: Anglin & Gençay (1996), *Journal of Applied Econometrics* 11(6):633–648 (pacote R AER).
- Extensões (só menção): Gaussian Mixtures (clusters probabilísticos), clustering hierárquico, t-SNE/UMAP, autoencoders (PCA não linear).""")
nb.foot("03")

out = Path(__file__).resolve().parents[2] / "02_unsupervised_learning" / "notebooks" / FILE
out.parent.mkdir(parents=True, exist_ok=True)
nb.save(out)
