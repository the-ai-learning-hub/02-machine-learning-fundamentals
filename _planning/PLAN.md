# PLAN · Lição 02 · Machine Learning: fundamentos

Briefing único para quem implementa (deck e notebooks). Não é entregue ao aluno; é apagado na limpeza final.

## 1. Identidade

- **Lição:** `Lição 02 · Machine Learning: fundamentos` (repo `the-ai-learning-hub/02-machine-learning-fundamentals`).
- **Autores:** Diogo Freitas & João Francisco Botas · Licença MIT.
- **Slides (GitHub Pages):** `https://the-ai-learning-hub.github.io/02-machine-learning-fundamentals/` (+ `#<id>`).
- **Colab:** `https://colab.research.google.com/github/the-ai-learning-hub/02-machine-learning-fundamentals/blob/main/<pasta>/notebooks/<nb>.ipynb`.
- **Idioma de tudo o que o aluno vê: PT-PT** (não PT-BR: "ficheiro", "ecrã", "utilizador", "equipa", "registo", "treinar", "previsão", "regressão", "aprendizagem"). Termos técnicos consagrados em inglês (*overfitting*, *feature*, *pipeline*, *threshold*, *batch*…) em itálico com explicação na primeira ocorrência. Código, variáveis e funções em inglês.
- **Fio condutor:** CRISP-DM (6 fases: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment). Nomes das fases em inglês + tradução PT-PT entre parênteses na primeira vez (Compreensão do negócio, Compreensão dos dados, Preparação dos dados, Modelação, Avaliação, Implementação).

## 2. Ordem da aula (≈ 6 h, duas sessões de 3 h) e ids do deck

| Bloco | Deck `#id` | Tempo teoria | Notebook | Tempo prática |
|---|---|---|---|---|
| Abertura | `#inicio`, `#agenda` | 15 min | — | — |
| CRISP-DM | `#crisp-dm` | 15 min | — | — |
| Preparação de dados | `#dados` | 25 min | — | — |
| Supervisionada | `#supervisionada` | 30 min | 01 House Prices | 45 min |
| Métricas | `#metricas` | 20 min | 02 Titanic | 45 min |
| *— fim da sessão 1 —* | | | | |
| Não supervisionada | `#nao-supervisionada` | 20 min | 03 Unsupervised | 25 min |
| Redes neuronais | `#redes-neuronais` | 30 min | 04 NN from scratch | 30 min |
| CPU vs GPU | `#cpu-gpu` | 10 min | 04 (parte 3) | — |
| Reinforcement learning | `#rl` | 25 min | 05 RL project | 30 min |
| Escolher modelo · Produção · Ética | `#escolher`, `#producao`, `#etica` | 25 min | — | — |
| Fecho | `#fecho` | 10 min | — | — |

## 3. Dados (decisão tomada; não mudar)

- **Casas:** (ronda 2: descarregadas da fonte oficial JAE `ag-data.zip` por `load_housing()`; não há CSV no repo) — Anglin & Gençay (1996), *J. Applied Econometrics* 11:633–648; 546 casas vendidas em **Windsor, Canadá, jul–set 1987**; preço em **dólares canadianos de 1987 (CAD)**; distribuído no pacote R **AER** (GPL-2 | GPL-3). Colunas: `price, lotsize (pés², do lote), bedrooms, bathrooms, stories, driveway, recreation, fullbase, gasheat, aircon, garage, prefer` (+ `rownames`, que o loader descarta). Sem NAs; 1 linha repetida (mantida: podem ser duas casas iguais). O antigo `Housing.csv` (Kaggle) é este mesmo dataset ×70, com colunas renomeadas, uma linha a menos e uma coluna `furnishingstatus` inventada → descartado por proveniência e licença.
- **Titanic:** `titanic3` (Harrell & Cason, Vanderbilt Biostatistics; 1309 passageiros, 14 colunas). **Não está no repo**: `src.data.load_titanic()` descarrega de `hbiostat.org` (fallback OpenML id 40945) e guarda em `data/titanic3.csv` (git-ignored). `boat` e `body` são **leakage** (só se sabem depois do naufrágio). 38,2 % sobreviveram.

## 4. API de `src/` (já escrita e testada — usar, não reescrever)

```python
from src.paths import find_root, data_dir, assets_dir, models_dir
from src.data import load_housing, load_titanic, HOUSING_NUMERIC, HOUSING_BINARY
from src.preprocessing import (make_housing_preprocessor, add_titanic_features, make_titanic_preprocessor,
    GroupMedianImputer, extract_title, TITANIC_FEATURES, TITANIC_NUMERIC, TITANIC_CATEGORICAL, TITANIC_LEAKY)
from src.metrics import confusion_at, threshold_table, best_threshold
from src.plotting import apply_style, PALETTE, CYCLE, MARKERS, save_fig, plot_decision_boundary, plot_residuals, plot_history
from src.nn_numpy import sigmoid, step, bce_loss, train_perceptron, MLP, gradient_check
from src.rl_utils import (GridWorld, DEFAULT_LAYOUT, COIN_LAYOUT, ARROWS, q_learning, evaluate, greedy_path,
    greedy_action, moving_average, epsilon_schedule, plot_q_values)
```
Lê as docstrings em `src/*.py` antes de usar. Se precisares de uma função nova reutilizável, diz-me no relatório (não alteres `src/` sem indicar).

## 5. Números canónicos (medidos; deck e notebooks têm de coincidir)

Seeds: `random_state=42` em splits/modelos sklearn; Q-learning `seed=0`; NumPy MLP `seed=0`.

**Casas** — `train_test_split(X, y, test_size=0.2, random_state=42)`; CV `KFold(5, shuffle=True, random_state=42)` no treino; modelos log usam `TransformedTargetRegressor(func=np.log, inverse_func=np.exp)` (avaliação sempre em CAD).

| Modelo | CV RMSE (treino) | Teste MAE | Teste RMSE | Teste R² |
|---|---|---|---|---|
| Baseline (média) — `DummyRegressor` | 26 825 ± 1 345 | 20 313 | 26 163 | −0,025 |
| Linear (log price) | 15 419 ± 1 134 | 10 750 | 15 475 | 0,642 |
| Ridge α=1 (log price) | 15 419 ± 1 125 | 10 735 | 15 457 | 0,642 |
| Random forest (300 árvores, `min_samples_leaf=3`, **sem log**: prevê CAD diretamente) | 16 406 ± 1 292 | 12 222 | 16 538 | 0,591 |

Mediana do preço 62 000 CAD, média 68 122, assimetria 1,21; corr(price, lotsize) = 0,54. Ridge (log): efeito multiplicativo (exp(coef)−1) por **1 desvio-padrão** nas numéricas / por ter a característica: gasheat +21 %, aircon +20 %, driveway +14 %, lotsize +12 %/DP, prefer +12 %, fullbase +11 %, stories +9 %/DP, recreation +8 %, bathrooms +8 %/DP, garage +5 %/DP, bedrooms +2 %/DP. Erro relativo médio (MAPE) ≈ 17,5 %.
**Mensagem:** o modelo linear simples ganha à random forest com 546 casas → "simplicidade primeiro".

**Titanic** — features `TITANIC_FEATURES` após `add_titanic_features`; `train_test_split(test_size=0.2, stratify=y, random_state=42)` (1047 treino / 262 teste); CV `StratifiedKFold(5, shuffle=True, random_state=42)`.

| Modelo | CV accuracy | CV ROC-AUC | Teste accuracy | Teste ROC-AUC | Teste PR-AUC (AP) |
|---|---|---|---|---|---|
| Baseline (classe maioritária) | 0,618 | 0,500 | 0,618 | 0,500 | 0,382 |
| Regressão logística (`max_iter=1000`) | 0,803 ± 0,012 | 0,848 | 0,840 | 0,885 | 0,856 |
| Random forest (300, `min_samples_leaf=3`) | 0,805 ± 0,011 | 0,853 | 0,851 | 0,900 | 0,877 |
| HistGradientBoosting | 0,770 ± 0,022 | 0,830 | 0,824 | 0,886 | 0,831 |

- Taxa de sobrevivência por classe × sexo: 1.ª mulher 0,97 · 1.ª homem 0,34 · 2.ª mulher 0,89 · 2.ª homem 0,15 · 3.ª mulher 0,49 · 3.ª homem 0,15.
- Títulos: Mr 757, Miss 264, Mrs 198, Master 61, Rare 29.
- **Leakage:** juntar `has_boat = boat.notna()` → accuracy no teste **0,966** (vs 0,840). Parece ótimo; é inútil (só se sabe depois).
- **Threshold por custo** (regressão logística; probabilidades *out-of-fold* `cross_val_predict` no treino = validação; depois aplicar ao teste). Positivo = "sobrevive". Cenário didático (custos **escolhidos para o exemplo**): uma equipa de socorro usa o modelo para decidir quem **não** precisa de ajuda prioritária; dizer "sobrevive" a quem não sobrevive (FP) custa 5, o contrário (FN) custa 1.
  - threshold 0,50 (teste): TP 78 · FP 20 · FN 22 · TN 142 → precision 0,796 · recall 0,780 · F1 0,788 · accuracy 0,840.
  - custos 1:1 → threshold 0,54 → TP 76 · FP 20 · FN 24 · TN 142.
  - custos FP=5, FN=1 → threshold **0,82** → TP 39 · FP 3 · FN 61 · TN 159 → precision 0,929 · recall 0,39 · accuracy 0,756.

**Redes neuronais (NumPy)** — perceptron (lr 0,1, 20 épocas, seed 0): AND e OR convergem (0 erros), XOR fica com 4 erros por época. `MLP(2, 4, seed=0)` no XOR, lr 0,5, 2000 épocas: loss 0,708 → 0,0022; saídas [0,000, 0,997, 0,997, 0,003]. Gradient check: diferença relativa máx. ≈ 3·10⁻⁹. `make_moons(400, noise=0.2, random_state=0)`, `MLP(2, 8, seed=0)`, lr 0,5, 3000 épocas: loss 0,091, accuracy 0,965 (treino).

**RL (gridworld)** — `GridWorld()` padrão: passo −1, objetivo +20, lava −20, máx. 100 passos. Q-learning: α=0,5, γ=0,95, ε linear 1,0→0,05 nos primeiros 60 % de 400 episódios.
- Seeds 0, 1, 2: retorno médio nos primeiros 50 episódios −57,5 / −53,8 / −48,2; últimos 50: ≈ 10,8; sucesso nos últimos 50: 98 % (ainda há ε=0,05).
- **Avaliação greedy (ε=0, separada do treino):** retorno 12,0, sucesso 100 %, 9 passos (ótimo: 8×(−1)+20=12). Caminho: sobe pela coluna da esquerda e segue pela linha de cima.
- Com `slip=0.1` na avaliação (200 episódios, seed 123): sucesso 97 %, retorno médio 9,9.
- **Reward hacking:** `COIN_LAYOUT` (moeda +5 que nunca desaparece em (4,2)): o agente aprende a ficar em cima da moeda a bater na parede de baixo → retorno de avaliação **494**, sucesso **0 %**, 100 passos. A métrica subiu, a tarefa falhou.

**CPU vs GPU** — ver `_planning/benchmark.md` (preenchido quando o benchmark correr; hardware: portátil com NVIDIA GeForce RTX 4060 Laptop GPU 8 GB).

## 6. Convenções visuais partilhadas

- Cores deck: `--page:#08111f; --slide:#0e1a2e; --card:#13223b; --ink:#f1f5fb; --body:#b4c2d8; --blue:#6aaef0; --yellow:#ffd43b; --green:#9bd67e; --red:#ff7a7a; --orange:#ffb547; --violet:#c4a2ff`. Fontes Inter Tight + JetBrains Mono (Google Fonts).
- Cor por bloco (`--acc`): CRISP-DM/dados `#ffd43b`, supervisionada `#6aaef0`, métricas `#8fc8ff`, não sup. `#9bd67e`, NN `#c4a2ff`, CPU/GPU `#ffb547`, RL `#ff9f7a`, produção/ética `#7fd6c8`.
- Notebooks: `apply_style()` + Okabe–Ito (`PALETTE`); classes distinguem-se por **marcador e cor**; eixos com unidades (CAD, pés², anos, épocas, episódios, ms).
- Notação comum: $\hat y$, $\mathcal L$, $\eta$ (learning rate), $W, b$, $\sigma$ sigmoide, $\gamma$ desconto, $\alpha$ taxa de aprendizagem RL, $\varepsilon$ exploração, $G_t$ retorno, $Q(s,a)$.

## 7. Modelo de notebook (estilo `02_Data_Structures.ipynb` da Lição 01)

Células, por ordem:
1. **Banner** (markdown HTML):
```html
<div align="center" style="padding:40px 24px;background:linear-gradient(135deg,#0e1a2e 0%,#13223b 100%);border-radius:14px;border-bottom:5px solid #ffd43b;">
  <p style="margin:0;color:#6aaef0;font-family:monospace;font-weight:700;font-size:14px;letter-spacing:.16em;text-transform:uppercase;">The AI Learning Hub · Lição 02 · Machine Learning: fundamentos</p>
  <h1 style="margin:14px 0 6px;color:#f1f5fb;font-size:42px;">01 · CRISP-DM: preços de casas</h1>
  <p style="margin:0;color:#b4c2d8;font-size:18px;">Que fatores mais pesam no preço de uma casa?</p>
</div>
```
2. **Badges:** `[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](<colab url>) [![Slides](https://img.shields.io/badge/slides-Lição_02-ffd43b)](https://the-ai-learning-hub.github.io/02-machine-learning-fundamentals/#<id>)`
3. **🎯 Objetivos de aprendizagem** + linha `**⏱️ Duração estimada:** … · **📋 Pré-requisitos:** … · **▶️ Como usar:** executa cada célula com `Shift`+`Enter`, por ordem.`
4. **📑 Índice** com links `#sec-N` (âncoras `<a id="sec-N"></a>` antes de cada `# N. Título`).
5. **⚙️ Preparar o ambiente** — célula de setup (igual em todos):
```python
# ⚙️ Preparar o ambiente: funciona localmente (VS Code/Jupyter) e no Google Colab
import subprocess
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    ROOT = Path("/content/02-machine-learning-fundamentals")
    if not ROOT.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/the-ai-learning-hub/02-machine-learning-fundamentals.git",
                        str(ROOT)], check=True)
else:
    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir() and (p / "data").is_dir())

try:  # o Colab já traz tudo; localmente usa `uv sync` (ver README)
    import sklearn, seaborn  # noqa: F401
except ImportError:  # plano B: instala as dependências declaradas no pyproject.toml
    import tomllib
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *deps], check=True)

sys.path.insert(0, str(ROOT))
print("Raiz do projeto:", ROOT.name, "(Colab)" if IN_COLAB else "(local)")
```
6. Secções numeradas com **CRISP-DM em cabeçalhos explícitos** (ex.: `# 2. 🧭 Business Understanding (compreensão do negócio)`).
7. Callouts em blockquote: `> 💡 **Dica:**`, `> ⚠️ **Cuidado:**`, `> 🤖 **Ligação à IA:**`, `> 📚 **Para saber mais:**`.
8. **💬 Discussão** (pelo menos uma por fase CRISP-DM):
```markdown
### 💬 Discussão — <título curto>
<pergunta aberta e interessante, 1–3 frases>

<details>
<summary>🧑‍🏫 Resposta-guia para o instrutor</summary>

<resposta apoiada em números/figuras **deste** notebook>
</details>
```
9. **✏️ Exercícios** (`### ✏️ Exercício N — Título` + `**Dificuldade:** ⭐`…`⭐⭐⭐` + célula inicial que já corre + `<details><summary>💡 Ver solução</summary> … </details>`).
10. **📌 Resumo** (tabela/checklist) → **📚 Para saber mais** (fontes reais, ver §9) → rodapé:
```markdown
---
<div align="center">

⬅️ [Anterior: …](…) &nbsp;·&nbsp; 🏠 [Lição 02](../../README.md) &nbsp;·&nbsp; [Seguinte: …](…) ➡️

**The AI Learning Hub** · Lição 02 — Machine Learning: fundamentos<br>
Diogo Freitas & João Francisco Botas · Licença MIT

</div>
```
Regras: o projeto usa **uv** (`pyproject.toml` + `uv.lock`; não há `requirements.txt`; instruções locais = `uv sync` e escolher o kernel `.venv`); código executado com outputs guardados; ≤ ~5 min em CPU; seeds fixas; sem `!pip` incondicional; sem caminhos absolutos; células de código curtas com propósito claro; nada de warnings ruidosos (filtra só os conhecidos e explica). Constrói cada notebook com um script Python em `_planning/builders/build_<nb>.py` (nbformat) e executa com
`"$PY" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 <nb>` e valida com `nbformat.validate`.

## 8. Caminhos e ficheiros

- Notebooks: `01_supervised_learning/notebooks/01_CRISP-DM_House_Prices.ipynb`, `…/02_CRISP-DM_Titanic_Survival.ipynb`, `02_unsupervised_learning/notebooks/03_Unsupervised_Learning.ipynb`, `03_neural_networks/notebooks/04_Neural_Networks_From_Scratch.ipynb`, `04_reinforcement_learning/notebooks/05_Reinforcement_Learning_Project.ipynb`.
- Links relativos entre notebooks: `../../02_unsupervised_learning/notebooks/03_Unsupervised_Learning.ipynb` etc. Raiz: `../../README.md`.
- Figuras que valem a pena guardar: `assets/figures/<nb>_<nome>.png` (leves, ≤ 150 KB) e `assets/rl/` (GIF do agente ≤ 1 MB). Modelos guardados em `models/` (git-ignored).
- Python de teste: `C:/Users/diogo/AppData/Local/Temp/claude/5120910ceae76249/68e8bf41-4274-4d58-8c46-670b2aeb583c/scratchpad/venv/Scripts/python.exe` (3.12; numpy 2.5, pandas 3.1, sklearn 1.9, matplotlib 3.11, seaborn 0.13, torch com CUDA). Corre a partir da raiz do repo.

## 9. Fontes («Para saber mais»; usar estas, reformular, não copiar)

1. Chapman et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS/NCR/DaimlerChrysler.
2. scikit-learn User Guide — https://scikit-learn.org/stable/user_guide.html (Pipelines/ColumnTransformer: `/modules/compose.html`; validação cruzada: `/modules/cross_validation.html`; métricas: `/modules/model_evaluation.html`; clustering: `/modules/clustering.html`; PCA: `/modules/decomposition.html#pca`; *Common pitfalls* (leakage): `/common_pitfalls.html`; escolher estimador: `/machine_learning_map.html`).
3. Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2.ª ed.), MIT Press — http://incompleteideas.net/book/the-book-2nd.html (cap. 3 MDP, cap. 6 Q-learning, §17.4 recompensas).
4. PyTorch — Learn the Basics: https://docs.pytorch.org/tutorials/beginner/basics/intro.html ; CUDA semantics (sincronização assíncrona): https://docs.pytorch.org/docs/stable/notes/cuda.html
5. Goodfellow, Bengio & Courville (2016). *Deep Learning*, MIT Press — https://www.deeplearningbook.org (cap. 6 MLP/backprop).
6. Dados: Anglin & Gençay (1996), *J. Applied Econometrics* 11(6):633–648 + pacote AER (Kleiber & Zeileis); titanic3: Harrell & Cason, Vanderbilt Biostatistics — https://hbiostat.org/data/
Extras de referência histórica (só menção): Rosenblatt 1958 (perceptron); Krizhevsky, Sutskever & Hinton 2012 (AlexNet/ImageNet); Vaswani et al. 2017 (*Attention Is All You Need*); Minsky & Papert 1969 (*Perceptrons*, XOR).
