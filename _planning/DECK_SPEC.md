# DECK_SPEC · `index.html` (guião slide a slide)

Lê primeiro `_planning/PLAN.md` (identidade, números canónicos, cores). Referência visual/técnica: `index1.html` (CSS, `.rail`, `.kicker`, `.hl`, `.term`, `.call`, `.cols`, `.steps`, `.tracker`, `.divider` com chuva de código, navegação, `fit()`, `#hash`, notas `aside.notes` + painel com tecla N). **Coerente, não cópia:** reescreve o CSS/JS que precisares; reaproveita tokens, tipografia, ritmo e componentes.

## Regras globais

- PT-PT em todo o texto visível e nas notas. `<html lang="pt-PT">`.
- 1 ideia por slide; corpo ≤ ~35 palavras; nunca scroll; texto mínimo 15 px no canvas 1280×720 (projetor).
- Visual > texto: SVG inline, mini-gráficos em SVG/JS, código curto (≤ 10 linhas) com o realce `pre.py` do `index1.html`.
- Cada secção: divisor (`.divider` com número grande e `.tracker` das secções) → conteúdo → **recap + 💬 pergunta de discussão** (um slide "Recap" com 3 bullets + caixa de pergunta).
- `aside.notes` em **todos** os slides de conteúdo: o que dizer, tempo sugerido (ex.: "⏱ 2 min") e, nos recaps, a pergunta de discussão com uma resposta curta.
- `data-id` nos primeiros slides de cada secção (ids do PLAN §2). Hash numérico e por id, como no `index1.html`.
- Rail: `Lição 02 · <secção>`; contador `SLIDE nn/NN`.
- Badges para notebooks: em cada divisor de secção e nos recaps, um `a.pill` "▶ Notebook 01 · Colab" (URL Colab do PLAN) e "GitHub" (link relativo `01_supervised_learning/notebooks/…ipynb` NÃO serve no Pages → usa URL absoluta `https://github.com/the-ai-learning-hub/02-machine-learning-fundamentals/blob/main/...`).
- Acessibilidade: contraste AA; nunca só cor (formas, rótulos, padrões); `:focus-visible` com contorno amarelo; botões reais (`<button>`) com `aria-label`; sliders `<input type=range>` com `<label>` e valor visível (`aria-live="polite"` no texto de resultado); teclado: ←/→/PageUp/PageDown/Espaço/Home/End/F/N. **Quando o foco está num `input`/`button` de um interativo, ←/→ não mudam de slide** (só se o foco estiver no body; documenta: "Esc devolve o foco ao deck").
- `prefers-reduced-motion`: sem animações automáticas (chuva, entradas, loops); os interativos ficam no estado final/estático e só avançam por clique ("Passo ▶").
- **Performance:** cada animação (`requestAnimationFrame`/`setInterval`) só corre quando o seu slide está ativo; para ao sair (hook `onEnter/onLeave` por slide). Nada de bibliotecas JS externas. Canvas com `devicePixelRatio`.
- Impressão: `@media print` simples como no `index1.html` (todos os slides, sem UI).
- `<head>`: `<title>Machine Learning: fundamentos · Lição 02</title>`, `meta description`, Open Graph (`og:title`, `og:description`, `og:type=website`, `og:url=https://the-ai-learning-hub.github.io/02-machine-learning-fundamentals/`), `theme-color #0e1a2e`, favicon SVG inline em data URI (ex.: nó de rede neuronal amarelo sobre azul-marinho).
- Só caminhos relativos; nenhum asset externo além de Google Fonts. Ficheiro único `index.html` (< 400 KB idealmente).

## Interativos (obrigatórios os 7; todos em JS puro, robustos, testáveis)

Cada interativo tem um `id` estável e um botão "Repor". Expõe estado legível no DOM (texto) para testes automáticos.

- **(g) CRISP-DM em anel** `#crisp-ring`: SVG com 6 setores em anel + "Dados" ao centro + setas do ciclo (incluindo as setas de volta Evaluation→Business e Data Prep↔Modeling). Cada setor é um `<button>`/`role=button` focável (Tab, Enter). Clique → painel lateral com: pergunta-chave, entregável, armadilha típica, **onde está nos notebooks** (ex.: "NB01 §2, NB02 §2"). Botão "▶ Percorrer o ciclo" anima a sequência.
- **(a) Grau do polinómio** `#poly-demo`: ~20 pontos gerados com seed fixa de `y = sin(2πx) + ruído` (x em [0,1]) divididos em treino (círculos) e validação (quadrados). Slider grau 1–15 → ajuste por mínimos quadrados (resolver equações normais com regularização mínima 1e-8 e x escalado para [-1,1] para estabilidade) desenhado como curva; mostra **RMSE treino** e **RMSE validação** e um mini-gráfico das duas curvas de erro vs. grau com marcador no grau atual; rótulo "underfit / bom ajuste / overfit".
- **(b) K-Means passo a passo** `#kmeans-demo`: canvas com 3 blobs (seed fixa, ~150 pontos), k ajustável 2–6, botões "Atribuir", "Atualizar centróides", "▶ Auto", "Novos centróides". Mostra iteração, inércia (WCSS) e se convergiu. Pontos com forma por cluster (círculo/quadrado/triângulo…) além da cor; centróides como ✕ grandes com rasto.
- **(c) Neurónio / MLP playground** `#neuron-demo`: neurónio de 2 entradas — sliders w1, w2, b (−5..5) + seletor de ativação (degrau, sigmoide, tanh, ReLU) + seletor de dataset (AND, OR, XOR). Desenha o plano 2D com a fronteira de decisão (sombreado da saída) e os 4 pontos com rótulos; mostra a tabela de verdade com a saída do neurónio e quantos acertos ("3/4"). Botão "Treinar (perceptrão)" para AND/OR que aplica a regra do perceptrão passo a passo; no XOR, mensagem: "Nenhuma reta separa o XOR → precisamos de uma camada escondida". Num segundo slide, **forward pass animado** com números (ver slide NN-forward).
- **(d) Gridworld Q-learning ao vivo** `#gridworld-demo`: **mesmo ambiente e hiperparâmetros do PLAN §5 RL** (layout 5×6, passo −1, objetivo +20, lava −20, máx. 100 passos, α=0,5, γ=0,95, ε 1,0→0,05 linear nos primeiros 60 % de 400 episódios, desempate aleatório como em `src/rl_utils.py`). Mostra: grelha com heatmap de max Q por célula, setas da política greedy, agente a mover-se, gráfico de retorno por episódio + média móvel (20), contador de episódio, ε atual, taxa de sucesso dos últimos 50. Controlos: ▶/⏸, velocidade (1×, 10×, máx = sem desenhar o agente), "Passo", "Repor", checkbox "🪙 Moeda (+5)" que muda para `COIN_LAYOUT` e repõe → mostra reward hacking (retorno sobe, sucesso 0 %). PRNG com seed (mulberry32) para reprodutibilidade; o JS não precisa de dar exatamente os mesmos números que o Python, mas o comportamento tem de bater certo (converge para retorno ≈ 11–12 sem moeda; com moeda fica a acampar na moeda).
- **(e) CPU vs GPU** `#cpugpu-demo`: dois painéis: CPU = 8 núcleos grandes, GPU = grelha de ~1000 núcleos pequenos; uma fila de N tarefas (blocos de uma multiplicação de matrizes). Botão "▶ Correr": CPU processa 8 tarefas de cada vez rápido; GPU processa centenas em paralelo, cada uma mais lenta, mais um custo fixo inicial de "transferência para a GPU" (barra de cópia). Slider "tamanho do trabalho" (pequeno → CPU ganha por causa da transferência/lançamento; grande → GPU ganha). Mostra os tempos simulados e quem ganhou; legenda "simulação didática — os números medidos estão no slide seguinte".
- **(f) Threshold** `#threshold-demo`: usa as **probabilidades reais do conjunto de teste do Titanic** (262 valores + rótulos) que estão em `_planning/titanic_test_probs.json` (embebe-as inline no JS, arredondadas a 3 casas). Slider 0,05–0,95: histograma das probabilidades por classe (sobreviveu/morreu, com padrões diferentes), linha do threshold, matriz de confusão 2×2 atualizada, precision, recall, F1, accuracy; inputs de custo FP e FN (por omissão 5 e 1) e custo total; botão "Threshold de custo mínimo". Em 0,50 tem de mostrar TP 78 · FP 20 · FN 22 · TN 142.

## Guião (≈ 60–70 slides). `→ NBxx §y` = ligação ao notebook.

### 0 · Abertura (`#inicio`, `#agenda`)
1. **Capa** — "Machine Learning: **fundamentos**". Kicker "The AI Learning Hub · Lição 02". Lead: dos dados à decisão — CRISP-DM, supervisionada, não supervisionada, redes neuronais e reinforcement learning. Pills. Terminal animado tipo `index1.html` (ex.: `model.fit(X, y)` → "A aprender…" → `accuracy: 0.84`). Chuva de código com palavras de ML.
2. **O que é ML?** — programação clássica (regras + dados → respostas) vs ML (dados + respostas → regras). Diagrama SVG lado a lado.
3. **IA ⊃ ML ⊃ DL ⊃ GenAI** — círculos concêntricos SVG com um exemplo em cada (xadrez com regras; filtro de spam; reconhecimento de imagem; ChatGPT/LLMs).
4. **ML no dia a dia** — 6 cartões: recomendações, deteção de fraude, tradução, previsão de procura, diagnóstico assistido, assistentes de voz. Cada um com a família (sup./não sup./RL).
5. **Mini-história** — linha temporal SVG: 1958 perceptrão (Rosenblatt) → 1969 Minsky & Papert (XOR) → invernos da IA (anos 70 e fim dos 80) → 1986 backprop popularizado → 2012 ImageNet/AlexNet → 2017 Transformers → 2022+ LLMs/GenAI.
6. **As 4 famílias** — supervisionada (com labels), não supervisionada (sem labels), auto-/semi-supervisionada (labels a partir dos próprios dados — como os LLMs aprendem), RL (recompensas). Ícone + exemplo + "nesta aula: NB01–02 / NB03 / menção / NB05".
7. **Agenda** — duas sessões com tempos (tabela do PLAN §2) + mapa dos 5 notebooks.

### 1 · CRISP-DM (`#crisp-dm`)
8. Divisor 01 · CRISP-DM.
9. **Porquê uma metodologia?** — a maioria dos projetos falha por pergunta errada/dados, não por algoritmo. Seis fases, iterativo, independente de ferramenta (Chapman et al., 2000).
10. **Interativo (g) anel CRISP-DM**.
11. **Fases → entregáveis → armadilhas** — tabela compacta 6 linhas (fase · pergunta · entregável · armadilha).
12. **É um ciclo, não uma cascata** — exemplo concreto: na Evaluation do NB01 descobrimos que falta localização → volta a Business/Data Understanding.
13. **Recap + 💬** "Em que fase achas que se gasta mais tempo num projeto real? Porquê?"

### 2 · Preparação de dados e fundamentos (`#dados`)
14. Divisor 02.
15. **Tipos de variáveis** — numéricas (contínuas/discretas), categóricas (nominais/ordinais), binárias, texto, datas. Exemplos dos 2 datasets (`lotsize`, `bedrooms`, `prefer`, `pclass`, `name`).
16. **Valores em falta: MCAR / MAR / MNAR** — definição curta + exemplo Titanic (`age` 263 NA, `cabin` 1014 NA, `embarked` 2). Estratégias: remover, imputar (mediana por título), indicador de falta.
17. **Outliers** — boxplot SVG de `price`; erro vs. caso raro; não apagar às cegas.
18. **Encoding** — one-hot (`embarked`), ordinal (`pclass`?), target encoding (cuidado com leakage). Mini-tabela antes/depois.
19. **Scaling** — porquê: kNN (distâncias), redes neuronais (gradientes), regularização (penalização justa). Gráfico SVG: `lotsize` (milhares) vs `bedrooms` (unidades) antes/depois de StandardScaler.
20. **Feature engineering** — `title` a partir de `name`, `family_size`, `deck`, `log(price)`. "Conhecimento do domínio > algoritmo".
21. **Treino / validação / teste** — barra 60/20/20 + regra de ouro: o teste só se usa uma vez, no fim.
22. **Validação cruzada (k-fold)** — animação SVG de 5 folds a rodar; resultado = média ± desvio.
23. **Data leakage** — o exemplo real: `boat` no Titanic → accuracy 0,966 vs 0,840. Outras fugas: scaler ajustado antes do split, duplicados, info do futuro.
24. **`Pipeline` + `ColumnTransformer`** — código curto (≤ 9 linhas) e diagrama: o pipeline faz fit só no treino de cada fold.
25. **Baseline "dummy"** — antes de qualquer modelo: média (casas: RMSE 26 163 CAD) / classe maioritária (Titanic: 61,8 %). "Se não bates o baseline, não tens modelo."
26. **Recap + 💬** "Encontra uma fuga de informação escondida neste pipeline" (mostra um snippet com `scaler.fit(X)` antes do split).

### 3 · Aprendizagem supervisionada (`#supervisionada`)
27. Divisor 03 + pills NB01/NB02.
28. **Regressão vs classificação** — dois mini-gráficos: preço (número) vs sobrevive (sim/não).
29. **Regressão linear** — reta + resíduos desenhados; minimizar soma dos quadrados; interpretação de coeficiente (casas: ar condicionado ≈ +20 % no preço, mantendo o resto igual).
30. **Regressão logística** — sigmoide SVG; saída é probabilidade; fronteira linear.
31. **Regularização** — Ridge (L2) encolhe, Lasso (L1) zera; λ controla a força. Mini-gráfico de coeficientes a encolher.
32. **Árvores de decisão** — árvore SVG com as primeiras divisões do Titanic (sexo → classe → idade/título); interpretáveis, mas instáveis/sobreajustam.
33. **Ensembles** — random forest (muitas árvores, votação, bagging) e gradient boosting (árvores em sequência a corrigir erros). Diagrama.
34. **Interativo (a) grau do polinómio**.
35. **Viés-variância** — curva em U (erro treino vs validação vs complexidade), alvo; ligação ao slider.
36. **Resultados reais — casas** — barras horizontais dos RMSE de teste (baseline 26 163 · linear 15 475 · ridge 15 457 · random forest 16 538). Mensagem: "o modelo simples ganhou" → NB01 §5.
37. **Desequilíbrio de classes** — 38 % vs 62 %; accuracy engana; estratificar, `class_weight`, métricas certas.
38. **Hiperparâmetros vs parâmetros** + tuning com CV (`GridSearchCV`), sem tocar no teste.
39. **Interpretação** — coeficientes vs permutation importance; correlação ≠ causa (introduz o ponto).
40. **Outros algoritmos (tabela pequena)** — kNN, SVM, Naive Bayes: quando considerar.
41. **Recap + 💬** "O modelo das casas diz que `gasheat` vale +21 %. Instalarias aquecimento a gás para vender mais caro?"

### 4 · Métricas (`#metricas`)
42. Divisor 04.
43. **Regressão: MAE, RMSE, R²** — fórmulas curtas + exemplo numérico com 4 casas (números didáticos, assinalados) mostrando como um erro grande pesa mais no RMSE. Resultado real: MAE 10 750 CAD, RMSE 15 475 CAD, R² 0,64.
44. **Matriz de confusão** — 2×2 do Titanic a 0,5 (TP 78 · FP 20 · FN 22 · TN 142) com nomes PT-PT + precision = 78/98 = 0,796, recall = 78/100 = 0,78, F1 0,788, accuracy 0,84.
45. **ROC-AUC e PR-AUC** — curvas SVG esquemáticas; AUC 0,885, AP 0,856 (logística, teste); quando preferir PR (classe rara).
46. **Interativo (f) threshold** — com custos FP=5, FN=1 → threshold de custo mínimo; nota: no notebook escolhemos 0,82 na validação (OOF) e só depois avaliamos no teste: precision 0,929 · recall 0,39.
47. **Métricas noutras famílias** — tabela: não sup. (silhouette, variância explicada), NN (curvas de loss treino/val), RL (retorno, taxa de sucesso). Pontes para as secções seguintes.
48. **Recap + 💬** "Num teste médico de rastreio, preferes alta precision ou alto recall? E num filtro de spam?"

### 5 · Não supervisionada (`#nao-supervisionada`)
49. Divisor 05 + NB03.
50. **Interativo (b) K-Means passo a passo**.
51. **Escolher k** — método do cotovelo (inércia) + silhouette; ambos são pistas, não verdades.
52. **A escala muda os clusters** — dois mini-scatters SVG: sem scaling (uma variável domina) vs com scaling.
53. **Quando o K-Means falha: DBSCAN** — luas (`make_moons`): K-Means corta mal; DBSCAN segue a densidade e marca ruído.
54. **PCA geometricamente** — nuvem 2D elíptica com eixos principais desenhados; projetar na direção de maior variância. Dígitos 8×8 (64 dimensões) → 2D. Variância explicada.
55. **Avaliar sem labels + extensões** — silhouette, estabilidade, utilidade para o negócio; menção: clustering hierárquico, t-SNE/UMAP, deteção de anomalias, regras de associação.
56. **Recap + 💬** "Uma loja segmenta clientes em 5 clusters. Como sabes se os clusters são úteis?"

### 6 · Redes neuronais (`#redes-neuronais`)
57. Divisor 06 + NB04.
58. **O neurónio** — soma ponderada + bias + ativação; diagrama SVG; analogia limitada com o neurónio biológico.
59. **Ativações** — 4 mini-gráficos (degrau, sigmoide, tanh, ReLU) e quando se usam.
60. **Interativo (c) neurónio playground** (AND/OR/XOR).
61. **XOR e a camada escondida** — 2 retas combinadas resolvem o XOR; MLP 2-4-1; NB04: loss 0,708 → 0,0022 em 2000 épocas.
62. **Forward pass com números** — rede 2-2-1 com pesos escolhidos à mão (didáticos, assinalados), entrada (1, 0): mostrar z, a, ŷ passo a passo (animado em 3 cliques/etapas). Verifica as contas.
63. **Loss e gradiente descendente** — superfície/curva 1D em SVG com uma bola a descer; η pequeno vs grande (diverge).
64. **Backprop = regra da cadeia** — ∂L/∂w = ∂L/∂ŷ · ∂ŷ/∂z · ∂z/∂w, com os números do slide 62; gradient check no NB04 ≈ 3·10⁻⁹.
65. **Overfitting em redes + regularização** — curvas treino/val divergem; dropout, weight decay, early stopping.
66. **De NumPy para PyTorch** — código lado a lado (≤ 8 linhas cada): `forward` manual vs `nn.Module` + `loss.backward()` + `optimizer.step()`.
67. **CNN e Transformers (intuição)** — CNN: filtros que deslizam e detetam padrões locais; Transformer: atenção — cada palavra "olha" para as outras. Um diagrama simples cada.
68. **Recap + 💬** "Se uma rede com 1 camada escondida pode aproximar qualquer função, porque usamos redes profundas?"

### 7 · CPU vs GPU (`#cpu-gpu`)
69. Divisor 07.
70. **Porque é que as GPUs aceleram ML** — multiplicação de matrizes = milhões de somas-produto independentes; latência vs throughput; memória (VRAM) e custo de transferência.
71. **Interativo (e) CPU vs GPU**.
72. **Benchmark medido** — dados de `_planning/benchmark.md` (hardware identificado; tabela/barras por tamanho de matriz; transferência). Mensagem: para matrizes pequenas a CPU chega/ganha; para grandes, a GPU ganha por ordens de grandeza. Se não houver GPU, explicar.
73. **Recap + 💬** "Treinar o modelo do Titanic numa GPU: vale a pena?"

### 8 · Reinforcement Learning (`#rl`)
74. Divisor 08 + NB05.
75. **Agente e ambiente** — ciclo estado → ação → recompensa → novo estado (SVG).
76. **MDP e retorno descontado** — S, A, P, R, γ; $G_t = r_{t+1} + γ r_{t+2} + γ^2 r_{t+3} + …$; exemplo numérico com γ=0,95 no gridworld (9 passos até ao objetivo).
77. **Política e funções de valor** — π(a|s), V(s), Q(s,a); equação de Bellman em palavras.
78. **Exploração vs aproveitamento** — ε-greedy; ε decai de 1,0 para 0,05.
79. **Uma atualização de Q-learning com números** — Q(s,→)=2,0, r=−1, γ=0,95, max Q(s′)=6,0, α=0,5 → alvo = −1 + 0,95·6 = 4,7; erro TD = 2,7; novo Q = 2,0 + 0,5·2,7 = 3,35 (números didáticos).
80. **Interativo (d) gridworld ao vivo**.
81. **Avaliar um agente** — treino (com exploração) ≠ avaliação (greedy, ε=0, sem aprender); retorno médio, média móvel, taxa de sucesso, várias seeds. NB05: avaliação 12,0 / 100 % / 9 passos; com escorregamento 10 % → 97 %.
82. **Reward shaping e reward hacking** — moeda +5: retorno 494, sucesso 0 %. "Recebes aquilo que recompensas, não aquilo que queres." Menção a shaping baseado em potencial.
83. **Pontes** — SARSA (on-policy), DQN (rede neuronal no lugar da tabela, Atari), PPO (gradiente de política, robótica), RLHF (afinar LLMs com preferências humanas). Limites: amostras, segurança, simulação ≠ realidade.
84. **Recap + 💬** "Que recompensa darias a um robô aspirador? Como é que ele a podia 'hackear'?"

### 9 · Escolher o modelo (`#escolher`)
85. **Cheat-sheet próprio** — fluxograma SVG: "Tens labels?" → sim: número ou categoria? → regressão/classificação → começa por linear/logística → árvores/boosting → NN se dados grandes/não estruturados; não → estrutura (clustering/PCA); "Aprendes por tentativa e recompensa?" → RL. Em cada folha: métrica + armadilha.
86. **No free lunch + simplicidade primeiro + custo vs ganho** — nenhum algoritmo é sempre melhor; casas: linear venceu a floresta; interpretar, manter, explicar.

### 10 · Produção (`#producao`)
87. **Do notebook à produção** — `joblib.dump(pipeline)` / ONNX; API (FastAPI numa linha); monitorizar; diagrama de pipeline MLOps num slide (dados → treino → registo/versão → deploy → monitorização → retreino).
88. **Drift** — data drift (mudam os inputs: casas de 2025 vs 1987!) vs concept drift (muda a relação). Retreino, versionamento, reprodutibilidade (seeds, requirements, dados versionados).
89. **Erros clássicos em produção** — skew treino/serviço, features indisponíveis no momento da previsão, sem baseline, sem monitorização, threshold esquecido.

### 11 · Ética (`#etica`)
90. **Fairness e viés** — Titanic: "mulheres e crianças primeiro" estava nos dados → o modelo aprende-o; num modelo de crédito isto seria discriminação. Proxies (código postal ↔ origem).
91. **Privacidade, explicabilidade, correlação ≠ causalidade, alucinação/excesso de confiança, impacto ambiental** (treinar modelos grandes consome energia; GPU ≠ grátis).
92. **Recap + 💬** "Quem é responsável quando um modelo erra?"

### 12 · Fecho (`#fecho`)
93. **Mapa final** — mapa visual: CRISP-DM no centro ligado às 4 famílias, métricas e notebooks.
94. **Desafios para casa** — 4–5 desafios concretos ligados aos notebooks (ex.: adicionar `garage`×`prefer`, testar kNN com e sem scaling, mudar γ no gridworld, treinar NN no Titanic).
95. **Para saber mais** — fontes do PLAN §9 (links reais).
96. **Próxima lição** — "Lição 03" (placeholder neutro: "Deep Learning na prática" pode ser sugerido como hipótese, assinalado como provisório) + obrigado.

Podes fundir/dividir slides se melhorar a clareza, mantendo todas as ideias e os 7 interativos. Total alvo 85–100 slides está OK se cada um for leve; se ficares acima de 100, funde.
