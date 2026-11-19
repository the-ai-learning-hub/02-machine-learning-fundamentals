# REVISION · ronda 2 (feedback do professor)

Feedback do utilizador (professor), por ordem:
1. **Muitos erros de visualização na apresentação** (ver auditoria abaixo).
2. **Os alunos estão a ver isto pela primeira vez: temos de explicar tudo, também nos slides.**
3. **Download dos dados: das fontes oficiais, não do repositório.** ✅ Já feito em `src/data.py`:
   casas ← arquivo oficial do *Journal of Applied Econometrics* (`http://qed.econ.queensu.ca/jae/1996-v11.6/anglin-gencay/ag-data.zip`, só http; fallback: cópia do pacote R AER); Titanic ← Vanderbilt (`https://hbiostat.org/data/repo/titanic3.csv`; fallback OpenML 40945). `data/HousePrices.csv` **foi removido** do repo; os ficheiros descarregados ficam em `data/` (git-ignored). Os dados são idênticos aos anteriores (números do PLAN §5 mantêm-se).
4. **Boas visualizações e explicações dos modelos e dos datasets, porque eles não os conhecem.**

Factos de referência para as novas explicações: `_planning/model_facts.json` (histograma de preços, 5 linhas reais de cada dataset, % de "yes" por coluna binária, sobrevivência por sexo/classe) e:
- **Árvore do Titanic** (mesma no deck e no NB02): `DecisionTreeClassifier(max_depth=3, random_state=42)` treinada no split de treino (80 %, estratificado, `random_state=42`) com 4 features simples: `is_female` (sex == female), `pclass` (1–3), `age` (imputada com `GroupMedianImputer` ajustado no treino), `family_size`. Accuracy treino 0,811 · teste 0,836. Estrutura (contagens [morreu, sobreviveu] no treino):
  - homem · idade ≤ 12,5 · família ≤ 4,5 → [5, 24] **sobrevive** (rapazes de famílias pequenas)
  - homem · idade ≤ 12,5 · família > 4,5 → [19, 1] morre
  - homem · idade > 12,5 · 1.ª classe → [92, 43] morre
  - homem · idade > 12,5 · 2.ª/3.ª → [427, 64] morre
  - mulher · 1.ª → [3, 107] sobrevive · mulher · 2.ª → [12, 72] sobrevive
  - mulher · 3.ª · família ≤ 4,5 → [67, 86] sobrevive · mulher · 3.ª · família > 4,5 → [22, 3] morre
- Sobrevivência: mulheres 72,7 %, homens 19,1 % · 1.ª classe 61,9 % (323 pass.), 2.ª 43,0 % (277), 3.ª 25,5 % (709) · 843 homens, 466 mulheres.
- **Gradient boosting, passo a passo** (casas, só `lotsize`, `GradientBoostingRegressor(n_estimators=3, max_depth=1, learning_rate=1.0, random_state=0)`): começa na média 68 122 CAD; casa nº 0 (real 42 000): 57 639 → 61 346 → 63 483; casa nº 10 (real 90 000): 88 745 → 92 452 → 87 156. Cada árvore pequena corrige o erro que sobrou. (Usa como exemplo didático; com 1 feature e 3 árvores não chega perto de todas as casas — diz isso.)

---

## A · DECK (`index.html`) — dono: agente do deck

### A1. Regras de design novas (aplicar a TODOS os slides)
- **Tamanhos mínimos no canvas 1280×720:** corpo/lead ≥ 20 px; texto em cartões/listas ≥ 18 px; legendas/rótulos de eixos/notas ≥ 15 px; texto SVG efetivo ≥ 16 px; código ≥ 16 px. Kickers/rail/pills podem ficar como estão.
- **Ocupar o slide:** o conteúdo deve usar a altura disponível (nada de 40 % de baixo vazio). Se um slide fica vazio, aumenta o visual, a letra, ou acrescenta a explicação (A2). Se um slide fica cheio demais com letra grande, divide-o em dois.
- **Explicar para principiantes:** cada slide de conteúdo tem uma frase "**Em palavras simples:** …" (ou um lead equivalente) que explica a ideia sem jargão, e **todo o termo técnico é definido na primeira vez que aparece** (ex.: "*feature* = coluna que o modelo usa para prever", "RMSE = erro típico, na mesma unidade do alvo"). Limite de palavras sobe para ~60 no corpo; mais do que isso → dois slides.
- **Sem sobreposições** de texto com texto, setas ou pontos. Nada cortado.
- **Chuva de código dos divisores:** sem palavras sobrepostas entre si (colisão/espaçamento por colunas), longe do título e do cartão de mapa (só na faixa direita/topo livre), opacidade baixa (≤ 0,35), menos palavras. No último slide (96) não pode tapar o título.

### A2. Auditoria — erros concretos a corrigir (nº = slide atual)
- **4, 6, 9, 13, 15, 16, 18, 20, 26, 37, 38, 40, 41, 43, 44, 47, 48, 55, 56, 68, 73, 76, 77, 81, 84, 86, 88, 90, 91, 92, 95:** conteúdo pequeno e encostado em cima, metade inferior vazia → aplicar A1 (letra maior, preencher, explicação simples).
- **Recaps (13, 26, 41, 48, 56, 68, 73, 84, 92):** bloco centrado com topo vazio; texto pequeno. Fazer layout a ocupar o slide, letra ≥ 20 px.
- **10 (anel CRISP-DM):** rótulo "pergunta mal respondida?" sobre a seta; setas pequenas; painel da direita com muito vazio e texto pequeno — mostrar por omissão a fase 1 selecionada.
- **12 (ciclo):** a seta tracejada vermelha atravessa o texto da caixa ("Erro médio de 17,5 %…"). Reposicionar.
- **19 (scaling):** os gráficos "antes" e "depois" são idênticos — não mostram nada. Refazer: "antes" com eixos na escala real (lotsize 1 650–16 200 no x, bedrooms 1–6 no y, desenhados à mesma escala física, para se ver que a distância é dominada por lotsize) e um exemplo numérico de distância entre duas casas antes/depois; "depois" com ambos ~[-2, 3].
- **28:** gráfico de classificação com "idade (anos) → esquema" confuso; explicar eixos.
- **31 (regularização), 33 (ensembles):** texto dos diagramas minúsculo.
- **34 (polinómio):** legenda no topo cortada ("amarelo = …"); mini-gráfico com rótulos sobrepostos.
- **45 (ROC/PR):** rótulo "AP = 0,856" por cima da curva; eixos sem valores.
- **46 (threshold):** rótulos do eixo x a colidir com a legenda; texto final minúsculo.
- **61 (XOR):** rótulos "OR"/"NAND" por cima dos pontos ("N..D"); diagrama confuso — refazer com os 4 pontos rotulados (0,0) (0,1) (1,0) (1,1) e as duas retas legendadas fora dos pontos.
- **62 (forward pass):** passos à direita com letra pequena e pouco contraste antes de clicar; mostrar também os pesos/bias de cada ligação legíveis.
- **69, 74, 96 (divisores):** chuva com palavras sobrepostas ("GFLOP/s GELOP/s", "gamma"), e a tapar o título no 96.
- **80 (gridworld):** gráfico vazio e texto minúsculo antes de treinar; legenda pequena. Mostrar eixos/legenda legíveis e uma frase "carrega em ▶ Treinar".
- **87 (produção):** seta tracejada atravessa "retreine quando os dados mudam"; linha de código cortada à direita; bullets espalhados.
- Verificar todos os outros slides com o mesmo critério.

### A3. Slides novos (explicar dados e modelos)
1. **Vocabulário essencial** (logo após "As 4 famílias"): uma tabela real (3 linhas das casas de Windsor, de `model_facts.json`) anotada visualmente: *linha = exemplo* (uma casa), *colunas = features*, *coluna alvo = label* (`price`, destacada), e definições curtas de: dataset, feature, label/alvo, modelo, treinar (`fit`), prever (`predict`), erro.
2. **Como aprende um modelo** (ciclo geral): prever → comparar com a resposta certa → medir o erro → ajustar os parâmetros → repetir. Visual em ciclo + exemplo com 1 casa.
3. **Nova secção "Conhecer os dados"** (`data-id="datasets"`, entre CRISP-DM e Preparação de dados; atualizar o mapa/tracker de secções e a agenda):
   - **Casas de Windsor (1/2):** o quê/onde/quando (Windsor, Ontário, Canadá; 546 casas vendidas jul–set 1987; preço em CAD de 1987; fonte oficial: arquivo do *Journal of Applied Econometrics*, Anglin & Gençay 1996). Tabela com as 5 primeiras linhas reais, cabeçalhos com tradução.
   - **Casas de Windsor (2/2):** as 12 colunas explicadas em grupos com ícones (tamanho: `lotsize` em pés² ≈ ×0,093 m²; divisões: quartos, casas de banho, pisos, garagem; extras sim/não com % de "yes"; alvo `price`) + histograma real dos preços (`price_hist`) com mediana 62 000 marcada. Pergunta do NB01.
   - **Titanic (1/2):** a história (15 de abril de 1912; 1309 passageiros no dataset; 38,2 % sobreviveram; "mulheres e crianças primeiro"); fonte oficial (Vanderbilt). Tabela de 5 passageiros reais com colunas traduzidas.
   - **Titanic (2/2):** colunas explicadas (classe, sexo, idade, irmãos/cônjuges, pais/filhos, tarifa, porto, cabine; `boat`/`body` marcadas como "só se sabem depois → proibidas") + barras reais de sobrevivência por sexo e por classe.
   - **Os outros dados da aula:** `make_blobs`/`make_moons` ("dados inventados de propósito: sabemos a resposta certa"), dígitos 8×8 (desenha 10 imagens REAIS de `sklearn.datasets.load_digits`, uma por dígito, como grelhas SVG de 8×8), e o mapa do gridworld com legenda (S, G, L, paredes).
4. **Fichas dos modelos** — um slide por modelo com o MESMO template visual ("Ficha do modelo"): **Ideia (1 frase simples)** · **Como aprende (passos 1-2-3)** · **Visual** · **Bom para / Cuidado com** · **Onde: NBxx §y**. Reaproveita/melhora os slides existentes onde já houver visual:
   - Baseline (dummy) · Regressão linear · Ridge/Lasso · Regressão logística · **Árvore de decisão** (usa a árvore REAL acima, com contagens e % de sobrevivência por folha, substituindo o esquema ilustrativo do slide 32) · **Random forest** (3 árvores diferentes a votar num passageiro concreto → votação) · **Gradient boosting** (os passos numéricos acima) · kNN (visual dos k vizinhos) · **K-Means** (ficha + interativo existente) · **DBSCAN** (visual com raio eps, pontos núcleo/fronteira/ruído) · **PCA** · **Perceptrão/neurónio** · **MLP** · **Q-learning**.
   - Um slide final de secção supervisionada: **tabela-resumo dos modelos** (modelo · ideia · interpretável? · precisa de scaling? · onde).
5. Atualizar os ids, o tracker, a agenda e as notas do orador (com tempos) para incluir tudo. O deck pode crescer para ~120–130 slides.

### A4. Já corrigido diretamente no `index.html` (porta para o teu `build.py` se regenerares!)
- Anel CRISP-DM: secções = Business "NB01 §1 · NB02 §1", Data "NB01 §2 · NB02 §2", Prep "NB01 §3 · NB02 §3–4", Modeling "NB01 §4 · NB02 §5", Evaluation "NB01 §5 · NB02 §6", Deployment "NB01 §6 · NB02 §7".
- `BENCH` preenchido (valores de `_planning/benchmark.md`, CPU "AMD Ryzen 7 7735HS (8 núcleos / 16 threads)", `transfer_ms: 11.059`, nota) e mensagem com o custo de transferência.
- Slide Q-learning: rótulo "Exemplo didático · α = 0,5 · γ = 0,95" com o α/γ fora do `text-transform:uppercase`.
- Slide Pipeline: acrescentado `from sklearn.linear_model import LogisticRegression`.
- Dados: as menções a "AER/GPL/no repositório" devem passar a "fonte oficial: arquivo do JAE / Vanderbilt; descarregado no 1.º uso".

### A5. Verificação
Corre os testes de `_planning/deck_tests/` (ajusta índices/ids) + uma verificação automática nova: **nenhum texto com tamanho efetivo < 15 px** (exceto kicker/rail/pill), **nenhuma sobreposição** entre caixas de texto (bounding boxes de elementos de texto folha que se intersectam) e **área vazia**: a caixa que envolve o conteúdo do `.main` deve ocupar ≥ 75 % da altura útil. Revê visualmente TODOS os slides (folhas de contacto 2×3 com PIL) e corrige o que vires.

---

## B · NOTEBOOKS — donos: agente NB01–02 e agente NB03–05

Para cada notebook (usar os builders em `_planning/builders/`, reconstruir e reexecutar com `uv run jupyter nbconvert …` a partir da raiz; `.venv` do projeto, só CPU — **exceto NB04**: executa-o com o Python do scratchpad que tem CUDA, para manter o benchmark GPU):
1. **Dados das fontes oficiais:** texto do notebook diz de onde vêm os dados (link da fonte oficial) e que são descarregados no 1.º uso para `data/`; nada sobre "ficheiro no repositório", AER/GPL como fonte principal, ou `HousePrices.csv`.
2. **🗂️ Conhecer os dados** (logo a seguir ao carregamento, antes da EDA técnica):
   - "Ficha do dataset" em markdown: O que é · De onde vem (fonte oficial + autores) · Quando/onde · Uma linha = … · Alvo = … · Tamanho · Unidades.
   - Tabela com TODAS as colunas: nome · significado em PT-PT · tipo · exemplo · unidade.
   - `df.head()` + uma figura-resumo "uma coluna por painel" com títulos em PT-PT (histogramas para numéricas, barras para categóricas) com 1 frase de leitura.
   - NB02: também sobrevivência por sexo e por classe em barras com %; NB03: imagens dos dígitos (grelha 2×5) e os datasets sintéticos explicados ("porque inventamos dados"); NB04: tabela-verdade XOR desenhada, luas, dígitos; NB05: mapa legendado.
3. **🧠 Como funciona: <modelo>** antes de usar cada modelo: markdown com ideia simples + analogia + passos + quando usar/cuidados, **e uma célula com uma visualização do mecanismo nos dados reais**:
   - NB01: baseline (histograma do preço com a linha da média), regressão linear com 1 feature (`lotsize` vs `price`, reta + resíduos desenhados), Ridge (caminho dos coeficientes vs α), árvore de decisão (`plot_tree` de profundidade 2 com nomes PT), random forest (previsões de 5 árvores individuais vs média para 3 casas), gradient boosting opcional (os passos de `REVISION` acima).
   - NB02: regressão logística (curva sigmoide numa variável, p.ex. tarifa ou idade, com os pontos 0/1), **a árvore de profundidade 3 definida acima** (`plot_tree`, com a leitura das folhas em texto), random forest (votos das árvores para 2 passageiros), HistGradientBoosting (accuracy de validação à medida que se juntam árvores, se disponível; senão `GradientBoostingClassifier` staged).
   - NB03: K-Means (já tem passos — acrescenta a explicação), DBSCAN (figura com o raio eps à volta de um ponto núcleo, um de fronteira e um de ruído), PCA (setas das componentes sobre uma nuvem 2D).
   - NB04: perceptrão (diagrama da soma ponderada), MLP (diagrama 2-4-1), gradiente descendente (curva + passos).
   - NB05: Q-learning (ciclo agente-ambiente e a tabela Q explicada).
4. **Explicar tudo:** cada termo técnico definido na 1.ª ocorrência; acrescenta no início do NB01 um "📖 Vocabulário" curto (dataset, linha, feature, alvo/label, modelo, treinar, prever, erro, baseline, overfitting) e nos outros notebooks um link para ele.
5. Manter tudo o que já existe (discussões, exercícios, números). Os números canónicos não mudam.
6. Reexecutar, validar (`nbformat.validate`), confirmar 0 erros/stderr e **nenhum caminho local** nos outputs (o setup imprime `ROOT.name`). Notebooks < 2,5 MB.
