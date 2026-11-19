# 01 · Aprendizagem supervisionada com CRISP-DM

Dois projetos completos, do problema de negócio ao modelo pronto a usar: um de **regressão** (prever um número) e um de **classificação** (prever uma categoria). 🏠 [Lição 02](../README.md) · 📖 [Slides](https://the-ai-learning-hub.github.io/02-machine-learning-fundamentals/#supervisionada)

| Notebook | Pergunta | Tempo | Abrir |
|---|---|---|---|
| [01 · CRISP-DM: preços de casas](notebooks/01_CRISP-DM_House_Prices.ipynb) | Que fatores mais pesam no preço de uma casa? | 45 min | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/02-machine-learning-fundamentals/blob/main/01_supervised_learning/notebooks/01_CRISP-DM_House_Prices.ipynb) |
| [02 · CRISP-DM: sobreviventes do Titanic](notebooks/02_CRISP-DM_Titanic_Survival.ipynb) | Que tipo de pessoa sobrevive? | 45 min | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/02-machine-learning-fundamentals/blob/main/01_supervised_learning/notebooks/02_CRISP-DM_Titanic_Survival.ipynb) |

**Ordem:** 01 → 02 (o 02 assume a ideia de *pipeline* e de *baseline* do 01).

## 🎯 Objetivos

- Percorrer as **seis fases do CRISP-DM** num caso real, com uma pergunta de discussão em cada fase.
- Explorar dados: distribuições, correlações, valores em falta, *outliers* e interações.
- Construir *pipelines* com `ColumnTransformer` + `Pipeline`, dividir treino/teste **antes** de qualquer `fit` e usar validação cruzada.
- Comparar modelos com um ***baseline***: regressão linear/Ridge, regressão logística, *random forest*, *gradient boosting*.
- Avaliar com as métricas certas: MAE, RMSE, R² e resíduos; *accuracy*, *precision*, *recall*, F1, ROC-AUC, PR-AUC e matriz de confusão.
- Escolher o ***threshold*** pelo custo dos erros e reconhecer ***data leakage*** (`boat` no Titanic).
- Guardar o modelo com `joblib` e discutir limites, *drift* e ética.

## 📋 Pré-requisitos

pandas e gráficos ([Lição 01](https://github.com/the-ai-learning-hub/01-python-as-ai-first-steps)) e os slides das secções CRISP-DM, Dados, Supervisionada e Métricas.

## Dados

[Casas de Windsor, 1987](../data/DATA_DICTIONARY.md#1-preços-de-casas-em-windsor-canadá) (arquivo oficial do *Journal of Applied Econometrics*) e [Titanic `titanic3`](../data/DATA_DICTIONARY.md#2-titanic-titanic3) (Vanderbilt University). O código descarrega ambos diretamente das fontes oficiais (nenhum ficheiro de dados fica no projeto).
