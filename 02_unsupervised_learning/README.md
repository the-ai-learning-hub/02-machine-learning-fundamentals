# 02 · Aprendizagem não supervisionada

O que se aprende quando **não há respostas certas** nos dados? 🏠 [Lição 02](../README.md) · 📖 [Slides](https://the-ai-learning-hub.github.io/02-machine-learning-fundamentals/#nao-supervisionada)

| Notebook | Pergunta | Tempo | Abrir |
|---|---|---|---|
| [03 · Aprendizagem não supervisionada](notebooks/03_Unsupervised_Learning.ipynb) | Que estrutura existe sem labels? | 25 min | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/02-machine-learning-fundamentals/blob/main/02_unsupervised_learning/notebooks/03_Unsupervised_Learning.ipynb) |

## 🎯 Objetivos

- Executar o **K-Means** passo a passo (atribuir → atualizar) e perceber porque converge.
- Escolher **k** com o método do cotovelo e o *silhouette*, conhecendo os limites de cada um.
- Ver como a **escala** das variáveis muda os clusters.
- Reconhecer quando o K-Means falha e o **DBSCAN** acerta (formas não esféricas, ruído).
- Usar **PCA** para comprimir e visualizar dados com 64 dimensões (dígitos 8×8) e ler a variância explicada.

## 📋 Pré-requisitos

Notebook 01 (pipelines e *scaling*) e os slides da secção Não supervisionada.

## Dados

Sintéticos (`make_blobs`, `make_moons`) e `load_digits` do scikit-learn: não é preciso descarregar nada.
