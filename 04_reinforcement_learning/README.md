# 04 · Reinforcement learning

Um agente aprende a atravessar um labirinto com lava **só com recompensas**, sem nunca lhe dizermos o caminho. 🏠 [Lição 02](../README.md) · 📖 [Slides](https://the-ai-learning-hub.github.io/02-machine-learning-fundamentals/#rl) (com a demonstração ao vivo)

| Notebook | Pergunta | Tempo | Abrir |
|---|---|---|---|
| [05 · Projeto de reinforcement learning](notebooks/05_Reinforcement_Learning_Project.ipynb) | Como aprende um agente por tentativa e erro? | 30 min | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/02-machine-learning-fundamentals/blob/main/04_reinforcement_learning/notebooks/05_Reinforcement_Learning_Project.ipynb) |

## 🎯 Objetivos

- Descrever um problema como **MDP**: estados, ações, recompensas, retorno descontado (γ).
- Escrever **Q-learning tabular** à mão e fazer uma atualização com números.
- Ler a **Q-table** como *heatmap*, as setas da política, a curva de retorno com média móvel e a taxa de sucesso.
- **Avaliar** a política separadamente do treino (ε = 0, várias seeds, ambiente escorregadio).
- Ver ***reward hacking*** ao vivo: uma moeda mal pensada faz o retorno subir e o sucesso cair para 0 %.

## 📋 Pré-requisitos

Python e NumPy (Lição 01) e os slides da secção Reinforcement learning. Corre em CPU em menos de um minuto.

Figuras e animações do projeto: [`assets/rl/`](../assets/rl/).
