# 03 · Redes neuronais

Abrir a caixa-preta: do perceptrão de 1958 a uma rede em PyTorch, e quando é que uma GPU compensa. 🏠 [Lição 02](../README.md) · 📖 [Slides](https://the-ai-learning-hub.github.io/02-machine-learning-fundamentals/#redes-neuronais)

| Notebook | Pergunta | Tempo | Abrir |
|---|---|---|---|
| [04 · Redes neuronais de raiz](notebooks/04_Neural_Networks_From_Scratch.ipynb) | O que acontece dentro da caixa-preta? | 30 min | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/02-machine-learning-fundamentals/blob/main/03_neural_networks/notebooks/04_Neural_Networks_From_Scratch.ipynb) |

## 🎯 Objetivos

- **Parte 1 · NumPy:** ver o perceptrão resolver AND/OR e falhar no XOR; escrever um MLP com *forward pass*, *loss*, *backprop* e gradiente descendente; confirmar o *backprop* com um **gradient check** numérico.
- **Parte 2 · PyTorch:** tensores, *autograd*, `nn.Module` e um ciclo de treino com curvas de treino e validação e regularização.
- **Parte 3 · CPU vs GPU:** medir a multiplicação de matrizes com aquecimento, repetições e sincronização, e perceber quando a CPU chega.

## 📋 Pré-requisitos

Notebooks 01–02 (treino/validação, *overfitting*) e os slides das secções Redes neuronais e CPU vs GPU.

## GPU (opcional)

O notebook corre só com CPU. Com GPU NVIDIA (CUDA) ou Apple Silicon (MPS), a Parte 3 compara automaticamente os dois. No Colab: *Runtime → Change runtime type → GPU*.
