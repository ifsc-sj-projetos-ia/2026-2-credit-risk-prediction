# 📚 Predição de Avaliação de Livros com Deep Learning (Goodbooks-10k)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([suspicious link removed])
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📌 Problema e Motivação
No mercado editorial e em plataformas de publicação independente (como webnovels), identificar precocemente o potencial de sucesso e aceitação de uma obra é fundamental para estratégias de marketing e recomendação. Este projeto desenvolve um modelo de **Deep Learning** para classificar se uma obra atingirá alta avaliação (Rating $\ge 4.0$) a partir de metadados tabulares.

## 📊 Conjunto de Dados
* **Fonte:** `goodbooks-10k` (Kaggle / GitHub)
* **Tamanho:** 10.000 livros com metadados de avaliações, revisões de texto, ano de lançamento e autores.
* **Tratamento:** Limpeza de nulos, normalização com `StandardScaler` e conversão do target para classificação binária.

## 🛠️ Tecnologias e Arquitetura
* **Linguagem:** Python 3 (Google Colab com GPU)
* **Modelagem:**
  * *Baseline:* Regressão Logística e MLP Simples (`scikit-learn`)
  * *Deep Learning:* Rede Neural Profunda Sequencial em `Keras/TensorFlow` com camadas `Dense` e `Dropout` para prevenção de overfitting.

## 📈 Resultados Mínimos Esperados
* Comparativo entre o modelo Baseline e a Rede Neural em Keras.
* Avaliação via Acurácia, F1-Score, Loss e Matriz de Confusão.

## 🚀 Como Executar
Clique no botão **Open in Colab** no topo deste README para abrir e executar diretamente no Google Colab.

## 📜 Licença
Este projeto está sob a licença MIT.
