# Relatório Técnico: Predição de Avaliação de Livros

## 1. Escopo e Requisitos Funcionais
O sistema tem como objetivo processar um dataset tabular em formato CSV contendo informações estatísticas de 10.000 livros para prever sua classe de avaliação.
* **Requisito 1:** Carregar e pré-processar os dados tratando valores nulos e escalonando colunas numéricas.
* **Requisito 2:** Treinar um modelo baseline com scikit-learn.
* **Requisito 3:** Treinar uma Rede Neural Profunda em Keras/TensorFlow, plotando as curvas de perda e acurácia por época.
* **Requisito 4:** Avaliar e comparar o desempenho final usando conjunto de teste isolado (20%).

## 2. Análise Ética e Limitações
* **Viés de Popularidade:** O dataset é composto por livros populares no Goodreads, o que pode inclinar as previsões a favorecer obras com maior número bruto de avaliações em detrimento de nichos específicos (como webnovels recentes).
* **Limitação de Atributos:** O modelo utiliza apenas estatísticas de engajamento e metadados, sem analisar o conteúdo textual do livro.
* **Uso Responsável:** A ferramenta deve servir como auxílio analítico preliminar, e não como único critério para decisões editoriais.
