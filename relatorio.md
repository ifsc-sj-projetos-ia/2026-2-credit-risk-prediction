# Relatório técnico: predição de risco de crédito e inadimplência

## Parte 1 — Problema, dados e metodologia

### Objetivo e escopo

O projeto compara técnicas de classificação supervisionada para sinalizar risco de inadimplência a partir de dados financeiros tabulares. A variável-alvo é explicitamente `loan_status`: **1 corresponde a default e 0 a não default**, não a crédito recusado/aprovado. A finalidade é acadêmica; não há aplicação de atendimento, API, concessão de crédito real ou recomendação financeira individual.

Como a técnica não é imposta pela disciplina, a solução privilegia modelos de árvores e uma regressão logística de referência. Dados tabulares de médio porte não justificam, por si só, redes profundas, GPU ou dependência de TensorFlow. O modelo recomendado deve emergir da avaliação, e não ser escolhido antecipadamente por complexidade.

### Requisitos funcionais

- **RF01 — Ingestão:** carregar a versão identificada da base, validar SHA-256, colunas e alvo; interromper a execução em caso de incompatibilidade, sem trocar de dataset silenciosamente.
- **RF02 — Limpeza:** remover duplicatas completas e registros incompatíveis com regras de plausibilidade; registrar quantidades removidas e preservar ausentes para imputação no treino.
- **RF03 — Exploração:** inspecionar distribuição de classes, renda, comprometimento de renda e ausentes somente no treino.
- **RF04 — Modelagem:** comparar DummyClassifier, regressão logística, árvore de decisão, Random Forest e HistGradientBoosting em pipelines independentes.
- **RF05 — Seleção e avaliação:** escolher modelo/limiar pelo F1 da classe 1 na validação; reportar precision, recall, F1, acurácia, ROC-AUC, Average Precision e matriz de confusão no teste reservado.
- **RF06 — Reprodutibilidade:** exportar resultados agregados, parâmetros, versões e figuras; fornecer notebook Colab válido, execução local automatizada e testes offline.

### Conjunto de dados e preparação

A fonte é o [Credit Risk Dataset de Lao Tse no Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset), versão 1, disponibilizado sob CC0. São **32.581 registros simulados**, 11 preditores e um alvo. Há 7.108 ocorrências da classe 1 na base original. A fonte não documenta moeda, horizonte temporal do default, representatividade geográfica ou mecanismo da simulação. Não tratamos o conjunto como carteira bancária real.

O arquivo original contém 895 ausentes em `person_emp_length` e 3.116 em `loan_int_rate`. A limpeza remove 165 duplicatas completas e 7 linhas inválidas. As regras fixas excluem idades fora de 18–100 anos, renda/valor solicitado não positivos, razão empréstimo/renda negativa e durações negativas ou superiores à idade menos 14. Este último limite é uma hipótese de plausibilidade, não um critério legal universal. Não removemos rendas altas com base em percentis ou no alvo.

Restam **32.409 registros**, divididos com estratificação e semente 42 em **19.445 para treino, 6.482 para validação e 6.482 para teste**. A prevalência da classe 1 permanece próxima de 21,9%. O notebook escolhe colunas pelo nome: nunca presume que o alvo seja a última coluna.

São utilizados seis atributos numéricos — idade, renda, tempo de emprego, valor do empréstimo, razão empréstimo/renda e duração do histórico — e três categóricos: moradia, finalidade e inadimplência anterior. `loan_grade` e `loan_int_rate` são excluídos porque podem incorporar uma avaliação prévia da instituição e não têm disponibilidade temporal assegurada. O histórico anterior é distinguido do desfecho do empréstimo atual.

A mediana dos numéricos, a moda dos categóricos e o vocabulário do One-Hot Encoder são aprendidos **somente no treino**, dentro de `Pipeline` e `ColumnTransformer`. Categorias desconhecidas são aceitas sem reestimativa do vocabulário. O StandardScaler é aplicado apenas à regressão logística; árvores não dependem da padronização z-score. Não há SMOTE nem rebalanceamento de validação/teste.

## Parte 2 — Comparação, resultados e interpretação

### Protocolo experimental

O DummyClassifier estima apenas a proporção de classes. A regressão logística é linear e regularizada, enquanto a árvore de decisão captura regras não lineares com profundidade máxima 5 e pelo menos 30 registros por folha. A Random Forest agrega 200 árvores com mínimo de dois registros por folha. O HistGradientBoosting usa até 200 iterações, taxa de aprendizado 0,08, 15 folhas por árvore e regularização L2 de 1. Os parâmetros completos são exportados; não houve busca extensa de hiperparâmetros.

Para cada modelo, o limiar é escolhido na validação maximizando o F1 da classe inadimplente. Empates de limiar favorecem o menor valor, portanto maior recall. A seleção do modelo usa F1 de validação; desempates usam Average Precision e nome. Modelo e limiar ficam congelados antes do teste, sem retreinar em treino+validação, pois isso alteraria a distribuição dos scores.

F1 combina precision e recall, mas não representa custo monetário. Sem dados de exposição, perda, receita e consequências sociais, seria incorreto inventar uma função de prejuízo financeiro. ROC-AUC usa scores, não classes previstas; Average Precision (AP) resume a curva Precision–Recall e não equivale à integração trapezoidal dessa curva. A acurácia é secundária em uma base desbalanceada.

### Resultados observados

Execução efetiva em Python 3.12.13, scikit-learn 1.6.1 e CPU, com os demais pacotes registrados em [`results/metrics.json`](results/metrics.json). A tabela mantém a ordem de seleção na validação. Os limiares usados no teste são os escolhidos anteriormente, não reajustados pelo teste.

| Modelo | F1 validação | Precision teste | Recall teste | F1 teste | ROC-AUC teste | AP teste |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **HistGradientBoosting** | **0,6957** | **0,6937** | **0,7059** | **0,6998** | **0,9016** | **0,8160** |
| Random Forest | 0,6713 | 0,7859 | 0,6058 | 0,6842 | 0,8758 | 0,7859 |
| Árvore de decisão | 0,6167 | 0,7900 | 0,5331 | 0,6366 | 0,8208 | 0,6938 |
| Regressão logística | 0,5685 | 0,5458 | 0,5719 | 0,5585 | 0,8041 | 0,6224 |
| Dummy (prior) | 0,3590 | 0,2188 | 1,0000 | 0,3590 | 0,5000 | 0,2188 |

O HistGradientBoosting foi selecionado com limiar **0,259819**. No teste, obteve 4.622 verdadeiros negativos, 442 falsos positivos, 417 falsos negativos e 1.001 verdadeiros positivos, totalizando 6.482 registros. Sua acurácia foi 86,75%; a Random Forest chegou a 87,77%, mas teve menor F1 e recall. A diferença mostra por que escolher o modelo apenas por acurácia seria inadequado ao critério definido.

No limiar padrão 0,5, o mesmo HistGradientBoosting teve precision 93,33%, recall 53,24% e F1 0,6780. O limiar validado elevou recall para 70,59%, com precision de 69,37%: os falsos negativos caíram de 663 para 417 e os falsos positivos aumentaram de 54 para 442. O compromisso foi aceito pelo critério F1, não por uma avaliação econômica comprovada.

Ao otimizar F1, o Dummy classifica todos como risco: seu recall 1 não indica qualidade. No limiar 0,5, prevê apenas adimplência, com F1 zero. A Random Forest apresenta F1 de treino 0,9682 contra 0,6713 na validação, sinal de sobreajuste. O boosting tem diferença menor: 0,7461 no treino contra 0,6957 na validação.

As quatro figuras exportadas mostram exploração do treino, comparação/limiares na validação, importância por permutação e avaliação final. A permutação usa uma amostra fixa de até mil registros da validação, três repetições e queda de AP; renda, moradia e comprometimento de renda se destacam nesta execução. Isso não implica causalidade nem explica individualmente uma decisão.

## Parte 3 — Ética, limitações e conclusão

### Riscos e uso responsável

Um falso negativo é inadimplência não sinalizada; um falso positivo é um caso adimplente sinalizado como risco. Ambos têm consequências possíveis, mas a classificação não é, por si só, aprovação ou recusa. Inadimplência pode decorrer de desemprego, doença ou mudanças econômicas; o modelo não mede honestidade ou intenção de pagamento.

Idade, renda, moradia e históricos podem reproduzir desigualdades socioeconômicas. A ausência de identificadores explícitos não elimina proxies nem demonstra equidade. Não foi realizada uma auditoria completa de subgrupos, pois faltam atributos, contexto e dados reais adequados. Não se pode concluir que o modelo é justo ou transportável para o Brasil.

Qualquer aplicação real exigiria fundamento legal e governança sob LGPD, controle de acesso e minimização de dados, explicações compreensíveis, supervisão humana e possibilidade de contestação. Seriam necessários monitoramento de drift, avaliações de impacto, validação externa e calibração dos scores. As saídas de `predict_proba` neste experimento não foram calibradas.

### Limitações e trabalhos futuros

Foi utilizada uma única divisão aleatória, sem intervalos de confiança ou testes de significância. A validação fica otimista após escolher nela modelo e limiar; a avaliação final é a do teste reservado. Não se deve reutilizar repetidamente esse teste para orientar ajustes. A remoção de duplicatas completas não identifica pessoas: podem restar perfis iguais após excluir colunas, e não há garantia de independência por cliente. Sem datas e IDs, não é possível realizar validação temporal ou por pessoa.

Como próximos passos, recomenda-se validação cruzada estratificada no desenvolvimento, avaliação de estabilidade por sementes e ajuste moderado de hiperparâmetros. CatBoost pode ser estudado para atributos categóricos; XGBoost e LightGBM são alternativas de boosting. Redes neurais só devem ser acrescentadas mediante hipótese e comparação justa. Novos experimentos precisam preservar um teste realmente independente.

**Conclusão:** neste protocolo e nesta base simulada, HistGradientBoosting foi a melhor alternativa pelo F1 da validação e manteve F1 próximo de 0,70 no teste reservado. O resultado sustenta sua escolha para a demonstração acadêmica, não para operação financeira. O notebook é executável em CPU/Colab; 14 testes offline e a execução integral em kernel novo verificam o pipeline. Nenhum resultado ou gráfico foi simulado para apresentação.

### Referências

1. [Lao Tse — Credit Risk Dataset, versão 1, Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset).
2. [scikit-learn 1.6 — Prevenção de vazamento de dados](https://scikit-learn.org/1.6/common_pitfalls.html#data-leakage).
3. [scikit-learn 1.6 — Ajuste do limiar de classificação](https://scikit-learn.org/1.6/modules/classification_threshold.html).
4. [scikit-learn 1.6 — HistGradientBoostingClassifier](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).
5. [scikit-learn 1.6 — Average Precision](https://scikit-learn.org/1.6/modules/generated/sklearn.metrics.average_precision_score.html).
6. [scikit-learn 1.6 — Importância por permutação](https://scikit-learn.org/1.6/modules/permutation_importance.html).
