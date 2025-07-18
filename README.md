# Credit Scoring Pipeline
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/joblib-1.4.2-blue.svg" alt="joblib 1.4.2"/>
  <img src="https://img.shields.io/badge/numpy-2.3.1-blue.svg" alt="numpy 2.3.1"/>
  <img src="https://img.shields.io/badge/pandas-2.3.1-blue.svg" alt="pandas 2.3.1"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.4.2-blue.svg" alt="scikit‑learn 1.4.2"/>
</p>

Este repositório contém o **pipeline completo** desenvolvido como projeto final da disciplina de Machine Learning. O objetivo foi criar um modelo enxuto, interpretável e bem calibrado para prever risco de crédito a partir de um conjunto de 51 variáveis.

---

## Contexto

- **Dados**: 2.100 amostras, 51 variáveis numéricas (`feat_1` … `feat_51`) e alvo binário (`class`).
- **Desafio**: evitar vazamento de informação, reduzir dimensionalidade, obter probabilidades confiáveis e garantir alta performance e interpretabilidade.

## Objetivo

1. **Filtrar** automaticamente features com correlação muito alta (|corr| > 0.9) em relação ao alvo;
2. **Selecionar** as 3 variáveis mais preditivas (`feat_50`, `feat_17`, `feat_8`) com feature\_importances\_ via árvore, e confirmar seus impactos com RFECV, permutation\_importance e SHAP;
3. **Treinar** um classificador KNN ajustado (parâmetros otimizados via GridSearchCV) usando RepeatedStratifiedKFold para validação cruzada;
4. **Recalibrar** probabilidades (Isotonic) para reduzir o Brier Score de 0.0063 para 0.0025;
5. **Validar** performance final com `cross_validate`, obtendo as seguintes métricas médias e desvios-padrão:
   - **Precision**: média = 98.49%, desvio padrão = 0.6515
   - **Accuracy**: média = 98.73%, desvio padrão = 0.5484
   - **Balanced Accuracy**: média = 98.62%, desvio padrão = 0.6912
   - **Recall**: média = 98.18%, desvio padrão = 1.3878
   - **F1 Score**: média = 98.33%, desvio padrão = 0.7325
   - **ROC-AUC**: média = 99.77%, desvio padrão = 0.2834

## Estrutura de diretórios

```
credit_scoring_project/
├── data/                         # Dados brutos (.csv) e dados de teste gerado (.csv)
├── model/                        # Modelos serializados (.joblib)
├── src/
│   ├── pipeline.py               # Build e treino do pipeline
│   └── predict.py                # Funções para carregar o modelo e gerar predições
├── output/                       # Exemplo de saídas do predict.py
├── test/                         # Função para gerar uma base de teste dirivada da original 
├── Exploração & Modelagem.ipynb  # Notebook Jupyter com a modelagem feita
├── requirements.txt              # Dependências Python
└── README.md
```

## Instalação

1. Clone este repositório:
   ```bash
   git clone https://seu-repo.git
   cd credit_scoring_project
   ```
2. Crie um ambiente virtual e instale dependências:
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Linux/macOS
   .\.venv\Scripts\activate   # Windows PowerShell
   pip install -r requirements.txt
   ```

## Como treinar o modelo

1. Certifique‑se de que os arquivos `pipeline.py` e `predict.py` estejam na raiz do projeto, ao lado de `README.md`.
2. Garanta que o CSV de dados brutos está em `data/raw/dados.csv`.
3. Abra o terminal na raiz do repositório e execute:
   ```bash
   python - << "EOF"
   from pipeline import train_and_export

   # Ajuste os caminhos conforme sua estrutura:
   train_and_export(
       df_path='data/dados.csv',
       model_path='model/knn_pipeline.joblib'
   )
   EOF
   ```

Isso irá:

1. Carregar `data/dados.csv`;
2. Separar em treino (80%), validação (10%) e teste (10%);
3. Filtrar features de alta correlação, selecionar variáveis via RandomForest + RFECV, treinar KNN e recalibrar probabilidades;
4. Avaliar performance em treino, validação e teste (mostrando um print de todas as métricas);
5. Salvar o pipeline final em `models/knn_pipeline.joblib`.

## Como gerar predições

Use o script `predict.py` na raiz do projeto em um script ou notebook:

```python
import pandas as pd
from predict import predict

# 1) Carregue novos dados com as mesmas colunas originais (feat_1 … feat_51):
df_new = pd.read_csv('data/test_data_for_inference.csv')

# 2) Gere predições e scores de crédito:
df_out = predict(df_new)

# 3) Salve resultados:
df_out.to_csv('predictions.csv', index=False)
```

O DataFrame retornado contém as colunas originais mais as seguintes colunas adicionais:

- `class_pred`: previsão da classe (0 ou 1);
- `prob`: probabilidade da classe positiva em porcentagem (valor de 0.00 a 100.00 com duas casas decimais);
- `score`: score de crédito que varia de 0 a 1000 (quanto menor, maior o risco), calculado como:

  score = (1 - prob) × 1000,

  onde `prob` é a probabilidade prevista de inadimplência (entre 0 e 1).
- `class_score`: faixa de risco associada ao score (`Risco muito baixo`, `Risco baixo`, `Risco médio`, `Risco alto` ou `Risco muito alto`)

Esta classificação de `score` segue o padrão de risco utilizado pela Serasa — conforme os limites abaixo — e pode ser facilmente adaptada para refletir qualquer outra política de crédito definida pela empresa:

```
if score >= 901:
    return "Risco muito baixo"
elif score >= 701:
    return "Risco baixo"
elif score >= 501:
    return "Risco médio"
elif score >= 301:
    return "Risco alto"
else:
    return "Risco muito alto"
```

## Resultados Finais

**Treinamento**

- Acurácia: 100.00%
- Precisão: 100.00%
- Acurácia Balanceada: 100.00%
- Recall: 100.00%
- F1 Score: 100.00%
- ROC-AUC: 100.00%

**Validação**

- Acurácia: 96.67%
- Precisão: 96.73%
- Acurácia Balanceada: 96.83%
- Recall: 96.67%
- F1 Score: 96.68%
- ROC-AUC: 98.95%

**Teste**

- Acurácia: 99.52%
- Precisão: 99.53%
- Acurácia Balanceada: 99.62%
- Recall: 99.52%
- F1 Score: 99.52%
- ROC-AUC: 100.00%

As três variáveis mais relevantes para o modelo são `feat_8`, `feat_17` e `feat_50`. Recomenda-se priorizar esses atributos tanto na etapa de treinamento quanto na análise dos perfis de clientes. É fundamental considerá-las tanto no treinamento quanto na análise dos clientes.

Na pasta `output` há um exemplo de saída das previsões do modelo com uma base de dados sintética gerada pela função `gerar_dados_sinteticos_para_testes.py` para simular novos dados a partir das informações da base `data/dados.csv`. Essa base gerada pode ser consultada em `data/test_data_for_inference.csv`.

---

## Contato

Se tiver dúvidas, sugestões ou quiser contribuir com o projeto, fique à vontade para abrir uma issue ou enviar um e‑mail para:

**Jonas Siqueira**\
[jonas11.siqueira@gmail.com](mailto\:jonas11.siqueira@gmail.com)

