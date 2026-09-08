# Modelo de Expected Goals (xG) com Aprendizado de Máquina

Trabalho de Conclusão de Curso (TCC) — Curso de Ciência da Computação
Universidade Tiradentes (UNIT) — Aracaju, Sergipe — 2026

**Título:** Modelo de Expected Goals com aprendizado de máquina: comparativo entre Regressão Logística, Random Forest e XGBoost

**Autor:** Paulo Gustavo Angelo de Barros
**Orientador:** Luiz Gomes Da Cunha Neto

## Sobre o projeto

Este projeto compara três algoritmos de aprendizado de máquina — Regressão Logística, Random Forest e XGBoost — para a modelagem de Expected Goals (xG) no futebol, utilizando dados públicos da StatsBomb Open Data. Além do comparativo experimental, o trabalho integra o modelo de melhor desempenho a um dashboard interativo desenvolvido em Streamlit.

## Base de dados

- FIFA World Cup 2018 — 64 partidas (treinamento)
- FIFA World Cup 2022 — 64 partidas (teste principal)
- UEFA Euro 2020 — 51 partidas (teste externo)

Dados obtidos via [StatsBomb Open Data](https://github.com/statsbomb/open-data) através da biblioteca `statsbombpy`.

## Estrutura do repositório

```
expected-goals-ml/
├── data/
│   ├── raw/                  # Dados brutos da StatsBomb (não versionados)
│   └── processed/            # Datasets com features (não versionados)
├── notebooks/                 # Jupyter para exploração
│   ├── 01_coleta.ipynb        # Coleta e exploração inicial da StatsBomb API
│   ├── 02_features.ipynb      # Exploração das features geométricas/categóricas
│   ├── 03_modelos.ipynb       # Treinamento e comparação dos 3 modelos
│   └── 04_analise_shap.ipynb  # Interpretabilidade (SHAP) do modelo campeão
├── src/                        # Código organizado
│   ├── data_loader.py          # Coleta via statsbombpy
│   ├── features.py             # Engenharia de features
│   ├── models.py               # Treinamento dos 3 modelos (grid search + K-Fold)
│   └── evaluation.py           # Avaliação consolidada, IC bootstrap, calibração
├── dashboard/                  # App Streamlit
│   └── app.py                  # Simulador de xG + comparação de modelos
├── results/
│   ├── figures/                     # Curvas de calibração (.png)
│   ├── models/                      # Modelos treinados (não versionados)
│   ├── tabela_comparativa_modelos.csv
│   └── tabela_avaliacao_completa.csv/.tex
├── requirements.txt
└── README.md
```

## Como executar

### Pré-requisitos

- Python 3.11 ou superior
- pip

### Instalação

```bash
# Clone o repositório
git clone https://github.com/PaulooBarros/expected-goals-ml.git
cd expected-goals-ml

# Crie um ambiente virtual
python -m venv venv

# Ative o ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

### Executar o pipeline

```bash
# 1. Coletar dados
python src/data_loader.py

# 2. Processar features
python src/features.py

# 3. Treinar modelos
python src/models.py

# 4. Avaliar e gerar métricas
python src/evaluation.py

# 5. Rodar o dashboard
streamlit run dashboard/app.py
```

## Métricas de avaliação

- AUC-ROC (Area Under the ROC Curve)
- Log-Loss
- Brier Score

## Resultados

Tabela completa em [`results/tabela_avaliacao_completa.csv`](results/tabela_avaliacao_completa.csv), gerada por `src/evaluation.py`:

| Modelo | Teste principal (Copa 2022) | Teste externo (Euro 2020) |
|---|---|---|
| Regressão Logística | AUC 0,714 | AUC 0,732 |
| **Random Forest** | **AUC 0,767** | **AUC 0,751** |
| XGBoost | AUC 0,746 | AUC 0,748 |

O **Random Forest** apresentou o melhor AUC-ROC nos dois cenários de teste (generalização temporal e entre torneios) e foi escolhido como modelo campeão, sendo o modelo integrado ao dashboard e analisado com SHAP em `notebooks/04_analise_shap.ipynb`.

## Tecnologias

- Python 3.11+
- statsbombpy, pandas, numpy
- scikit-learn, xgboost, shap
- matplotlib, mplsoccer, seaborn, plotly
- streamlit

## Licença

Este projeto é acadêmico. Os dados utilizados são da StatsBomb e devem ser creditados conforme os [termos de uso](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).
