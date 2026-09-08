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
tcc-xg-model/
├── data/
│   ├── raw/                  # Dados brutos da StatsBomb (não versionados)
│   └── processed/            # Datasets pré-processados
├── notebooks/                # Jupyter para exploração
│   ├── 01_coleta.ipynb
│   ├── 02_features.ipynb
│   ├── 03_modelos.ipynb
│   └── 04_analise_shap.ipynb
├── src/                      # Código organizado
│   ├── data_loader.py        # Coleta via statsbombpy
│   ├── features.py           # Engenharia de features
│   ├── models.py             # Treinamento dos 3 modelos
│   └── evaluation.py         # Métricas e validação cruzada
├── dashboard/                # App Streamlit
│   └── app.py
├── results/
│   ├── figures/              # Gráficos gerados
│   └── models/               # Modelos treinados (.pkl)
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
git clone https://github.com/[usuario]/tcc-xg-model.git
cd tcc-xg-model

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

## Tecnologias

- Python 3.11+
- statsbombpy, pandas, numpy
- scikit-learn, xgboost, shap
- matplotlib, mplsoccer, plotly
- streamlit

## Licença

Este projeto é acadêmico. Os dados utilizados são da StatsBomb e devem ser creditados conforme os [termos de uso](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).
