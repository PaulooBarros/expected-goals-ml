# CLAUDE.md

Contexto do projeto para Claude (ou outros assistentes de IA) usarem como referência ao ajudar no desenvolvimento.

## Sobre o projeto

Este é o repositório do Trabalho de Conclusão de Curso (TCC) do curso de Ciência da Computação da Universidade Tiradentes (UNIT), Aracaju/SE.

**Título:** Modelo de Expected Goals com aprendizado de máquina: comparativo entre Regressão Logística, Random Forest e XGBoost.

**Autor:** Paulo Gustavo Angelo de Barros
**Orientador:** Luiz Gomes Da Cunha Neto (luiz.gomes@souunit.com.br)
**Instituição:** Universidade Tiradentes (UNIT) — Aracaju, Sergipe
**Linha de pesquisa:** Inteligência Artificial
**Formato do TCC:** Artigo científico no padrão SBC (Sociedade Brasileira de Computação), 10–15 páginas
**Defesa:** Novembro de 2026

## Objetivo do trabalho

Desenvolver e avaliar um modelo de Expected Goals (xG) baseado em aprendizado de máquina, com foco em duas contribuições:

1. Estudo comparativo entre três algoritmos: Regressão Logística (baseline), Random Forest e XGBoost.
2. Integração do modelo campeão a um dashboard interativo (Streamlit) para aplicação prática na análise esportiva.

## Base de dados

- **Fonte:** StatsBomb Open Data (via biblioteca `statsbombpy`)
- **Treinamento:** FIFA World Cup 2018 (64 partidas, sem dados 360)
- **Teste principal:** FIFA World Cup 2022 (64 partidas, com dados 360)
- **Teste externo:** UEFA Euro 2020 (51 partidas, com dados 360)
- **Volume estimado:** ~4.400 finalizações
- **Restrição importante:** apenas features comuns às três competições são usadas (a Copa 2018 não tem dados 360)

## Metodologia

**Problema:** classificação binária (gol / não-gol), com saída sendo a probabilidade da classe positiva.

**Features (por categoria):**
- Geométricas: distância ao gol, ângulo de visão, coordenadas X/Y
- Técnicas: parte do corpo utilizada, tipo de assistência
- Contextuais: situação de jogo, pattern do chute, sob pressão
- Temporais: minuto da partida, período (1º/2º tempo, prorrogação)

**Pré-processamento:**
- Filtragem apenas de eventos tipo `Shot`
- Remoção de pênaltis (padrão da literatura em xG)
- One-hot encoding para variáveis categóricas
- Padronização z-score para features numéricas (apenas para Regressão Logística)

**Split:**
- Treino: Copa 2018 → validação cruzada K-Fold (K=5) para tuning de hiperparâmetros
- Teste principal: Copa 2022 → generalização temporal
- Teste externo: Euro 2020 → generalização entre torneios diferentes
- Grid search para tuning do Random Forest e XGBoost

**Métricas de avaliação:**
- AUC-ROC (discriminação)
- Log-Loss (qualidade das probabilidades)
- Brier Score (calibração)

**Interpretabilidade:** SHAP (SHapley Additive exPlanations) aplicado ao modelo campeão.

## Stack tecnológico

- Python 3.11+
- **Coleta:** `statsbombpy`
- **Dados:** `pandas`, `numpy`
- **Modelagem:** `scikit-learn`, `xgboost`, `shap`
- **Visualização:** `matplotlib`, `mplsoccer`, `seaborn`, `plotly`
- **Dashboard:** `streamlit`
- **Ambiente:** VS Code + Python local (Linux Ubuntu/Debian)

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
├── README.md
└── CLAUDE.md                 # Este arquivo
```

## Convenções e estilo

**Nomenclatura:**
- Variáveis, funções e nomes de arquivos em português (feature branch: `features.py`, função: `calcular_distancia_gol`).
- Comentários e docstrings em português.
- Nomes de classes de features (categorias, valores) mantidos em inglês quando vêm direto da StatsBomb (`shot_body_part`, `under_pressure`).

**Código Python:**
- Docstrings estilo Google/NumPy.
- Type hints sempre que possível.
- Formatação com `black` ou `ruff format` (linha máx 100 caracteres).
- Imports organizados: stdlib → terceiros → locais.
- Preferir `pathlib.Path` a strings de caminho.

**Notebooks:**
- Sempre começar com célula de contexto (título, autor, descrição).
- Manter células curtas e com uma responsabilidade clara.
- Documentar decisões em células Markdown.

**Git:**
- Commits em português no imperativo: "Adiciona coleta da Copa 2022", "Corrige cálculo de ângulo".
- Branches para features grandes; `main` estável.
- Nunca commitar dados brutos (`data/raw/*.parquet`), modelos (`results/models/*.pkl`) ou credenciais.

## Restrições e cuidados

- **StatsBomb Open Data** exige atribuição em publicações. Sempre creditar como fonte.
- **Não modificar hiperparâmetros manualmente** sem justificativa metodológica — usar grid search.
- **Pênaltis não entram** na modelagem (decisão metodológica documentada no artigo).
- **Reprodutibilidade:** todo experimento deve ter `random_state` fixado.
- **Salvamento de modelos:** usar `joblib` para modelos scikit-learn, formato nativo para XGBoost.

## Como ajudar

Ao contribuir com código ou análises neste projeto, considere:

1. **Sempre respeitar as decisões metodológicas** já documentadas — se sugerir algo diferente, justificar claramente.
2. **Pensar em reprodutibilidade** — outra pessoa deve conseguir rodar o código e chegar aos mesmos resultados.
3. **Documentar o "porquê"** das escolhas técnicas, não só o "o quê".
4. **Considerar o contexto acadêmico** — o trabalho será defendido em uma banca, então clareza é mais importante que "esperteza" no código.
5. **Priorizar leitura sobre escrita** — código será lido por orientador e banca, não só executado.

## Contexto do artigo (LaTeX)

O artigo científico do TCC é escrito em LaTeX e está em repositório separado (Overleaf). O código deste repositório gera os resultados, tabelas e figuras que alimentam o artigo. Ao gerar saídas, considere que elas podem ir para o artigo:

- **Figuras:** salvar em PDF (vetorial) para o artigo e PNG (raster) para o dashboard/README.
- **Tabelas:** exportar como CSV limpo para conversão fácil em LaTeX (via `to_latex()` do pandas).
- **Valores numéricos:** sempre com 3-4 casas decimais e reprodutíveis com `random_state`.

---

**Última atualização:** 08 Setembro de 2026.