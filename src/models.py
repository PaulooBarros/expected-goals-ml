"""
models.py
Treinamento e comparação dos três modelos de Expected Goals (xG):
Regressão Logística, Random Forest e XGBoost.

Uso:
    python src/models.py

Treina em FIFA World Cup 2018 (com validação cruzada K-Fold para tuning),
avalia em FIFA World Cup 2022 (teste principal) e UEFA Euro 2020 (teste
externo), e salva os modelos treinados em results/models/.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODELS_DIR = RESULTS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5

TREINO = "world_cup_2018"
TESTE_PRINCIPAL = "world_cup_2022"
TESTE_EXTERNO = "euro_2020"
COMPETICOES = [TREINO, TESTE_PRINCIPAL, TESTE_EXTERNO]

FEATURES_NUMERICAS = ["x", "y", "distancia_gol", "angulo_gol", "minute", "period", "under_pressure"]
PREFIXOS_CATEGORICOS = ("shot_body_part_", "shot_type_", "play_pattern_", "shot_technique_")


def carregar_dataset(nome: str) -> pd.DataFrame:
    """Carrega o dataset processado (com features) de uma competição."""
    return pd.read_parquet(PROCESSED_DIR / f"{nome}_features.parquet")


def selecionar_features_comuns(dfs: dict[str, pd.DataFrame]) -> list[str]:
    """
    Retorna a lista de features (numéricas + one-hot) presentes nas três
    competições. Necessário porque a Copa 2018 não possui dados 360 e nem
    toda categoria (ex.: `shot_type_Corner`) aparece nas três bases.
    """
    conjuntos_por_competicao = []
    for df in dfs.values():
        colunas = {
            coluna
            for coluna in df.columns
            if coluna in FEATURES_NUMERICAS or coluna.startswith(PREFIXOS_CATEGORICOS)
        }
        conjuntos_por_competicao.append(colunas)

    features_comuns = sorted(set.intersection(*conjuntos_por_competicao))
    return features_comuns


def preparar_X_y(df: pd.DataFrame, colunas_features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features (X) e variável alvo (y) de um dataset processado."""
    X = df[colunas_features].astype(float)
    y = df["gol"].astype(int)
    return X, y


def calcular_metricas(y: pd.Series, proba: pd.Series) -> dict[str, float]:
    """Calcula AUC-ROC, Log-Loss e Brier Score a partir das probabilidades preditas."""
    return {
        "auc_roc": roc_auc_score(y, proba),
        "log_loss": log_loss(y, proba),
        "brier_score": brier_score_loss(y, proba),
    }


def avaliar_cv(modelo, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> dict[str, float]:
    """Avalia um modelo (não ajustado) via probabilidades out-of-fold da validação cruzada."""
    proba = cross_val_predict(clone(modelo), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return calcular_metricas(y, proba)


def avaliar_holdout(modelo, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Avalia um modelo já treinado em um conjunto de teste held-out."""
    proba = modelo.predict_proba(X)[:, 1]
    return calcular_metricas(y, proba)


def treinar_regressao_logistica(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Treina a Regressão Logística (baseline), com padronização z-score das
    features numéricas — única entre os três modelos que exige isso.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def treinar_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Treina o Random Forest com grid search (AUC-ROC) via K-Fold (K=5)."""
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 6, 10],
        "min_samples_leaf": [1, 5],
    }
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    busca = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    busca.fit(X_train, y_train)
    print(f"    Melhores parâmetros (RF): {busca.best_params_} | AUC-ROC (CV): {busca.best_score_:.4f}")
    return busca.best_estimator_


def treinar_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Treina o XGBoost com grid search (AUC-ROC) via K-Fold (K=5)."""
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    busca = GridSearchCV(
        XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
        param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    busca.fit(X_train, y_train)
    print(f"    Melhores parâmetros (XGB): {busca.best_params_} | AUC-ROC (CV): {busca.best_score_:.4f}")
    return busca.best_estimator_


def salvar_modelo(modelo, nome_modelo: str) -> None:
    """Salva o modelo treinado: joblib para scikit-learn, formato nativo para XGBoost."""
    if nome_modelo == "xgboost":
        modelo.save_model(MODELS_DIR / "xgboost_model.json")
    else:
        joblib.dump(modelo, MODELS_DIR / f"{nome_modelo}_model.pkl")


def main() -> None:
    """Executa o pipeline completo: carrega dados, treina, avalia e salva os três modelos."""
    dfs = {nome: carregar_dataset(nome) for nome in COMPETICOES}
    colunas_features = selecionar_features_comuns(dfs)
    print(f"[+] {len(colunas_features)} features comuns às três competições selecionadas.")

    X_train, y_train = preparar_X_y(dfs[TREINO], colunas_features)
    X_teste_principal, y_teste_principal = preparar_X_y(dfs[TESTE_PRINCIPAL], colunas_features)
    X_teste_externo, y_teste_externo = preparar_X_y(dfs[TESTE_EXTERNO], colunas_features)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    treinadores = {
        "regressao_logistica": treinar_regressao_logistica,
        "random_forest": treinar_random_forest,
        "xgboost": treinar_xgboost,
    }

    linhas_resultado = []
    for nome_modelo, treinar in treinadores.items():
        print(f"\n[+] Treinando {nome_modelo}...")
        modelo = treinar(X_train, y_train)
        salvar_modelo(modelo, nome_modelo)

        conjuntos_avaliacao = {
            "treino_2018_cv": avaliar_cv(modelo, X_train, y_train, cv),
            "teste_principal_2022": avaliar_holdout(modelo, X_teste_principal, y_teste_principal),
            "teste_externo_euro2020": avaliar_holdout(modelo, X_teste_externo, y_teste_externo),
        }
        for conjunto, metricas in conjuntos_avaliacao.items():
            linhas_resultado.append({"modelo": nome_modelo, "conjunto": conjunto, **metricas})

    tabela_resultados = pd.DataFrame(linhas_resultado).round(4)
    arquivo_resultados = RESULTS_DIR / "tabela_comparativa_modelos.csv"
    tabela_resultados.to_csv(arquivo_resultados, index=False)

    print("\n=== Tabela comparativa (AUC-ROC / Log-Loss / Brier Score) ===")
    print(tabela_resultados.to_string(index=False))
    print(f"\n[✓] Modelos salvos em: {MODELS_DIR}")
    print(f"[✓] Tabela salva em: {arquivo_resultados}")


if __name__ == "__main__":
    main()
