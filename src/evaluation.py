"""
evaluation.py
Avaliação consolidada dos modelos de xG já treinados.

Uso:
    python src/evaluation.py

Pressupõe que `python src/models.py` já foi executado (modelos salvos em
results/models/). Recarrega os modelos, recalcula as métricas nos
conjuntos de teste com intervalos de confiança (bootstrap) para o
AUC-ROC, gera curvas de calibração e exporta as tabelas finais (CSV e
LaTeX) para uso no artigo.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

from models import (
    COMPETICOES,
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TESTE_EXTERNO,
    TESTE_PRINCIPAL,
    calcular_metricas,
    carregar_dataset,
    preparar_X_y,
    selecionar_features_comuns,
)

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_ITERACOES_BOOTSTRAP = 1000


def carregar_modelos_treinados() -> dict:
    """Recarrega os três modelos salvos por src/models.py."""
    modelo_xgb = XGBClassifier()
    modelo_xgb.load_model(MODELS_DIR / "xgboost_model.json")

    return {
        "regressao_logistica": joblib.load(MODELS_DIR / "regressao_logistica_model.pkl"),
        "random_forest": joblib.load(MODELS_DIR / "random_forest_model.pkl"),
        "xgboost": modelo_xgb,
    }


def bootstrap_ic_auc(
    y: pd.Series, proba: np.ndarray, n_iteracoes: int = N_ITERACOES_BOOTSTRAP
) -> tuple[float, float]:
    """Intervalo de confiança de 95% para o AUC-ROC via bootstrap não-paramétrico."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(RANDOM_STATE)
    y = np.asarray(y)
    proba = np.asarray(proba)
    n = len(y)

    aucs = []
    for _ in range(n_iteracoes):
        indices = rng.integers(0, n, n)
        if len(np.unique(y[indices])) < 2:
            continue
        aucs.append(roc_auc_score(y[indices], proba[indices]))

    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def avaliar_com_intervalo(modelo, X: pd.DataFrame, y: pd.Series) -> dict:
    """Avalia um modelo treinado e adiciona o intervalo de confiança do AUC-ROC."""
    proba = modelo.predict_proba(X)[:, 1]
    metricas = calcular_metricas(y, proba)
    ic_inferior, ic_superior = bootstrap_ic_auc(y, proba)
    metricas["auc_roc_ic95_inf"] = ic_inferior
    metricas["auc_roc_ic95_sup"] = ic_superior
    return metricas


def gerar_curva_calibracao(modelos: dict, X: pd.DataFrame, y: pd.Series, titulo: str, arquivo: Path) -> None:
    """Gera e salva o gráfico de calibração (reliability diagram) dos três modelos."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for nome_modelo, modelo in modelos.items():
        proba = modelo.predict_proba(X)[:, 1]
        fracao_observada, probabilidade_media = calibration_curve(y, proba, n_bins=10, strategy="quantile")
        ax.plot(probabilidade_media, fracao_observada, marker="o", label=nome_modelo)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfeitamente calibrado")
    ax.set_xlabel("Probabilidade média predita")
    ax.set_ylabel("Fração observada de gols")
    ax.set_title(titulo)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150)
    plt.close(fig)


def main() -> None:
    """Recarrega modelos, recalcula métricas com IC e gera curvas de calibração."""
    dfs = {nome: carregar_dataset(nome) for nome in COMPETICOES}
    colunas_features = selecionar_features_comuns(dfs)

    X_teste_principal, y_teste_principal = preparar_X_y(dfs[TESTE_PRINCIPAL], colunas_features)
    X_teste_externo, y_teste_externo = preparar_X_y(dfs[TESTE_EXTERNO], colunas_features)

    modelos = carregar_modelos_treinados()

    conjuntos_teste = {
        "teste_principal_2022": (X_teste_principal, y_teste_principal),
        "teste_externo_euro2020": (X_teste_externo, y_teste_externo),
    }

    linhas_resultado = []
    for nome_modelo, modelo in modelos.items():
        for nome_conjunto, (X, y) in conjuntos_teste.items():
            metricas = avaliar_com_intervalo(modelo, X, y)
            linhas_resultado.append({"modelo": nome_modelo, "conjunto": nome_conjunto, **metricas})

    tabela = pd.DataFrame(linhas_resultado).round(4)

    arquivo_csv = RESULTS_DIR / "tabela_avaliacao_completa.csv"
    tabela.to_csv(arquivo_csv, index=False)

    arquivo_tex = RESULTS_DIR / "tabela_avaliacao_completa.tex"
    tabela.to_latex(arquivo_tex, index=False, float_format="%.4f")

    gerar_curva_calibracao(
        modelos,
        X_teste_principal,
        y_teste_principal,
        "Calibração — Teste principal (Copa 2022)",
        FIGURES_DIR / "calibracao_copa2022.png",
    )
    gerar_curva_calibracao(
        modelos,
        X_teste_externo,
        y_teste_externo,
        "Calibração — Teste externo (Euro 2020)",
        FIGURES_DIR / "calibracao_euro2020.png",
    )

    print(tabela.to_string(index=False))
    print(f"\n[✓] Tabela CSV salva em: {arquivo_csv}")
    print(f"[✓] Tabela LaTeX salva em: {arquivo_tex}")
    print(f"[✓] Curvas de calibração salvas em: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
