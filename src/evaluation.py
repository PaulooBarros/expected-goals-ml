"""
evaluation.py
Avaliação consolidada dos modelos de xG já treinados.

Uso:
    python src/evaluation.py

Pressupõe que `python src/models.py` já foi executado (modelos salvos em
results/models/). Recarrega os modelos, recalcula as métricas nos
conjuntos de teste com intervalos de confiança (bootstrap) para o
AUC-ROC, gera as figuras de diagnóstico (calibração com faixa de
confiança, curvas ROC comparativas e matriz de confusão do modelo
campeão) e exporta as tabelas finais (CSV e LaTeX) para uso no artigo.

Todas as figuras saem em PDF (vetorial, para o LaTeX) e PNG (raster, para
o README e o dashboard), ambos a 300 dpi.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
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
from plot_style import (
    CMAP_DENSIDADE,
    COR_REFERENCIA,
    FIGSIZE_DUPLA,
    FIGSIZE_SIMPLES,
    configurar_matplotlib,
    cor_modelo,
    estilo_modelo,
    limitar_eixos_aos_dados,
    nome_exibicao,
    numero_ptbr,
    salvar_figura,
    usar_virgula_decimal,
)

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_ITERACOES_BOOTSTRAP = 1000
N_BINS_CALIBRACAO = 10

#: Modelo de melhor AUC-ROC nos dois cenários de teste (ver README).
MODELO_CAMPEAO = "random_forest"

#: Rótulos legíveis dos conjuntos de teste, usados nos títulos das figuras.
ROTULOS_CONJUNTOS = {
    "teste_principal_2022": "Teste principal — Copa 2022",
    "teste_externo_euro2020": "Teste externo — Euro 2020",
}


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


# --------------------------------------------------------------------------
# Calibração
# --------------------------------------------------------------------------

def intervalo_wilson(n_sucessos: np.ndarray, n_total: np.ndarray, z: float = 1.96) -> tuple:
    """
    Intervalo de confiança de Wilson para uma proporção binomial.

    Preferido à aproximação normal porque as frações observadas de gol por
    bin são pequenas (tipicamente 0,02 a 0,35) e o número de chutes por bin
    é modesto (~140). Nessa faixa a aproximação normal produz limites fora
    de [0, 1] e cobertura ruim; o intervalo de Wilson permanece válido.

    Args:
        n_sucessos: gols observados em cada bin.
        n_total: chutes em cada bin.
        z: quantil normal (1,96 para 95%).

    Returns:
        Tupla (limite_inferior, limite_superior), ambos arrays.
    """
    n_total = np.maximum(n_total, 1)
    proporcao = n_sucessos / n_total

    denominador = 1 + z**2 / n_total
    centro = (proporcao + z**2 / (2 * n_total)) / denominador
    meia_largura = (
        z / denominador * np.sqrt(proporcao * (1 - proporcao) / n_total + z**2 / (4 * n_total**2))
    )

    return np.clip(centro - meia_largura, 0, 1), np.clip(centro + meia_largura, 0, 1)


def curva_calibracao_com_ic(
    y: np.ndarray, proba: np.ndarray, n_bins: int = N_BINS_CALIBRACAO
) -> dict[str, np.ndarray]:
    """
    Calcula a curva de calibração com intervalo de confiança por bin.

    Usa binning por quantis (bins de frequência aproximadamente igual), o
    mesmo critério do `calibration_curve(strategy="quantile")` do
    scikit-learn, mas devolve também a contagem por bin — necessária para
    o intervalo de Wilson, que o scikit-learn não expõe.

    Returns:
        Dicionário com `probabilidade_media`, `fracao_observada`,
        `ic_inferior`, `ic_superior` e `n_por_bin`.
    """
    y = np.asarray(y, dtype=float)
    proba = np.asarray(proba, dtype=float)

    # Bordas por quantil; duplicatas são removidas para evitar bins vazios
    # quando muitas predições compartilham o mesmo valor.
    bordas = np.unique(np.quantile(proba, np.linspace(0, 1, n_bins + 1)))
    indices_bin = np.clip(np.digitize(proba, bordas[1:-1], right=True), 0, len(bordas) - 2)

    probabilidade_media, fracao_observada, n_por_bin, n_gols = [], [], [], []
    for indice in range(len(bordas) - 1):
        mascara = indices_bin == indice
        if not mascara.any():
            continue
        probabilidade_media.append(proba[mascara].mean())
        fracao_observada.append(y[mascara].mean())
        n_por_bin.append(int(mascara.sum()))
        n_gols.append(int(y[mascara].sum()))

    n_por_bin = np.array(n_por_bin)
    ic_inferior, ic_superior = intervalo_wilson(np.array(n_gols), n_por_bin)

    return {
        "probabilidade_media": np.array(probabilidade_media),
        "fracao_observada": np.array(fracao_observada),
        "ic_inferior": ic_inferior,
        "ic_superior": ic_superior,
        "n_por_bin": n_por_bin,
    }


def gerar_curva_calibracao(
    modelos: dict, X: pd.DataFrame, y: pd.Series, titulo: str, nome_base: str
) -> list[Path]:
    """
    Gera o diagrama de confiabilidade dos três modelos, com faixa de
    confiança de 95% (Wilson) em torno da fração observada de cada bin.

    Os eixos são recortados ao intervalo efetivamente ocupado pelos dados:
    como praticamente nenhuma finalização tem xG acima de 0,4, fixar os
    eixos em [0, 1] deixaria dois terços da figura vazios.
    """
    curvas = {}
    valor_maximo = 0.0
    for nome_modelo, modelo in modelos.items():
        proba = modelo.predict_proba(X)[:, 1]
        curva = curva_calibracao_com_ic(y, proba)
        curvas[nome_modelo] = curva
        valor_maximo = max(
            valor_maximo,
            curva["probabilidade_media"].max(),
            curva["ic_superior"].max(),
        )

    fig, ax = plt.subplots(figsize=FIGSIZE_SIMPLES)

    limite = limitar_eixos_aos_dados(ax, valor_maximo)
    ax.plot(
        [0, limite], [0, limite],
        linestyle=":", color=COR_REFERENCIA, linewidth=1.2,
        label="Perfeitamente calibrado", zorder=1,
    )

    for nome_modelo, curva in curvas.items():
        cor = cor_modelo(nome_modelo)
        ax.fill_between(
            curva["probabilidade_media"],
            curva["ic_inferior"],
            curva["ic_superior"],
            color=cor, alpha=0.15, linewidth=0, zorder=2,
        )
        ax.plot(
            curva["probabilidade_media"],
            curva["fracao_observada"],
            markerfacecolor="white", markeredgecolor=cor, zorder=3,
            **estilo_modelo(nome_modelo),
        )

    ax.set_xlabel("Probabilidade média predita (xG)")
    ax.set_ylabel("Fração observada de gols")
    ax.set_title(titulo)
    usar_virgula_decimal(ax)
    # Canto inferior direito: em um diagrama de confiabilidade os pontos se
    # concentram sobre a diagonal, deixando esse triângulo livre. No canto
    # superior esquerdo a legenda cobriria os bins de xG mais alto.
    ax.legend(loc="lower right")

    n_bins_efetivo = len(next(iter(curvas.values()))["probabilidade_media"])
    fig.text(
        0.5, -0.04,
        f"Bins por quantil (n = {n_bins_efetivo}); faixa sombreada = IC 95% de Wilson "
        f"para a fração observada.",
        ha="center", fontsize=8, color=COR_REFERENCIA,
    )
    fig.tight_layout()

    return salvar_figura(fig, FIGURES_DIR, nome_base)


# --------------------------------------------------------------------------
# Curvas ROC
# --------------------------------------------------------------------------

def gerar_curva_roc(modelos: dict, conjuntos: dict, nome_base: str = "roc_comparativa") -> list[Path]:
    """
    Gera as curvas ROC dos três modelos, um painel por conjunto de teste.

    Diferente da calibração — que mede se as probabilidades têm o valor
    certo —, a ROC mede apenas a ordenação dos chutes. Os dois diagnósticos
    são complementares e ambos entram no artigo.
    """
    fig, eixos = plt.subplots(1, len(conjuntos), figsize=FIGSIZE_DUPLA)
    eixos = np.atleast_1d(eixos)

    for ax, (nome_conjunto, (X, y)) in zip(eixos, conjuntos.items()):
        ax.plot(
            [0, 1], [0, 1],
            linestyle=":", color=COR_REFERENCIA, linewidth=1.2,
            label="Classificador aleatório", zorder=1,
        )

        for nome_modelo, modelo in modelos.items():
            proba = modelo.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, proba)
            auc = roc_auc_score(y, proba)

            estilo = estilo_modelo(nome_modelo)
            estilo["label"] = f"{nome_exibicao(nome_modelo)} — AUC {numero_ptbr(auc)}"
            # A ROC tem centenas de pontos: sem espaçar, os marcadores viram
            # uma mancha contínua. Cerca de 12 marcadores por curva preservam
            # a leitura em escala de cinza sem poluir a figura.
            ax.plot(
                fpr, tpr,
                markevery=max(1, len(fpr) // 12),
                markerfacecolor="white",
                markeredgecolor=estilo["color"],
                zorder=3,
                **estilo,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Taxa de falsos positivos (1 − especificidade)")
        ax.set_ylabel("Taxa de verdadeiros positivos (sensibilidade)")
        ax.set_title(ROTULOS_CONJUNTOS.get(nome_conjunto, nome_conjunto))
        usar_virgula_decimal(ax)
        # Corpo menor e handles curtos: com o texto do AUC, a legenda em
        # tamanho cheio fica tão larga quanto o eixo e cobre a subida inicial
        # das curvas, que é justamente a região de interesse da ROC.
        ax.legend(loc="lower right", fontsize=8.5, handlelength=1.8, borderpad=0.4)

    fig.tight_layout()
    return salvar_figura(fig, FIGURES_DIR, nome_base)


# --------------------------------------------------------------------------
# Matriz de confusão
# --------------------------------------------------------------------------

def limiar_youden(y: np.ndarray, proba: np.ndarray) -> float:
    """
    Limiar que maximiza o índice J de Youden (sensibilidade + especificidade − 1).

    Serve como alternativa ao corte de 0,5, que é praticamente degenerado
    em xG: como apenas ~10% das finalizações viram gol, quase nenhum chute
    ultrapassa 0,5 e o modelo classifica quase tudo como "não-gol".
    """
    fpr, tpr, limiares = roc_curve(y, proba)
    return float(limiares[np.argmax(tpr - fpr)])


def _desenhar_matriz(ax: plt.Axes, matriz: np.ndarray, titulo: str, valor_maximo: int) -> None:
    """
    Desenha uma matriz de confusão 2x2 anotada com contagem e percentual da linha.

    `valor_maximo` é compartilhado entre os painéis da figura: sem uma escala
    de cor comum, cada matriz seria normalizada pelo próprio máximo e as cores
    dos dois limiares não poderiam ser comparadas visualmente.
    """
    ax.imshow(matriz, cmap=CMAP_DENSIDADE, vmin=0, vmax=valor_maximo)
    ax.grid(False)

    rotulos = ["Não-gol", "Gol"]
    ax.set_xticks([0, 1], labels=rotulos)
    ax.set_yticks([0, 1], labels=rotulos)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Observado")
    ax.set_title(titulo)

    limite_texto_claro = valor_maximo / 2
    total_por_linha = matriz.sum(axis=1, keepdims=True)
    for linha in range(2):
        for coluna in range(2):
            valor = matriz[linha, coluna]
            percentual = 100 * valor / total_por_linha[linha, 0]
            ax.text(
                coluna, linha,
                f"{valor}\n({numero_ptbr(percentual, casas=1)}%)",
                ha="center", va="center",
                color="white" if valor > limite_texto_claro else "black",
                fontsize=10,
            )


def gerar_matriz_confusao(
    modelo, X: pd.DataFrame, y: pd.Series, nome_modelo: str,
    nome_conjunto: str, nome_base: str = "matriz_confusao_campeao",
) -> tuple[list[Path], float]:
    """
    Gera a matriz de confusão do modelo campeão em dois limiares.

    O painel da esquerda usa o corte convencional de 0,5; o da direita usa
    o limiar de Youden. A comparação é deliberada: ela torna visível que a
    matriz de confusão de um modelo de xG depende inteiramente do limiar
    escolhido — o modelo entrega probabilidades calibradas, não uma
    classificação binária.

    Returns:
        Tupla (caminhos gravados, limiar de Youden usado).
    """
    proba = modelo.predict_proba(X)[:, 1]
    limiar_otimo = limiar_youden(np.asarray(y), proba)

    matriz_padrao = confusion_matrix(y, (proba >= 0.5).astype(int), labels=[0, 1])
    matriz_otima = confusion_matrix(y, (proba >= limiar_otimo).astype(int), labels=[0, 1])
    valor_maximo = int(max(matriz_padrao.max(), matriz_otima.max()))

    fig, (ax_padrao, ax_otimo) = plt.subplots(1, 2, figsize=FIGSIZE_DUPLA)

    _desenhar_matriz(ax_padrao, matriz_padrao, "Limiar convencional (0,50)", valor_maximo)
    _desenhar_matriz(
        ax_otimo, matriz_otima, f"Limiar de Youden ({numero_ptbr(limiar_otimo)})", valor_maximo
    )

    fig.suptitle(
        f"Matriz de confusão — {nome_exibicao(nome_modelo)} | "
        f"{ROTULOS_CONJUNTOS.get(nome_conjunto, nome_conjunto)}"
    )
    fig.text(
        0.5, -0.02,
        "Percentuais calculados sobre o total de cada linha (recall por classe).",
        ha="center", fontsize=8, color=COR_REFERENCIA,
    )
    fig.tight_layout()

    return salvar_figura(fig, FIGURES_DIR, nome_base), limiar_otimo


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def main() -> None:
    """Recarrega modelos, recalcula métricas com IC e regenera todas as figuras."""
    configurar_matplotlib()

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

    arquivos_gerados = []
    arquivos_gerados += gerar_curva_calibracao(
        modelos, X_teste_principal, y_teste_principal,
        "Calibração — Teste principal (Copa 2022)", "calibracao_copa2022",
    )
    arquivos_gerados += gerar_curva_calibracao(
        modelos, X_teste_externo, y_teste_externo,
        "Calibração — Teste externo (Euro 2020)", "calibracao_euro2020",
    )
    arquivos_gerados += gerar_curva_roc(modelos, conjuntos_teste)

    caminhos_matriz, limiar_otimo = gerar_matriz_confusao(
        modelos[MODELO_CAMPEAO], X_teste_principal, y_teste_principal,
        MODELO_CAMPEAO, "teste_principal_2022",
    )
    arquivos_gerados += caminhos_matriz

    print(tabela.to_string(index=False))
    print(f"\n[✓] Tabela CSV salva em: {arquivo_csv}")
    print(f"[✓] Tabela LaTeX salva em: {arquivo_tex}")
    print(f"[+] Limiar de Youden do modelo campeão: {limiar_otimo:.4f}")
    print(f"\n[✓] {len(arquivos_gerados)} arquivos de figura gerados em: {FIGURES_DIR}")
    for caminho in arquivos_gerados:
        print(f"    - {caminho.name}")


if __name__ == "__main__":
    main()
