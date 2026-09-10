"""
visualizations.py
Gráficos "no campo" das finalizações, com mplsoccer.

Uso:
    python src/visualizations.py

As funções retornam objetos `Figure` em vez de salvar diretamente, para
que possam ser reaproveitadas tanto no pipeline do artigo (via
`plot_style.salvar_figura`) quanto no dashboard Streamlit (via
`st.pyplot`). Executado como script, gera as figuras de campo das duas
competições de teste usando o modelo campeão.

Convenção de coordenadas: sistema StatsBomb, campo 120x80, com o gol
atacado em x = 120. As figuras mostram apenas o meio-campo ofensivo,
onde estão praticamente todas as finalizações.
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import VerticalPitch

# Permite executar como script (`python src/visualizations.py`) e também
# importar como `src.visualizations` a partir do dashboard.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import (  # noqa: E402
    COMPETICOES,
    MODELS_DIR,
    RESULTS_DIR,
    TESTE_EXTERNO,
    TESTE_PRINCIPAL,
    carregar_dataset,
    preparar_X_y,
    selecionar_features_comuns,
)
from plot_style import (  # noqa: E402
    CMAP_DENSIDADE,
    CMAP_XG,
    COR_FUNDO,
    COR_LINHA_CAMPO,
    COR_TEXTO_SECUNDARIO,
    FIGSIZE_SIMPLES,
    configurar_matplotlib,
    numero_ptbr,
    salvar_figura,
    usar_virgula_decimal,
)

FIGURES_DIR = RESULTS_DIR / "figures"

#: Início do meio-campo ofensivo no sistema StatsBomb.
CAMPO_X_MINIMO = 60.0
CAMPO_X_MAXIMO = 120.0
CAMPO_Y_MINIMO = 0.0
CAMPO_Y_MAXIMO = 80.0

#: Resolução padrão do mapa de calor (colunas no eixo x, linhas no eixo y).
BINS_PADRAO = (12, 16)

#: Modelo usado para estimar o xG das figuras de campo.
ARQUIVO_MODELO_CAMPEAO = "random_forest_model.pkl"


def _criar_campo(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes, VerticalPitch]:
    """
    Desenha meio campo vertical no padrão visual do TCC.

    `line_zorder=2` mantém as linhas do campo acima do mapa de calor — sem
    isso o heatmap cobre a grande área e o gráfico perde a referência
    espacial que justifica plotar em um campo.
    """
    pitch = VerticalPitch(
        pitch_type="statsbomb",
        half=True,
        pitch_color=COR_FUNDO,
        line_color=COR_LINHA_CAMPO,
        linewidth=1.0,
        line_zorder=2,
        pad_bottom=-5,
    )
    fig, ax = pitch.draw(figsize=figsize)
    ax.grid(False)
    return fig, ax, pitch


def _bordas_dos_bins(bins: tuple[int, int]) -> list[np.ndarray]:
    """
    Converte (n_colunas, n_linhas) em bordas explícitas restritas ao meio-campo
    ofensivo. Sem isso o mplsoccer distribuiria os bins pelo campo inteiro e
    metade da resolução seria gasta fora da área visível.
    """
    n_x, n_y = bins
    return [
        np.linspace(CAMPO_X_MINIMO, CAMPO_X_MAXIMO, n_x + 1),
        np.linspace(CAMPO_Y_MINIMO, CAMPO_Y_MAXIMO, n_y + 1),
    ]


def _filtrar_meio_campo(df: pd.DataFrame, coluna_x: str, coluna_y: str) -> pd.DataFrame:
    """Descarta finalizações fora do meio-campo ofensivo (raras, mas distorcem a escala)."""
    return df[(df[coluna_x] >= CAMPO_X_MINIMO) & (df[coluna_x] <= CAMPO_X_MAXIMO)]


def mapa_calor_chutes(
    df: pd.DataFrame,
    titulo: str,
    coluna_x: str = "x",
    coluna_y: str = "y",
    bins: tuple[int, int] = BINS_PADRAO,
    figsize: tuple[float, float] = FIGSIZE_SIMPLES,
) -> plt.Figure:
    """
    Mapa de calor da densidade de finalizações no meio-campo ofensivo.

    Args:
        df: DataFrame de finalizações, com as coordenadas já extraídas
            (colunas `x` e `y`, produzidas por `features.py`).
        titulo: título da figura.
        coluna_x, coluna_y: nomes das colunas de coordenadas.
        bins: resolução da grade (colunas, linhas).
        figsize: dimensões da figura em polegadas.

    Returns:
        A figura, pronta para `salvar_figura` ou `st.pyplot`.
    """
    dados = _filtrar_meio_campo(df, coluna_x, coluna_y)
    fig, ax, pitch = _criar_campo(figsize)

    estatisticas = pitch.bin_statistic(
        dados[coluna_x].to_numpy(),
        dados[coluna_y].to_numpy(),
        statistic="count",
        bins=_bordas_dos_bins(bins),
    )
    malha = pitch.heatmap(
        estatisticas, ax=ax, cmap=CMAP_DENSIDADE, edgecolors=COR_FUNDO, linewidth=0.3, zorder=1,
    )

    barra = fig.colorbar(malha, ax=ax, shrink=0.7, pad=0.02)
    barra.set_label("Finalizações por célula", fontsize=9)
    barra.ax.tick_params(labelsize=8)
    barra.outline.set_visible(False)

    ax.set_title(titulo)
    fig.text(
        0.5, 0.02,
        f"n = {len(dados)} finalizações (pênaltis já removidos).",
        ha="center", fontsize=8, color=COR_TEXTO_SECUNDARIO,
    )
    fig.tight_layout()
    return fig


def scatter_chutes_xg(
    df: pd.DataFrame,
    titulo: str,
    coluna_xg: str = "xg_previsto",
    coluna_x: str = "x",
    coluna_y: str = "y",
    figsize: tuple[float, float] = FIGSIZE_SIMPLES,
) -> plt.Figure:
    """
    Dispersão das finalizações no campo, cada chute colorido pelo xG estimado.

    Cor e tamanho codificam a mesma variável (xG) de propósito: a redundância
    mantém a figura legível em impressão em escala de cinza, onde a diferença
    de cor sozinha desapareceria.

    Args:
        df: DataFrame de finalizações contendo as coordenadas e uma coluna
            com o xG previsto pelo modelo.
        titulo: título da figura.
        coluna_xg: coluna com a probabilidade estimada de gol.
        coluna_x, coluna_y: nomes das colunas de coordenadas.
        figsize: dimensões da figura em polegadas.

    Returns:
        A figura, pronta para `salvar_figura` ou `st.pyplot`.

    Raises:
        KeyError: se `coluna_xg` não existir no DataFrame.
    """
    if coluna_xg not in df.columns:
        raise KeyError(
            f"Coluna '{coluna_xg}' ausente. Calcule o xG antes de plotar, por exemplo: "
            f"df['{coluna_xg}'] = modelo.predict_proba(X)[:, 1]"
        )

    dados = _filtrar_meio_campo(df, coluna_x, coluna_y).sort_values(coluna_xg)
    fig, ax, pitch = _criar_campo(figsize)

    valores_xg = dados[coluna_xg].to_numpy()
    pontos = pitch.scatter(
        dados[coluna_x].to_numpy(),
        dados[coluna_y].to_numpy(),
        c=valores_xg,
        cmap=CMAP_XG,
        vmin=0.0,
        vmax=float(valores_xg.max()),
        # Marcadores pequenos: com ~1.400 finalizações concentradas na
        # entrada da área, pontos grandes viram uma mancha sólida e escondem
        # justamente os chutes de xG alto que a figura quer destacar.
        s=8 + 130 * valores_xg,
        edgecolors=COR_TEXTO_SECUNDARIO,
        linewidth=0.2,
        alpha=0.75,
        ax=ax,
        zorder=3,
    )

    barra = fig.colorbar(pontos, ax=ax, shrink=0.7, pad=0.02)
    barra.set_label("xG estimado", fontsize=9)
    barra.ax.tick_params(labelsize=8)
    barra.outline.set_visible(False)
    usar_virgula_decimal(barra.ax, casas=1, eixos="y")

    ax.set_title(titulo)
    fig.text(
        0.5, 0.02,
        f"n = {len(dados)} finalizações | xG médio = {numero_ptbr(valores_xg.mean())} | "
        f"máximo = {numero_ptbr(valores_xg.max())}",
        ha="center", fontsize=8, color=COR_TEXTO_SECUNDARIO,
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Geração das figuras do artigo
# --------------------------------------------------------------------------

def carregar_competicao_com_xg(nome: str, colunas_features: list[str], modelo) -> pd.DataFrame:
    """Carrega uma competição processada e anexa o xG previsto pelo modelo campeão."""
    df = carregar_dataset(nome)
    X, _ = preparar_X_y(df, colunas_features)
    df = df.copy()
    df["xg_previsto"] = modelo.predict_proba(X)[:, 1]
    return df


def main() -> None:
    """Gera as figuras de campo das duas competições de teste, em PDF e PNG."""
    configurar_matplotlib()

    dfs = {nome: carregar_dataset(nome) for nome in COMPETICOES}
    colunas_features = selecionar_features_comuns(dfs)
    modelo = joblib.load(MODELS_DIR / ARQUIVO_MODELO_CAMPEAO)

    competicoes_alvo = {
        TESTE_PRINCIPAL: ("Copa 2022", "copa2022"),
        TESTE_EXTERNO: ("Euro 2020", "euro2020"),
    }

    arquivos_gerados = []
    for nome_competicao, (rotulo, sufixo) in competicoes_alvo.items():
        df = carregar_competicao_com_xg(nome_competicao, colunas_features, modelo)

        figura_calor = mapa_calor_chutes(df, f"Densidade de finalizações — {rotulo}")
        arquivos_gerados += salvar_figura(figura_calor, FIGURES_DIR, f"mapa_calor_chutes_{sufixo}")

        figura_scatter = scatter_chutes_xg(df, f"Finalizações por xG estimado — {rotulo}")
        arquivos_gerados += salvar_figura(figura_scatter, FIGURES_DIR, f"scatter_chutes_xg_{sufixo}")

    print(f"[✓] {len(arquivos_gerados)} arquivos de figura gerados em: {FIGURES_DIR}")
    for caminho in arquivos_gerados:
        print(f"    - {caminho.name}")


if __name__ == "__main__":
    main()
