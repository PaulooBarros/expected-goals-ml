"""
plot_style.py
Identidade visual das figuras do TCC de Expected Goals (xG).

Centraliza paleta de cores, configuração global do matplotlib e o
salvamento em formato duplo (PDF vetorial para o artigo LaTeX, PNG raster
para o README e o dashboard). Importar e chamar `configurar_matplotlib()`
uma única vez, antes de qualquer plotagem:

    from plot_style import configurar_matplotlib
    configurar_matplotlib()

A paleta usa as cores de Okabe & Ito (2008), desenhadas para permanecerem
distinguíveis nas três formas mais comuns de daltonismo (protanopia,
deuteranopia e tritanopia). Como cor sozinha não basta em impressão em
escala de cinza, cada modelo também recebe um marcador e um estilo de
linha próprios — redundância recomendada pelos guias de acessibilidade
de periódicos científicos.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------------
# Dimensões padrão de figura (polegadas)
# --------------------------------------------------------------------------

#: Figura de painel único — ocupa uma coluna do artigo SBC.
FIGSIZE_SIMPLES = (6, 4)

#: Figura de painel duplo (dois eixos lado a lado) — ocupa a largura total.
FIGSIZE_DUPLA = (12, 4)

# --------------------------------------------------------------------------
# Paleta de cores
# --------------------------------------------------------------------------

#: Cor primária de cada modelo (Okabe & Ito — seguras para daltonismo).
CORES_MODELOS: dict[str, str] = {
    "regressao_logistica": "#0072B2",  # azul
    "random_forest": "#D55E00",        # vermelho-alaranjado
    "xgboost": "#009E73",              # verde-azulado
}

#: Marcador de cada modelo — garante leitura em escala de cinza.
MARCADORES_MODELOS: dict[str, str] = {
    "regressao_logistica": "o",
    "random_forest": "s",
    "xgboost": "^",
}

#: Estilo de linha de cada modelo — segunda camada de redundância.
LINHAS_MODELOS: dict[str, str] = {
    "regressao_logistica": "-",
    "random_forest": "-",
    "xgboost": "--",
}

#: Rótulos legíveis para exibição (o código usa snake_case internamente).
NOMES_MODELOS: dict[str, str] = {
    "regressao_logistica": "Regressão Logística",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

#: Cores neutras de apoio (linhas de referência, grade, anotações).
COR_REFERENCIA = "#4D4D4D"
COR_GRADE = "#C8C8C8"
COR_TEXTO_SECUNDARIO = "#595959"
COR_FUNDO = "#FFFFFF"

#: Cor das linhas do campo nas figuras com mplsoccer.
COR_LINHA_CAMPO = "#8C8C8C"

#: Mapa sequencial para valores de xG (claro = xG baixo, escuro = xG alto).
CMAP_XG = LinearSegmentedColormap.from_list(
    "xg_sequencial",
    ["#FFF7EC", "#FDD49E", "#FC8D59", "#D55E00", "#7F2704"],
)

#: Mapa sequencial para densidade de finalizações (mapas de calor).
CMAP_DENSIDADE = LinearSegmentedColormap.from_list(
    "densidade_sequencial",
    ["#F7FBFF", "#C6DBEF", "#6BAED6", "#0072B2", "#08306B"],
)


def cor_modelo(nome_modelo: str) -> str:
    """Retorna a cor do modelo, com cinza neutro para nomes desconhecidos."""
    return CORES_MODELOS.get(nome_modelo, COR_REFERENCIA)


def nome_exibicao(nome_modelo: str) -> str:
    """Converte o identificador interno em rótulo legível para legendas."""
    return NOMES_MODELOS.get(nome_modelo, nome_modelo.replace("_", " ").title())


def estilo_modelo(nome_modelo: str) -> dict:
    """
    Retorna cor, marcador, estilo de linha e rótulo de um modelo, prontos
    para expandir como kwargs em `ax.plot(...)`.
    """
    return {
        "color": cor_modelo(nome_modelo),
        "marker": MARCADORES_MODELOS.get(nome_modelo, "o"),
        "linestyle": LINHAS_MODELOS.get(nome_modelo, "-"),
        "label": nome_exibicao(nome_modelo),
    }


# --------------------------------------------------------------------------
# Configuração global do matplotlib
# --------------------------------------------------------------------------

def configurar_matplotlib() -> None:
    """
    Aplica a identidade visual do TCC a todas as figuras subsequentes.

    Fonte serifada (combina com o corpo de texto do artigo SBC), corpo 11,
    grade sutil atrás dos dados, bordas superior e direita removidas e
    saída em 300 dpi. As fontes são embutidas como TrueType (`fonttype 42`)
    porque o padrão do matplotlib (Type 3) é rejeitado por boa parte das
    editoras científicas e por alguns validadores de PDF.
    """
    mpl.rcParams.update(
        {
            # Tipografia
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman", "serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.titlesize": 13,
            # Eixos e bordas
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": COR_TEXTO_SECUNDARIO,
            "axes.labelcolor": "black",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.axisbelow": True,  # grade atrás dos dados
            # Grade sutil
            "axes.grid": True,
            "axes.grid.axis": "both",
            "grid.color": COR_GRADE,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.5,
            # Marcas de escala
            "xtick.color": COR_TEXTO_SECUNDARIO,
            "ytick.color": COR_TEXTO_SECUNDARIO,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            # Linhas e marcadores
            "lines.linewidth": 1.7,
            "lines.markersize": 5,
            "lines.markeredgewidth": 0.8,
            # Legenda
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": COR_GRADE,
            "legend.facecolor": COR_FUNDO,
            "legend.borderpad": 0.5,
            "legend.handlelength": 2.2,
            # Figura e exportação
            "figure.figsize": FIGSIZE_SIMPLES,
            "figure.dpi": 300,
            "figure.facecolor": COR_FUNDO,
            "figure.constrained_layout.use": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "savefig.facecolor": COR_FUNDO,
            "savefig.transparent": False,
            # Fontes embutidas como TrueType (exigência de editoras)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Ciclo de cores padrão = paleta dos modelos
            "axes.prop_cycle": mpl.cycler(color=list(CORES_MODELOS.values())),
        }
    )


# --------------------------------------------------------------------------
# Exportação
# --------------------------------------------------------------------------

def salvar_figura(
    fig: plt.Figure,
    diretorio: Path,
    nome_base: str,
    formatos: tuple[str, ...] = ("pdf", "png"),
    fechar: bool = True,
) -> list[Path]:
    """
    Salva uma figura em múltiplos formatos, todos a 300 dpi.

    PDF é vetorial e vai para o artigo em LaTeX; PNG é raster e alimenta o
    README e o dashboard Streamlit. O `nome_base` não deve conter extensão.

    Args:
        fig: figura a salvar.
        diretorio: pasta de destino (criada se não existir).
        nome_base: nome do arquivo sem extensão (ex.: "calibracao_copa2022").
        formatos: extensões a gerar.
        fechar: se True, fecha a figura após salvar (evita acúmulo de memória
            ao gerar muitas figuras em sequência).

    Returns:
        Lista dos caminhos gravados.
    """
    diretorio = Path(diretorio)
    diretorio.mkdir(parents=True, exist_ok=True)

    caminhos = []
    for formato in formatos:
        caminho = diretorio / f"{nome_base}.{formato}"
        fig.savefig(caminho, format=formato, dpi=300)
        caminhos.append(caminho)

    if fechar:
        plt.close(fig)

    return caminhos


def numero_ptbr(valor: float, casas: int = 3) -> str:
    """Formata um número com vírgula decimal, como pede a norma para textos em português."""
    return f"{valor:.{casas}f}".replace(".", ",")


def usar_virgula_decimal(ax: plt.Axes, casas: int = 1, eixos: str = "both") -> None:
    """
    Troca o ponto decimal por vírgula nos rótulos dos eixos.

    O artigo do TCC é escrito em português, onde o separador decimal é a
    vírgula (ABNT NBR 5892). O matplotlib usa ponto por padrão, o que deixa
    as figuras em desacordo com o corpo do texto.

    Args:
        ax: eixo a formatar.
        casas: casas decimais nos rótulos.
        eixos: "x", "y" ou "both".
    """
    formatador = FuncFormatter(lambda valor, _: numero_ptbr(valor, casas))
    if eixos in ("x", "both"):
        ax.xaxis.set_major_formatter(formatador)
    if eixos in ("y", "both"):
        ax.yaxis.set_major_formatter(formatador)


def limitar_eixos_aos_dados(
    ax: plt.Axes, valor_maximo: float, margem: float = 0.05, minimo_absoluto: float = 0.1
) -> float:
    """
    Ajusta os eixos x e y ao intervalo realmente ocupado pelos dados.

    Em xG quase todas as probabilidades ficam abaixo de 0,4; manter os eixos
    fixos em [0, 1] desperdiça a maior parte da área do gráfico e achata as
    diferenças entre os modelos. Mantém a escala quadrada (mesmo limite nos
    dois eixos) porque as figuras de calibração dependem da diagonal a 45°.

    Args:
        ax: eixo a ajustar.
        valor_maximo: maior valor plotado (entre x e y).
        margem: folga proporcional acima do máximo.
        minimo_absoluto: limite mínimo, para não gerar eixos degenerados.

    Returns:
        O limite superior aplicado aos dois eixos.
    """
    limite = max(minimo_absoluto, float(valor_maximo) * (1 + margem))
    limite = min(1.0, limite)
    ax.set_xlim(0, limite)
    ax.set_ylim(0, limite)
    ax.set_aspect("equal", adjustable="box")
    return limite
