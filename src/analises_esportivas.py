"""
analises_esportivas.py
Análises esportivas da Copa do Mundo de 2022, usando o modelo campeão
(Random Forest, treinado na Copa 2018) para estimar o xG de cada
finalização do torneio.

Uso:
    python src/analises_esportivas.py

Pressupõe que `python src/models.py` já foi executado (modelo salvo em
results/models/random_forest_model.pkl). Gera tabelas (CSV + XLSX) e
figuras (PDF + PNG) em results/analises_esportivas/.

Escopo: apenas Copa do Mundo de 2022 (competition_id=43, season_id=106).
Pênaltis (incluindo disputas por pênaltis) já são excluídos, seguindo a
mesma convenção metodológica usada no treinamento dos modelos.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.figure import Figure
from statsbombpy import sb

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import (  # noqa: E402
    PROCESSED_DIR,
    RAW_DIR,
    criar_variavel_alvo,
    extrair_features_geometricas,
    preparar_categoricas,
)
from models import (  # noqa: E402
    COMPETICOES,
    MODELS_DIR,
    RESULTS_DIR,
    carregar_dataset,
    selecionar_features_comuns,
)
from plot_style import (  # noqa: E402
    COR_FUNDO,
    COR_REFERENCIA,
    COR_TEXTO_SECUNDARIO,
    FIGSIZE_SIMPLES,
    configurar_matplotlib,
)
from visualizations import _criar_campo  # noqa: E402

RANDOM_STATE = 42

ANALISES_DIR = RESULTS_DIR / "analises_esportivas"
TABELAS_DIR = ANALISES_DIR / "tabelas"
FIGURAS_PDF_DIR = ANALISES_DIR / "figuras_pdf"
FIGURAS_PNG_DIR = ANALISES_DIR / "figuras_png"
for diretorio in (TABELAS_DIR, FIGURAS_PDF_DIR, FIGURAS_PNG_DIR):
    diretorio.mkdir(parents=True, exist_ok=True)

ARQUIVO_MODELO_CAMPEAO = "random_forest_model.pkl"

#: Limite de x (sistema StatsBomb, campo 120x80) que define a grande área.
LIMITE_GRANDE_AREA_X = 102.0

#: Chutes mínimos para uma seleção entrar no perfil tático (Análise 3).
MINIMO_CHUTES_PERFIL_TATICO = 20

#: Seleções usadas no radar de perfis táticos — escolhidas pelo contraste
#: tático (times de posse/combinação vs. times mais diretos/físicos).
SELECOES_RADAR = ["Argentina", "Brazil", "Morocco", "Germany", "France", "Croatia"]

#: Paleta local (Okabe-Ito) para as análises esportivas — distinta da
#: paleta de modelos (plot_style.CORES_MODELOS) para não sugerir uma
#: relação entre "modelo" e "seleção"/"categoria de chute" que não existe.
COR_ACIMA_EXPECTATIVA = "#009E73"  # verde: mais gols que xG sugeria
COR_ABAIXO_EXPECTATIVA = "#CC79A7"  # roxo: menos gols que xG sugeria
COR_GOLS_IMPROVAVEIS = "#E69F00"  # laranja: baixo xG, resultou em gol
COR_GOLS_PERDIDOS = "#D55E00"  # vermelho: alto xG, não resultou em gol
CORES_RADAR = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
COR_MANDANTE = "#0072B2"
COR_VISITANTE = "#CC79A7"

#: Ordem cronológica das fases do mata-mata (nomes exatos de `competition_stage`).
ORDEM_FASES_MATA_MATA = [
    "Round of 16", "Quarter-finals", "Semi-finals", "3rd Place Final", "Final",
]
CACHE_MATA_MATA = PROCESSED_DIR / "world_cup_2022_mata_mata.parquet"


# ----------------------------------------------------------------------------
# Utilidades de carregamento e exportação
# ----------------------------------------------------------------------------


def carregar_copa2022_com_xg() -> pd.DataFrame:
    """
    Carrega as finalizações da Copa 2022 (sem pênaltis) com as features
    geométricas, a variável alvo e o xG previsto pelo modelo campeão.

    Recalcula a partir do parquet bruto (em vez de usar diretamente o
    parquet processado) para preservar as colunas categóricas em texto
    (`team`, `player`, `shot_body_part`, `play_pattern` etc.), que o
    one-hot encoding de `features.py` substitui por colunas dummy.
    """
    df = pd.read_parquet(RAW_DIR / "world_cup_2022_shots.parquet")
    df = df[df["shot_type"] != "Penalty"].copy()
    df = extrair_features_geometricas(df)
    df = criar_variavel_alvo(df)

    dfs_treino = {nome: carregar_dataset(nome) for nome in COMPETICOES}
    colunas_features = selecionar_features_comuns(dfs_treino)

    X = preparar_categoricas(df.copy()).reindex(columns=colunas_features, fill_value=0).astype(float)
    modelo = joblib.load(MODELS_DIR / ARQUIVO_MODELO_CAMPEAO)
    df["xg_previsto"] = modelo.predict_proba(X)[:, 1]

    return df


def construir_confrontos(df: pd.DataFrame) -> dict[int, str]:
    """Mapeia cada `match_id` a um rótulo "TimeA x TimeB" (sem indicar mandante)."""
    confrontos = {}
    for match_id, grupo in df.groupby("match_id"):
        times = sorted(grupo["team"].unique())
        confrontos[match_id] = " x ".join(times)
    return confrontos


def salvar_tabela(df: pd.DataFrame, nome_base: str) -> tuple[Path, Path]:
    """Salva uma tabela em CSV (UTF-8) e XLSX, ambos em results/analises_esportivas/tabelas/."""
    caminho_csv = TABELAS_DIR / f"{nome_base}.csv"
    caminho_xlsx = TABELAS_DIR / f"{nome_base}.xlsx"
    df.to_csv(caminho_csv, index=False, encoding="utf-8")
    df.to_excel(caminho_xlsx, index=False)
    return caminho_csv, caminho_xlsx


def salvar_figura_esportiva(fig: Figure, nome_base: str) -> tuple[Path, Path]:
    """Salva uma figura em PDF (figuras_pdf/) e PNG (figuras_png/), ambos a 300 dpi."""
    caminho_pdf = FIGURAS_PDF_DIR / f"{nome_base}.pdf"
    caminho_png = FIGURAS_PNG_DIR / f"{nome_base}.png"
    fig.savefig(caminho_pdf, format="pdf", dpi=300)
    fig.savefig(caminho_png, format="png", dpi=300)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return caminho_pdf, caminho_png


# ----------------------------------------------------------------------------
# Análise 1 — Quem superou expectativas
# ----------------------------------------------------------------------------


def analise_eficiencia_selecoes(df: pd.DataFrame) -> pd.DataFrame:
    """Ranqueia as seleções pela diferença entre gols reais e xG total acumulado."""
    tabela = (
        df.groupby("team")
        .agg(total_chutes=("gol", "size"), gols_reais=("gol", "sum"), xg_total=("xg_previsto", "sum"))
        .rename_axis("selecao")
        .reset_index()
    )
    tabela["xg_medio_por_chute"] = tabela["xg_total"] / tabela["total_chutes"]
    tabela["diferenca_absoluta"] = tabela["gols_reais"] - tabela["xg_total"]
    tabela["diferenca_relativa_pct"] = tabela["diferenca_absoluta"] / tabela["xg_total"] * 100
    tabela = tabela.sort_values("diferenca_relativa_pct", ascending=False).reset_index(drop=True)

    colunas_arredondar = ["xg_total", "xg_medio_por_chute", "diferenca_absoluta", "diferenca_relativa_pct"]
    tabela[colunas_arredondar] = tabela[colunas_arredondar].round(3)
    return tabela


def figura_eficiencia_selecoes(tabela: pd.DataFrame) -> Figure:
    """Scatter xG total vs. gols reais, com diagonal de referência e destaque dos extremos."""
    configurar_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6.5))

    limite = float(max(tabela["xg_total"].max(), tabela["gols_reais"].max()) * 1.12)
    ax.plot([0, limite], [0, limite], linestyle=":", color=COR_REFERENCIA, linewidth=1.2, zorder=1)
    ax.annotate(
        "xG = Gols", (limite * 0.97, limite * 0.97), fontsize=8, color=COR_REFERENCIA,
        ha="right", va="bottom", style="italic",
    )

    selecoes_acima = set(tabela.nlargest(3, "diferenca_relativa_pct")["selecao"])
    selecoes_abaixo = set(tabela.nsmallest(3, "diferenca_relativa_pct")["selecao"])

    textos = []
    for _, linha in tabela.iterrows():
        destaque = linha["selecao"] in selecoes_acima or linha["selecao"] in selecoes_abaixo
        if linha["selecao"] in selecoes_acima:
            cor, marcador, tamanho = COR_ACIMA_EXPECTATIVA, "^", 60
        elif linha["selecao"] in selecoes_abaixo:
            cor, marcador, tamanho = COR_ABAIXO_EXPECTATIVA, "v", 60
        else:
            cor, marcador, tamanho = COR_TEXTO_SECUNDARIO, "o", 26

        ax.scatter(
            linha["xg_total"], linha["gols_reais"], color=cor, marker=marcador, s=tamanho,
            edgecolors=COR_FUNDO, linewidth=0.5, alpha=0.95, zorder=3,
        )
        texto = ax.text(
            linha["xg_total"], linha["gols_reais"], linha["selecao"],
            fontsize=8 if destaque else 6.3,
            fontweight="bold" if destaque else "normal",
            color=cor if destaque else COR_TEXTO_SECUNDARIO,
            zorder=4,
        )
        textos.append(texto)

    # As 32 seleções produzem muitos pontos próximos (times com o mesmo
    # número de gols, por exemplo); adjust_text reposiciona os rótulos para
    # eliminar sobreposição e traça uma linha fina até o ponto original
    # quando o rótulo precisa se afastar dele.
    adjust_text(
        textos, ax=ax,
        arrowprops={"arrowstyle": "-", "color": COR_TEXTO_SECUNDARIO, "lw": 0.4, "alpha": 0.6},
    )

    ax.set_xlabel("xG total acumulado")
    ax.set_ylabel("Gols reais marcados")
    ax.set_xlim(0, limite)
    ax.set_ylim(0, limite)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Análise 2 — O gol mais improvável e o gol perdido
# ----------------------------------------------------------------------------

COLUNAS_CASOS_EXTREMOS = [
    "partida", "minuto", "jogador", "xg_estimado", "resultado_real",
    "x", "y", "parte_do_corpo", "situacao_jogo",
]


def _tabela_casos_extremos(df: pd.DataFrame, confrontos: dict[int, str]) -> pd.DataFrame:
    """Monta a tabela com as colunas de identificação pedidas para um subconjunto de chutes."""
    tabela = df.copy()
    tabela["partida"] = tabela["match_id"].map(confrontos)
    tabela["minuto"] = tabela["minute"]
    tabela["jogador"] = tabela["player"]
    tabela["xg_estimado"] = tabela["xg_previsto"].round(4)
    tabela["resultado_real"] = tabela["shot_outcome"]
    tabela["parte_do_corpo"] = tabela["shot_body_part"]
    tabela["situacao_jogo"] = tabela["play_pattern"]
    return tabela[COLUNAS_CASOS_EXTREMOS + ["xg_previsto"]]


def analise_casos_extremos(df: pd.DataFrame, confrontos: dict[int, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top 10 grandes chances perdidas (alto xG sem gol) e top 10 gols improváveis (baixo xG com gol)."""
    sem_gol = df[df["gol"] == 0]
    com_gol = df[df["gol"] == 1]

    gols_perdidos = _tabela_casos_extremos(sem_gol.nlargest(10, "xg_previsto"), confrontos)
    gols_improvaveis = _tabela_casos_extremos(com_gol.nsmallest(10, "xg_previsto"), confrontos)

    gols_perdidos = gols_perdidos.sort_values("xg_previsto", ascending=False).drop(columns="xg_previsto")
    gols_improvaveis = gols_improvaveis.sort_values("xg_previsto", ascending=True).drop(columns="xg_previsto")

    return gols_perdidos.reset_index(drop=True), gols_improvaveis.reset_index(drop=True)


def figura_casos_extremos_campo(gols_perdidos: pd.DataFrame, gols_improvaveis: pd.DataFrame) -> Figure:
    """Mapa de campo com as 20 finalizações extremas (10 perdidas + 10 improváveis)."""
    configurar_matplotlib()
    fig, ax, pitch = _criar_campo(FIGSIZE_SIMPLES)

    pitch.scatter(
        gols_perdidos["x"], gols_perdidos["y"], ax=ax, color=COR_GOLS_PERDIDOS, marker="D",
        s=70, edgecolors="black", linewidth=0.4, alpha=0.85, zorder=3,
        label="Grande chance perdida (alto xG, sem gol)",
    )
    pitch.scatter(
        gols_improvaveis["x"], gols_improvaveis["y"], ax=ax, color=COR_GOLS_IMPROVAVEIS, marker="o",
        s=70, edgecolors="black", linewidth=0.4, alpha=0.85, zorder=3,
        label="Gol improvável (baixo xG, com gol)",
    )

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), fontsize=7.5, frameon=True)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Análise 3 — Assinaturas táticas
# ----------------------------------------------------------------------------

METRICAS_PERFIL_TATICO = [
    "distancia_media", "angulo_medio", "pct_grande_area",
    "pct_contra_ataque", "pct_cabeceio", "xg_medio",
]
ROTULOS_METRICAS_PERFIL = [
    "Distância\nmédia (m)", "Ângulo\nmédio (°)", "% grande\nárea",
    "% contra-\nataque", "% cabeceio", "xG médio\npor chute",
]


def analise_perfis_taticos(df: pd.DataFrame) -> pd.DataFrame:
    """Perfil tático por seleção (mín. 20 chutes): geometria, origem da jogada, xG médio."""
    linhas = []
    for time, grupo in df.groupby("team"):
        if len(grupo) < MINIMO_CHUTES_PERFIL_TATICO:
            continue
        linhas.append(
            {
                "selecao": time,
                "total_chutes": len(grupo),
                "distancia_media": grupo["distancia_gol"].mean(),
                "angulo_medio": grupo["angulo_gol"].mean(),
                "pct_grande_area": (grupo["x"] >= LIMITE_GRANDE_AREA_X).mean() * 100,
                "pct_contra_ataque": (grupo["play_pattern"] == "From Counter").mean() * 100,
                "pct_cabeceio": (grupo["shot_body_part"] == "Head").mean() * 100,
                "xg_medio": grupo["xg_previsto"].mean(),
            }
        )

    tabela = pd.DataFrame(linhas).sort_values("xg_medio", ascending=False).reset_index(drop=True)
    tabela[METRICAS_PERFIL_TATICO] = tabela[METRICAS_PERFIL_TATICO].round(3)
    return tabela


def figura_perfis_taticos_radar(tabela: pd.DataFrame, selecoes: list[str]) -> Figure:
    """
    Radar comparando o perfil tático de seleções contrastantes.

    Cada métrica é normalizada (min-max) usando TODAS as seleções
    qualificadas (>= 20 chutes) como referência de escala — não apenas as
    exibidas no radar — para que a posição de cada seleção reflita sua
    posição real na distribuição do torneio, não apenas relativa às
    outras seleções destacadas.
    """
    configurar_matplotlib()
    import matplotlib.pyplot as plt

    base = tabela.set_index("selecao")[METRICAS_PERFIL_TATICO]
    normalizado = (base - base.min()) / (base.max() - base.min())

    angulos = np.linspace(0, 2 * np.pi, len(METRICAS_PERFIL_TATICO), endpoint=False).tolist()
    angulos += angulos[:1]

    fig, ax = plt.subplots(figsize=FIGSIZE_SIMPLES, subplot_kw={"polar": True})

    for cor, selecao in zip(CORES_RADAR, selecoes):
        if selecao not in normalizado.index:
            continue
        valores = normalizado.loc[selecao, METRICAS_PERFIL_TATICO].tolist()
        valores += valores[:1]
        ax.plot(angulos, valores, color=cor, linewidth=1.8, label=selecao, zorder=3)
        ax.fill(angulos, valores, color=cor, alpha=0.10, zorder=2)

    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(ROTULOS_METRICAS_PERFIL, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0,25", "0,50", "0,75", "1,00"], fontsize=6.5, color=COR_TEXTO_SECUNDARIO)
    ax.set_rlabel_position(210)  # entre "% contra-ataque" e "% cabeceio", região menos congestionada
    ax.grid(color="#C8C8C8", alpha=0.6)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.12), fontsize=7.5, frameon=True)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Análise 4 — O modelo prevê o campeão?
# ----------------------------------------------------------------------------


def obter_partidas_mata_mata() -> pd.DataFrame:
    """
    Carrega mandante/visitante/placar/fase das 16 partidas do mata-mata.

    `data_loader.py` só persistiu `match_id` e `match_date` de `sb.matches()`
    — o resto (mandante, visitante, placar, fase) nunca foi salvo. Como esses
    metadados não mudam, busca uma única vez e cacheia em
    `data/processed/world_cup_2022_mata_mata.parquet` para não repetir a
    consulta à API a cada execução.
    """
    if CACHE_MATA_MATA.exists():
        return pd.read_parquet(CACHE_MATA_MATA)

    partidas = sb.matches(competition_id=43, season_id=106)
    mata_mata = (
        partidas[partidas["competition_stage"] != "Group Stage"]
        [["match_id", "match_date", "competition_stage", "home_team", "away_team", "home_score", "away_score"]]
        .reset_index(drop=True)
    )
    mata_mata.to_parquet(CACHE_MATA_MATA, index=False)
    return mata_mata


def _contar_gols_penaltis(match_ids: list[int]) -> dict[int, dict[str, int]]:
    """
    Conta gols na disputa de pênaltis (period == 5) por partida e seleção.

    `home_score`/`away_score` de `sb.matches()` registram apenas o placar de
    tempo normal + prorrogação — 4 das 16 partidas do mata-mata da Copa 2022
    empataram nesse placar e foram decididas nos pênaltis. Os chutes da
    disputa estão no parquet bruto (excluídos do restante da análise, pois
    `shot_type == "Penalty"`), então o vencedor real é recuperado dali em
    vez de assumido externamente.
    """
    df_raw = pd.read_parquet(RAW_DIR / "world_cup_2022_shots.parquet")
    disputas = df_raw[
        (df_raw["shot_type"] == "Penalty")
        & (df_raw["period"] == 5)
        & (df_raw["match_id"].isin(match_ids))
    ]
    gols_por_partida: dict[int, dict[str, int]] = {}
    for match_id, grupo in disputas.groupby("match_id"):
        gols = grupo[grupo["shot_outcome"] == "Goal"].groupby("team").size()
        gols_por_partida[int(match_id)] = gols.to_dict()
    return gols_por_partida


def _determinar_vencedor(partida: pd.Series, gols_penaltis: dict[int, dict[str, int]]) -> str:
    """Vencedor real da partida, resolvendo empates via disputa de pênaltis."""
    if partida["home_score"] != partida["away_score"]:
        return partida["home_team"] if partida["home_score"] > partida["away_score"] else partida["away_team"]

    gols = gols_penaltis.get(int(partida["match_id"]), {})
    gols_mandante = gols.get(partida["home_team"], 0)
    gols_visitante = gols.get(partida["away_team"], 0)
    return partida["home_team"] if gols_mandante > gols_visitante else partida["away_team"]


def analise_mata_mata(df: pd.DataFrame) -> pd.DataFrame:
    """xG acumulado de cada seleção nas 16 partidas do mata-mata vs. o resultado real."""
    partidas = obter_partidas_mata_mata()
    gols_penaltis = _contar_gols_penaltis(partidas["match_id"].tolist())
    xg_por_time_partida = df.groupby(["match_id", "team"])["xg_previsto"].sum()

    linhas = []
    for _, partida in partidas.iterrows():
        match_id, mandante, visitante = partida["match_id"], partida["home_team"], partida["away_team"]
        xg_mandante = float(xg_por_time_partida.get((match_id, mandante), 0.0))
        xg_visitante = float(xg_por_time_partida.get((match_id, visitante), 0.0))

        vencedor_real = _determinar_vencedor(partida, gols_penaltis)
        if xg_mandante == xg_visitante:
            favorito_xg = "empate_xg"
        else:
            favorito_xg = mandante if xg_mandante > xg_visitante else visitante

        linhas.append(
            {
                "fase": partida["competition_stage"],
                "data": partida["match_date"],
                "mandante": mandante,
                "visitante": visitante,
                "gols_mandante": int(partida["home_score"]),
                "gols_visitante": int(partida["away_score"]),
                "decidida_penaltis": partida["home_score"] == partida["away_score"],
                "vencedor_real": vencedor_real,
                "xg_mandante": round(xg_mandante, 3),
                "xg_visitante": round(xg_visitante, 3),
                "favorito_pelo_xg": favorito_xg,
                "modelo_acertou_vencedor": favorito_xg == vencedor_real,
                "divergencia_xg_placar": round(
                    abs((xg_mandante - xg_visitante) - (partida["home_score"] - partida["away_score"])), 3
                ),
            }
        )

    tabela = pd.DataFrame(linhas)
    tabela["fase"] = pd.Categorical(tabela["fase"], categories=ORDEM_FASES_MATA_MATA, ordered=True)
    return tabela.sort_values(["fase", "data"]).reset_index(drop=True)


def figura_mata_mata(tabela: pd.DataFrame) -> Figure:
    """
    Linha horizontal por partida com 4 marcadores: xG e gols de mandante e
    visitante. Losango = xG previsto: círculo = gols reais. A linha fina
    entre os dois marcadores de uma mesma seleção mostra visualmente o
    tamanho do "erro" do modelo naquela partida.
    """
    configurar_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n = len(tabela)
    fig, ax = plt.subplots(figsize=(7, 0.4 * n + 1.0))

    posicoes_y = list(range(n, 0, -1))
    cor_fase_alternada = ["#F2F2F2", COR_FUNDO]
    for indice_fase, fase in enumerate(ORDEM_FASES_MATA_MATA):
        ys = [y for y, (_, p) in zip(posicoes_y, tabela.iterrows()) if p["fase"] == fase]
        if not ys:
            continue
        ax.axhspan(
            min(ys) - 0.5, max(ys) + 0.5,
            color=cor_fase_alternada[indice_fase % 2], zorder=0, alpha=0.6,
        )
        ax.text(
            1.01, (min(ys) + max(ys)) / 2, fase, transform=ax.get_yaxis_transform(),
            fontsize=7, color=COR_TEXTO_SECUNDARIO, va="center", ha="left",
        )

    for y, (_, partida) in zip(posicoes_y, tabela.iterrows()):
        ax.plot(
            [partida["xg_mandante"], partida["gols_mandante"]], [y, y],
            color=COR_MANDANTE, lw=1.1, alpha=0.5, zorder=1,
        )
        ax.plot(
            [partida["xg_visitante"], partida["gols_visitante"]], [y, y],
            color=COR_VISITANTE, lw=1.1, alpha=0.5, zorder=1,
        )
        ax.scatter(partida["xg_mandante"], y, marker="D", color=COR_MANDANTE, s=45, zorder=3,
                   edgecolors=COR_FUNDO, linewidth=0.5)
        ax.scatter(partida["gols_mandante"], y, marker="o", color=COR_MANDANTE, s=60, zorder=3,
                   edgecolors=COR_FUNDO, linewidth=0.5)
        ax.scatter(partida["xg_visitante"], y, marker="D", color=COR_VISITANTE, s=45, zorder=3,
                   edgecolors=COR_FUNDO, linewidth=0.5)
        ax.scatter(partida["gols_visitante"], y, marker="o", color=COR_VISITANTE, s=60, zorder=3,
                   edgecolors=COR_FUNDO, linewidth=0.5)

    rotulos = [
        f"{p['mandante']} {int(p['gols_mandante'])}-{int(p['gols_visitante'])} {p['visitante']}"
        + (" (pên.)" if p["decidida_penaltis"] else "")
        for _, p in tabela.iterrows()
    ]
    ax.set_yticks(posicoes_y)
    ax.set_yticklabels(rotulos, fontsize=7.5)
    ax.set_ylim(0.3, n + 0.7)
    ax.set_xlabel("Gols / xG acumulado na partida")

    legenda = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COR_REFERENCIA, markersize=6, label="Gols reais"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COR_REFERENCIA, markersize=6, label="xG previsto"),
        Line2D([0], [0], color=COR_MANDANTE, lw=2, label="Mandante"),
        Line2D([0], [0], color=COR_VISITANTE, lw=2, label="Visitante"),
    ]
    ax.legend(handles=legenda, loc="upper center", bbox_to_anchor=(0.42, -0.09), ncol=4, fontsize=7.5, frameon=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


# ----------------------------------------------------------------------------
# Execução principal
# ----------------------------------------------------------------------------


def main() -> None:
    print("[+] Carregando Copa 2022 e aplicando o modelo campeão (Random Forest)...")
    df = carregar_copa2022_com_xg()
    confrontos = construir_confrontos(df)
    print(f"    {len(df)} finalizações (sem pênaltis) em {df['match_id'].nunique()} partidas.")

    # --- Análise 1 -----------------------------------------------------
    print("\n[+] Análise 1 — Eficiência de finalização por seleção")
    tabela_eficiencia = analise_eficiencia_selecoes(df)
    salvar_tabela(tabela_eficiencia, "tabela_eficiencia_selecoes")
    fig1 = figura_eficiencia_selecoes(tabela_eficiencia)
    salvar_figura_esportiva(fig1, "eficiencia_selecoes_scatter")

    print("    Top 3 acima da expectativa (gols - xG):")
    print(tabela_eficiencia.head(3)[["selecao", "gols_reais", "xg_total", "diferenca_relativa_pct"]].to_string(index=False))
    print("    Bottom 3 abaixo da expectativa:")
    print(tabela_eficiencia.tail(3)[["selecao", "gols_reais", "xg_total", "diferenca_relativa_pct"]].to_string(index=False))

    # --- Análise 2 -----------------------------------------------------
    print("\n[+] Análise 2 — Casos extremos (gol improvável / chance perdida)")
    gols_perdidos, gols_improvaveis = analise_casos_extremos(df, confrontos)
    salvar_tabela(gols_perdidos, "tabela_gols_perdidos")
    salvar_tabela(gols_improvaveis, "tabela_gols_improvaveis")
    fig2 = figura_casos_extremos_campo(gols_perdidos, gols_improvaveis)
    salvar_figura_esportiva(fig2, "casos_extremos_campo")

    print("    Maior chance perdida:")
    print(gols_perdidos.iloc[[0]].to_string(index=False))
    print("    Gol mais improvável:")
    print(gols_improvaveis.iloc[[0]].to_string(index=False))

    # --- Análise 3 -----------------------------------------------------
    print("\n[+] Análise 3 — Assinaturas táticas por seleção")
    tabela_perfis = analise_perfis_taticos(df)
    salvar_tabela(tabela_perfis, "tabela_perfis_taticos")
    fig3 = figura_perfis_taticos_radar(tabela_perfis, SELECOES_RADAR)
    salvar_figura_esportiva(fig3, "perfis_taticos_radar")

    perfis_radar = tabela_perfis[tabela_perfis["selecao"].isin(SELECOES_RADAR)]
    distancias_normalizadas = (
        perfis_radar.set_index("selecao")[METRICAS_PERFIL_TATICO]
        - tabela_perfis[METRICAS_PERFIL_TATICO].min()
    ) / (tabela_perfis[METRICAS_PERFIL_TATICO].max() - tabela_perfis[METRICAS_PERFIL_TATICO].min())
    dispersao = distancias_normalizadas.std(axis=1).sort_values(ascending=False)
    print(f"    Seleções com perfis mais distintos (maior dispersão entre métricas normalizadas): "
          f"{dispersao.index[0]} e {dispersao.index[1]}")

    # --- Análise 4 -------------------------------------------------------
    print("\n[+] Análise 4 — O modelo prevê o campeão? (mata-mata: xG vs. placar)")
    tabela_mata_mata = analise_mata_mata(df)
    salvar_tabela(tabela_mata_mata, "tabela_mata_mata")
    fig4 = figura_mata_mata(tabela_mata_mata)
    salvar_figura_esportiva(fig4, "mata_mata_xg_vs_placar")

    acertos = tabela_mata_mata["modelo_acertou_vencedor"].sum()
    print(f"    Time com maior xG venceu em {acertos}/{len(tabela_mata_mata)} partidas do mata-mata.")
    maior_divergencia = tabela_mata_mata.loc[tabela_mata_mata["divergencia_xg_placar"].idxmax()]
    print("    Maior divergência entre xG e placar:")
    print(
        f"    {maior_divergencia['mandante']} {maior_divergencia['gols_mandante']}-"
        f"{maior_divergencia['gols_visitante']} {maior_divergencia['visitante']} ({maior_divergencia['fase']}) — "
        f"xG {maior_divergencia['xg_mandante']:.2f} x {maior_divergencia['xg_visitante']:.2f}"
    )

    _imprimir_resumo(df, tabela_mata_mata)


def _imprimir_resumo(df: pd.DataFrame, tabela_mata_mata: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("RESUMO — results/analises_esportivas/")
    print("=" * 78)
    print(f"tabelas/         {sorted(p.name for p in TABELAS_DIR.glob('*'))}")
    print(f"figuras_pdf/     {sorted(p.name for p in FIGURAS_PDF_DIR.glob('*'))}")
    print(f"figuras_png/     {sorted(p.name for p in FIGURAS_PNG_DIR.glob('*'))}")
    print(f"\nTotal de finalizações analisadas (Copa 2022, sem pênaltis): {len(df)}")
    print(f"Partidas do mata-mata analisadas: {len(tabela_mata_mata)}")


if __name__ == "__main__":
    main()
