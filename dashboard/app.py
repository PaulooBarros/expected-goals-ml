"""
app.py
Dashboard interativo (Streamlit) do modelo de Expected Goals (xG).

Uso:
    streamlit run dashboard/app.py

Integra o modelo campeão (Random Forest) treinado em src/models.py:
permite simular a probabilidade de gol de uma finalização e compara
o desempenho dos três modelos avaliados no TCC.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from mplsoccer import Pitch

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROJETO))
sys.path.insert(0, str(RAIZ_PROJETO / "src"))

from src.features import calcular_angulo_gol, calcular_distancia_gol
from src.models import (
    COMPETICOES,
    MODELS_DIR,
    carregar_dataset,
    selecionar_features_comuns,
)

RESULTS_DIR = RAIZ_PROJETO / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

PREFIXOS_CATEGORICOS = {
    "Parte do corpo": "shot_body_part_",
    "Tipo de chute": "shot_type_",
    "Situação de jogo": "play_pattern_",
    "Técnica": "shot_technique_",
}


@st.cache_resource
def carregar_modelo_campeao():
    """Carrega o modelo Random Forest treinado (modelo campeão do TCC)."""
    return joblib.load(MODELS_DIR / "random_forest_model.pkl")


@st.cache_data
def carregar_colunas_features() -> list[str]:
    """Recalcula a lista de features comuns às três competições (mesma lógica do treino)."""
    dfs = {nome: carregar_dataset(nome) for nome in COMPETICOES}
    return selecionar_features_comuns(dfs)


def opcoes_categoricas(colunas_features: list[str], prefixo: str) -> list[str]:
    """Extrai as categorias válidas (presentes no treino) para um dado prefixo one-hot."""
    return sorted(coluna[len(prefixo):] for coluna in colunas_features if coluna.startswith(prefixo))


def montar_vetor_features(entrada: dict, colunas_features: list[str]) -> pd.DataFrame:
    """Monta uma linha de features no mesmo formato usado no treinamento do modelo."""
    linha = dict.fromkeys(colunas_features, 0.0)

    linha["x"] = entrada["x"]
    linha["y"] = entrada["y"]
    linha["distancia_gol"] = calcular_distancia_gol(entrada["x"], entrada["y"])
    linha["angulo_gol"] = calcular_angulo_gol(entrada["x"], entrada["y"])
    linha["minute"] = entrada["minute"]
    linha["period"] = entrada["period"]
    linha["under_pressure"] = int(entrada["under_pressure"])

    for rotulo, prefixo in PREFIXOS_CATEGORICOS.items():
        coluna_dummy = f"{prefixo}{entrada[rotulo]}"
        if coluna_dummy in linha:
            linha[coluna_dummy] = 1.0

    return pd.DataFrame([linha])[colunas_features]


def desenhar_chute_no_campo(x: float, y: float, xg: float):
    """Desenha a posição do chute no meio-campo ofensivo (coordenadas StatsBomb)."""
    pitch = Pitch(pitch_type="statsbomb", pitch_color="grass", line_color="white", half=True)
    fig, ax = pitch.draw(figsize=(6, 4.5))
    pitch.scatter(
        x, y, ax=ax, s=400, color="red", edgecolors="black", linewidth=1.5, zorder=3,
        alpha=0.5 + 0.5 * xg,
    )
    return fig


def pagina_simulador(colunas_features: list[str], modelo) -> None:
    """Página de simulação: usuário define uma finalização e vê o xG previsto."""
    st.header("Simulador de finalização")
    st.caption(
        "Defina as características de um chute para estimar a probabilidade de gol (xG) "
        "segundo o modelo Random Forest treinado na Copa do Mundo de 2018."
    )

    coluna_form, coluna_campo = st.columns([1, 1])

    with coluna_form:
        x = st.slider("Posição X (0 = fundo do próprio campo, 120 = linha do gol adversário)", 0.0, 120.0, 105.0, 0.5)
        y = st.slider("Posição Y (0 a 80, gol centrado em 40)", 0.0, 80.0, 40.0, 0.5)
        minute = st.slider("Minuto da partida", 0, 120, 45)
        period = st.selectbox("Período", options=[1, 2, 3, 4, 5], format_func=lambda p: f"{p}º tempo")
        under_pressure = st.checkbox("Sob pressão", value=False)

        entrada = {
            "x": x,
            "y": y,
            "minute": minute,
            "period": period,
            "under_pressure": under_pressure,
        }
        for rotulo, prefixo in PREFIXOS_CATEGORICOS.items():
            opcoes = opcoes_categoricas(colunas_features, prefixo)
            entrada[rotulo] = st.selectbox(rotulo, options=opcoes)

    X_entrada = montar_vetor_features(entrada, colunas_features)
    xg_previsto = float(modelo.predict_proba(X_entrada)[0, 1])

    with coluna_campo:
        st.metric("xG previsto", f"{xg_previsto:.1%}")
        st.pyplot(desenhar_chute_no_campo(x, y, xg_previsto))

    with st.expander("Ver vetor de features enviado ao modelo"):
        st.dataframe(X_entrada.T.rename(columns={0: "valor"}))


def pagina_comparacao_modelos() -> None:
    """Página de comparação: tabela de métricas e curvas de calibração dos três modelos."""
    st.header("Comparação entre os três modelos")
    st.caption("Regressão Logística (baseline), Random Forest e XGBoost — treinados na Copa 2018.")

    arquivo_tabela = RESULTS_DIR / "tabela_avaliacao_completa.csv"
    if arquivo_tabela.exists():
        tabela = pd.read_csv(arquivo_tabela)
        st.dataframe(tabela, width="stretch")
    else:
        st.warning("Tabela de avaliação não encontrada. Rode `python src/evaluation.py` primeiro.")

    st.subheader("Curvas de calibração")
    coluna_1, coluna_2 = st.columns(2)
    figura_copa2022 = FIGURES_DIR / "calibracao_copa2022.png"
    figura_euro2020 = FIGURES_DIR / "calibracao_euro2020.png"

    if figura_copa2022.exists():
        coluna_1.image(str(figura_copa2022), caption="Teste principal — Copa 2022")
    if figura_euro2020.exists():
        coluna_2.image(str(figura_euro2020), caption="Teste externo — Euro 2020")


def main() -> None:
    st.set_page_config(page_title="Modelo de xG — TCC", layout="wide")
    st.title("⚽ Modelo de Expected Goals (xG)")
    st.caption("TCC — Ciência da Computação (UNIT) | Paulo Gustavo Angelo de Barros")

    colunas_features = carregar_colunas_features()
    modelo = carregar_modelo_campeao()

    aba_simulador, aba_comparacao = st.tabs(["Simulador de Chute", "Comparação de Modelos"])
    with aba_simulador:
        pagina_simulador(colunas_features, modelo)
    with aba_comparacao:
        pagina_comparacao_modelos()


if __name__ == "__main__":
    main()
