"""
features.py
Engenharia de features para o modelo de Expected Goals (xG).

Uso:
    python src/features.py

Carrega os parquets brutos de data/raw/, aplica as transformações
geométricas, categóricas e de variável alvo, e salva os datasets
processados em data/processed/.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

# Diretórios de entrada/saída
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Coordenadas do gol no sistema StatsBomb (pitch 120x80)
GOL_X = 120.0
GOL_Y = 40.0
TRAVE_ESQUERDA = (120.0, 36.0)
TRAVE_DIREITA = (120.0, 44.0)

# Colunas categóricas usadas na modelagem
COLUNAS_CATEGORICAS = ["shot_body_part", "shot_type", "play_pattern", "shot_technique"]

COMPETICOES = ["world_cup_2018", "world_cup_2022", "euro_2020"]


def calcular_distancia_gol(x: float, y: float) -> float:
    """Calcula a distância euclidiana entre a posição do chute e o centro do gol (120, 40)."""
    return math.hypot(GOL_X - x, GOL_Y - y)


def calcular_angulo_gol(x: float, y: float) -> float:
    """
    Calcula o ângulo de visão do gol (em graus), formado pelas retas entre a
    posição do chute e as duas traves (120, 36) e (120, 44).
    """
    vetor_esquerda = np.array([TRAVE_ESQUERDA[0] - x, TRAVE_ESQUERDA[1] - y])
    vetor_direita = np.array([TRAVE_DIREITA[0] - x, TRAVE_DIREITA[1] - y])

    norma_esquerda = np.linalg.norm(vetor_esquerda)
    norma_direita = np.linalg.norm(vetor_direita)
    if norma_esquerda == 0 or norma_direita == 0:
        return 0.0

    cos_angulo = np.dot(vetor_esquerda, vetor_direita) / (norma_esquerda * norma_direita)
    cos_angulo = np.clip(cos_angulo, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angulo)))


def extrair_features_geometricas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai as coordenadas X/Y da coluna `location` e calcula distância e
    ângulo até o gol para cada finalização.

    Adiciona as colunas: x, y, distancia_gol, angulo_gol.
    """
    df = df.copy()
    coordenadas = np.stack(df["location"].to_numpy())
    df["x"] = coordenadas[:, 0]
    df["y"] = coordenadas[:, 1]

    df["distancia_gol"] = df.apply(
        lambda linha: calcular_distancia_gol(linha["x"], linha["y"]), axis=1
    )
    df["angulo_gol"] = df.apply(
        lambda linha: calcular_angulo_gol(linha["x"], linha["y"]), axis=1
    )

    return df


def preparar_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica one-hot encoding às features categóricas técnicas e contextuais:
    shot_body_part, shot_type, play_pattern, shot_technique.

    As colunas originais são substituídas pelas colunas dummy geradas.
    """
    df = df.copy()
    colunas_presentes = [coluna for coluna in COLUNAS_CATEGORICAS if coluna in df.columns]
    df = pd.get_dummies(df, columns=colunas_presentes, prefix=colunas_presentes)
    return df


def criar_variavel_alvo(df: pd.DataFrame) -> pd.DataFrame:
    """Cria a coluna `gol` (0/1) a partir de shot_outcome: 1 se Goal, 0 caso contrário."""
    df = df.copy()
    df["gol"] = (df["shot_outcome"] == "Goal").astype(int)
    return df


def processar_competicao(nome: str) -> pd.DataFrame:
    """
    Pipeline completo de engenharia de features para uma competição.

    Carrega data/raw/{nome}_shots.parquet, remove pênaltis, aplica as
    transformações geométricas/categóricas/de alvo e salva o resultado
    em data/processed/{nome}_features.parquet.

    Args:
        nome: identificador da competição (ex.: "world_cup_2018").

    Returns:
        DataFrame processado, pronto para a modelagem.
    """
    arquivo_entrada = RAW_DIR / f"{nome}_shots.parquet"
    df = pd.read_parquet(arquivo_entrada)
    n_bruto = len(df)

    df = df[df["shot_type"] != "Penalty"].copy()
    n_sem_penaltis = len(df)

    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)

    df = extrair_features_geometricas(df)
    df = criar_variavel_alvo(df)
    df = preparar_categoricas(df)

    arquivo_saida = PROCESSED_DIR / f"{nome}_features.parquet"
    df.to_parquet(arquivo_saida, index=False)

    print(f"[+] {nome}: {n_bruto} chutes -> {n_sem_penaltis} após remover pênaltis")
    print(f"    {df.shape[1]} colunas no dataset final. Salvo em: {arquivo_saida}")

    return df


def main() -> None:
    """Processa as features das três competições."""
    for nome in COMPETICOES:
        processar_competicao(nome)
    print("\n[✓] Engenharia de features finalizada.")


if __name__ == "__main__":
    main()
