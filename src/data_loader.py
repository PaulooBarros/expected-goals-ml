"""
data_loader.py
Módulo de coleta de dados da StatsBomb Open Data.

Uso:
    python src/data_loader.py

Baixa eventos de finalização (Shot) das três competições selecionadas
e salva como parquet em data/raw/.
"""

from pathlib import Path

import pandas as pd
from statsbombpy import sb
from tqdm import tqdm

# Diretório de saída
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Competições selecionadas (competition_id, season_id, nome)
COMPETICOES = [
    (43, 3, "world_cup_2018"),   # FIFA World Cup 2018
    (43, 106, "world_cup_2022"),  # FIFA World Cup 2022
    (55, 43, "euro_2020"),        # UEFA Euro 2020
]


def listar_partidas(competition_id: int, season_id: int) -> pd.DataFrame:
    """Retorna DataFrame com as partidas de uma competição/temporada."""
    return sb.matches(competition_id=competition_id, season_id=season_id)


def coletar_eventos_partida(match_id: int) -> pd.DataFrame:
    """Coleta eventos de uma partida específica."""
    return sb.events(match_id=match_id)


def coletar_finalizacoes(competition_id: int, season_id: int, nome: str) -> pd.DataFrame:
    """
    Coleta todas as finalizações (Shots) de uma competição.

    Retorna DataFrame com uma linha por finalização, incluindo
    identificação da competição e partida.
    """
    print(f"\n[+] Coletando {nome}...")
    partidas = listar_partidas(competition_id, season_id)
    print(f"    {len(partidas)} partidas encontradas.")

    todos_chutes = []
    for _, partida in tqdm(partidas.iterrows(), total=len(partidas), desc=f"    {nome}"):
        try:
            eventos = coletar_eventos_partida(partida["match_id"])
            chutes = eventos[eventos["type"] == "Shot"].copy()
            chutes["competition"] = nome
            chutes["match_id"] = partida["match_id"]
            chutes["match_date"] = partida["match_date"]
            todos_chutes.append(chutes)
        except Exception as e:  # noqa: BLE001
            print(f"    [!] Erro na partida {partida['match_id']}: {e}")

    df = pd.concat(todos_chutes, ignore_index=True)
    print(f"    {len(df)} finalizações coletadas.")
    return df


def main() -> None:
    """Coleta todas as competições e salva em parquet separado."""
    for competition_id, season_id, nome in COMPETICOES:
        df = coletar_finalizacoes(competition_id, season_id, nome)
        arquivo = DATA_DIR / f"{nome}_shots.parquet"
        df.to_parquet(arquivo, index=False)
        print(f"    Salvo em: {arquivo}")

    print("\n[✓] Coleta finalizada.")


if __name__ == "__main__":
    main()
