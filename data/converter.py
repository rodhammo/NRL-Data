"""
Convert NRL JSON data to TXT/CSV formats.

Usage:
    python converter.py                          # defaults: match + player for 2023
    python converter.py --type match --years 2023
    python converter.py --type player --years 2023
    python converter.py --type all --years 2022 2023
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import ENVIRONMENT_VARIABLES as EV

from loader import TEAMS, MATCH_VARIABLES, load_match_data


DATA_DIR = os.path.dirname(__file__)
TXT_DIR = os.path.join(DATA_DIR, "txt")
PLAYERS_TXT_DIR = os.path.join(TXT_DIR, "players", "txt")
PLAYERS_CSV_DIR = os.path.join(TXT_DIR, "players", "csv")


def convert_match(years):
    """Convert match JSON to TXT and CSV files."""
    for year in years:
        json_path = os.path.join(DATA_DIR, f"nrl_data_multi_years_{year}.json")
        if not os.path.exists(json_path):
            print(f"Skipping match conversion for {year}: {json_path} not found")
            continue

        df = load_match_data(json_path, [year])

        os.makedirs(TXT_DIR, exist_ok=True)
        df.to_csv(os.path.join(TXT_DIR, f"match_data_{year}.txt"), sep="\t", index=False)
        df.to_csv(os.path.join(TXT_DIR, f"match_data_{year}.csv"), sep="\t", index=False)
        print(f"Match data for {year} saved to {TXT_DIR}")


def convert_player(years):
    """Convert player statistics JSON to per-player TXT and CSV files."""
    player_variables = ["Name"] + EV.PLAYER_LABELS

    for year in years:
        json_path = os.path.join(DATA_DIR, f"player_statistics_{year}.json")
        if not os.path.exists(json_path):
            print(f"Skipping player conversion for {year}: {json_path} not found")
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        data = data["PlayerStats"]
        years_arr = {year: data[0][str(year)]}

        player_stats = defaultdict(list)

        for round_idx in range(0, 26):
            try:
                round_data = years_arr[year][round_idx]
                round_data = round_data[str(round_idx)]
            except (IndexError, KeyError):
                continue

            for round_game in round_data:
                for game in round_game:
                    game_split = game.split("-")
                    game_round = game_split[1]

                    game_split = game.split("v")
                    home_team = " ".join(game_split[0].split("-")[2:]).replace("-", " ").strip()
                    away_team = " ".join(game_split[-1:]).replace("-", " ").strip()

                    players = round_game[game]
                    player_round_stats = {}
                    for player in players:
                        vals = [player[val] for val in player_variables]
                        player_round_stats[vals[0]] = vals[1:]

                    player_round_stats = list(player_round_stats.items())
                    home_players = player_round_stats[:18]
                    away_players = player_round_stats[18:]

                    for p in home_players:
                        player_stats[p[0]].append([p[1], game_round, home_team, away_team])
                    for p in away_players:
                        player_stats[p[0]].append([p[1], game_round, away_team, home_team])

        headers = EV.PLAYER_LABELS + ["Round", "Team", "Opposition"]

        os.makedirs(PLAYERS_TXT_DIR, exist_ok=True)
        os.makedirs(PLAYERS_CSV_DIR, exist_ok=True)

        for player, values in player_stats.items():
            rows = [[*r[0], *r[1:]] for r in values]
            df = pd.DataFrame(rows, columns=headers)
            df.to_csv(os.path.join(PLAYERS_TXT_DIR, f"{player}.txt"), encoding="utf-8", sep="\t", index=False)
            df.to_csv(os.path.join(PLAYERS_CSV_DIR, f"{player}.csv"), encoding="utf-8", sep="\t", index=False)

        print(f"Player data for {year} saved ({len(player_stats)} players)")


def main():
    parser = argparse.ArgumentParser(description="Convert NRL JSON data to TXT/CSV")
    parser.add_argument("--type", choices=["match", "player", "all"], default="all")
    parser.add_argument("--years", nargs="+", type=int, default=[2023])
    args = parser.parse_args()

    if args.type in ("match", "all"):
        convert_match(args.years)
    if args.type in ("player", "all"):
        convert_player(args.years)


if __name__ == "__main__":
    main()
