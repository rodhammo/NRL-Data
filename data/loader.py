"""
Shared data loading and feature engineering for NRL data.

Provides functions to load match/player JSON data into DataFrames
and compute features used by prediction models.
"""

import json
import os

import numpy as np
import pandas as pd

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import ENVIRONMENT_VARIABLES as EV

TEAMS = EV.TEAMS
MATCH_VARIABLES = ["Year", "Win", "Defense", "Attack", "Margin", "Home", "Versus", "Round"]

DATA_DIR = os.path.dirname(__file__)


def _load_year_data(json_path, year):
    """Load round data for a single year from a JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    nrl = data["NRL"]

    # Find the entry containing this year
    for entry in nrl:
        if str(year) in entry:
            return entry[str(year)]

    raise KeyError(f"Year {year} not found in {json_path}")


def load_match_data(source, years, competition="NRL"):
    """
    Load match JSON data and return a DataFrame with one row per round,
    columns for each team's variables (e.g. 'Broncos Year', 'Broncos Win', ...).

    Parameters
    ----------
    source : str
        Either a path to a single combined JSON file, or a directory
        containing per-year folders (e.g. 'data/NRL/2025/NRL_data_2025.json').
        If a directory, it will look for files at:
            {source}/{competition}/{year}/{competition}_data_{year}.json
    years : list[int]
        List of years to load.
    competition : str
        Competition name (default 'NRL'). Used when source is a directory.

    Returns
    -------
    pd.DataFrame
    """
    years_arr = {}

    if os.path.isfile(source):
        # Legacy single-file format
        with open(source, "r") as f:
            data = json.load(f)
        nrl = data["NRL"]
        for year in years:
            years_arr[year] = nrl[years.index(year)][str(year)]
    else:
        # Per-year file structure: source/{competition}/{year}/{competition}_data_{year}.json
        for year in years:
            json_path = os.path.join(source, competition, str(year), f"{competition}_data_{year}.json")
            if not os.path.exists(json_path):
                print(f"Warning: {json_path} not found, skipping year {year}")
                continue
            years_arr[year] = _load_year_data(json_path, year)

    columns = [f"{team} {var}" for team in TEAMS for var in MATCH_VARIABLES]
    rows = []

    for year in years:
        if year not in years_arr:
            continue

        year_data = years_arr[year]

        for round_idx in range(len(year_data)):
            try:
                round_data = year_data[round_idx][str(round_idx + 1)]
            except (IndexError, KeyError):
                continue

            round_store = np.zeros(len(TEAMS) * len(MATCH_VARIABLES), dtype=int)
            round_teams = []

            for game in round_data:
                h_team = game["Home"]
                h_score = int(game["Home_Score"])
                a_team = game["Away"]
                a_score = int(game["Away_Score"])

                if h_team not in TEAMS or a_team not in TEAMS:
                    continue

                h_team_win = int(h_score >= a_score)
                a_team_win = int(a_score >= h_score)

                h_versus = TEAMS.index(a_team)
                a_versus = TEAMS.index(h_team)

                round_teams.extend([h_team, a_team])

                for team, values in [
                    (h_team, [year, h_team_win, a_score, h_score, h_score - a_score, 1, h_versus, round_idx + 1]),
                    (a_team, [year, a_team_win, h_score, a_score, a_score - h_score, 0, a_versus, round_idx + 1]),
                ]:
                    idx = TEAMS.index(team) * len(MATCH_VARIABLES)
                    for offset, val in enumerate(values):
                        round_store[idx + offset] = val

            bye_teams = set(TEAMS) - set(round_teams)
            for bye_team in bye_teams:
                idx = TEAMS.index(bye_team) * len(MATCH_VARIABLES)
                for offset, val in enumerate([year, -1, -1, -1, 0, -1, -1, round_idx + 1]):
                    round_store[idx + offset] = val

            rows.append(round_store)

    return pd.DataFrame(rows, columns=columns)


def get_game_history(df, year, round_, team, game_history=3):
    """
    Compute rolling statistics for a team's recent games.

    Parameters
    ----------
    df : pd.DataFrame
        Match DataFrame from load_match_data().
    year : int
    round_ : int
    team : str
    game_history : int
        Number of recent games to consider.

    Returns
    -------
    tuple
        (win_rate, defense_median, attack_median, margin_median,
         byes, home_rate, defense_mean, attack_mean, margin_mean, year)
    """
    filtered = df[df[f"{team} Year"] == year]
    filtered = filtered.iloc[round_ - game_history - 1 : round_ - 1]

    byes = len(filtered[filtered[f"{team} Win"] == -1])
    filtered = filtered[filtered[f"{team} Win"] != -1]

    win = filtered[f"{team} Win"].mean()
    defense = filtered[f"{team} Defense"].median()
    attack = filtered[f"{team} Attack"].median()
    margin = filtered[f"{team} Margin"].median()

    defense_mean = filtered[f"{team} Defense"].mean()
    attack_mean = filtered[f"{team} Attack"].mean()
    margin_mean = filtered[f"{team} Margin"].mean()

    games_at_home = filtered[f"{team} Home"].mean()

    return win, defense, attack, margin, byes, games_at_home, defense_mean, attack_mean, margin_mean, year


def build_training_data(df, game_history=3):
    """
    Build X (features) and y (labels) arrays from the match DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Match DataFrame from load_match_data().
    game_history : int
        Number of recent games to use for rolling stats.

    Returns
    -------
    (list, list)
        X features and y labels.
    """
    X, y = [], []

    for team in TEAMS:
        versed_teams = df[f"{team} Versus"]
        wins = df[f"{team} Win"]
        rounds = df[f"{team} Round"]
        years = df[f"{team} Year"]
        margins = df[f"{team} Margin"]

        c_team_idx = TEAMS.index(team)

        for versed_team, win, round_, year, margin in zip(versed_teams, wins, rounds, years, margins):
            if win == -1 or round_ <= game_history:
                continue

            v_win_ = 0 if win == 1 else 1
            big_win = 1 if abs(margin) > 13 else 0

            X.append([
                c_team_idx,
                versed_team,
                *get_game_history(df, year, round_, team, game_history),
                *get_game_history(df, year, round_, TEAMS[int(versed_team)], game_history),
            ])
            y.append([c_team_idx, versed_team, win, v_win_, big_win])

    return X, y
