"""
NRL Round Predictor

Predicts for each upcoming match:
  1. Winner and win probability
  2. Predicted margin
  3. Most likely first try scorer

Uses historical match data, detailed team stats, and player statistics.

Usage:
    python predict_round.py
"""

import sys
import os
import json
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import requests
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import ENVIRONMENT_VARIABLES as EV

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TEAMS = EV.TEAMS


# ── Utility ────────────────────────────────────────────────────────────────────

def parse_num(val, default=0.0):
    """Parse a stat value that might be a string like '1,822', '78%', '3.72s', or -1."""
    if val is None or val == -1 or val == -10 or val == "-1":
        return default
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).replace(",", "").replace("%", "").replace("s", "").strip()
    try:
        return float(val)
    except ValueError:
        return default


def parse_possession(val, default=50.0):
    """Parse possession time like '28:33' to minutes, or '52%' to percentage."""
    if val is None or val == -1:
        return default
    val = str(val).strip()
    if ":" in val:
        parts = val.split(":")
        return float(parts[0]) + float(parts[1]) / 60
    return parse_num(val, default)


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_all_match_data(years):
    """Load match results for multiple years."""
    all_matches = []
    for year in years:
        path = os.path.join(DATA_DIR, "NRL", str(year), f"NRL_data_{year}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        rounds = data["NRL"][0][str(year)]
        for rnd in rounds:
            rnd_num = list(rnd.keys())[0]
            for game in rnd[rnd_num]:
                h_score = int(game["Home_Score"])
                a_score = int(game["Away_Score"])
                if h_score == 0 and a_score == 0:
                    continue
                if game["Home"] not in TEAMS or game["Away"] not in TEAMS:
                    continue
                all_matches.append({
                    "year": year, "round": int(rnd_num),
                    "home": game["Home"], "away": game["Away"],
                    "home_score": h_score, "away_score": a_score,
                })
    return all_matches


def load_detailed_team_stats(years):
    """Load detailed team stats per match, keyed by (year, round, home, away)."""
    stats = {}
    for year in years:
        path = os.path.join(DATA_DIR, "NRL", str(year), f"NRL_detailed_match_data_{year}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for rnd in data["NRL"]:
            rnd_key = list(rnd.keys())[0]
            for game in rnd[rnd_key]:
                game_name = list(game.keys())[0]
                match_data = game[game_name]

                parts = game_name.split(" v ")
                if len(parts) != 2:
                    continue
                home_team = parts[0].strip()
                away_team = parts[1].strip()

                stats[(year, int(rnd_key), home_team, away_team)] = match_data
    return stats


def extract_team_stats(side_data):
    """Extract numeric stats from a home/away side's detailed data."""
    return {
        "completion_rate": parse_num(side_data.get("Completion Rate"), 75),
        "effective_tackle": parse_num(side_data.get("Effective_Tackle"), 85),
        "possession": parse_possession(side_data.get("time_in_possession"), 30),
        "all_runs": parse_num(side_data.get("all_runs"), 150),
        "all_run_metres": parse_num(side_data.get("all_run_metres"), 1400),
        "post_contact_metres": parse_num(side_data.get("post_contact_metres"), 500),
        "line_breaks": parse_num(side_data.get("line_breaks"), 4),
        "tackle_breaks": parse_num(side_data.get("tackle_breaks"), 30),
        "avg_set_distance": parse_num(side_data.get("average_set_distance"), 40),
        "kick_return_metres": parse_num(side_data.get("kick_return_metres"), 150),
        "offloads": parse_num(side_data.get("offloads"), 8),
        "kicks": parse_num(side_data.get("kicks"), 20),
        "kicking_metres": parse_num(side_data.get("kicking_metres"), 500),
        "forced_drop_outs": parse_num(side_data.get("forced_drop_outs"), 2),
        "tackles_made": parse_num(side_data.get("tackles_made"), 300),
        "missed_tackles": parse_num(side_data.get("missed_tackles"), 25),
        "errors": parse_num(side_data.get("errors"), 10),
        "penalties": parse_num(side_data.get("penalties_conceded"), 5),
        "play_ball_speed": parse_num(side_data.get("Average_Play_Ball_Speed"), 3.5),
        "tries": parse_num(side_data.get("tries"), 3),
    }


def build_team_detailed_history(matches, detailed_stats):
    """Build per-team rolling history of detailed stats."""
    team_detailed = defaultdict(list)

    for m in sorted(matches, key=lambda x: (x["year"], x["round"])):
        home, away = m["home"], m["away"]
        year, rnd = m["year"], m["round"]

        detail = detailed_stats.get((year, rnd, home, away))
        if detail and "home" in detail and "away" in detail:
            home_stats = extract_team_stats(detail["home"])
            away_stats = extract_team_stats(detail["away"])
            home_stats["scored"] = m["home_score"]
            home_stats["conceded"] = m["away_score"]
            away_stats["scored"] = m["away_score"]
            away_stats["conceded"] = m["home_score"]
            team_detailed[home].append(home_stats)
            team_detailed[away].append(away_stats)

    return team_detailed


def load_player_stats(years):
    """Load player statistics and build per-player try scoring history."""
    player_data = defaultdict(lambda: {
        "tries": 0, "games": 0, "first_tries": 0,
        "positions": [], "teams": [], "try_rate_recent": [],
    })

    for year in years:
        path = os.path.join(DATA_DIR, "NRL", str(year), f"NRL_player_statistics_{year}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)

        ps = data["PlayerStats"][0][str(year)]
        for rnd in ps:
            rnd_key = list(rnd.keys())[0]
            for game in rnd[rnd_key]:
                game_name = list(game.keys())[0]
                players = game[game_name]
                parts = game_name.split("v")
                home_team = " ".join(parts[0].split("-")[2:]).replace("-", " ").strip()
                away_team = " ".join(parts[-1:]).replace("-", " ").strip()

                for i, p in enumerate(players):
                    name = p.get("Name", "")
                    if not name:
                        continue
                    team = home_team if i < 18 else away_team
                    tries_str = p.get("Tries", "-")
                    tries = int(tries_str) if tries_str not in ("-", "", "0") else 0
                    pos = p.get("Position", "")

                    entry = player_data[name]
                    entry["games"] += 1
                    entry["tries"] += tries
                    if pos:
                        entry["positions"].append(pos)
                    entry["teams"].append(team)
                    entry["try_rate_recent"].append(tries)

        # First try scorer data
        detail_path = os.path.join(DATA_DIR, "NRL", str(year), f"NRL_detailed_match_data_{year}.json")
        if os.path.exists(detail_path):
            with open(detail_path) as f:
                detail_data = json.load(f)
            for rnd in detail_data["NRL"]:
                for game in rnd[list(rnd.keys())[0]]:
                    match = game[list(game.keys())[0]]
                    fts = match["match"].get("overall_first_try_scorer", "")
                    if fts and fts != "-1" and fts in player_data:
                        player_data[fts]["first_tries"] += 1

    return dict(player_data)


def get_upcoming_matches():
    """Find unplayed matches from the most recent season data."""
    nrl_dir = os.path.join(DATA_DIR, "NRL")
    years = sorted([int(y) for y in os.listdir(nrl_dir) if y.isdigit()], reverse=True)

    for year in years:
        path = os.path.join(nrl_dir, str(year), f"NRL_data_{year}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        rounds = data["NRL"][0][str(year)]
        for rnd in rounds:
            rnd_num = list(rnd.keys())[0]
            games = rnd[rnd_num]
            unplayed = [g for g in games if int(g["Home_Score"]) == 0 and int(g["Away_Score"]) == 0]
            if unplayed:
                return year, int(rnd_num), unplayed
    return None, None, []


def fetch_team_lists(year, rnd, games):
    """Fetch team lists for upcoming matches from NRL match centre pages.

    Returns a dict keyed by (home, away) with value:
        {"home_players": [...], "away_players": [...]}
    Each player dict has: firstName, lastName, fullName, position, number, isOnField.
    """
    team_lists = {}

    for game in games:
        home = game["Home"]
        away = game["Away"]
        url_home = home.lower().replace(" ", "-")
        url_away = away.lower().replace(" ", "-")
        url = f"{EV.NRL_WEBSITE}{year}/round-{rnd}/{url_home}-v-{url_away}/"

        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
            el = soup.find(id="vue-match-centre")
            if not el:
                print(f"    No match centre data for {home} v {away}")
                continue

            data = json.loads(el.get("q-data", "{}"))
            match_data = data.get("match", {})

            def extract_players(team_data):
                players = []
                for p in team_data.get("players", []):
                    players.append({
                        "firstName": p.get("firstName", ""),
                        "lastName": p.get("lastName", ""),
                        "fullName": f"{p.get('firstName', '')} {p.get('lastName', '')}",
                        "position": p.get("position", ""),
                        "number": p.get("number", 0),
                        "isOnField": p.get("isOnField", False),
                    })
                return players

            home_players = extract_players(match_data.get("homeTeam", {}))
            away_players = extract_players(match_data.get("awayTeam", {}))

            team_lists[(home, away)] = {
                "home_players": home_players,
                "away_players": away_players,
            }
            print(f"    {home} v {away}: {len(home_players)} + {len(away_players)} players")

        except Exception as e:
            print(f"    Error fetching team list for {home} v {away}: {e}")

    return team_lists


# ── Feature Engineering ────────────────────────────────────────────────────────

DETAILED_STAT_KEYS = [
    "completion_rate", "effective_tackle", "possession",
    "all_run_metres", "post_contact_metres", "line_breaks", "tackle_breaks",
    "avg_set_distance", "kicking_metres", "forced_drop_outs",
    "tackles_made", "missed_tackles", "errors", "penalties",
    "play_ball_speed", "tries", "scored", "conceded", "offloads",
]


def avg_stats(history, n=5):
    """Average the last n entries of detailed stat history."""
    recent = history[-n:] if len(history) >= n else history
    if not recent:
        return {k: 0.0 for k in DETAILED_STAT_KEYS}
    result = {}
    for k in DETAILED_STAT_KEYS:
        vals = [g.get(k, 0) for g in recent]
        result[k] = np.mean(vals)
    return result


def build_feature_vector(home, away, team_detailed, matches):
    """Build feature vector combining form and detailed stats."""
    h_hist = team_detailed.get(home, [])
    a_hist = team_detailed.get(away, [])

    if len(h_hist) < 3 or len(a_hist) < 3:
        return None

    h = avg_stats(h_hist)
    a = avg_stats(a_hist)

    # Head-to-head from recent matches
    h2h_home_wins, h2h_total, h2h_margin = 0, 0, 0.0
    for m in matches[-300:]:
        if (m["home"] == home and m["away"] == away) or (m["home"] == away and m["away"] == home):
            h2h_total += 1
            if m["home"] == home:
                h2h_margin += m["home_score"] - m["away_score"]
                if m["home_score"] > m["away_score"]:
                    h2h_home_wins += 1
            else:
                h2h_margin += m["away_score"] - m["home_score"]
                if m["away_score"] > m["home_score"]:
                    h2h_home_wins += 1
    h2h_rate = h2h_home_wins / h2h_total if h2h_total > 0 else 0.5
    h2h_avg_margin = h2h_margin / h2h_total if h2h_total > 0 else 0

    # Win rate from recent score history
    h_recent_results = h_hist[-5:]
    a_recent_results = a_hist[-5:]
    h_win_rate = np.mean([1 if g["scored"] > g["conceded"] else 0 for g in h_recent_results])
    a_win_rate = np.mean([1 if g["scored"] > g["conceded"] else 0 for g in a_recent_results])

    features = [
        # -- Home team form --
        h_win_rate,
        h["scored"], h["conceded"], h["scored"] - h["conceded"],
        # -- Home team detailed stats --
        h["completion_rate"], h["effective_tackle"], h["possession"],
        h["all_run_metres"], h["post_contact_metres"],
        h["line_breaks"], h["tackle_breaks"],
        h["avg_set_distance"], h["kicking_metres"], h["forced_drop_outs"],
        h["tackles_made"], h["missed_tackles"],
        h["errors"], h["penalties"],
        h["play_ball_speed"], h["offloads"],
        # -- Away team form --
        a_win_rate,
        a["scored"], a["conceded"], a["scored"] - a["conceded"],
        # -- Away team detailed stats --
        a["completion_rate"], a["effective_tackle"], a["possession"],
        a["all_run_metres"], a["post_contact_metres"],
        a["line_breaks"], a["tackle_breaks"],
        a["avg_set_distance"], a["kicking_metres"], a["forced_drop_outs"],
        a["tackles_made"], a["missed_tackles"],
        a["errors"], a["penalties"],
        a["play_ball_speed"], a["offloads"],
        # -- Differentials (home advantage perspective) --
        h["completion_rate"] - a["completion_rate"],
        h["effective_tackle"] - a["effective_tackle"],
        h["all_run_metres"] - a["all_run_metres"],
        h["line_breaks"] - a["line_breaks"],
        h["errors"] - a["errors"],
        h["missed_tackles"] - a["missed_tackles"],
        (h["scored"] - h["conceded"]) - (a["scored"] - a["conceded"]),
        # -- Head-to-head --
        h2h_rate, h2h_avg_margin,
        # -- Home advantage --
        1.0,
    ]
    return features


NUM_FEATURES = 49  # must match the length of features above


# ── Model Training ─────────────────────────────────────────────────────────────

def train_model(matches, detailed_stats):
    """Train model using rolling detailed stats as features."""
    rolling_detailed = defaultdict(list)
    X, y_win, y_margin = [], [], []
    matches_so_far = []

    for m in sorted(matches, key=lambda x: (x["year"], x["round"])):
        home, away = m["home"], m["away"]
        year, rnd = m["year"], m["round"]

        h_hist = rolling_detailed[home]
        a_hist = rolling_detailed[away]

        # Only train if we have enough history
        if len(h_hist) >= 3 and len(a_hist) >= 3:
            h = avg_stats(h_hist)
            a = avg_stats(a_hist)

            # H2H
            h2h_hw, h2h_t, h2h_m = 0, 0, 0.0
            for past in matches_so_far:
                if (past["home"] == home and past["away"] == away) or \
                   (past["home"] == away and past["away"] == home):
                    h2h_t += 1
                    if past["home"] == home:
                        h2h_m += past["home_score"] - past["away_score"]
                        if past["home_score"] > past["away_score"]:
                            h2h_hw += 1
                    else:
                        h2h_m += past["away_score"] - past["home_score"]
                        if past["away_score"] > past["home_score"]:
                            h2h_hw += 1

            h2h_rate = h2h_hw / h2h_t if h2h_t > 0 else 0.5
            h2h_avg = h2h_m / h2h_t if h2h_t > 0 else 0

            h_recent = h_hist[-5:]
            a_recent = a_hist[-5:]
            h_wr = np.mean([1 if g["scored"] > g["conceded"] else 0 for g in h_recent])
            a_wr = np.mean([1 if g["scored"] > g["conceded"] else 0 for g in a_recent])

            features = [
                h_wr, h["scored"], h["conceded"], h["scored"] - h["conceded"],
                h["completion_rate"], h["effective_tackle"], h["possession"],
                h["all_run_metres"], h["post_contact_metres"],
                h["line_breaks"], h["tackle_breaks"],
                h["avg_set_distance"], h["kicking_metres"], h["forced_drop_outs"],
                h["tackles_made"], h["missed_tackles"],
                h["errors"], h["penalties"],
                h["play_ball_speed"], h["offloads"],
                a_wr, a["scored"], a["conceded"], a["scored"] - a["conceded"],
                a["completion_rate"], a["effective_tackle"], a["possession"],
                a["all_run_metres"], a["post_contact_metres"],
                a["line_breaks"], a["tackle_breaks"],
                a["avg_set_distance"], a["kicking_metres"], a["forced_drop_outs"],
                a["tackles_made"], a["missed_tackles"],
                a["errors"], a["penalties"],
                a["play_ball_speed"], a["offloads"],
                h["completion_rate"] - a["completion_rate"],
                h["effective_tackle"] - a["effective_tackle"],
                h["all_run_metres"] - a["all_run_metres"],
                h["line_breaks"] - a["line_breaks"],
                h["errors"] - a["errors"],
                h["missed_tackles"] - a["missed_tackles"],
                (h["scored"] - h["conceded"]) - (a["scored"] - a["conceded"]),
                h2h_rate, h2h_avg,
                1.0,
            ]

            X.append(features)
            y_win.append(1 if m["home_score"] > m["away_score"] else 0)
            y_margin.append(m["home_score"] - m["away_score"])

        # Update rolling detailed history for this match
        detail = detailed_stats.get((year, rnd, home, away))
        if detail and "home" in detail and "away" in detail:
            h_entry = extract_team_stats(detail["home"])
            a_entry = extract_team_stats(detail["away"])
        else:
            h_entry = {k: 0 for k in DETAILED_STAT_KEYS}
            a_entry = {k: 0 for k in DETAILED_STAT_KEYS}
        h_entry["scored"] = m["home_score"]
        h_entry["conceded"] = m["away_score"]
        a_entry["scored"] = m["away_score"]
        a_entry["conceded"] = m["home_score"]
        rolling_detailed[home].append(h_entry)
        rolling_detailed[away].append(a_entry)
        matches_so_far.append(m)

    X = np.array(X, dtype=np.float32)
    y = np.column_stack([y_win, y_margin]).astype(np.float32)

    print(f"  Training samples: {len(X)} | Features per sample: {X.shape[1]}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = nn.Sequential(
        nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 2),  # [win_logit, margin]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 50
    patience_counter = 0

    print("  Training model...")
    for epoch in range(400):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = bce(pred[:, 0], yb[:, 0]) + 0.01 * mse(pred[:, 1], yb[:, 1])
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate every 20 epochs
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = bce(val_pred[:, 0], y_val_t[:, 0]).item()
                val_win_pred = (torch.sigmoid(val_pred[:, 0]) > 0.5).float()
                accuracy = (val_win_pred == y_val_t[:, 0]).float().mean().item()
                margin_mae = torch.abs(val_pred[:, 1] - y_val_t[:, 1]).mean().item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 20

            if (epoch + 1) % 100 == 0:
                print(f"    Epoch {epoch+1}/400 - Accuracy: {accuracy:.1%} - Margin MAE: {margin_mae:.1f} pts - Val Loss: {val_loss:.4f}")

            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    # Report final accuracy
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_win_pred = (torch.sigmoid(val_pred[:, 0]) > 0.5).float()
        accuracy = (val_win_pred == y_val_t[:, 0]).float().mean().item()
    print(f"  Best val accuracy: {accuracy:.1%}")

    return model, scaler


# ── First Try Scorer Prediction ────────────────────────────────────────────────

def _match_player_name(player_full_name, squad_names):
    """Check if a historical player name matches any name in the squad list.

    Uses exact match first, then falls back to matching where both first
    name initial AND full last name match (handles multi-part surnames).
    """
    pn = player_full_name.strip().lower()
    for sn in squad_names:
        sn_lower = sn.lower()
        # Exact match
        if pn == sn_lower:
            return sn
        # First-initial + full last name match (e.g. "J Hughes" vs "Jahrome Hughes")
        p_parts = pn.split()
        s_parts = sn_lower.split()
        if len(p_parts) >= 2 and len(s_parts) >= 2:
            # Compare first initial and everything after the first name
            p_last = " ".join(p_parts[1:])
            s_last = " ".join(s_parts[1:])
            if p_last == s_last and p_parts[0][0] == s_parts[0][0]:
                return sn
    return None


def predict_first_try_scorer(home, away, player_data, squad=None):
    """Predict most likely first try scorers for a match.

    If squad is provided, only players named in the squad are considered.
    squad should be a dict with 'home_players' and 'away_players' lists
    from fetch_team_lists().
    """
    # Build set of squad player names and their positions/teams if available
    squad_lookup = {}  # fullName -> {team, position}
    if squad:
        for p in squad.get("home_players", []):
            squad_lookup[p["fullName"]] = {"team": home, "position": p["position"]}
        for p in squad.get("away_players", []):
            squad_lookup[p["fullName"]] = {"team": away, "position": p["position"]}
        squad_names = list(squad_lookup.keys())
    else:
        squad_names = None

    candidates = []

    for name, stats in player_data.items():
        if stats["games"] < 3:
            continue

        # If we have a squad list, only consider players in it
        if squad_names is not None:
            matched = _match_player_name(name, squad_names)
            if not matched:
                continue
            squad_info = squad_lookup[matched]
            team = squad_info["team"]
            pos = squad_info["position"]
        else:
            recent_teams = stats["teams"][-10:] if stats["teams"] else []
            if home not in recent_teams and away not in recent_teams:
                continue
            team = home if recent_teams.count(home) >= recent_teams.count(away) else away
            pos = max(set(stats["positions"]), key=stats["positions"].count) if stats["positions"] else "Unknown"

        try_rate = stats["tries"] / stats["games"] if stats["games"] > 0 else 0
        recent_tries = stats["try_rate_recent"][-10:]
        recent_rate = np.mean(recent_tries) if recent_tries else 0
        fts_rate = stats["first_tries"] / stats["games"] if stats["games"] > 0 else 0

        pos_bonus = {
            "Winger": 1.5, "Fullback": 1.3, "Centre": 1.2,
            "Five-Eighth": 1.0, "Halfback": 0.9,
            "Lock": 0.8, "Second Row": 0.8, "2nd Row": 0.8,
            "Prop": 0.5, "Hooker": 0.7,
            "Interchange": 0.6, "Reserve": 0.3,
        }
        bonus = pos_bonus.get(pos, 0.8)
        score = (try_rate * 0.4 + recent_rate * 0.3 + fts_rate * 3.0 + 0.1) * bonus

        candidates.append({
            "name": name, "team": team, "position": pos,
            "try_rate": try_rate, "fts_rate": fts_rate,
            "recent_rate": recent_rate, "score": score,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[:10]
    total_score = sum(c["score"] for c in top)
    for c in top:
        c["probability"] = c["score"] / total_score * 100 if total_score > 0 else 0

    return top


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  NRL ROUND PREDICTOR")
    print("=" * 80)

    pred_year, pred_round, upcoming = get_upcoming_matches()
    if not upcoming:
        print("No upcoming matches found!")
        return

    print(f"\n  Predicting: {pred_year} Round {pred_round}")
    print(f"  Matches: {len(upcoming)}")

    all_years = list(range(2018, pred_year + 1))
    print(f"\n  Loading data from {all_years[0]}-{all_years[-1]}...")

    matches = load_all_match_data(all_years)
    print(f"  Loaded {len(matches)} historical matches")

    detailed_stats = load_detailed_team_stats(all_years)
    print(f"  Loaded {len(detailed_stats)} detailed match stat records")

    player_data = load_player_stats(all_years)
    print(f"  Loaded stats for {len(player_data)} players")

    # Build team detailed history for prediction
    team_detailed = build_team_detailed_history(matches, detailed_stats)

    # Fetch team lists for upcoming matches
    print(f"\n  Fetching team lists for Round {pred_round}...")
    team_lists = fetch_team_lists(pred_year, pred_round, upcoming)
    print(f"  Retrieved team lists for {len(team_lists)}/{len(upcoming)} matches")

    # Train
    print()
    model, scaler = train_model(matches, detailed_stats)

    # Predict
    print("\n" + "=" * 80)
    print(f"  PREDICTIONS - {pred_year} ROUND {pred_round}")
    print("=" * 80)

    model.eval()
    for game in upcoming:
        home = game["Home"]
        away = game["Away"]
        venue = game["Venue"]

        print(f"\n  {'-' * 76}")
        print(f"  {home} (Home) vs {away} (Away)")
        print(f"  Venue: {venue}")
        print(f"  {'-' * 76}")

        # Show team lists if available
        squad = team_lists.get((home, away))
        if squad:
            print(f"\n  TEAM LISTS:")
            print(f"  {home}:")
            for p in squad["home_players"]:
                marker = "*" if p["isOnField"] else " "
                print(f"    {marker} #{p['number']:2d} {p['fullName']:25s} {p['position']}")
            print(f"  {away}:")
            for p in squad["away_players"]:
                marker = "*" if p["isOnField"] else " "
                print(f"    {marker} #{p['number']:2d} {p['fullName']:25s} {p['position']}")

        features = build_feature_vector(home, away, team_detailed, matches)

        if features is None:
            print(f"  WARNING: Insufficient data to predict this match")
            continue

        with torch.no_grad():
            x_np = np.array([features], dtype=np.float32)
            x_scaled = scaler.transform(x_np)
            x_t = torch.tensor(x_scaled, dtype=torch.float32)
            raw = model(x_t).numpy()[0]

        raw_prob = 1 / (1 + np.exp(-raw[0]))
        # Clamp to realistic NRL range — even heavy favourites rarely
        # exceed ~80% implied probability (odds ~$1.25)
        home_win_prob = np.clip(raw_prob, 0.15, 0.85)
        away_win_prob = 1 - home_win_prob
        pred_margin = raw[1]

        winner = home if home_win_prob > 0.5 else away
        win_prob = max(home_win_prob, away_win_prob) * 100
        margin = abs(pred_margin)

        print(f"\n  WINNER:  {winner} ({win_prob:.1f}% confidence)")
        print(f"  MARGIN:  {margin:.0f} points")
        print(f"     {home}: {home_win_prob*100:.1f}%  |  {away}: {away_win_prob*100:.1f}%")

        fts = predict_first_try_scorer(home, away, player_data, squad=squad)
        if fts:
            source = "from named squad" if squad else "from historical data"
            print(f"\n  FIRST TRY SCORER PREDICTIONS ({source}):")
            for i, c in enumerate(fts[:5]):
                print(f"     {i+1}. {c['name']:25s} ({c['team']:15s} | {c['position']:12s}) - {c['probability']:.1f}%  (try rate: {c['try_rate']:.2f}/game)")

    print(f"\n{'=' * 80}")
    print(f"  Features used: team form, completion rate, tackle efficiency,")
    print(f"  run metres, line breaks, errors, penalties, kicking, head-to-head")
    print(f"  Based on {len(matches)} matches with detailed stats")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
