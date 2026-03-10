"""
SuperCoach NRL Team Optimizer

Fetches player data from the SuperCoach API, predicts points per player
using historical NRL stats and SuperCoach scoring, then uses constrained
optimization (scipy MILP) to select an optimal 26-man squad.

Usage:
    python supercoach.py
    python supercoach.py --squad "Player1, Player2, ..."   # trade advice
    python supercoach.py --offline                          # use cached API data
"""

import sys
import os
import json
import argparse
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import requests
from scipy.optimize import milp, LinearConstraint, Bounds

import ENVIRONMENT_VARIABLES as EV

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ── Constants ──────────────────────────────────────────────────────────────────

SUPERCOACH_API = (
    "https://www.supercoach.com.au/2026/api/nrl/classic/v1/players"
    "?embed=stats,positions,player_stats"
)

SALARY_CAP = 11_950_000

POSITION_REQUIREMENTS = {
    "FLB": 2,
    "CTW": 7,
    "5/8": 2,
    "HFB": 2,
    "2RF": 6,
    "FRF": 4,
    "HOK": 2,
}

SQUAD_SIZE = 26  # 25 positional + 1 flex
TOTAL_SEASON_TRADES = 46
MULTI_BYE_ROUNDS = {12, 15, 18}  # 3 trades allowed instead of 2

SQUAD_FILE = os.path.join(DATA_DIR, "my_supercoach_squad.json")


# ── Utility ────────────────────────────────────────────────────────────────────

def parse_num(val, default=0.0):
    """Parse a stat value that might be a string like '1,822', '78%', '3.72s', or '-'."""
    if val is None or val == -1 or val == -10 or val == "-1" or val == "-":
        return default
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).replace(",", "").replace("%", "").replace("s", "").strip()
    try:
        return float(val)
    except ValueError:
        return default


def parse_mins(val):
    """Parse minutes played field like '80:00' → 80.0."""
    if not val or val == "-":
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).strip()
    if ":" in val:
        parts = val.split(":")
        return float(parts[0]) + float(parts[1]) / 60
    return parse_num(val)


def match_name(sc_name, nrl_names):
    """Match a SuperCoach player name to NRL stats player names.

    Uses exact match first, then first-initial + full last name match.
    """
    sc_lower = sc_name.strip().lower()
    for nrl_name in nrl_names:
        nrl_lower = nrl_name.lower()
        if sc_lower == nrl_lower:
            return nrl_name
        sc_parts = sc_lower.split()
        nrl_parts = nrl_lower.split()
        if len(sc_parts) >= 2 and len(nrl_parts) >= 2:
            sc_last = " ".join(sc_parts[1:])
            nrl_last = " ".join(nrl_parts[1:])
            if sc_last == nrl_last and sc_parts[0][0] == nrl_parts[0][0]:
                return nrl_name
    return None


# ── Squad Persistence ──────────────────────────────────────────────────────────

def save_squad(squad, trades_used=0, trade_history=None):
    """Save the current squad to disk for use in future rounds."""
    # Load existing data to preserve trade history
    existing = _load_squad_file()
    if existing:
        trades_used = existing.get("trades_used", 0)
        trade_history = existing.get("trade_history", [])

    data = {
        "squad": [
            {
                "name": p["name"],
                "team": p["team"],
                "positions": p["positions"],
                "assigned_position": p["assigned_position"],
                "price": p["price"],
                "starter": bool(p.get("starter", True)),
            }
            for p in squad
        ],
        "trades_used": trades_used,
        "trade_history": trade_history or [],
    }
    with open(SQUAD_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Squad saved to {SQUAD_FILE}")


def _load_squad_file():
    """Load raw squad data from disk, or None if not found / invalid."""
    if not os.path.exists(SQUAD_FILE):
        return None
    try:
        with open(SQUAD_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        return None


def load_my_squad():
    """Load saved squad and trade tracking info."""
    data = _load_squad_file()
    if not data:
        return None, 0, []
    return data["squad"], data.get("trades_used", 0), data.get("trade_history", [])


def record_trades(trade_list, current_round):
    """Append completed trades to the saved squad file."""
    data = _load_squad_file()
    if not data:
        return
    for t in trade_list:
        data["trade_history"].append({
            "round": current_round,
            "out": t["out"]["name"],
            "in": t["in"]["name"],
            "pts_gain": round(t["pts_gain"], 1),
        })
        data["trades_used"] += 1
    with open(SQUAD_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Data Fetching ──────────────────────────────────────────────────────────────

def fetch_supercoach_data():
    """Fetch fresh player data from the SuperCoach API."""
    print("  Fetching SuperCoach player data...")
    resp = requests.get(
        SUPERCOACH_API,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    players = resp.json()

    cache_path = os.path.join(DATA_DIR, "supercoach_players_2026.json")
    with open(cache_path, "w") as f:
        json.dump(players, f, indent=2)

    print(f"  Fetched {len(players)} players")
    return players


def parse_supercoach_players(raw_players):
    """Parse raw SuperCoach API data into a structured list of eligible players."""
    players = []
    for p in raw_players:
        if not p.get("active", False):
            continue

        positions = [pos["position"] for pos in p.get("positions", [])]
        if not positions:
            continue

        team_name = p.get("team", {}).get("name", "")
        stats = p["player_stats"][0] if p.get("player_stats") else {}

        price = stats.get("price", 0)
        if not price or price <= 0:
            continue

        played_status = p.get("played_status", {}).get("status", "pre")

        players.append({
            "name": f"{p['first_name']} {p['last_name']}",
            "team": team_name,
            "positions": positions,
            "price": price,
            "previous_average": p.get("previous_average") or 0,
            "previous_games": p.get("previous_games") or 0,
            "current_avg": stats.get("avg") or 0,
            "current_games": stats.get("total_games") or 0,
            "std": stats.get("std") or 0,
            "status": p.get("injury_suspension_status", ""),
            "status_text": p.get("injury_suspension_status_text", ""),
            "locked": bool(p.get("locked", False)),
            "played": played_status == "post",
            "ownership": stats.get("own") or 0,
            "breakeven": stats.get("be1") or 0,
        })

    return players


# ── Historical Stats → SuperCoach Points ───────────────────────────────────────

def compute_sc_points(game_stats):
    """Compute estimated SuperCoach points from a single game's NRL player stats."""
    g = lambda key: parse_num(game_stats.get(key))

    tries = g("Tries")
    try_assists = g("Try Assists")
    conversions = g("Conversions")
    conv_attempts = g("Conversion Attempts")
    penalty_goals = g("Penalty Goals")
    fg_1pt = g("1 Point Field Goals")
    fg_2pt = g("2 Point Field Goals")
    tackles = g("Tackles Made")
    missed_tackles = g("Missed Tackles")
    tackle_breaks = g("Tackle Breaks")
    forced_drop_outs = g("Forced Drop Outs")
    offloads = g("Offloads")
    line_breaks = g("Line Breaks")
    line_break_assists = g("Line Break Assists")
    forty_twenty = g("40/20")
    twenty_forty = g("20/40")
    intercepts = g("Intercepts")
    kicked_dead = g("Kicked Dead")
    penalties = g("Penalties")
    errors = g("Errors")
    sin_bins = g("Sin Bins")
    send_offs = g("Send Offs")

    # Estimate runs over/under 8 m from totals
    all_runs = g("All Runs")
    all_run_metres = g("All Run Metres")
    if all_runs > 0:
        avg_per_run = all_run_metres / all_runs
        frac_over = max(0.0, min(1.0, (avg_per_run - 4) / 8))
        runs_over = all_runs * frac_over
        runs_under = all_runs - runs_over
    else:
        runs_over = runs_under = 0.0

    missed_goals = max(0, conv_attempts - conversions)

    return (
        tries * 17
        + try_assists * 12
        + conversions * 4
        + penalty_goals * 4
        + missed_goals * (-2)
        + fg_1pt * 5
        + fg_2pt * 10
        + tackles * 1
        + missed_tackles * (-1)
        + tackle_breaks * 2
        + forced_drop_outs * 6
        + offloads * 4
        + line_breaks * 10
        + line_break_assists * 8
        + forty_twenty * 10
        + twenty_forty * 10
        + runs_over * 2
        + runs_under * 1
        + intercepts * 5
        + kicked_dead * (-3)
        + penalties * (-2)
        + errors * (-2)
        + sin_bins * (-8)
        + send_offs * (-16)
    )


def load_historical_sc_points(years):
    """Load NRL player stats and compute SC points per player per game.

    Returns dict: player_name → list of {year, round, points, mins}.
    """
    player_points = defaultdict(list)

    for year in years:
        path = os.path.join(
            DATA_DIR, "NRL", str(year), f"NRL_player_statistics_{year}.json"
        )
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)

        ps = data["PlayerStats"][0][str(year)]
        for rnd in ps:
            rnd_key = list(rnd.keys())[0]
            for game in rnd[rnd_key]:
                game_name = list(game.keys())[0]
                for p in game[game_name]:
                    name = p.get("Name", "")
                    if not name:
                        continue
                    mins = parse_mins(p.get("Mins Played"))
                    if mins <= 0:
                        continue
                    player_points[name].append({
                        "year": year,
                        "round": int(rnd_key),
                        "points": compute_sc_points(p),
                        "mins": mins,
                    })

    return dict(player_points)


# ── Point Prediction ───────────────────────────────────────────────────────────

def predict_player_points(sc_player, historical, nrl_names):
    """Predict SuperCoach points per game for a player.

    Blends SuperCoach previous_average with estimated SC points from
    historical NRL stats.
    """
    prev_avg = sc_player["previous_average"]
    prev_games = sc_player["previous_games"]
    current_avg = sc_player["current_avg"]
    current_games = sc_player["current_games"]

    # If the 2026 season is under way, weight current form
    if current_games >= 3:
        return current_avg * 0.6 + prev_avg * 0.4
    if current_games > 0 and current_avg > 0:
        return current_avg * 0.3 + prev_avg * 0.7

    # Look up historical NRL stats for this player
    matched = match_name(sc_player["name"], nrl_names)
    hist_avg = 0.0
    if matched and matched in historical:
        games = historical[matched]
        recent = [g["points"] for g in games if g["year"] >= 2025]
        older = [g["points"] for g in games if g["year"] < 2025]
        recent_avg = np.mean(recent) if recent else 0.0
        older_avg = np.mean(older[-20:]) if older else 0.0
        hist_avg = recent_avg * 0.7 + older_avg * 0.3 if recent else older_avg

    if prev_avg > 0 and prev_games >= 5:
        if hist_avg > 0:
            return prev_avg * 0.7 + hist_avg * 0.3
        return prev_avg

    if hist_avg > 0:
        return hist_avg

    # Rookie / unknown – rough estimate from price
    return max(15.0, sc_player["price"] / 10_000)


# ── Squad Optimization ─────────────────────────────────────────────────────────

SCORING_17 = 17  # only best 17 from Starting 18 actually score


def optimize_squad(players, strategy="points", current_squad_names=None,
                   max_trades=None):
    """Use MILP to find the optimal 26-man squad under the salary cap.

    Only the best 17 scores from your Starting 18 count each round, so
    the objective maximises the top-17 predicted points while filling the
    remaining 9 bench spots as cheaply as possible.

    Parameters
    ----------
    strategy : str
        'points' – maximise predicted weekly SuperCoach score.
        'growth' – maximise price-growth potential (predicted - breakeven).
    current_squad_names : set[str] | None
        If provided, limits changes from the current squad to max_trades.
    max_trades : int | None
        Maximum number of players that can be swapped out of current_squad.

    Variables
    ---------
    x[i, p]  binary – player *i* is assigned to position slot *p* (in the 26).
    s[i]     binary – player *i* is a scoring starter (in the best 17).

    Constraints
    -----------
    * Each player assigned to at most one position.
    * Each position filled to its required count, +1 flex.
    * Total squad = 26, exactly 17 starters.
    * Starter only if selected: s[i] <= sum_p x[i, p].
    * Total salary <= cap.
    """
    N = len(players)
    positions = list(POSITION_REQUIREMENTS.keys())
    P = len(positions)
    # Variables: [x_0_0 .. x_0_P-1, x_1_0 .. x_N-1_P-1, s_0 .. s_N-1]
    num_x = N * P
    num_vars = num_x + N  # x variables + s (starter) variables

    # ── Objective ─────────────────────────────────────────────────────────
    # Only starters contribute to the score (milp minimises → negate).
    c = np.zeros(num_vars)
    for i, pl in enumerate(players):
        if strategy == "growth":
            score = (
                pl["predicted_points"] * 0.4
                + max(0, pl.get("growth", 0)) * 0.6
            )
        else:
            score = pl["predicted_points"]
        c[num_x + i] = -score  # only the s[i] variable carries the objective

    # ── Bounds ────────────────────────────────────────────────────────────
    ub = np.zeros(num_vars)
    # x bounds: eligible (player, position) pairs only
    for i, pl in enumerate(players):
        for j, pos in enumerate(positions):
            if pos in pl["positions"]:
                ub[i * P + j] = 1.0
    # s bounds: all players can potentially be starters
    for i in range(N):
        ub[num_x + i] = 1.0

    bounds = Bounds(lb=0.0, ub=ub)
    integrality = np.ones(num_vars)

    # ── Constraints ───────────────────────────────────────────────────────

    # 1. Each player selected at most once across positions
    A_player = np.zeros((N, num_vars))
    for i in range(N):
        for j in range(P):
            A_player[i, i * P + j] = 1.0

    # 2. Position requirements (min = required, max = required + 1 for flex)
    A_pos = np.zeros((P, num_vars))
    for j in range(P):
        for i in range(N):
            A_pos[j, i * P + j] = 1.0
    pos_lb = np.array([POSITION_REQUIREMENTS[p] for p in positions], dtype=float)
    pos_ub = pos_lb + 1

    # 3. Total squad = 26
    A_total = np.zeros((1, num_vars))
    A_total[0, :num_x] = 1.0

    # 4. Salary cap (applies to all 26 selected players)
    A_salary = np.zeros((1, num_vars))
    for i, pl in enumerate(players):
        for j in range(P):
            A_salary[0, i * P + j] = pl["price"]

    # 5. Starter only if selected: s[i] <= sum_p x[i,p]
    #    Rewritten as: s[i] - sum_p x[i,p] <= 0
    A_starter = np.zeros((N, num_vars))
    for i in range(N):
        A_starter[i, num_x + i] = 1.0        # +s[i]
        for j in range(P):
            A_starter[i, i * P + j] = -1.0    # -x[i,p]

    # 6. Exactly 17 starters
    A_scoring = np.zeros((1, num_vars))
    A_scoring[0, num_x:] = 1.0

    constraints = [
        LinearConstraint(A_player, 0, 1),
        LinearConstraint(A_pos, pos_lb, pos_ub),
        LinearConstraint(A_total, SQUAD_SIZE, SQUAD_SIZE),
        LinearConstraint(A_salary, 0, SALARY_CAP),
        LinearConstraint(A_starter, -np.inf, 0),       # s[i] <= selected[i]
        LinearConstraint(A_scoring, SCORING_17, SCORING_17),
    ]

    # 7. Trade constraint: keep at least (26 - max_trades) current squad members
    trade_msg = ""
    if current_squad_names and max_trades is not None:
        A_keep = np.zeros((1, num_vars))
        current_count = 0
        for i, pl in enumerate(players):
            if pl["name"] in current_squad_names:
                for j in range(P):
                    A_keep[0, i * P + j] = 1.0
                current_count += 1
        min_keep = max(0, current_count - max_trades)
        constraints.append(LinearConstraint(A_keep, min_keep, current_count))
        trade_msg = f", max {max_trades} trades"

    print(f"  Optimising squad ({N} players, {num_vars} variables, best-17 scoring{trade_msg})...")
    result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

    if not result.success:
        print(f"  Optimisation failed: {result.message}")
        return []

    # Extract selected players, their position, and starter status
    x = result.x
    squad = []
    for i, pl in enumerate(players):
        for j, pos in enumerate(positions):
            if x[i * P + j] > 0.5:
                is_starter = x[num_x + i] > 0.5
                squad.append({
                    **pl,
                    "assigned_position": pos,
                    "starter": is_starter,
                })
                break

    squad.sort(key=lambda p: (
        list(POSITION_REQUIREMENTS.keys()).index(p["assigned_position"]),
        not p["starter"],  # starters first within each position
    ))
    return squad


# ── Captain & Trade Advice ─────────────────────────────────────────────────────

def recommend_captain(squad):
    """Return the player with the highest predicted points (captain gets 2×)."""
    return max(squad, key=lambda p: p["predicted_points"])


def suggest_trades(current_squad_names, all_players, max_trades=5):
    """Compare current squad against the full pool and suggest value trades."""
    current_set = {n.lower().strip() for n in current_squad_names}

    current = [p for p in all_players if p["name"].lower() in current_set]
    available = [p for p in all_players if p["name"].lower() not in current_set]

    if not current:
        print("  Could not match any current squad player names.")
        return []

    trades = []
    for pos in POSITION_REQUIREMENTS:
        in_pos = sorted(
            [p for p in current if pos in p["positions"]],
            key=lambda p: p["value"],
        )
        avail_pos = sorted(
            [p for p in available if pos in p["positions"]],
            key=lambda p: p["predicted_points"],
            reverse=True,
        )
        if not in_pos or not avail_pos:
            continue

        worst = in_pos[0]
        for cand in avail_pos[:5]:
            if cand["predicted_points"] > worst["predicted_points"]:
                trades.append({
                    "out": worst,
                    "in": cand,
                    "price_diff": cand["price"] - worst["price"],
                    "pts_gain": cand["predicted_points"] - worst["predicted_points"],
                    "position": pos,
                })

    trades.sort(key=lambda t: t["pts_gain"], reverse=True)
    return trades[:max_trades]


# ── Output ─────────────────────────────────────────────────────────────────────

STATUS_FLAGS = {
    "PlayingNextRound": "",
    "EmergencyNextRound": "[EMG]",
    "NotPlayingNextRound": "[OUT]",
    "Bye": "[BYE]",
    "Injury": "[INJ]",
    "Suspended": "[SUS]",
}


def _status_flag(player):
    """Return a short flag string for a player's availability status."""
    return STATUS_FLAGS.get(player.get("status", ""), "")


def print_squad(squad):
    """Print the optimised squad in a formatted table."""
    starters = [p for p in squad if p.get("starter", True)]
    bench = [p for p in squad if not p.get("starter", True)]
    total_price = sum(p["price"] for p in squad)
    starter_points = sum(p["predicted_points"] for p in starters)
    total_growth = sum(p.get("growth", 0) for p in starters)

    print(f"\n  {'-' * 120}")
    print(
        f"  {'Player':<25s} {'Team':<15s} {'Pos':<8s} "
        f"{'Price':>10s} {'Pred Pts':>8s} {'BE':>5s} {'Growth':>7s} "
        f"{'Pts/$100k':>9s}  {'Role':<6s} {'Status'}"
    )
    print(f"  {'-' * 120}")

    current_pos = None
    warnings = []
    for p in squad:
        if p["assigned_position"] != current_pos:
            current_pos = p["assigned_position"]
            req = POSITION_REQUIREMENTS[current_pos]
            count = sum(1 for x in squad if x["assigned_position"] == current_pos)
            flex = " (+FLEX)" if count > req else ""
            print(f"  {current_pos}{flex}:")

        flag = _status_flag(p)
        growth = p.get("growth", 0)
        growth_str = f"{growth:+.0f}" if growth != 0 else "0"
        role = "START" if p.get("starter", True) else "BENCH"
        print(
            f"    {p['name']:<25s} {p['team']:<15s} "
            f"{'/'.join(p['positions']):<8s} "
            f"${p['price']:>9,d} {p['predicted_points']:>7.1f} "
            f"{p.get('breakeven', 0):>5.0f} {growth_str:>7s} "
            f"{p['value']:>9.1f}  {role:<6s} {flag}"
        )
        if flag:
            warnings.append(f"    ! {p['name']}: {p.get('status_text', p['status'])}")

    print(f"  {'-' * 120}")
    bench_price = sum(p["price"] for p in bench)
    print(
        f"  {'SCORING 17':<25s} {'':15s} {'':8s} "
        f"{'':>10s} {starter_points:>7.1f} "
        f"{'':>5s} {total_growth:>+7.0f}"
    )
    print(
        f"  {'TOTAL (26)':<25s} {'':15s} {'':8s} "
        f"${total_price:>9,d}"
    )
    print(f"  Salary cap remaining: ${SALARY_CAP - total_price:,d}")
    print(f"  Bench spend: ${bench_price:,d} on {len(bench)} players")

    if warnings:
        print(f"\n  AVAILABILITY WARNINGS:")
        for w in warnings:
            print(w)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SuperCoach NRL Team Optimizer")
    parser.add_argument(
        "--offline", action="store_true",
        help="Use cached SuperCoach data instead of fetching fresh",
    )
    parser.add_argument(
        "--playing-only", action="store_true",
        help="Only include players confirmed to play this round "
             "(excludes byes, emergencies, and not-selected)",
    )
    parser.add_argument(
        "--strategy", choices=["points", "growth"], default="points",
        help="'points' = maximise predicted weekly score (default). "
             "'growth' = maximise price-growth potential.",
    )
    parser.add_argument(
        "--trades", action="store_true",
        help="Load saved squad and suggest optimal trades for next round.",
    )
    parser.add_argument(
        "--round", type=int, default=None,
        help="Current round number (affects max trades: 3 in bye rounds 12/15/18).",
    )
    parser.add_argument(
        "--boost", action="store_true",
        help="Activate a Trade Boost for +1 extra trade this round.",
    )
    parser.add_argument(
        "--confirm-trades", action="store_true",
        help="Actually save the recommended trades (without this, trades are preview only).",
    )
    args = parser.parse_args()

    trade_mode = args.trades
    current_round = args.round or 1

    # Prices don't move until after Round 3, so growth strategy is
    # ineffective before then — override to points automatically.
    PRICE_FREEZE_ROUNDS = 3
    strategy = args.strategy
    if strategy == "growth" and current_round <= PRICE_FREEZE_ROUNDS:
        strategy = "points"
        print(f"  NOTE: Prices are frozen until Round {PRICE_FREEZE_ROUNDS + 1}. "
              f"Overriding strategy to 'points'.")

    title = "TRADE ADVISOR" if trade_mode else "TEAM OPTIMIZER"

    print("=" * 96)
    print(f"  SUPERCOACH NRL {title}  (strategy: {strategy}, round {current_round})")
    print("=" * 96)

    # ── 1. Load SuperCoach player data ────────────────────────────────────
    if args.offline:
        cache_path = os.path.join(DATA_DIR, "supercoach_players_2026.json")
        with open(cache_path) as f:
            raw_players = json.load(f)
        print(f"  Loaded {len(raw_players)} players from cache")
    else:
        try:
            raw_players = fetch_supercoach_data()
        except Exception as e:
            print(f"  API fetch failed ({e}), falling back to cache...")
            cache_path = os.path.join(DATA_DIR, "supercoach_players_2026.json")
            with open(cache_path) as f:
                raw_players = json.load(f)

    sc_players = parse_supercoach_players(raw_players)
    print(f"  {len(sc_players)} active players with valid prices")

    # ── 2. Load historical NRL stats ──────────────────────────────────────
    years = list(range(2023, 2027))
    print(f"  Loading historical NRL stats ({years[0]}-{years[-1]})...")
    historical = load_historical_sc_points(years)
    nrl_names = list(historical.keys())
    print(f"  Computed SC points for {len(historical)} players from NRL stats")

    # ── 3. Predict points per player ──────────────────────────────────────
    print("  Predicting player points...")
    for p in sc_players:
        p["predicted_points"] = predict_player_points(p, historical, nrl_names)
        p["value"] = (
            p["predicted_points"] / (p["price"] / 100_000) if p["price"] > 0 else 0
        )
        be = p["breakeven"]
        p["growth"] = p["predicted_points"] - be if be > 0 else 0

    # ── 4. Eligibility filter ─────────────────────────────────────────────
    ALWAYS_EXCLUDE = ("Suspended", "Injury")
    PLAYING_ONLY_ALLOW = ("PlayingNextRound",)
    playing_only = getattr(args, "playing_only", False) or strategy == "growth"

    def is_eligible(p):
        if p["locked"]:
            return False
        if p["status"] in ALWAYS_EXCLUDE:
            return False
        if playing_only and p["status"] not in PLAYING_ONLY_ALLOW:
            return False
        return True

    eligible = [p for p in sc_players if is_eligible(p)]
    excluded = len(sc_players) - len(eligible)
    mode = "playing-only" if playing_only else "default"
    print(f"  {len(eligible)} eligible players ({mode} filter, {excluded} excluded)")

    # ── 5. Trade mode or fresh squad ──────────────────────────────────────
    if trade_mode:
        my_squad, trades_used, trade_history = load_my_squad()
        if not my_squad:
            print("  No saved squad found! Run without --trades first to build one.")
            return

        trades_remaining = TOTAL_SEASON_TRADES - trades_used
        base_trades = 3 if current_round in MULTI_BYE_ROUNDS else 2
        max_trades = base_trades + (1 if args.boost else 0)
        max_trades = min(max_trades, trades_remaining)

        current_names = {p["name"] for p in my_squad}
        print(f"  Loaded saved squad ({len(my_squad)} players)")
        print(
            f"  Round {current_round}: "
            f"up to {max_trades} trades this round "
            f"({trades_remaining}/{TOTAL_SEASON_TRADES} remaining for season)"
        )

        # Only allow trading IN players who are confirmed to play.
        # Current squad members stay in the pool regardless of status
        # (they're already on our team and can't be un-rostered).
        trade_in_eligible = [
            p for p in eligible
            if p["name"] in current_names or p["status"] in PLAYING_ONLY_ALLOW
        ]
        pool = list(trade_in_eligible)
        for p in sc_players:
            if p["name"] in current_names and p["name"] not in {x["name"] for x in pool}:
                pool.append(p)

        playing_in = sum(1 for p in pool if p["name"] not in current_names)
        print(f"  {playing_in} confirmed-playing trade-in candidates")

        new_squad = optimize_squad(
            pool,
            strategy=strategy,
            current_squad_names=current_names,
            max_trades=max_trades,
        )
        if not new_squad:
            print("  Optimisation failed!")
            return

        # Identify actual trades
        new_names = {p["name"] for p in new_squad}
        traded_out = current_names - new_names
        traded_in = new_names - current_names

        if not traded_out:
            print("\n  NO TRADES RECOMMENDED - current squad is optimal.")
            print_squad(new_squad)
        else:
            print(f"\n  RECOMMENDED TRADES ({len(traded_out)}):")
            out_players = [p for p in sc_players if p["name"] in traded_out]
            in_players = [p for p in new_squad if p["name"] in traded_in]
            for o, i in zip(
                sorted(out_players, key=lambda x: x["predicted_points"]),
                sorted(in_players, key=lambda x: x["predicted_points"], reverse=True),
            ):
                pts_diff = i["predicted_points"] - o["predicted_points"]
                print(
                    f"    OUT: {o['name']:<25s} ({o['team']:<15s} "
                    f"${o['price']:>9,d}  {o['predicted_points']:>5.1f}pts)"
                )
                print(
                    f"    IN:  {i['name']:<25s} ({i['team']:<15s} "
                    f"${i['price']:>9,d}  {i['predicted_points']:>5.1f}pts)  "
                    f"gain: {pts_diff:+.1f}pts"
                )
                print()

            print("  NEW SQUAD AFTER TRADES:")
            print_squad(new_squad)

            print(
                f"\n  To confirm these trades, re-run with --confirm-trades."
                f"\n  ({trades_remaining - len(traded_out)}/{TOTAL_SEASON_TRADES} "
                f"trades would remain)"
            )

            if getattr(args, "confirm_trades", False):
                save_squad(new_squad)
                trade_list = [
                    {"out": o, "in": i, "pts_gain": i["predicted_points"] - o["predicted_points"]}
                    for o, i in zip(
                        sorted(out_players, key=lambda x: x["predicted_points"]),
                        sorted(in_players, key=lambda x: x["predicted_points"], reverse=True),
                    )
                ]
                record_trades(trade_list, current_round)
                print(
                    f"  CONFIRMED. Trades recorded. "
                    f"{trades_remaining - len(traded_out)}/{TOTAL_SEASON_TRADES} "
                    f"trades remaining."
                )
    else:
        squad = optimize_squad(eligible, strategy=strategy)
        if not squad:
            print("  Failed to find optimal squad!")
            return

        print_squad(squad)
        save_squad(squad)

    # ── 6. Captain recommendation ─────────────────────────────────────────
    final_squad = new_squad if trade_mode else squad
    captain = recommend_captain(final_squad)
    by_pts = sorted(final_squad, key=lambda p: p["predicted_points"], reverse=True)
    vice = by_pts[1] if len(by_pts) > 1 else None

    print(f"\n  CAPTAIN RECOMMENDATION:")
    print(
        f"    {captain['name']} ({captain['team']}) - "
        f"{captain['predicted_points']:.1f} pts "
        f"(doubled: {captain['predicted_points'] * 2:.1f} pts)"
    )
    if vice:
        print(
            f"    Vice: {vice['name']} ({vice['team']}) - "
            f"{vice['predicted_points']:.1f} pts"
        )

    # ── 7. Top value players not selected ─────────────────────────────────
    final_names = {p["name"] for p in final_squad}
    not_selected = sorted(
        [p for p in eligible if p["name"] not in final_names],
        key=lambda p: p["value"],
        reverse=True,
    )
    print(f"\n  TOP VALUE PLAYERS NOT IN SQUAD:")
    for p in not_selected[:10]:
        print(
            f"    {p['name']:<25s} {p['team']:<15s} "
            f"{'/'.join(p['positions']):<8s} "
            f"${p['price']:>9,d} {p['predicted_points']:>5.1f} pts  "
            f"value: {p['value']:.1f}"
        )

    print(f"\n{'=' * 96}")


if __name__ == "__main__":
    main()
