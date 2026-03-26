"""
NRL Data & SuperCoach Dashboard

A Streamlit web app wrapping the CLI tools:
  - Scrape NRL data
  - Predict round outcomes
  - SuperCoach trade advisor
  - Squad sync

Run with:  streamlit run app.py
"""

import subprocess
import sys
import os
import json

import streamlit as st

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_DIR = os.path.join(ROOT_DIR, "predictions")
SCRAPING_DIR = os.path.join(ROOT_DIR, "scraping")
DATA_DIR = os.path.join(ROOT_DIR, "data")
SQUAD_FILE = os.path.join(DATA_DIR, "my_supercoach_squad.json")

st.set_page_config(page_title="NRL SuperCoach Dashboard", page_icon="🏉", layout="wide")


def detect_current_round():
    """Detect the current (upcoming) round from the match data file.

    Returns (year, round_num) where round_num is the first round where
    ALL games have 0-0 scores (i.e. unplayed). Falls back to (2026, 1)
    if no data file exists or all rounds have been played.
    """
    from datetime import datetime

    year = datetime.now().year
    data_file = os.path.join(DATA_DIR, "NRL", str(year), f"NRL_data_{year}.json")
    if not os.path.exists(data_file):
        return year, 1

    try:
        with open(data_file) as f:
            data = json.load(f)
        rounds = data["NRL"][0][str(year)]
        for rd in rounds:
            for round_num, games in rd.items():
                if all(g["Home_Score"] == 0 and g["Away_Score"] == 0 for g in games):
                    return year, int(round_num)
        # All rounds have scores — return next round after the last one
        last_round = max(int(k) for rd in rounds for k in rd)
        return year, last_round + 1
    except (KeyError, IndexError, json.JSONDecodeError):
        return year, 1


def run_command(cmd, cwd=None):
    """Run a command and stream output to a Streamlit container."""
    output_area = st.empty()
    lines = []
    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd or ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            lines.append(line)
            output_area.code("".join(lines), language="text")
        process.wait()
        if process.returncode != 0:
            st.error(f"Process exited with code {process.returncode}")
        else:
            st.success("Done!")
    except Exception as e:
        st.error(f"Error: {e}")
    return "".join(lines)


def load_squad():
    """Load the saved squad file."""
    if not os.path.exists(SQUAD_FILE):
        return None
    with open(SQUAD_FILE) as f:
        return json.load(f)


# ── Sidebar Navigation ────────────────────────────────────────────────────────

st.sidebar.title("🏉 NRL Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Scrape NRL Data", "Predict Round", "SuperCoach Trades", "Sync Squad"],
)

# ── Home ───────────────────────────────────────────────────────────────────────

if page == "Home":
    st.title("🏉 NRL SuperCoach Dashboard")
    st.markdown(
        """
        Welcome! Use the sidebar to navigate:

        | Action | Description |
        |--------|-------------|
        | **Scrape NRL Data** | Update match & player data from NRL.com |
        | **Predict Round** | Run the neural-net match predictor |
        | **SuperCoach Trades** | Get optimal trade suggestions |
        | **Sync Squad** | Pull your real SuperCoach roster |
        """
    )

    squad_data = load_squad()
    if squad_data:
        squad = squad_data.get("squad", [])
        trades_used = squad_data.get("trades_used", 0)
        total_salary = sum(p["price"] for p in squad)
        starters = [p for p in squad if p.get("starter")]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Squad Size", len(squad))
        col2.metric("Starters", len(starters))
        col3.metric("Salary", f"${total_salary:,}")
        col4.metric("Trades Used", f"{trades_used}/46")
    else:
        st.info("No squad synced yet. Go to **Sync Squad** to get started.")


# ── Scrape NRL Data ────────────────────────────────────────────────────────────

elif page == "Scrape NRL Data":
    st.title("📡 Scrape NRL Data")
    st.markdown("Update match results, detailed stats, and player statistics from NRL.com.")

    detected_year, detected_round = detect_current_round()

    col1, col2 = st.columns(2)
    with col1:
        scrape_year = st.number_input("Year", min_value=2012, max_value=2030, value=detected_year)
    with col2:
        scrape_round = st.number_input("Round", min_value=1, max_value=30, value=detected_round)

    st.caption("Scrapes a single round and merges with existing data.")

    if st.button("🚀 Start Scraping", type="primary"):
        st.info(f"Scraping {scrape_year} Round {scrape_round}...")

        script = f"""
import sys, os
sys.path.insert(0, r"{SCRAPING_DIR}")
os.chdir(r"{SCRAPING_DIR}")
from match_data_select import match_data_select
from match_data_detailed_select import match_data_detailed_select
from player_data_select import player_data_select
year = {scrape_year}
round_num = {scrape_round}
directory_path = r"{os.path.join(DATA_DIR, 'NRL', str(scrape_year))}"
os.makedirs(directory_path, exist_ok=True)
print(f"Scraping Year: {{year}}, Round: {{round_num}}")
match_data_select(year, round_num, 'NRL')
match_data_detailed_select(year, round_num, 'NRL')
player_data_select(year, round_num, 'NRL')
print("Data scraping process completed successfully.")
"""
        run_command([sys.executable, "-c", script], cwd=SCRAPING_DIR)


# ── Predict Round ──────────────────────────────────────────────────────────────

elif page == "Predict Round":
    st.title("🔮 NRL Round Predictor")
    st.markdown("Predict winners, margins, and first try scorers for the upcoming round.")

    if st.button("🏟️ Run Predictions", type="primary"):
        run_command(
            [sys.executable, os.path.join(PREDICTIONS_DIR, "predict_round.py")],
            cwd=PREDICTIONS_DIR,
        )


# ── SuperCoach Trades ──────────────────────────────────────────────────────────

elif page == "SuperCoach Trades":
    st.title("📊 SuperCoach Trade Advisor")

    squad_data = load_squad()
    if not squad_data:
        st.warning("No squad found. Sync your squad first, or run without trades to build an initial squad.")

    _, detected_sc_round = detect_current_round()

    col1, col2 = st.columns(2)
    with col1:
        sc_round = st.number_input("Round", min_value=1, max_value=30, value=detected_sc_round)
    with col2:
        strategy = st.selectbox("Strategy", ["points", "growth"])

    trade_mode = st.checkbox("Trade mode (suggest trades for existing squad)", value=True)
    boost = st.checkbox("Trade Boost (+1 extra trade)")

    if st.button("🔍 Preview Trades", type="primary"):
        cmd = [sys.executable, os.path.join(PREDICTIONS_DIR, "supercoach.py")]
        if trade_mode:
            cmd.append("--trades")
        cmd.extend(["--round", str(sc_round), "--strategy", strategy])
        if boost:
            cmd.append("--boost")
        run_command(cmd, cwd=PREDICTIONS_DIR)

    st.divider()
    st.subheader("Confirm Trades")
    st.markdown("⚠️ This will **save** the recommended trades to your squad file.")

    if st.button("✅ Confirm Trades", type="secondary"):
        cmd = [sys.executable, os.path.join(PREDICTIONS_DIR, "supercoach.py")]
        cmd.extend([
            "--trades", "--round", str(sc_round),
            "--strategy", strategy, "--confirm-trades",
        ])
        if boost:
            cmd.append("--boost")
        run_command(cmd, cwd=PREDICTIONS_DIR)


# ── Sync Squad ─────────────────────────────────────────────────────────────────

elif page == "Sync Squad":
    st.title("🔄 Sync SuperCoach Squad")
    st.markdown(
        """
        Pull your actual SuperCoach roster from the API.

        **To get your token:**
        1. Open [supercoach.com.au](https://www.supercoach.com.au) and log in
        2. Press **F12** → **Network** tab → refresh the page
        3. Click any request to `supercoach.com.au/2026/api/...`
        4. In the Headers, copy the value after `Authorization: Bearer `
        """
    )

    token = st.text_input("Bearer Token", type="password", placeholder="Paste your token here")

    if st.button("🔄 Sync Squad", type="primary"):
        if not token.strip():
            st.warning("Please enter a token.")
        else:
            cmd = [
                sys.executable, os.path.join(PREDICTIONS_DIR, "supercoach.py"),
                "--sync-squad", "--token", token.strip(),
            ]
            run_command(cmd, cwd=PREDICTIONS_DIR)

    # Show current squad
    squad_data = load_squad()
    if squad_data:
        st.divider()
        st.subheader("Current Squad")
        squad = squad_data.get("squad", [])
        trades_used = squad_data.get("trades_used", 0)
        st.caption(f"Trades used: {trades_used}/46")

        starters = sorted(
            [p for p in squad if p.get("starter")],
            key=lambda p: p.get("assigned_position", ""),
        )
        bench = sorted(
            [p for p in squad if not p.get("starter")],
            key=lambda p: p.get("assigned_position", ""),
        )

        st.markdown("**Starters**")
        for p in starters:
            pos = p.get("assigned_position", p["positions"][0])
            st.text(f"  {p['name']:<28s} {p['team']:<15s} {pos:<5s} ${p['price']:>9,d}")

        st.markdown("**Bench**")
        for p in bench:
            pos = p.get("assigned_position", p["positions"][0])
            st.text(f"  {p['name']:<28s} {p['team']:<15s} {pos:<5s} ${p['price']:>9,d}")
