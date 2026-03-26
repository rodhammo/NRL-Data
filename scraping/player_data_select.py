"""
Optimized Web Scraper for NRL Player Statistics (Fast Execution, Saves Per Round, Includes Year)
"""

from bs4 import BeautifulSoup
import json
import sys
import os
from utilities.set_up_driver import set_up_driver

sys.path.append("..")
import ENVIRONMENT_VARIABLES as EV


def player_data_select(SELECT_YEAR, round_num, SELECTION_TYPE):
    # ============================================
    # ============================================
    # Do not edit below (unless modifying code)
    # ============================================
    # ============================================

    selection_mapping = {
        'NRLW': (EV.NRLW_TEAMS, EV.NRLW_WEBSITE),
        'KNOCKON': (EV.KNOCKON_TEAMS, EV.KNOCKON_WEBSITE),
        'HOSTPLUS': (EV.HOSTPLUS_TEAMS, EV.HOSTPLUS_WEBSITE)
    }

    WEBSITE = EV.NRL_WEBSITE
    # Team name selecter
    TEAMS = EV.TEAMS
    TEAMS, WEBSITE = selection_mapping.get(SELECTION_TYPE, (TEAMS, WEBSITE))


    # List of variables for data extraction
    variables = ["Year", "Win", "Versus", "Round"]

    # Define file path for player statistics
    player_stats_file = f"../data/{SELECTION_TYPE}/{SELECT_YEAR}/{SELECTION_TYPE}_player_statistics_{SELECT_YEAR}.json"

    # Load existing player stats if file exists
    player_stats = None
    if os.path.exists(player_stats_file):
        try:
            with open(player_stats_file, "r") as file:
                player_stats = json.load(file)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading existing player stats, starting fresh: {e}")

    if player_stats is None:
        player_stats = {"PlayerStats": [{str(SELECT_YEAR): []}]}

    # Load NRL match data
    with open(f"../data/{SELECTION_TYPE}/{SELECT_YEAR}/{SELECTION_TYPE}_data_{SELECT_YEAR}.json", "r") as file:
        data = json.load(file)
        data = data[f"{SELECTION_TYPE}"]

    # Store match data for the selected year
    years_arr = {SELECT_YEAR: data[0][str(SELECT_YEAR)]}

    # Find the round data by key from match data
    round_key = str(round_num)
    round_data_from_match = None
    for rd in years_arr[SELECT_YEAR]:
        if round_key in rd:
            round_data_from_match = rd[round_key]
            break

    if round_data_from_match is None:
        print(f"Round {round_num} not found in match data.")
        return

    # Convert round_num (1-indexed) to 0-indexed key for player stats
    round_index_key = str(round_num - 1)

    # **Start WebDriver once and reuse it**
    driver = set_up_driver()

    print(f"Fetching player statistics for {SELECTION_TYPE} {SELECT_YEAR}, Round {round_num}...")

    try:
        round_results = []

        for game in round_data_from_match:
            h_team, a_team = [game[x].replace(" ", "-") for x in ["Home", "Away"]]
            match_key = f"{SELECT_YEAR}-{round_num}-{h_team}-v-{a_team}"

            url = f"{WEBSITE}{SELECT_YEAR}/round-{round_num}/{h_team}-v-{a_team}/"
            print(f"Fetching: {url}")

            # Use existing WebDriver (runs headless for speed)
            driver.get(url)
            soup = BeautifulSoup(driver.page_source, "html.parser")

            # Extract player data
            rows = soup.find_all("tr", class_="table-tbody__tr")
            players_info = []

            for row in rows:
                player_info = {}
                player_name_elem = row.find("a", class_="table__content-link")

                if player_name_elem:
                    player_info["Name"] = player_name_elem.get_text(strip=True, separator=" ")

                statistics = row.find_all("td", class_="table__cell table-tbody__td")

                for i, label in enumerate(EV.PLAYER_LABELS):
                    player_info[label] = statistics[i].get_text(strip=True) if i < len(statistics) else "na"

                players_info.append(player_info)

            # Store match data for this round
            round_results.append({match_key: players_info})
            print(f"Processed match: {match_key}")

        new_round_entry = {round_index_key: round_results}

        # Merge into existing data
        year_index = 0
        year_rounds = player_stats["PlayerStats"][year_index][str(SELECT_YEAR)]

        # Check if round already exists, replace if so
        replaced = False
        for i, rd in enumerate(year_rounds):
            if round_index_key in rd:
                year_rounds[i] = new_round_entry
                replaced = True
                break
        if not replaced:
            year_rounds.append(new_round_entry)

        # Sort rounds by numeric key
        year_rounds.sort(key=lambda rd: int(list(rd.keys())[0]))

        # **Write to file**
        with open(player_stats_file, "w") as file:
            json.dump(player_stats, file, indent=4)

        print(f"Round {round_num} data saved.")

    except Exception as ex:
        print(f"Error: {ex}")

    # **Close WebDriver after all matches are processed**
    driver.quit()

    print(f"Final player statistics saved to {player_stats_file}")
