import json
import pandas as pd
import numpy as np
from utilities.get_detailed_match_data import get_detailed_nrl_data
from utilities.set_up_driver import set_up_driver
import sys

sys.path.append("..")
import ENVIRONMENT_VARIABLES as EV


def match_data_detailed_select(SELECT_YEAR, round_num, SELECTION_TYPE):

    VARIABLES = ["Year", "Win", "Defense", "Attack", "Margin", "Home", "Versus", "Round"]
    JSON_FILE_PATH = f"../data/{SELECTION_TYPE}/{SELECT_YEAR}/{SELECTION_TYPE}_data_{SELECT_YEAR}.json"
    OUTPUT_FILE_PATH = f"../data/{SELECTION_TYPE}/{SELECT_YEAR}/{SELECTION_TYPE}_detailed_match_data_{SELECT_YEAR}.json"


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


    # Load NRL match data
    try:
        with open(JSON_FILE_PATH, "r") as file:
            data = json.load(file)[f"{SELECTION_TYPE}"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"Error loading JSON data: {e}")
        sys.exit(1)

    # Extract data for the selected year
    try:
        years_arr = {SELECT_YEAR: data[0][str(SELECT_YEAR)]}
    except IndexError as e:
        print(f"Error accessing year data: {e}")
        sys.exit(1)

    # Find the round data by key from match data
    round_key = str(round_num)
    round_data_from_match = None
    for rd in years_arr[SELECT_YEAR]:
        if round_key in rd:
            round_data_from_match = rd[round_key]
            break

    if round_data_from_match is None:
        print(f"Round {round_num} not found in match data.")
        sys.exit(1)

    # Create DataFrame with appropriate columns
    df = pd.DataFrame(columns=[f"{team} {variable}" for team in TEAMS for variable in VARIABLES])


    # ** Function to Fetch Data for a Single Match (Using Persistent WebDriver) **
    def fetch_match_data(driver, game, rnd):
        h_team, a_team = game["Home"], game["Away"]

        # Try fetching data twice before failing
        game_data = None
        for attempt in range(2):
            try:
                game_data = get_detailed_nrl_data(
                    round=rnd, year=SELECT_YEAR,
                    home_team=h_team.lower(), away_team=a_team.lower(),
                    driver=driver, nrl_website=WEBSITE
                )
                if "match" in game_data:
                    return {f"{h_team} v {a_team}": game_data}  
            except Exception as ex:
                print(f"Attempt {attempt + 1} failed for {h_team} vs {a_team}: {ex}")

        return None 


    # Load existing output data if it exists
    existing_data = None
    import os
    if os.path.exists(OUTPUT_FILE_PATH):
        try:
            with open(OUTPUT_FILE_PATH, "r") as file:
                existing_data = json.load(file)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading existing detailed data, starting fresh: {e}")

    # ** Keep Selenium WebDriver Open **
    driver = set_up_driver()

    print(f"Fetching detailed match data for {SELECTION_TYPE} {SELECT_YEAR}, Round {round_num}...")

    try:
        round_data_scores = []

        for game in round_data_from_match:
            match_data = fetch_match_data(driver, game, round_num)
            if match_data:
                round_data_scores.append(match_data)

        new_round_entry = {round_num: round_data_scores}

        # Merge into existing data
        if existing_data and SELECTION_TYPE in existing_data:
            match_json_datas = existing_data[SELECTION_TYPE]
            # Check if round already exists, replace if so
            replaced = False
            for i, rd in enumerate(match_json_datas):
                if round_num in rd:
                    match_json_datas[i] = new_round_entry
                    replaced = True
                    break
            if not replaced:
                match_json_datas.append(new_round_entry)
            # Sort rounds by numeric key
            match_json_datas.sort(key=lambda rd: int(list(rd.keys())[0]))
        else:
            existing_data = {SELECTION_TYPE: [new_round_entry]}

        with open(OUTPUT_FILE_PATH, "w") as file:
            json.dump(existing_data, file, indent=4)
        print(f"Round {round_num} data saved.")

    except Exception as ex:
        print(f"Error processing round {round_num}: {ex}")

    # ** Close WebDriver after processing **
    driver.quit()
    print(f"Final detailed match data saved to {OUTPUT_FILE_PATH}")
