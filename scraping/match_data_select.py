"""
This script fetches NRL  match data for the selected year and saves it to a JSON file
"""

# Imports
from utilities.get_nrl_data import get_nrl_data
import json
import sys
sys.path.append('..')
import ENVIRONMENT_VARIABLES as EV
import os


def match_data_select(SELECT_YEAR, round_num, SELECTION_TYPE):
    """
    Fetches NRL match data for a single round and merges it into the existing JSON file.
    """
    try:
        COMPETITION_TYPE = EV.COMPETITION[SELECTION_TYPE]
    except (TypeError, KeyError):
        print(f"Unknown Competition Type: {SELECTION_TYPE}")
        return

    print(f"Fetching data for {SELECTION_TYPE} {SELECT_YEAR}, Round {round_num}...")

    directory_path = os.path.abspath(f"../data/{SELECTION_TYPE}/{SELECT_YEAR}/")
    os.makedirs(directory_path, exist_ok=True)
    file_path = os.path.join(directory_path, f"{SELECTION_TYPE}_data_{SELECT_YEAR}.json")

    # Load existing data if file exists
    existing_data = None
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                existing_data = json.load(file)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading existing file, starting fresh: {e}")

    # Fetch the single round
    try:
        match_json = get_nrl_data(round_num, SELECT_YEAR, COMPETITION_TYPE)
    except Exception as ex:
        print(f"Error fetching round {round_num}: {ex}")
        return

    # Merge into existing data
    if existing_data and SELECTION_TYPE in existing_data:
        year_entry = None
        for entry in existing_data[SELECTION_TYPE]:
            if str(SELECT_YEAR) in entry:
                year_entry = entry
                break

        if year_entry:
            rounds_list = year_entry[str(SELECT_YEAR)]
            # Check if round already exists, replace it if so
            round_key = str(round_num)
            replaced = False
            for i, rd in enumerate(rounds_list):
                if round_key in rd:
                    rounds_list[i] = match_json
                    replaced = True
                    break
            if not replaced:
                rounds_list.append(match_json)
            # Sort rounds by numeric key
            rounds_list.sort(key=lambda rd: int(list(rd.keys())[0]))
        else:
            existing_data[SELECTION_TYPE].append({str(SELECT_YEAR): [match_json]})

        overall_data = existing_data
    else:
        overall_data = {SELECTION_TYPE: [{str(SELECT_YEAR): [match_json]}]}

    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(overall_data, file, ensure_ascii=False, separators=(',', ':'))
        print(f"Saved match data to: {file_path}")
    except Exception as e:
        print(f"Error writing file: {e}")
