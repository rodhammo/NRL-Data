"""
Script to run the data scraper for match and player data.
"""

import os

from match_data_select import match_data_select
from match_data_detailed_select import match_data_detailed_select
from player_data_select import player_data_select

# Define the selection type for the dataset
# Options: 'NRL', 'NRLW', 'HOSTPLUS', 'KNOCKON'
SELECTION_TYPE = 'NRL'

# Define the year and single round to scrape
SELECT_YEAR = 2026
SELECT_ROUND = 3  # The round to scrape

print(f"Starting data collection for Year: {SELECT_YEAR}, Round: {SELECT_ROUND}")

# Define the directory path for storing scraped data
directory_path = f"../data/{SELECTION_TYPE}/{SELECT_YEAR}/"

# Ensure the directory exists; create it if it doesn't
os.makedirs(directory_path, exist_ok=True)

# Call functions to scrape and process match and player data
match_data_select(SELECT_YEAR, SELECT_ROUND, SELECTION_TYPE)            # Basic match data
match_data_detailed_select(SELECT_YEAR, SELECT_ROUND, SELECTION_TYPE)   # Detailed match data
player_data_select(SELECT_YEAR, SELECT_ROUND, SELECTION_TYPE)           # Player statistics

print("Data scraping process completed successfully.")
