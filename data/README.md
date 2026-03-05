# Data

This directory stores all scraped NRL data and provides shared utilities for loading and converting it.

## Files

### `loader.py`
Shared data loading and feature engineering module used by prediction models and converters.

**Key functions:**
- `load_match_data(source, years)` — Load match JSON into a pandas DataFrame with one row per round. Supports both a single combined JSON file and the per-year directory structure (`data/NRL/{year}/NRL_data_{year}.json`).
- `get_game_history(df, year, round_, team)` — Compute rolling statistics for a team's recent games
- `build_training_data(df, game_history=3)` — Build X/y arrays for model training

**Usage:**
```python
from data.loader import TEAMS, load_match_data, build_training_data

# Per-year directory structure (preferred)
df = load_match_data('data', [2022, 2023])

# Legacy single-file format
df = load_match_data('data/nrl_data_multi_years_2.json', [2022, 2023])

X, y = build_training_data(df)
```

### `converter.py`
Converts JSON data to TXT/CSV formats. Replaces the legacy Jupyter notebooks in `/converters/`.

**Usage:**
```bash
python data/converter.py --type all --years 2023
python data/converter.py --type match --years 2022 2023
python data/converter.py --type player --years 2023
```

**Output:**
- Match data → `data/txt/match_data_{year}.txt` and `.csv`
- Player data → `data/txt/players/txt/{player}.txt` and `data/txt/players/csv/{player}.csv`

## Directory Structure
```
data/
├── loader.py              # Shared data loading module
├── converter.py           # JSON → TXT/CSV converter
├── txt/                   # Converted output files
│   └── players/
│       ├── txt/
│       └── csv/
├── NRL/                   # Scraped NRL data (gitignored)
│   ├── 2024/
│   │   ├── NRL_data_2024.json
│   │   ├── NRL_detailed_match_data_2024.json
│   │   └── NRL_player_statistics_2024.json
│   └── ...
├── NRLW/
├── HOSTPLUS/
└── KNOCKON/
```

> JSON, TXT, and CSV files are gitignored. Download data from [nrlpredictions.net](https://nrlpredictions.net/sport) or use `scraping/downloader.py`.
