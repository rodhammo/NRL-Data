# Predictions

NRL machine learning models built with **PyTorch**.

## `predict_round.py` — Main Prediction Script

The primary way to run predictions. Auto-detects the next unplayed round and outputs winner, margin, and first try scorer predictions for each match.

```bash
python predictions/predict_round.py
```

### What it does
1. Scans `data/NRL/{year}/` to find the first round with unplayed matches (scores of 0-0)
2. Loads historical match data, detailed team stats, and player statistics (2018–present)
3. Fetches live team lists from NRL match centre pages via the `q-data` attribute (plain HTTP, no Selenium)
4. Trains a neural network on the historical data
5. Predicts winner, margin, and first try scorer for each upcoming match

### Model Architecture
- **Network:** 2-layer dense NN (128 → 64 → 2) with ReLU, Dropout (0.4/0.3), and weight decay (1e-3)
- **Outputs:** Win logit + predicted margin
- **Training:** Adam optimiser, StepLR scheduler, BCE + MSE loss, early stopping (patience=50), 400 max epochs
- **Probability clamping:** Raw sigmoid output is clamped to 15–85% to reflect realistic NRL uncertainty

### Features (49 per match)
- **Per-team form (×2):** Rolling 5-game win rate, points scored/conceded, point differential
- **Per-team detailed stats (×2):** Completion rate, tackle efficiency, possession, run metres, post-contact metres, line breaks, tackle breaks, set distance, kicking metres, forced drop-outs, tackles made, missed tackles, errors, penalties, play-the-ball speed, offloads
- **Stat differentials (×7):** Completion rate, tackle efficiency, run metres, line breaks, errors, missed tackles, point differential
- **Head-to-head (×2):** Win rate and average margin from recent matchups
- **Home advantage (×1)**

### Team List Integration
Team lists are scraped from the `q-data` JSON attribute on NRL match centre pages. First try scorer predictions are restricted to players named in the weekly squad, with fuzzy name matching between historical data and squad lists.

### First Try Scorer Model
Scores candidates based on career try rate (40%), recent form (30%), historical FTS rate (×3 boost), and a position bonus (wingers and fullbacks weighted higher). Only players with 3+ career games are considered.

## Legacy Notebooks

### `model_1.ipynb` — Match Prediction
Earlier match prediction model using `data.loader` for feature engineering.

- **Features:** Team indices, rolling win rate, attack/defense medians and means, home advantage, margin history
- **Architecture:** Dense neural network (128 → 64 → 32 → 5) with BatchNorm and Dropout

### `model_1_players WIP.ipynb` — Player-Based Prediction (WIP)
Uses individual player statistics to predict match outcomes. Currently a work-in-progress.

## Shared Data Loading
The notebooks import from `data.loader`:

```python
from data.loader import TEAMS, load_match_data, build_training_data

df = load_match_data('../data', [2022, 2023])
X, y = build_training_data(df)
```

`predict_round.py` has its own data loading that reads directly from the per-year directory structure.
