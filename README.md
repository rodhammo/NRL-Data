# NRL Machine Learning Models, Data Analytics and Data Scraper

⚠️ This library is still a Work-In-Progress. Feel free to help out by adding to the repository. ⚠️

## Description
This project is a web scraper for NRL data, and provides PyTorch machine learning models for NRL related predictions.

## How to Use

### Installation
```bash
pip install -r requirements.txt
```

### Scraping Data
See [Scraping](./scraping/README.md) for information on how to download the data or run the web scraper.

### Running Predictions
The easiest way to run predictions is with `predict_round.py`, which auto-detects the next unplayed round:

```bash
python predictions/predict_round.py
```

This trains a PyTorch neural network on historical data, fetches live team lists from the NRL website, and predicts winner/margin/first-try-scorer for each match.

Jupyter notebooks in `/predictions/` are also available for exploratory analysis.

## Project Structure

```
NRL-Data/
├── scraping/          # Web scrapers and data downloader
├── data/
│   ├── loader.py      # Shared data loading and feature engineering
│   ├── converter.py   # Convert JSON data to TXT/CSV formats
│   └── {JSON files}   # Scraped data (gitignored)
├── predictions/       # PyTorch ML models (predict_round.py + notebooks)
├── visualisations/    # Interactive Plotly charts (Jupyter notebooks)
├── converters/        # Legacy converter notebooks (deprecated, use data/converter.py)
├── reports/           # Ad-hoc analytics using NRL data
├── ENVIRONMENT_VARIABLES.py  # Teams, URLs, player labels, team colours
└── requirements.txt   # Python dependencies
```

## Data
All data for this project is hosted on [this website](https://nrlpredictions.net/sport).
I personally host this website with all data being stored in a S3 instance.

The following data is available:
* NRL : 2001 - 2026
* NRLW: 2022 - 2024
* KNOCKON: 2022 - 2024
* HOSTPLUS: 2022 - 2024

### 📂 Available Data Files

| Data Type                        | Description                                                                              |
|----------------------------------|------------------------------------------------------------------------------------------|
| **📊 Detailed Match Data**       | In-depth statistics for each match, including team performance metrics and match events.  |
| **📊 General Match Data**        | Match data for every game from the selected years.                                       |
| **👤 Player Statistics**         | Individual player performance data, including tries, tackles, run meters, and more.      |

> **Note:** Player data requires match data to be retrieved first.

You can view player data on the website by selecting a year, type, and Player Statistics.

### Match Data
<details>
<summary>Match Data JSON Schema</summary>

```json
{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "object",
  "properties": {
    "NRL": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "2024": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "1": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "properties": {
                      "Details": { "type": "string" },
                      "Date": { "type": "string" },
                      "Home": { "type": "string" },
                      "Home_Score": { "type": "string" },
                      "Away": { "type": "string" },
                      "Away_Score": { "type": "string" },
                      "Venue": { "type": "string" }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
```

</details>

## Web Scraping
This project utilises Selenium and Requests for web scraping NRL data from the NRL website. This code is located in `/scraping/`.

## Machine Learning
This code is located in `/predictions/`. Models use **PyTorch** for neural networks.

The primary model is `predict_round.py` — a 2-layer neural network (128→64→2) that uses 49 features including rolling 5-game averages, head-to-head records, and stat differentials to predict match winners and margins. It uses early stopping, probability clamping (15–85%), and regularisation (dropout + weight decay) to produce realistic confidence levels. First try scorer predictions are filtered to players named in the weekly squad (fetched live from the NRL website).

Legacy notebooks are also available:
* **Match based** (`model_1.ipynb`): Earlier match prediction model.
* **Player and Match based** (`model_1_players WIP.ipynb`): Player-based prediction (WIP).
* **Anytime Try Scorer** (`antyime_try_scorer_model.ipynb`): Try scorer probability model (project root).

## Data Conversion
JSON is the default format for all scraped data. To convert to TXT/CSV:

```bash
python data/converter.py --type all --years 2023
```

See [Data](./data/README.md) for more details.

## Visualisations
Interactive charts and data viewers are located in `/visualisations/`.

## Future Tasks
* ~~Update the machine learning model to work with current data~~ ✅
* ~~Anytime Try Scorer Probability model~~ ✅
* ~~Integrate team lists into predictions~~ ✅
* Update the website to display prediction results
* Optimise the current machine learning model
* NRLW prediction models
* Try Location Data

## Help
I intend for this project to be open source, so help is always handy!
