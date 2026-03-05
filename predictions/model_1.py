"""
NRL Feature Map and Machine Learning Model

Trains a PyTorch neural network on historical NRL match data
and predicts match outcomes with win probabilities.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.loader import TEAMS, load_match_data, get_game_history, build_training_data


# ── Load Data ──────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
print("Loading match data...")
df = load_match_data(DATA_DIR, years)
print(f"Loaded {len(df)} rounds across {len(years)} seasons")


# ── Build Feature Map ──────────────────────────────────────────────────────────

GAME_HISTORY = 3
print("Building training data...")
X, y = build_training_data(df, game_history=GAME_HISTORY)
print(f"Training samples: {len(X)}")


# ── Train the Model ────────────────────────────────────────────────────────────

X_train, X_val, y_train, y_val = train_test_split(
    np.array(X), np.array(y), test_size=0.3, shuffle=True
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

input_size = X_train_scaled.shape[1]

model = nn.Sequential(
    nn.Linear(input_size, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.BatchNorm1d(64),
    nn.Dropout(0.2),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 5),
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
criterion = nn.MSELoss()

num_epochs = 1000
train_r2_scores = []
val_r2_scores = []
train_losses = []
previous_loss = None
no_loss_change_epochs = 0
loss_change_threshold = 1e-5

print("\nTraining model...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1

    scheduler.step()
    avg_loss = epoch_loss / num_batches
    train_losses.append(avg_loss)

    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train_t).numpy()
        y_val_pred = model(X_val_t).numpy()

    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    train_r2_scores.append(train_r2)
    val_r2_scores.append(val_r2)

    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f} - Train R²: {train_r2:.4f} - Val R²: {val_r2:.4f}")

    if previous_loss is not None and abs(previous_loss - avg_loss) < loss_change_threshold:
        no_loss_change_epochs += 1
    else:
        no_loss_change_epochs = 0
    previous_loss = avg_loss

    if no_loss_change_epochs >= 5:
        print(f"  Training stopped early at epoch {epoch + 1} due to no significant loss change.")
        break

print(f"\nFinal Training R²:    {train_r2_scores[-1]:.4f}")
print(f"Final Validation R²:  {val_r2_scores[-1]:.4f}")


# ── Visualise Training ─────────────────────────────────────────────────────────

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Training Loss over Epochs")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(train_r2_scores, label="Training R-squared")
plt.plot(val_r2_scores, label="Validation R-squared")
plt.xlabel("Epoch")
plt.ylabel("R-squared Score")
plt.title("R-squared Score over Epochs")
plt.legend()
plt.grid(True)
plt.show()

model.eval()
with torch.no_grad():
    y_train_pred = model(X_train_t).numpy()
    y_val_pred = model(X_val_t).numpy()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].scatter(y_train[:, 0], y_train_pred[:, 0], alpha=0.5)
axs[0, 0].set_title("Target 1")
axs[0, 1].scatter(y_train[:, 1], y_train_pred[:, 1], alpha=0.5)
axs[0, 1].set_title("Target 2")
axs[1, 0].scatter(y_train[:, 2], y_train_pred[:, 2], alpha=0.5)
axs[1, 0].set_title("Target 3")
axs[1, 1].scatter(y_train[:, 3], y_train_pred[:, 3], alpha=0.5)
axs[1, 1].set_title("Target 4")

for ax in axs.flat:
    ax.set(xlabel="True Values", ylabel="Predicted Values")

plt.tight_layout()
plt.show()


# ── Predict Upcoming Matches ──────────────────────────────────────────────────

PREDICT_YEAR = 2025


def get_latest_round(team):
    """Get the last round a team actually played (not a bye) in the prediction year."""
    team_df = df[(df[f"{team} Year"] == PREDICT_YEAR) & (df[f"{team} Win"] != -1)]
    return int(team_df[f"{team} Round"].max())


wkd_matches = [
    ["Broncos", "Rabbitohs"],
    ["Sharks", "Bulldogs"],
    ["Panthers", "Eels"],
    ["Raiders", "Wests Tigers"],
    ["Cowboys", "Knights"],
    ["Storm", "Warriors"],
    ["Sea Eagles", "Roosters"],
    ["Dolphins", "Dragons"],
]

print("\n" + "=" * 80)
print(f"  NRL Match Predictions (based on {PREDICT_YEAR} form)")
print("=" * 80)

model.eval()
for wkd_match in wkd_matches:
    team_1 = TEAMS.index(wkd_match[0])
    team_2 = TEAMS.index(wkd_match[1])

    round_1 = get_latest_round(TEAMS[team_1])
    round_2 = get_latest_round(TEAMS[team_2])

    pred_in = [
        team_1, team_2,
        *get_game_history(df, PREDICT_YEAR, round_1, TEAMS[team_1]),
        *get_game_history(df, PREDICT_YEAR, round_2, TEAMS[team_2]),
    ]

    with torch.no_grad():
        pred_tensor = torch.tensor([pred_in], dtype=torch.float32)
        raw = model(pred_tensor).numpy()[0]

    # Convert raw scores to win probabilities using sigmoid
    prob_1 = 1 / (1 + np.exp(-raw[2]))
    prob_2 = 1 / (1 + np.exp(-raw[3]))
    # Normalise so they sum to 100%
    total = prob_1 + prob_2
    pct_1 = prob_1 / total * 100
    pct_2 = prob_2 / total * 100

    big = "Yes" if raw[4] > 0.5 else "No"
    winner = TEAMS[team_1] if pct_1 > pct_2 else TEAMS[team_2]
    print(f"  {winner} wins\t\t {TEAMS[team_1]}: {pct_1:.1f}%\t{TEAMS[team_2]}: {pct_2:.1f}%\t\tBig Win: {big}")
