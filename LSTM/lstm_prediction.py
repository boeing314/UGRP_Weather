import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pickle
import os
from conversion_function import year_conv, time_conv, compute_humidity_ratio

# --- Configuration ---
DATA_FILE = 'data_new.csv'
MODEL_PATH = 'lstm_weather_model_best.pth'
PARAMS_PATH = 'lstm_params_best.pkl'
TEST_RESULTS_PATH = 'lstm_predictions.csv'
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. Load Saved Parameters ---
if not os.path.exists(PARAMS_PATH) or not os.path.exists(MODEL_PATH):
    print("Error: Model or Parameter files not found. Run training first.")
    exit()

print(f"Loading parameters from {PARAMS_PATH}...")
with open(PARAMS_PATH, 'rb') as f:
    params = pickle.load(f)

# Extract parameters
seq_length = params['seq_length']
input_dim = params['input_dim']
output_dim = params['output_dim']
hidden_size = params['hidden_size']
num_layers = params['num_layers']
dropout = params['dropout']
feature_columns = params['feature_columns']
target_columns = params['target_columns']
mean_val = params['mean_val']
std_val = params['std_val']
target_indices = [feature_columns.index(col) for col in target_columns]

# --- 2. Define Classes  ---

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, target_indices):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length
        self.target_indices = target_indices

    def __len__(self):
        if self.data.shape[0] <= self.seq_length:
            return 0
        return self.data.shape[0] - self.seq_length

    def __getitem__(self, index):
        x = self.data[index : index + self.seq_length]
        y = self.data[index + self.seq_length, self.target_indices]
        return x, y

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob if layer_dim > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :]) 
        out = self.fc(out)
        return out

# --- 3. Data Loading and Preprocessing ---
print("Loading and preprocessing data...")
df1 = pd.read_csv(DATA_FILE)

# Feature Engineering
df1['date'] = df1['date'].apply(year_conv)
df1['time'] = df1['time'].apply(time_conv)
df1['hr'] = compute_humidity_ratio(df1['d2m'], df1.get('sp', 101325))

# Scaling (Standardization) using loaded mean/std
numerical_features_for_scaling = ["d2m", "t2m", "DNI", "DHI", "GHI", "hr"]
for hd in numerical_features_for_scaling:
    df1[hd] = (df1[hd] - mean_val[hd]) / std_val[hd]

# Cyclic Time Features
df1['year_sin'] = np.sin(df1['date'])
df1['year_cos'] = np.cos(df1['date'])
df1['day_sin'] = np.sin(df1['time'])
df1['day_cos'] = np.cos(df1['time'])

raw_data = df1[feature_columns].values

# --- 4. Prepare Test Dataset ---
full_dataset = TimeSeriesDataset(raw_data, seq_length, target_indices)
dataset_size = len(full_dataset)
indices = list(range(dataset_size))

# Calculate the split index exactly as done in training
split_index = int(np.floor(0.85 * dataset_size))

# Select the last 15%
test_indices = indices[split_index:]
print(f"Total samples: {dataset_size}")
print(f"Test samples (last 15%): {len(test_indices)}")

test_subset = Subset(full_dataset, test_indices)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

# --- 5. Load Model ---
print(f"Loading model weights from {MODEL_PATH}...")
model = LSTMModel(input_dim, hidden_size, num_layers, output_dim, dropout).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- 6. Prediction Loop ---
print("Generating predictions on test set...")
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.numpy())

# Concatenate batches
if len(all_preds) > 0:
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
else:
    print("Error: No predictions generated. Check dataset size.")
    exit()

# --- 7. Inverse Scaling (Un-normalize) ---
# We need to convert the scaled values back to real units (Kelvin, Watts, etc.)
print("Inverse scaling results...")

# Create empty dictionaries to hold the columns
results_data = {}

for i, col_name in enumerate(target_columns):
    # Retrieve scaling params for this specific target
    mu = mean_val[col_name]
    sigma = std_val[col_name]
    
    # Inverse transform: X_real = X_scaled * std + mean
    real_preds = (all_preds[:, i] * sigma) + mu
    real_targets = (all_targets[:, i] * sigma) + mu
    
    results_data[f"Actual_{col_name}"] = real_targets
    results_data[f"Predicted_{col_name}"] = real_preds

# --- 8. Save to CSV ---
results_df = pd.DataFrame(results_data)

# Optional: Calculate metrics for the user to see immediately
mse_loss = np.mean((all_preds - all_targets)**2) # MSE on scaled data
print(f"Test Set MSE (Scaled): {mse_loss:.6f}")

results_df.to_csv(TEST_RESULTS_PATH, index=False)
print(f"Successfully saved test predictions to '{TEST_RESULTS_PATH}'.")

print(results_df.head())
