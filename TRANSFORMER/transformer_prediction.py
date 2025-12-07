import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import os
import math

# --- Import helper functions ---
from conversion_function import year_conv, time_conv, compute_humidity_ratio

# --- Configuration ---
DATA_FILE = 'data_new.csv'
MODEL_DIR = 'transformer_single'
OUTPUT_FILE = 'transformer_predictions.csv'
BATCH_SIZE = 64

# PREDICTION SETTING
# 0.15 = Predict only the last 15% (the unseen test set)
# 0.0  = Predict everything possible (from row [seq_len] to the end)
TEST_SPLIT_RATIO = 0.15 

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. Define Model Architecture (Must match training exactly) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, dim_feedforward, dropout, seq_len):
        super(TimeSeriesTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 50)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                    dim_feedforward=dim_feedforward, 
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, src):
        src = self.input_linear(src)
        src = self.relu(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        last_step_output = output[:, -1, :]
        prediction = self.decoder(self.dropout(last_step_output))
        return prediction

# --- 2. Load Config & Model ---

config_path = os.path.join(MODEL_DIR, 'transformer_config.pkl')
weights_path = os.path.join(MODEL_DIR, 'best_transformer.pth')

if not os.path.exists(config_path) or not os.path.exists(weights_path):
    print("Error: Model files not found. Please train the model first.")
    exit()

print("Loading configuration...")
with open(config_path, 'rb') as f:
    config = pickle.load(f)

# Extract params
seq_len = config['seq_len']
mean_val = config['mean_val']
std_val = config['std_val']
feature_cols = config['feature_columns']
target_cols = config['target_columns']

print(f"Model Configuration: Seq_Len={seq_len}, Heads={config['nhead']}, Layers={config['num_layers']}")

# Initialize Model
model = TimeSeriesTransformer(
    input_dim=config['input_dim'],
    output_dim=config['output_dim'],
    d_model=config['d_model'],
    nhead=config['nhead'],
    num_layers=config['num_layers'],
    dim_feedforward=config['dim_feedforward'],
    dropout=config['dropout'],
    seq_len=seq_len
).to(device)

# Load Weights
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# --- 3. Data Processing ---

print("Processing data...")
df1 = pd.read_csv(DATA_FILE)

# Apply Conversions
df1['date_processed'] = df1['date'].apply(year_conv)
df1['time_processed'] = df1['time'].apply(time_conv)
df1['hr'] = compute_humidity_ratio(df1['d2m'], df1.get('sp', 101325))

# Create copy for scaling
df_scaled = df1.copy()
numerical_features = ["d2m", "t2m", "DNI", "DHI", "GHI", "hr"]

# Scale using saved stats
for hd in numerical_features:
    df_scaled[hd] = (df_scaled[hd] - mean_val[hd]) / std_val[hd]

# Feature Engineering
df_scaled['year_sin'] = np.sin(df_scaled['date_processed'])
df_scaled['year_cos'] = np.cos(df_scaled['date_processed'])
df_scaled['day_sin'] = np.sin(df_scaled['time_processed'])
df_scaled['day_cos'] = np.cos(df_scaled['time_processed'])

raw_data = df_scaled[feature_cols].values

# --- 4. Prepare Inference Dataset ---

# Determine where to start predicting
total_len = len(raw_data)
if TEST_SPLIT_RATIO > 0:
    # Example: If 1000 rows, split 0.15, we predict from index 850 to 1000
    start_index = int(total_len * (1 - TEST_SPLIT_RATIO))
    # Ensure start_index is at least seq_len
    if start_index < seq_len:
        start_index = seq_len
else:
    start_index = seq_len

indices_to_predict = list(range(start_index, total_len))
print(f"Predicting {len(indices_to_predict)} samples (from row {start_index} to {total_len})")

class InferenceDataset(Dataset):
    def __init__(self, data, indices, seq_length):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.indices = indices
        self.seq_length = seq_length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # The row we want to predict is `target_row`
        # The input window is [target_row - seq_len : target_row]
        target_row = self.indices[idx]
        start_x = target_row - self.seq_length
        end_x = target_row
        
        return self.data[start_x : end_x]

ds = InferenceDataset(raw_data, indices_to_predict, seq_len)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

# --- 5. Prediction Loop ---

all_preds = []

print("Running inference...")
with torch.no_grad():
    for batch_x in loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        all_preds.append(outputs.cpu().numpy())

# Combine batches
predictions = np.concatenate(all_preds, axis=0)

# --- 6. Inverse Scaling & Saving ---

print("Inverting scaling and saving...")

# Create DataFrame
results_df = pd.DataFrame(predictions, columns=[f"Pred_{c}" for c in target_cols])

# Inverse Transform: Real = (Norm * Std) + Mean
for col in target_cols:
    pred_col = f"Pred_{col}"
    results_df[pred_col] = (results_df[pred_col] * std_val[col]) + mean_val[col]

# Get the corresponding original rows (Date/Time/Actuals)
# We use .iloc with the exact indices we predicted
reference_data = df1.iloc[indices_to_predict].reset_index(drop=True)

# Combine Date/Time, Actual Values, and Predicted Values
final_output = pd.concat([reference_data[['date', 'time']], results_df], axis=1)

# Add Actual columns for comparison
for col in target_cols:
    final_output[f"Actual_{col}"] = reference_data[col]

# Save
final_output.to_csv(OUTPUT_FILE, index=False)
print(f"\nSuccess! Predictions saved to: {OUTPUT_FILE}")
print(final_output.head())