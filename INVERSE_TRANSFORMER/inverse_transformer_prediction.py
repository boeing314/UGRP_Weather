import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
import pickle
from conversion_function import year_conv, time_conv, compute_humidity_ratio

# --- Configuration ---
DATA_FILE = 'dataset.csv'
MODEL_DIR = 'itransformer_models'
CONFIG_FILE = os.path.join(MODEL_DIR, 'itransformer_config.pkl')
MODEL_FILE = os.path.join(MODEL_DIR, 'best_itransformer.pth')
OUTPUT_FILE = 'itransformer_final_predictions.csv'

BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. Load Configuration ---
if not os.path.exists(CONFIG_FILE):
    print(f"Error: Configuration file {CONFIG_FILE} not found. Run train_itransformer.py first.")
    exit()

print("Loading model configuration...")
with open(CONFIG_FILE, 'rb') as f:
    config = pickle.load(f)

# Extract parameters from loaded config
SEQ_LENGTH = config['seq_len']
D_MODEL = config['d_model']
NHEAD = config['nhead']
NUM_LAYERS = config['num_layers']
DIM_FEEDFORWARD = config['dim_feedforward']
DROPOUT = config['dropout']
MEAN_VAL = config['mean_val']
STD_VAL = config['std_val']
FEATURE_COLUMNS = config['feature_columns']
TARGET_COLUMNS = config['target_columns']
TARGET_INDICES = config['target_indices']
TRAIN_SPLIT_RATIO = config['train_split_ratio']

print(f"Loaded Params -> Seq Len: {SEQ_LENGTH}, D_Model: {D_MODEL}, Layers: {NUM_LAYERS}")

# --- 2. Model Architecture (Must match training script) ---
class InverseTransformer(nn.Module):
    def __init__(self, seq_len, num_all_features, target_indices, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(InverseTransformer, self).__init__()
        self.target_indices = target_indices
        self.feature_embedding = nn.Linear(seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projector = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x_enc = self.feature_embedding(x)
        enc_out = self.encoder(x_enc)
        target_tokens = enc_out[:, self.target_indices, :]
        out = self.projector(target_tokens)
        return out.squeeze(-1)

# --- 3. Dataset Class ---
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, target_indices):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length
        self.target_indices = target_indices

    def __len__(self):
        return max(0, self.data.shape[0] - self.seq_length)

    def __getitem__(self, index):
        x = self.data[index : index + self.seq_length]
        y = self.data[index + self.seq_length, self.target_indices]
        return x, y

# --- 4. Main Execution ---

# A. Data Loading & Preprocessing
print("Loading and preprocessing data...")
df = pd.read_csv(DATA_FILE)

df['date_processed'] = df['date'].apply(year_conv)
df['time_processed'] = df['time'].apply(time_conv)
df['hr'] = compute_humidity_ratio(df['d2m'], df.get('sp', 101325))

# Scale using the LOADED statistics
df_scaled = df.copy()
numerical_cols = ["d2m", "t2m", "DNI", "DHI", "GHI", "hr"]
for col in numerical_cols:
    df_scaled[col] = (df_scaled[col] - MEAN_VAL[col]) / STD_VAL[col]

df_scaled['year_sin'] = np.sin(df_scaled['date_processed'])
df_scaled['year_cos'] = np.cos(df_scaled['date_processed'])
df_scaled['day_sin'] = np.sin(df_scaled['time_processed'])
df_scaled['day_cos'] = np.cos(df_scaled['time_processed'])

raw_data = df_scaled[FEATURE_COLUMNS].values

# B. Initialize Model & Load Weights
model = InverseTransformer(
    seq_len=SEQ_LENGTH,
    num_all_features=len(FEATURE_COLUMNS),
    target_indices=TARGET_INDICES,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT
).to(device)

print(f"Loading weights from {MODEL_FILE}...")
model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
model.eval()

# C. Prepare Test Data
# Determine the index where training ended
total_rows = len(raw_data)
train_rows = int(total_rows * TRAIN_SPLIT_RATIO)

# We want predictions for rows: train_rows -> end
# Dataset logic: input at index 'i' predicts target at 'i + seq_len'
# To predict target at 'train_rows', we need input starting at 'train_rows - seq_len'
full_dataset = TimeSeriesDataset(raw_data, SEQ_LENGTH, TARGET_INDICES)
start_pred_idx = train_rows - SEQ_LENGTH
end_pred_idx = len(full_dataset) # This goes up to total_rows - seq_len

test_indices = list(range(start_pred_idx, end_pred_idx))
test_loader = DataLoader(Subset(full_dataset, test_indices), batch_size=BATCH_SIZE, shuffle=False)

print(f"Generating predictions for {len(test_indices)} time steps...")

# D. Prediction Loop
all_preds = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        output = model(batch_x)
        all_preds.append(output.cpu().numpy())

if len(all_preds) == 0:
    print("Error: No predictions generated. Check split ratio and data size.")
    exit()

predictions = np.concatenate(all_preds, axis=0)

# E. Inverse Scaling
results_df = pd.DataFrame(predictions, columns=[f"Pred_{c}" for c in TARGET_COLUMNS])

for col in TARGET_COLUMNS:
    pred_col = f"Pred_{col}"
    results_df[pred_col] = (results_df[pred_col] * STD_VAL[col]) + MEAN_VAL[col]

# F. Align with Actuals (Timestamps)
# The predictions correspond to original DataFrame rows from index [train_rows] to end
ref_data = df.iloc[train_rows:].reset_index(drop=True)

# Create output DataFrame
output_df = pd.DataFrame()
output_df['Date'] = ref_data['date']
output_df['Time'] = ref_data['time']

# Add Predictions and Actuals
for col in TARGET_COLUMNS:
    output_df[f"Actual_{col}"] = ref_data[col]
    output_df[f"Pred_{col}"] = results_df[f"Pred_{col}"]

# Save
output_df.to_csv(OUTPUT_FILE, index=False)
print(f"Successfully saved predictions to {OUTPUT_FILE}")
print(output_df.head())
