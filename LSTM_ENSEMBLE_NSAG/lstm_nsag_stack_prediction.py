import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from sklearn.metrics import mean_squared_error
from conversion_function import year_conv, time_conv, compute_humidity_ratio
from nsag_functions import merge_file, reduce_size, new_child, update_file

# --- Configuration ---
DATA_FILE = 'dataset.csv'
ENSEMBLE_DIR = 'ensemble_models_trained'
WEIGHTS_SAVE_PATH = 'ensemble_best_weights.pkl'
FINAL_PREDS_PATH = 'lstm_nsag_predictions.csv'

# Same configs as training
ENSEMBLE_CONFIGS = [
    {'id': 1, 'hidden_size': 62, 'num_layers': 1, 'dropout': 0.03735, 'seq_length': 75},
    {'id': 2, 'hidden_size': 39, 'num_layers': 1, 'dropout': 0.03088, 'seq_length': 11},
    {'id': 3, 'hidden_size': 41, 'num_layers': 1, 'dropout': 0.00995, 'seq_length': 11},
    {'id': 4, 'hidden_size': 54, 'num_layers': 1, 'dropout': 0.02046, 'seq_length': 51},
    {'id': 5, 'hidden_size': 54, 'num_layers': 1, 'dropout': 0.03160, 'seq_length': 31},
]

numerical_features_for_scaling = ["d2m", "t2m", "DNI", "DHI", "GHI", "hr"]
feature_columns = ['year_sin','year_cos','day_sin','day_cos',"t2m","d2m","DNI","DHI","GHI","hr"]
target_columns = ["t2m", "DNI", "DHI", "GHI"]

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. Load Data ---
print("Loading Data...")
df1 = pd.read_csv(DATA_FILE)

# Scaling Stats
mean_val={"d2m":296.72937075746887,"t2m":301.567039019018,"DNI":210.59884837107128,"DHI":74.42870504330628,"GHI":219.5148092104472,"hr":0.01852766471396759}
std_val={"d2m":2.14331552045601950,"t2m":2.7471734820877556,"DNI":276.0952387638112,"DHI":94.91021039014818,"GHI":297.0297968459997,"hr":0.0023766051788255255}

df1['date'] = df1['date'].apply(year_conv)
df1['time'] = df1['time'].apply(time_conv)
df1['hr'] = compute_humidity_ratio(df1['d2m'], df1.get('sp', 101325))

for hd in numerical_features_for_scaling:
    df1[hd] = (df1[hd] - mean_val[hd]) / std_val[hd]

df1['year_sin'] = np.sin(df1['date'])
df1['year_cos'] = np.cos(df1['date'])
df1['day_sin'] = np.sin(df1['time'])
df1['day_cos'] = np.cos(df1['time'])

raw_data = df1[feature_columns].values
input_size = len(feature_columns)
output_dim = len(target_columns)
target_indices = [feature_columns.index(col) for col in target_columns]

# --- 2. Load Weights ---
if not os.path.exists(WEIGHTS_SAVE_PATH):
    print("Error: Weights file not found. Run training script first.")
    exit()

with open(WEIGHTS_SAVE_PATH, 'rb') as f:
    final_weights = pickle.load(f)
print("Ensemble weights loaded.")

# --- 3. Model Class (Must match) ---
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

# --- 4. Predict on Test Set ---
print("Generating Predictions on Test Set (Last 15%)...")

total_len = len(raw_data)
split_idx = int(total_len * 0.85)
test_target_indices = np.arange(split_idx, total_len)
gt_test = raw_data[test_target_indices][:, target_indices]

test_preds_stack = []

for config in ENSEMBLE_CONFIGS:
    model_id = config['id']
    seq_len = config['seq_length']
    model_path = os.path.join(ENSEMBLE_DIR, f"model_{model_id}.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Model {model_id} file not found.")
        exit()

    model = LSTMModel(input_size, config['hidden_size'], config['num_layers'], output_dim, config['dropout']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    preds = []
    with torch.no_grad():
        batch_inputs = []
        for target_idx in test_target_indices:
            # Input slice for this specific target time
            input_slice = raw_data[target_idx - seq_len : target_idx]
            batch_inputs.append(input_slice)
            
            if len(batch_inputs) == 1024 or target_idx == test_target_indices[-1]:
                bx = torch.tensor(np.array(batch_inputs), dtype=torch.float32).to(device)
                out = model(bx)
                preds.append(out.cpu().numpy())
                batch_inputs = []
    test_preds_stack.append(np.concatenate(preds, axis=0))

# Convert to array: [Models, Samples, Targets]
test_preds_stack = np.array(test_preds_stack)

# Apply Weights
final_preds_scaled = np.average(test_preds_stack, axis=0, weights=final_weights)

# MSE
mse_scaled = mean_squared_error(gt_test, final_preds_scaled)
print(f"Ensemble Test MSE (Scaled): {mse_scaled:.6f}")

# --- 5. Inverse Scale & Save ---
print("Inverse scaling and saving CSV...")
results_df = pd.DataFrame()

# Re-read raw DF to get timestamps easily for the test set
df_raw = pd.read_csv(DATA_FILE)
test_rows = df_raw.iloc[split_idx:].reset_index(drop=True)

results_df['Date'] = test_rows['date']
results_df['Time'] = test_rows['time']

for i, col in enumerate(target_columns):
    # Unscale
    pred_real = (final_preds_scaled[:, i] * std_val[col]) + mean_val[col]
    act_real = (gt_test[:, i] * std_val[col]) + mean_val[col]
    
    results_df[f'Actual_{col}'] = act_real
    results_df[f'Pred_{col}'] = pred_real

results_df.to_csv(FINAL_PREDS_PATH, index=False)
print(f"Results saved to {FINAL_PREDS_PATH}")
print(results_df.head())
