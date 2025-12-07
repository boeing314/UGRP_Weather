import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import pickle
from conversion_function import year_conv, time_conv, compute_humidity_ratio
from nsag_functions import merge_file, reduce_size, new_child, update_file

# --- Configuration ---
DATA_FILE = 'dataset.csv'
ENSEMBLE_DIR = 'ensemble_models_trained'
WEIGHTS_SAVE_PATH = 'ensemble_best_weights.pkl'
os.makedirs(ENSEMBLE_DIR, exist_ok=True)

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS_PER_MODEL = 50 

# The 5 Best Configurations
ENSEMBLE_CONFIGS = [
    {'id': 1, 'hidden_size': 62, 'num_layers': 1, 'dropout': 0.03735, 'seq_length': 75},
    {'id': 2, 'hidden_size': 39, 'num_layers': 1, 'dropout': 0.03088, 'seq_length': 11},
    {'id': 3, 'hidden_size': 41, 'num_layers': 1, 'dropout': 0.00995, 'seq_length': 11},
    {'id': 4, 'hidden_size': 54, 'num_layers': 1, 'dropout': 0.02046, 'seq_length': 51},
    {'id': 5, 'hidden_size': 54, 'num_layers': 1, 'dropout': 0.03160, 'seq_length': 31},
]

# Features
numerical_features_for_scaling = ["d2m", "t2m", "DNI", "DHI", "GHI", "hr"]
feature_columns = ['year_sin','year_cos','day_sin','day_cos',"t2m","d2m","DNI","DHI","GHI","hr"]
target_columns = ["t2m", "DNI", "DHI", "GHI"]

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. Data Loading & Preprocessing ---
print("\n[1/4] Loading Data...")
df1 = pd.read_csv(DATA_FILE)

# Stats
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
target_indices = [feature_columns.index(col) for col in target_columns]
input_size = len(feature_columns)
output_dim = len(target_columns)

# Split: 85% Train/Val (Used for training and weighting), 15% Test (Reserved for prediction script)
total_len = len(raw_data)
split_idx = int(total_len * 0.85)

# --- Classes ---
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

# --- 2. Train Base Models ---
print(f"\n[2/4] Training {len(ENSEMBLE_CONFIGS)} Base Models on first 85% of data...")

for config in ENSEMBLE_CONFIGS:
    model_id = config['id']
    seq_len = config['seq_length']
    save_path = os.path.join(ENSEMBLE_DIR, f"model_{model_id}.pth")
    
    print(f"  > Training Model {model_id}...")
    
    dataset = TimeSeriesDataset(raw_data, seq_len, target_indices)
    # Train only on the 85% split
    train_indices = list(range(split_idx - seq_len)) 
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True)
    
    model = LSTMModel(input_size, config['hidden_size'], config['num_layers'], output_dim, config['dropout']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(EPOCHS_PER_MODEL):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    torch.save(model.state_dict(), save_path)
    print(f"    Saved to {save_path}")

# --- 3. Generate Predictions for Optimization ---
print(f"\n[3/4] Generating predictions for Optuna Weighting...")
# We generate predictions on the *validation* portion of the 85% split
# to determine which model to trust more.
max_seq_len = max(c['seq_length'] for c in ENSEMBLE_CONFIGS)
validation_target_indices = np.arange(max_seq_len, split_idx)
gt_values = raw_data[validation_target_indices][:, target_indices]

model_predictions_stack = []

for config in ENSEMBLE_CONFIGS:
    model_id = config['id']
    seq_len = config['seq_length']
    model_path = os.path.join(ENSEMBLE_DIR, f"model_{model_id}.pth")
    
    model = LSTMModel(input_size, config['hidden_size'], config['num_layers'], output_dim, config['dropout']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    preds = []
    with torch.no_grad():
        batch_inputs = []
        for target_idx in validation_target_indices:
            input_slice = raw_data[target_idx - seq_len : target_idx]
            batch_inputs.append(input_slice)
            
            if len(batch_inputs) == 1024 or target_idx == validation_target_indices[-1]:
                bx = torch.tensor(np.array(batch_inputs), dtype=torch.float32).to(device)
                out = model(bx)
                preds.append(out.cpu().numpy())
                batch_inputs = []
    model_predictions_stack.append(np.concatenate(preds, axis=0))

model_predictions_stack = np.array(model_predictions_stack) # [Models, Samples, Targets]

# --- 4. Optuna Optimization ---
print(f"\n[4/4] Optimizing Ensemble Weights...")

def objective(trial):
    weights = []
    for i in range(len(ENSEMBLE_CONFIGS)):
        weights.append(trial.suggest_float(f'w_{i}', 0.0, 1.0))
    
    weights = np.array(weights)
    if np.sum(weights) == 0: return 1e9
    weights /= np.sum(weights)
    
    tscv = TimeSeriesSplit(n_splits=5)
    mse_scores = []
    
    for train_idx, val_idx in tscv.split(gt_values):
        fold_preds = model_predictions_stack[:, val_idx, :]
        fold_gt = gt_values[val_idx]
        ensemble_pred = np.average(fold_preds, axis=0, weights=weights)
        mse_scores.append(mean_squared_error(fold_gt, ensemble_pred))
        
    return np.mean(mse_scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
raw_weights = [best_params[f'w_{i}'] for i in range(len(ENSEMBLE_CONFIGS))]
total = sum(raw_weights)
final_weights = [w/total for w in raw_weights]

print("\nBest Weights:")
for i, w in enumerate(final_weights):
    print(f"  Model {ENSEMBLE_CONFIGS[i]['id']}: {w:.4f}")

with open(WEIGHTS_SAVE_PATH, 'wb') as f:
    pickle.dump(final_weights, f)
print(f"Weights saved to {WEIGHTS_SAVE_PATH}")
