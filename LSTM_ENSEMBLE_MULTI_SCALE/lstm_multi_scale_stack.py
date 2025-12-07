import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
import optuna
from conversion_function import year_conv, time_conv, compute_humidity_ratio

# --- Configuration ---
DATA_FILE = 'dataset.csv'
STACKING_SAVE_PATH = 'lstm_stacking_meta_model.pkl' 
OPTUNA_DB_PATH = 'sqlite:///optuna_stacking_study.db'

# Update this list with the intervals you actually trained 
TRAINED_HOUR_INTERVALS = [1, 24, 48, 72] 

# Cross-Validation Settings
N_SPLITS = 4
BATCH_SIZE_FIXED = 32
N_TRIALS_TUNING = 20 

# Features
numerical_features_for_scaling = ["d2m", "t2m", "DNI", "DHI", "GHI", "hr"]
feature_columns = ['year_sin','year_cos','day_sin','day_cos',"t2m","d2m","DNI","DHI","GHI","hr"]
target_columns = ["t2m", "DNI", "DHI", "GHI"]

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# --- 1. Data Loading ---
print(f"Loading data from {DATA_FILE}...")
try:
    df1 = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: '{DATA_FILE}' not found.")
    exit()

# Load scaling params from the FIRST available model
mean_val, std_val, INDIVIDUAL_MODEL_POOL_RATIO = None, None, 0.85
for interval in TRAINED_HOUR_INTERVALS:
    param_file = f'lstm_params_best_{interval}h.pkl'
    if os.path.exists(param_file):
        with open(param_file, 'rb') as f:
            sample_params = pickle.load(f)
            mean_val = sample_params['mean_val']
            std_val = sample_params['std_val']
            if 'overall_cv_data_pool_ratio' in sample_params:
                INDIVIDUAL_MODEL_POOL_RATIO = sample_params['overall_cv_data_pool_ratio']
            break
if mean_val is None:
    print("Error: Could not find base model parameters.")
    exit()

# Apply Preprocessing
df1['date'] = df1['date'].apply(year_conv)
df1['time'] = df1['time'].apply(time_conv)
df1['hr'] = compute_humidity_ratio(df1['d2m'], df1.get('sp', 101325))
for hd in numerical_features_for_scaling:
    df1[hd] = (df1[hd] - mean_val[hd]) / std_val[hd]
df1['year_sin'] = np.sin(df1['date'])
df1['year_cos'] = np.cos(df1['date'])
df1['day_sin'] = np.sin(df1['time'])
df1['day_cos'] = np.cos(df1['time'])

target_indices = [feature_columns.index(col) for col in target_columns]
raw_data = df1[feature_columns].values
dataset_size_full = len(raw_data)

# --- 2. Model Classes (Must match base training) ---
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

class DynamicInputDataset(Dataset):
    def __init__(self, raw_data, target_indices_to_predict, model_params, hour_interval):
        self.raw_data = torch.tensor(raw_data, dtype=torch.float32)
        self.input_sequences = []
        self.seq_length = model_params['seq_length']
        self.hour_interval = hour_interval
        for target_idx in target_indices_to_predict:
            input_start_idx = target_idx - (self.seq_length * self.hour_interval)
            if input_start_idx >= 0:
                x_indices = torch.arange(input_start_idx, input_start_idx + self.seq_length * self.hour_interval, self.hour_interval)
                if torch.all(x_indices < len(self.raw_data)):
                    self.input_sequences.append((self.raw_data[x_indices], target_idx))
    def __len__(self): return len(self.input_sequences)
    def __getitem__(self, idx): return self.input_sequences[idx]

# --- 3. Helper Functions ---
def get_aligned_predictions(target_data_indices, current_raw_data, current_target_indices, current_loaded_models):
    if len(target_data_indices) == 0: return {}, np.array([]), []
    temp_preds = {interval: {} for interval in current_loaded_models}
    valid_indices_set = set()
    first_model = True

    with torch.no_grad():
        for interval, info in current_loaded_models.items():
            model = info['model']
            ds = DynamicInputDataset(current_raw_data, target_data_indices, info['params'], interval)
            if len(ds) == 0: continue
            loader = DataLoader(ds, batch_size=BATCH_SIZE_FIXED, shuffle=False)
            current_model_indices = []
            for batch_x, batch_target_ids in loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x).cpu().numpy()
                ids_np = batch_target_ids.cpu().numpy()
                for i, t_id in enumerate(ids_np):
                    temp_preds[interval][t_id] = outputs[i]
                    current_model_indices.append(t_id)
            
            if first_model:
                valid_indices_set = set(current_model_indices)
                first_model = False
            else:
                valid_indices_set = valid_indices_set.intersection(set(current_model_indices))

    final_indices = sorted(list(valid_indices_set))
    if not final_indices: return {}, np.array([]), []

    y_true = current_raw_data[final_indices][:, current_target_indices]
    aligned_preds = {}
    for interval in current_loaded_models:
        aligned_preds[interval] = np.array([temp_preds[interval][idx] for idx in final_indices])

    return aligned_preds, y_true, final_indices

def prepare_stacking_matrix(aligned_preds, intervals):
    list_of_preds = []
    for interval in intervals:
        if interval in aligned_preds:
            list_of_preds.append(aligned_preds[interval])
        else: return None
    return np.concatenate(list_of_preds, axis=1)

# --- 4. Load Models & Tune ---
loaded_models = {}
for interval in TRAINED_HOUR_INTERVALS:
    model_path = f'lstm_weather_model_best_{interval}h.pth'
    params_path = f'lstm_params_best_{interval}h.pkl'
    if os.path.exists(model_path) and os.path.exists(params_path):
        with open(params_path, 'rb') as f: params = pickle.load(f)
        model = LSTMModel(params['input_dim'], params['hidden_size'], params['num_layers'], params['output_dim'], params['dropout']).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        loaded_models[interval] = {'model': model, 'params': params}

cv_pool_end_idx = int(INDIVIDUAL_MODEL_POOL_RATIO * dataset_size_full)
cv_pool_indices = np.arange(cv_pool_end_idx)

def objective(trial):
    ridge_alpha = trial.suggest_float('ridge_alpha', 0.01, 100.0, log=True)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_mses = []
    for train_idx, val_idx in tscv.split(cv_pool_indices):
        preds_train, y_train, _ = get_aligned_predictions(cv_pool_indices[train_idx], raw_data, target_indices, loaded_models)
        preds_val, y_val, _ = get_aligned_predictions(cv_pool_indices[val_idx], raw_data, target_indices, loaded_models)
        if not preds_train or not preds_val: continue
        
        X_train = prepare_stacking_matrix(preds_train, TRAINED_HOUR_INTERVALS)
        X_val = prepare_stacking_matrix(preds_val, TRAINED_HOUR_INTERVALS)
        
        meta = MultiOutputRegressor(Ridge(alpha=ridge_alpha))
        meta.fit(X_train, y_train)
        fold_mses.append(mean_squared_error(y_val, meta.predict(X_val)))
    return np.mean(fold_mses) if fold_mses else float('inf')

if __name__ == "__main__":
    print("Tuning Stacking Weights...")
    study = optuna.create_study(direction='minimize', storage=OPTUNA_DB_PATH, study_name="stacking_weights", load_if_exists=True)
    study.optimize(objective, n_trials=N_TRIALS_TUNING)
    
    best_alpha = study.best_params['ridge_alpha']
    print(f"Best Ridge Alpha: {best_alpha}")

    print("Training Final Meta-Model...")
    preds_full, y_full, _ = get_aligned_predictions(cv_pool_indices, raw_data, target_indices, loaded_models)
    X_full = prepare_stacking_matrix(preds_full, TRAINED_HOUR_INTERVALS)
    
    final_meta_model = MultiOutputRegressor(Ridge(alpha=best_alpha))
    final_meta_model.fit(X_full, y_full)
    
    with open(STACKING_SAVE_PATH, 'wb') as f:
        pickle.dump({
            'meta_model': final_meta_model,
            'intervals': TRAINED_HOUR_INTERVALS,
            'mean_val': mean_val, # Saving scaling params for prediction script
            'std_val': std_val
        }, f)

    print(f"Meta-model saved to {STACKING_SAVE_PATH}")
