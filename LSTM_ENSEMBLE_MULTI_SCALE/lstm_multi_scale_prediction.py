import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge # Required for loading pickle
from sklearn.multioutput import MultiOutputRegressor
from gen_funct import year_conv, time_conv, compute_humidity_ratio

# --- Configuration ---
DATA_FILE = 'data_new.csv'
STACKING_MODEL_PATH = 'lstm_stacking_meta_model.pkl'
RESULTS_CSV_PATH = 'lstm_stacking_test_results.csv'
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# --- 1. Load Meta-Model and Config ---
if not os.path.exists(STACKING_MODEL_PATH):
    print("Error: Meta-model not found. Run train_stacking.py first.")
    exit()

print(f"Loading Meta-Model from {STACKING_MODEL_PATH}...")
with open(STACKING_MODEL_PATH, 'rb') as f:
    meta_data = pickle.load(f)

meta_model = meta_data['meta_model']
trained_intervals = meta_data['intervals']
mean_val = meta_data['mean_val']
std_val = meta_data['std_val']

# Define columns (Hardcoded or saved/loaded, ensuring consistency)
numerical_features_for_scaling = ["d2m", "t2m", "DNI", "DHI", "GHI", "hr"]
feature_columns = ['year_sin','year_cos','day_sin','day_cos',"t2m","d2m","DNI","DHI","GHI","hr"]
target_columns = ["t2m", "DNI", "DHI", "GHI"]

# --- 2. Model Classes (Copy from training) ---
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

# --- 3. Data Loading ---
print("Loading and Preprocessing Data...")
df1 = pd.read_csv(DATA_FILE)

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

# --- 4. Load Base Models ---
loaded_models = {}
for interval in trained_intervals:
    model_path = f'lstm_weather_model_best_{interval}h.pth'
    params_path = f'lstm_params_best_{interval}h.pkl'
    if os.path.exists(model_path) and os.path.exists(params_path):
        with open(params_path, 'rb') as f: params = pickle.load(f)
        model = LSTMModel(params['input_dim'], params['hidden_size'], params['num_layers'], params['output_dim'], params['dropout']).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        loaded_models[interval] = {'model': model, 'params': params}
        print(f"Loaded base model for interval: {interval}h")

# --- 5. Prediction Helpers ---
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
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
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
        list_of_preds.append(aligned_preds[interval])
    return np.concatenate(list_of_preds, axis=1)

# --- 6. Run Prediction on Test Set (Last 15%) ---
print("Running prediction on Hold-out Test Set...")
dataset_size = len(raw_data)
# We assume the last 15% is testing, matching the training logic
test_start_idx = int(np.floor(0.85 * dataset_size))
test_indices = list(range(test_start_idx, dataset_size))

# Get Base Predictions
preds_test, y_test_true, _ = get_aligned_predictions(test_indices, raw_data, target_indices, loaded_models)

if not preds_test:
    print("Error: No valid test data generated (check sequence lengths).")
    exit()

# Stack and Predict with Meta-Model
X_test = prepare_stacking_matrix(preds_test, trained_intervals)
final_preds_scaled = meta_model.predict(X_test)

print(f"MSE on Scaled Data: {mean_squared_error(y_test_true, final_preds_scaled):.6f}")

# --- 7. Inverse Scale and Save ---
print("Inverse scaling results...")
results_data = {}

for i, col_name in enumerate(target_columns):
    mu = mean_val[col_name]
    sigma = std_val[col_name]
    
    # Unscale True Values
    results_data[f"Actual_{col_name}"] = (y_test_true[:, i] * sigma) + mu
    # Unscale Predicted Values
    results_data[f"Predicted_{col_name}"] = (final_preds_scaled[:, i] * sigma) + mu

results_df = pd.DataFrame(results_data)
results_df.to_csv(RESULTS_CSV_PATH, index=False)
print(f"Predictions saved to {RESULTS_CSV_PATH}")
print(results_df.head())