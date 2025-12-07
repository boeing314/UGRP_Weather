import pandas as pd
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import optuna
import pickle
import os

# Ensure gen_funct.py is available in the same directory
from conversion_function import year_conv, time_conv, compute_humidity_ratio

# --- Configuration ---

# HOUR_INTERVAL: The step size between data points in the sequence.
# 1 = Continuous hours (t, t+1, t+2...)
# 24 = Same hour every day (t, t+24, t+48...)
HOUR_INTERVAL = 1 

DATA_FILE = 'data_new.csv'
# Append interval to filenames to distinguish between different time-step models
MODEL_SAVE_PATH = f'lstm_weather_model_best_{HOUR_INTERVAL}h.pth'
PARAMS_SAVE_PATH = f'lstm_params_best_{HOUR_INTERVAL}h.pkl'
CV_LOSSES_SAVE_PATH = f'cv_mse_losses_best_params_{HOUR_INTERVAL}h.csv'
OPTUNA_DB_PATH = f'sqlite:///optuna_lstm_study_{HOUR_INTERVAL}h.db'

# Features for standardization
numerical_features_for_scaling = ["d2m", "t2m", "DNI", "DHI", "GHI", "hr"]
feature_columns = ['year_sin','year_cos','day_sin','day_cos',"t2m","d2m","DNI","DHI","GHI","hr"]
target_columns = ["t2m", "DNI", "DHI", "GHI"]

# Fixed training parameters
BATCH_SIZE_FIXED = 32
N_SPLITS = 4 

# Early Stopping
PATIENCE = 10 
MIN_DELTA = 1e-4 

# Optuna parameters
N_TRIALS = 50 
NUM_EPOCHS_TUNING = 60 

# Final model training
NUM_EPOCHS_FINAL = 100 

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
np.random.seed(0) 
torch.manual_seed(0) 
print(f"Using device: {device}\n")

# --- Data Loading and Preprocessing ---
print(f"Loading and preprocessing data from {DATA_FILE}...")
try:
    df1 = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: '{DATA_FILE}' not found.")
    exit()

try:
    # Statistical constants for normalization (derived from training data analysis)
    mean_val={"d2m":296.72937075746887,"t2m":301.567039019018,"DNI":210.59884837107128,"DHI":74.42870504330628,"GHI":219.5148092104472,"hr":0.01852766471396759}
    std_val={"d2m":2.14331552045601950,"t2m":2.7471734820877556,"DNI":276.0952387638112,"DHI":94.91021039014818,"GHI":297.0297968459997,"hr":0.0023766051788255255}

    # Feature Engineering
    df1['date'] = df1['date'].apply(year_conv)
    df1['time'] = df1['time'].apply(time_conv)
    df1['hr'] = compute_humidity_ratio(df1['d2m'], df1.get('sp', 101325))

    # Normalize numerical features
    for hd in numerical_features_for_scaling:
        df1[hd] = (df1[hd] - mean_val[hd]) / std_val[hd]

    # Cyclical encoding for time
    df1['year_sin'] = np.sin(df1['date'])
    df1['year_cos'] = np.cos(df1['date'])
    df1['day_sin'] = np.sin(df1['time'])
    df1['day_cos'] = np.cos(df1['time'])

    target_indices = [feature_columns.index(col) for col in target_columns]
    input_size = len(feature_columns)
    output_dim = len(target_columns)

except Exception as e:
    print(f"Preprocessing Error: {e}")
    exit()

raw_data = df1[feature_columns].values

# --- Dataset and Model Classes ---

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, target_indices, hour_interval=1):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length
        self.target_indices = target_indices
        self.hour_interval = hour_interval 

    def __len__(self):
        # Calculate total time span needed for one sequence + target
        required_span = self.seq_length * self.hour_interval
        if self.data.shape[0] <= required_span:
            return 0
        return self.data.shape[0] - required_span

    def __getitem__(self, index):
        # Generate indices with the specified interval (stride)
        # e.g., if interval=24, get rows 0, 24, 48...
        x_indices = torch.arange(index, index + self.seq_length * self.hour_interval, self.hour_interval)
        x = self.data[x_indices]
        # Target is the next step after the sequence ends
        y = self.data[index + self.seq_length * self.hour_interval, self.target_indices]
        return x, y

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob if layer_dim > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :]) # Use output of the last time step
        out = self.fc(out)
        return out


# --- Optuna Objective Function ---
def objective(trial: optuna.trial.Trial):
    # Search Space
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.05, 0.4)
    seq_length = trial.suggest_int('seq_length', 8, 48) 
    learning_rate = trial.suggest_float('learning_rate', 5e-5, 1e-2, log=True)

    print(f"  Trial {trial.number}: Starting CV. Params: {trial.params}")

    # Create dataset using the specific HOUR_INTERVAL
    full_dataset = TimeSeriesDataset(raw_data, seq_length, target_indices, hour_interval=HOUR_INTERVAL)

    # Safety checks for dataset size
    if len(full_dataset) == 0:
        raise optuna.exceptions.TrialPruned("Not enough data for parameters.")

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    # Reserve last 15% for final testing (Hold-out), use 85% for CV
    cv_data_pool_indices = indices[:int(np.floor(0.85 * dataset_size))]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_validation_losses = [] 
    global_step_counter = 0 

    # Cross-Validation Loop
    for fold, (train_ids, val_ids) in enumerate(tscv.split(cv_data_pool_indices)):
        train_subset = Subset(full_dataset, [cv_data_pool_indices[i] for i in train_ids])
        val_subset = Subset(full_dataset, [cv_data_pool_indices[i] for i in val_ids])

        if len(train_subset) == 0 or len(val_subset) == 0:
            fold_validation_losses.append(float('nan'))
            continue

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE_FIXED, shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE_FIXED, shuffle=False)

        model = LSTMModel(input_size, hidden_size, num_layers, output_dim, dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=0)

        best_val_loss_fold = float('inf')
        epochs_no_improve = 0

        # Training Epochs for this Fold
        for epoch in range(NUM_EPOCHS_TUNING):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            current_epoch_val_loss = 0
            with torch.no_grad():
                for batch_x_val, batch_y_val in val_loader:
                    batch_x_val, batch_y_val = batch_x_val.to(device), batch_y_val.to(device)
                    preds_val = model(batch_x_val)
                    loss_val = criterion(preds_val, batch_y_val)
                    current_epoch_val_loss += loss_val.item() * batch_x_val.size(0)
            current_epoch_val_loss /= len(val_subset)

            trial.report(current_epoch_val_loss, global_step_counter)
            global_step_counter += 1

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if current_epoch_val_loss < best_val_loss_fold - MIN_DELTA:
                best_val_loss_fold = current_epoch_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                break
        
        fold_validation_losses.append(best_val_loss_fold)

    avg_trial_val_loss = np.nanmean(fold_validation_losses)
    if np.isnan(avg_trial_val_loss):
        raise optuna.exceptions.TrialPruned("All folds resulted in NaN.")
    
    return avg_trial_val_loss


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting hyperparameter tuning with Optuna...")
    
    study = optuna.create_study(
        study_name='lstm_tuning_study',
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1),
        storage=OPTUNA_DB_PATH,
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)

    print("\nTuning complete.")
    best_params = study.best_trial.params
    print(f"Best Params: {best_params}")

    # Save study results
    study.trials_dataframe().to_csv("optuna_lstm_study_results.csv", index=False)


    # --- Final Validation with Best Hyperparameters ---
    print("\n--- Final CV with Best Params ---")

    all_fold_train_mses_final = []
    all_fold_val_mses_final = []

    # Re-init dataset with best params and the specified interval
    full_dataset_final = TimeSeriesDataset(raw_data, best_params['seq_length'], target_indices, hour_interval=HOUR_INTERVAL)
    
    dataset_size_final = len(full_dataset_final)
    indices_final = list(range(dataset_size_final))
    cv_data_pool_indices_final = indices_final[:int(np.floor(0.85 * dataset_size_final))]

    tscv_final = TimeSeriesSplit(n_splits=N_SPLITS)

    # Final CV Loop (to generate loss curves)
    for fold, (train_ids, val_ids) in enumerate(tscv_final.split(cv_data_pool_indices_final)):
        print(f"Final Model CV Fold {fold + 1}/{N_SPLITS}")

        train_subset = Subset(full_dataset_final, [cv_data_pool_indices_final[i] for i in train_ids])
        val_subset = Subset(full_dataset_final, [cv_data_pool_indices_final[i] for i in val_ids])

        if len(train_subset) == 0 or len(val_subset) == 0:
            continue

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE_FIXED, shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE_FIXED, shuffle=False)

        model = LSTMModel(input_size, best_params['hidden_size'], best_params['num_layers'], output_dim, best_params['dropout']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        criterion = nn.MSELoss()

        fold_train_mses = []
        fold_val_mses = []
        best_mse = float('inf')
        no_imp = 0

        for epoch in range(NUM_EPOCHS_FINAL):
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            
            fold_train_mses.append(train_loss / len(train_subset))

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
            
            avg_val_loss = val_loss / len(val_subset)
            fold_val_mses.append(avg_val_loss)

            # Early stopping check for final CV
            if avg_val_loss < best_mse - MIN_DELTA:
                best_mse = avg_val_loss
                no_imp = 0
            else:
                no_imp += 1

            if no_imp >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        all_fold_train_mses_final.append(fold_train_mses)
        all_fold_val_mses_final.append(fold_val_mses)

    # Save CV Loss Data
    if all_fold_train_mses_final:
        max_len = max(len(l) for l in all_fold_train_mses_final)
        # Pad shorter epochs with NaN for DataFrame alignment
        padded_train = [l + [np.nan]*(max_len-len(l)) for l in all_fold_train_mses_final]
        padded_val = [l + [np.nan]*(max_len-len(l)) for l in all_fold_val_mses_final]

        losses_df = pd.DataFrame({
            'Epoch': range(1, max_len + 1),
            'Average_Training_MSE': np.nanmean(padded_train, axis=0),
            'Average_Validation_MSE': np.nanmean(padded_val, axis=0)
        })
        losses_df.to_csv(CV_LOSSES_SAVE_PATH, index=False)
        print(f"Saved CV losses to {CV_LOSSES_SAVE_PATH}")


    # --- Deployment Model Training ---
    print("\nTraining final deployment model on all available training data...")
    
    deployment_dataset = TimeSeriesDataset(raw_data, best_params['seq_length'], target_indices, hour_interval=HOUR_INTERVAL)
    deployment_loader = DataLoader(
        Subset(deployment_dataset, cv_data_pool_indices_final),
        batch_size=BATCH_SIZE_FIXED, shuffle=True
    )

    final_model = LSTMModel(input_size, best_params['hidden_size'], best_params['num_layers'], output_dim, best_params['dropout']).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
    criterion = nn.MSELoss()

    for epoch in range(NUM_EPOCHS_FINAL):
        final_model.train()
        epoch_loss = 0
        for batch_X, batch_y in deployment_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            final_optimizer.zero_grad()
            outputs = final_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            final_optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS_FINAL}, Loss: {epoch_loss / len(deployment_loader):.4f}")


    # --- Save Artifacts ---
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
    
    best_model_params = {
        'input_dim': input_size, 
        'output_dim': output_dim,
        'hidden_size': best_params['hidden_size'],
        'num_layers': best_params['num_layers'],
        'dropout': best_params['dropout'],
        'seq_length': best_params['seq_length'],
        'hour_interval': HOUR_INTERVAL, # Saved for use in testing script
        'learning_rate': best_params['learning_rate'],
        'feature_columns': feature_columns,
        'target_columns': target_columns,
        'mean_val': mean_val,
        'std_val': std_val,
        'overall_cv_data_pool_ratio': 0.85
    }
    
    with open(PARAMS_SAVE_PATH, 'wb') as f:
        pickle.dump(best_model_params, f)

    print("Training complete. All artifacts saved.")