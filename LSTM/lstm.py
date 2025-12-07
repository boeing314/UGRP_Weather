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

from conversion_function import year_conv, time_conv, compute_humidity_ratio

# --- Configuration ---
DATA_FILE = 'data_new.csv'
MODEL_SAVE_PATH = 'lstm_weather_model_best.pth'
PARAMS_SAVE_PATH = 'lstm_params_best.pkl'
CV_LOSSES_SAVE_PATH = 'cv_mse_losses_best_params.csv'
OPTUNA_DB_PATH = 'sqlite:///optuna_lstm_study.db'

# Define numerical features for scaling and target identification
numerical_features_for_scaling = ["d2m", "t2m", "DNI", "DHI", "GHI", "hr"]
feature_columns = ['year_sin','year_cos','day_sin','day_cos',"t2m","d2m","DNI","DHI","GHI","hr"]
target_columns = ["t2m", "DNI", "DHI", "GHI"]

# Fixed training parameters
BATCH_SIZE_FIXED = 32
N_SPLITS = 4 

# Early Stopping Parameters
PATIENCE = 10 
MIN_DELTA = 1e-4 

# Optuna tuning parameters
N_TRIALS = 1 
NUM_EPOCHS_TUNING = 60 

# Final model training after tuning
NUM_EPOCHS_FINAL = 100 

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
np.random.seed(0)
torch.manual_seed(0)
print(f"Using device: {device}\n")

# --- Data Loading and Preprocessing ---
print(f"Loading and preprocessing data from {DATA_FILE}...")
try:
    df1 = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: '{DATA_FILE}' not found. Please ensure it's in the same directory.")
    exit()
except Exception as e:
    print(f"An error occurred during initial data loading: {e}")
    exit()

try:
    mean_val={"d2m":296.72937075746887,"t2m":301.567039019018,"DNI":210.59884837107128,"DHI":74.42870504330628,"GHI":219.5148092104472,"hr":0.01852766471396759}
    std_val={"d2m":2.14331552045601950,"t2m":2.7471734820877556,"DNI":276.0952387638112,"DHI":94.91021039014818,"GHI":297.0297968459997,"hr":0.0023766051788255255}

    df1['date']=df1['date'].apply(year_conv)
    df1['time']=df1['time'].apply(time_conv)
    df1['hr'] = compute_humidity_ratio(df1['d2m'], df1.get('sp', 101325))

    for hd in numerical_features_for_scaling:
        df1[hd]=(df1[hd]-mean_val[hd])/std_val[hd]

    df1['year_sin']=np.sin(df1['date'])
    df1['year_cos']=np.cos(df1['date'])
    df1['day_sin']=np.sin(df1['time'])
    df1['day_cos']=np.cos(df1['time'])

    print(df1.head())

    target_indices = [feature_columns.index(col) for col in target_columns]
    input_size = len(feature_columns)
    output_dim = len(target_columns)

except KeyError as e:
    print(f"Column not found during preprocessing: {e}")
    exit()
except Exception as e:
    print(f"An error occurred during data preprocessing: {e}")
    exit()

raw_data = df1[feature_columns].values

# --- Define Dataset and Model Classes ---
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, target_indices):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length
        self.target_indices = target_indices
        # Removed n_alt

    def __len__(self):
        # Determine valid starting indices. 
        # We need seq_length for input + 1 for target
        if self.data.shape[0] <= self.seq_length:
            return 0
        return self.data.shape[0] - self.seq_length

    def __getitem__(self, index):
        # Take a continuous slice of data for input
        x = self.data[index : index + self.seq_length]
        # Target is the immediate next time step
        y = self.data[index + self.seq_length, self.target_indices]
        return x, y

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()
        # If layer_dim is 1, dropout should be 0 to avoid errors in nn.LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob if layer_dim > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :]) # Take the output from the last time step
        out = self.fc(out)
        return out


# --- Optuna Objective Function for Hyperparameter Tuning ---
def objective(trial: optuna.trial.Trial):
    # Hyperparameters to tune
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.05, 0.4)
    seq_length = trial.suggest_int('seq_length', 8, 48) 
    learning_rate = trial.suggest_float('learning_rate', 5e-5, 1e-2, log=True)

    print(f"  Trial {trial.number}: Starting internal CV for params: {trial.params}")

    # Create dataset with trial's sequence parameters (Continuous 1-hour intervals)
    full_dataset = TimeSeriesDataset(raw_data, seq_length, target_indices) 

    if len(full_dataset) == 0:
        print(f"    Warning: Not enough data for seq_length={seq_length}. Pruning trial.")
        raise optuna.exceptions.TrialPruned("Not enough data for given sequence parameters.")

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    # Using 85% of the full_dataset for CV
    cv_data_pool_indices = indices[:int(np.floor(0.85 * dataset_size))]

    if len(cv_data_pool_indices) == 0:
        print(f"    Warning: CV data pool is empty for seq_length={seq_length}. Pruning trial.")
        raise optuna.exceptions.TrialPruned("CV data pool is empty for given sequence parameters.")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_validation_losses = [] 

    global_step_counter = 0 

    for fold, (train_ids, val_ids) in enumerate(tscv.split(cv_data_pool_indices)):
        train_subset = Subset(full_dataset, [cv_data_pool_indices[i] for i in train_ids])
        val_subset = Subset(full_dataset, [cv_data_pool_indices[i] for i in val_ids])

        if len(train_subset) == 0 or len(val_subset) == 0:
            print(f"    Fold {fold+1} skipped (empty train or validation set).")
            for _ in range(NUM_EPOCHS_TUNING):
                trial.report(float('nan'), global_step_counter)
                global_step_counter += 1
            fold_validation_losses.append(float('nan'))
            continue

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE_FIXED, shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE_FIXED, shuffle=False)

        model = LSTMModel(input_size, hidden_size, num_layers, output_dim, dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=0)

        best_val_loss_fold = float('inf')
        epochs_no_improve = 0

        for epoch in range(NUM_EPOCHS_TUNING):
            model.train()
            train_loss_epoch = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item() * batch_x.size(0)

            model.eval()
            current_epoch_val_loss = 0
            with torch.no_grad():
                for batch_x_val, batch_y_val in val_loader:
                    batch_x_val, batch_y_val = batch_x_val.to(device), batch_y_val.to(device)
                    preds_val = model(batch_x_val)
                    loss_val = criterion(preds_val, batch_y_val)
                    current_epoch_val_loss += loss_val.item() * batch_x_val.size(0)
            current_epoch_val_loss /= len(val_subset)
            
            # Reduce verbosity for tuning
            # print(f"val loss at epoch {epoch}: {current_epoch_val_loss}")
            trial.report(current_epoch_val_loss, global_step_counter)
            global_step_counter += 1

            if trial.should_prune():
                print(f"    Trial {trial.number} Fold {fold+1} pruned by Optuna at epoch {epoch+1}.")
                raise optuna.exceptions.TrialPruned()

            if current_epoch_val_loss < best_val_loss_fold - MIN_DELTA:
                best_val_loss_fold = current_epoch_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                # print(f"    Early stopping triggered for Trial {trial.number} Fold {fold+1} at epoch {epoch+1}.")
                break
        
        fold_validation_losses.append(best_val_loss_fold)

    avg_trial_val_loss = np.nanmean(fold_validation_losses)

    if np.isnan(avg_trial_val_loss):
        raise optuna.exceptions.TrialPruned("All folds resulted in NaN validation loss or were skipped.")
    
    print(f"  Trial {trial.number} finished with average best validation MSE: {avg_trial_val_loss:.6f}")
    return avg_trial_val_loss


# --- Main Optimization Block ---
if __name__ == "__main__":
    print("Starting hyperparameter tuning with Optuna (optimizing for average best validation MSE)...")
    
    study_name = 'lstm_tuning_study'

    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1),
        storage=OPTUNA_DB_PATH,
        load_if_exists=True
    )
    
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        gc_after_trial=True
    )

    print("\nHyperparameter tuning complete.")
    best_trial = study.best_trial
    print(f"  Value (Best Average Validation MSE from tuning): {best_trial.value:.6f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    best_params = best_trial.params

    df_study = study.trials_dataframe()
    df_study.to_csv("optuna_lstm_study_results.csv", index=False)


    # --- Train Final Model with Best Hyperparameters ---
    print("\n--- Performing Final Time Series Cross-Validation with best hyperparameters ---")

    all_fold_train_mses_final = []
    all_fold_val_mses_final = []

    # Re-create full_dataset with best seq_length (Continuous 1-hr intervals)
    full_dataset_final = TimeSeriesDataset(raw_data, best_params['seq_length'], target_indices)
    
    if len(full_dataset_final) == 0:
        print(f"Error: Not enough data for the best sequence parameters (seq_length={best_params['seq_length']}). Cannot train final model.")
        exit()

    dataset_size_final = len(full_dataset_final)
    indices_final = list(range(dataset_size_final))
    cv_data_pool_indices_final = indices_final[:int(np.floor(0.85 * dataset_size_final))]

    if len(cv_data_pool_indices_final) == 0:
        print("Error: CV data pool is empty for final model training. Exiting.")
        exit()

    tscv_final = TimeSeriesSplit(n_splits=N_SPLITS)

    for fold, (train_ids, val_ids) in enumerate(tscv_final.split(cv_data_pool_indices_final)):
        print(f"\n--- Final Model CV Fold {fold + 1}/{N_SPLITS} ---")

        train_subset_final = Subset(full_dataset_final, [cv_data_pool_indices_final[i] for i in train_ids])
        val_subset_final = Subset(full_dataset_final, [cv_data_pool_indices_final[i] for i in val_ids])

        if len(train_subset_final) == 0 or len(val_subset_final) == 0:
            print(f"Skipping fold {fold+1}: Empty train or validation set.")
            all_fold_train_mses_final.append([float('nan')] * NUM_EPOCHS_FINAL)
            all_fold_val_mses_final.append([float('nan')] * NUM_EPOCHS_FINAL)
            continue

        print(f"Fold {fold+1} Training data size: {len(train_subset_final)}")
        print(f"Fold {fold+1} Validation data size: {len(val_subset_final)}")

        train_loader_final = DataLoader(train_subset_final, batch_size=BATCH_SIZE_FIXED, shuffle=False)
        val_loader_final = DataLoader(val_subset_final, batch_size=BATCH_SIZE_FIXED, shuffle=False)

        fold_model_final = LSTMModel(
            input_size,
            best_params['hidden_size'],
            best_params['num_layers'],
            output_dim,
            best_params['dropout']
        ).to(device)
        fold_optimizer_final = torch.optim.Adam(fold_model_final.parameters(), lr=best_params['learning_rate'])
        criterion = nn.MSELoss()

        fold_best_val_mse = float('inf')
        epochs_no_improve_final = 0

        fold_train_mses_epoch = []
        fold_val_mses_epoch = []

        for epoch in range(NUM_EPOCHS_FINAL):
            fold_model_final.train()
            train_loss_epoch = 0
            for batch_X, batch_y in train_loader_final:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                fold_optimizer_final.zero_grad()
                outputs = fold_model_final(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                fold_optimizer_final.step()
                train_loss_epoch += loss.item() * batch_X.size(0)
            avg_train_loss_epoch = train_loss_epoch / len(train_subset_final)
            fold_train_mses_epoch.append(avg_train_loss_epoch)

            fold_model_final.eval()
            current_epoch_val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader_final:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = fold_model_final(batch_X)
                    loss_val = criterion(outputs, batch_y)
                    current_epoch_val_loss += loss_val.item() * batch_X.size(0)
            avg_val_loss_epoch = current_epoch_val_loss / len(val_subset_final)
            fold_val_mses_epoch.append(avg_val_loss_epoch)

            if avg_val_loss_epoch < fold_best_val_mse - MIN_DELTA:
                fold_best_val_mse = avg_val_loss_epoch
                epochs_no_improve_final = 0
            else:
                epochs_no_improve_final += 1

            if epochs_no_improve_final >= PATIENCE:
                print(f"    Early stopping triggered for Final CV Fold {fold+1} at epoch {epoch+1} (no improvement for {PATIENCE} epochs).")
                break
        
        all_fold_train_mses_final.append(fold_train_mses_epoch)
        all_fold_val_mses_final.append(fold_val_mses_epoch)

    print("\nFinal Time Series Cross-Validation complete.")

    if all_fold_train_mses_final and all_fold_val_mses_final:
        max_len_train = max(len(l) for l in all_fold_train_mses_final)
        max_len_val = max(len(l) for l in all_fold_val_mses_final)
        max_len = max(max_len_train, max_len_val)

        padded_train_mses = [l + [np.nan]*(max_len-len(l)) for l in all_fold_train_mses_final]
        padded_val_mses = [l + [np.nan]*(max_len-len(l)) for l in all_fold_val_mses_final]

        avg_train_mses_final = np.nanmean(padded_train_mses, axis=0)
        avg_val_mses_final = np.nanmean(padded_val_mses, axis=0)

        losses_df = pd.DataFrame({
            'Epoch': range(1, len(avg_train_mses_final) + 1),
            'Average_Training_MSE': avg_train_mses_final,
            'Average_Validation_MSE': avg_val_mses_final
        })
        losses_df.to_csv(CV_LOSSES_SAVE_PATH, index=False)
        print(f"Average training and validation MSE for best params saved to {CV_LOSSES_SAVE_PATH}.")
    else:
        print("No CV loss data for best params to save.")


    # --- Train one final deployment model on the entire Training-Validation Pool with best params ---
    print("\nTraining final deployment model on the entire Training-Validation Pool with best params...")
    
    # Re-create deployment_full_dataset using the best sequence parameters
    deployment_full_dataset = TimeSeriesDataset(raw_data, best_params['seq_length'], target_indices)
    
    deployment_train_loader = DataLoader(
        Subset(deployment_full_dataset, cv_data_pool_indices_final),
        batch_size=BATCH_SIZE_FIXED, shuffle=True
    )

    final_deployment_model = LSTMModel(
        input_size,
        best_params['hidden_size'],
        best_params['num_layers'],
        output_dim,
        best_params['dropout']
    ).to(device)
    final_deployment_optimizer = torch.optim.Adam(final_deployment_model.parameters(), lr=best_params['learning_rate'])
    criterion = nn.MSELoss()

    for epoch in range(NUM_EPOCHS_FINAL):
        final_deployment_model.train()
        train_loss_epoch_deployment = 0
        num_batches = 0
        for batch_X, batch_y in deployment_train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            final_deployment_optimizer.zero_grad()
            outputs = final_deployment_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            final_deployment_optimizer.step()
            train_loss_epoch_deployment += loss.item()
            num_batches += 1
        
        avg_train_loss_deployment = train_loss_epoch_deployment / num_batches if num_batches > 0 else float('nan')
        if (epoch + 1) % 10 == 0:
            print(f"  Final Deployment Model Epoch {epoch + 1}/{NUM_EPOCHS_FINAL}, Train Loss: {avg_train_loss_deployment:.4f}")
    print("Final deployment model training complete.")


    # --- Save Final Deployment Model and Best Parameters ---
    print(f"Saving final deployment model state_dict to {MODEL_SAVE_PATH}...")
    torch.save(final_deployment_model.state_dict(), MODEL_SAVE_PATH)
    
    best_model_params_to_save = {
        'input_dim': input_size, 'output_dim': output_dim,
        'hidden_size': best_params['hidden_size'],
        'num_layers': best_params['num_layers'],
        'dropout': best_params['dropout'],
        'seq_length': best_params['seq_length'],
        # Removed n_alt from saved parameters
        'learning_rate': best_params['learning_rate'],
        'feature_columns': feature_columns,
        'target_columns': target_columns,
        'mean_val': mean_val,
        'std_val': std_val,
        'overall_cv_data_pool_ratio': 0.85
    }
    print(f"Saving best model parameters to {PARAMS_SAVE_PATH}...")
    with open(PARAMS_SAVE_PATH, 'wb') as f:
        pickle.dump(best_model_params_to_save, f)

    print("All final deployment model artifacts saved successfully.")