import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import optuna
import pickle
import os
import math
from conversion_function import year_conv, time_conv, compute_humidity_ratio

# --- Configuration ---
DATA_FILE = 'dataset.csv'
SAVE_DIR = 'transformer_single'
os.makedirs(SAVE_DIR, exist_ok=True)

# Tuning Settings
N_TRIALS = 100          # How many hyperparameter combinations to try
EPOCHS_PER_TRIAL = 100   # Epochs for tuning (keep low for speed)
FINAL_EPOCHS = 200       # Epochs for the final best model
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. Data Loading & Preprocessing ---
def load_data():
    df1 = pd.read_csv(DATA_FILE)
    
    # Hardcoded scaling values (from your previous scripts)
    mean_val={"d2m":296.72937075746887,"t2m":301.567039019018,"DNI":210.59884837107128,"DHI":74.42870504330628,"GHI":219.5148092104472,"hr":0.01852766471396759}
    std_val={"d2m":2.14331552045601950,"t2m":2.7471734820877556,"DNI":276.0952387638112,"DHI":94.91021039014818,"GHI":297.0297968459997,"hr":0.0023766051788255255}

    df1['date'] = df1['date'].apply(year_conv)
    df1['time'] = df1['time'].apply(time_conv)
    df1['hr'] = compute_humidity_ratio(df1['d2m'], df1.get('sp', 101325))

    numerical_features = ["d2m", "t2m", "DNI", "DHI", "GHI", "hr"]
    for hd in numerical_features:
        df1[hd] = (df1[hd] - mean_val[hd]) / std_val[hd]

    df1['year_sin'] = np.sin(df1['date'])
    df1['year_cos'] = np.cos(df1['date'])
    df1['day_sin'] = np.sin(df1['time'])
    df1['day_cos'] = np.cos(df1['time'])

    feature_columns = ['year_sin','year_cos','day_sin','day_cos',"t2m","d2m","DNI","DHI","GHI","hr"]
    target_columns = ["t2m", "DNI", "DHI", "GHI"]
    
    raw_data = df1[feature_columns].values
    target_indices = [feature_columns.index(col) for col in target_columns]
    
    return raw_data, target_indices, len(feature_columns), len(target_columns), mean_val, std_val

# Load data once globally
RAW_DATA, TARGET_INDICES, INPUT_DIM, OUTPUT_DIM, MEAN_VAL, STD_VAL = load_data()

# --- 2. Dataset Class ---
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

# --- 3. Transformer Model Architecture ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, dim_feedforward, dropout, seq_len):
        super(TimeSeriesTransformer, self).__init__()
        
        # 1. Input Embedding: Project features to d_model size
        self.input_linear = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 50)
        
        # 3. Transformer Encoder Layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 4. Output Head
        self.decoder = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, src):
        # src: [batch, seq_len, features]
        
        # Embed and add position info
        src = self.input_linear(src)
        src = self.relu(src)
        src = self.pos_encoder(src)
        
        # Pass through Transformer
        # output: [batch, seq_len, d_model]
        output = self.transformer_encoder(src)
        
        # We only care about the prediction based on the LAST time step
        # Take the last time step feature vector
        last_step_output = output[:, -1, :]
        
        prediction = self.decoder(self.dropout(last_step_output))
        return prediction

# --- 4. Optuna Objective ---

def objective(trial):
    # --- Hyperparameters Search Space ---
    seq_length = trial.suggest_categorical('seq_length', [24, 48, 72, 96])
    
    # Transformer Architecture
    # Note: d_model must be divisible by nhead
    nhead = trial.suggest_categorical('nhead', [2, 4])
    d_model_multiplier = trial.suggest_int('d_model_mult', 8, 32) 
    d_model = nhead * d_model_multiplier
    
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dim_feedforward = trial.suggest_int('dim_feedforward', 64, 256)
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)

    # --- Data Setup ---
    dataset = TimeSeriesDataset(RAW_DATA, seq_length, TARGET_INDICES)
    
    # Train/Val Split
    train_size = int(len(dataset) * 0.8)
    indices = list(range(len(dataset)))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Model Setup ---
    model = TimeSeriesTransformer(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        seq_len=seq_length
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # --- Training Loop (Short) ---
    for epoch in range(EPOCHS_PER_TRIAL):
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
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Pruning (stop bad trials early)
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_val_loss

# --- 5. Main Execution ---

if __name__ == "__main__":
    print("------------------------------------------------")
    print(" Step 1: Hyperparameter Tuning (Optuna)")
    print("------------------------------------------------")
    
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)

    print("\nBest Trial:")
    print(f"  Value: {study.best_value}")
    print("  Params: ")
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"    {key}: {value}")

    # Reconstruct derived params
    best_nhead = best_params['nhead']
    best_d_model = best_nhead * best_params['d_model_mult']
    best_seq_len = best_params['seq_length']

    print("\n------------------------------------------------")
    print(" Step 2: Training Final Model with Best Params")
    print("------------------------------------------------")

    # Full Dataset with best seq_len
    full_dataset = TimeSeriesDataset(RAW_DATA, best_seq_len, TARGET_INDICES)
    
    # We still split for monitoring, but you could train on all data if preferred
    train_size = int(len(full_dataset) * 0.85)
    train_loader = DataLoader(Subset(full_dataset, list(range(train_size))), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, list(range(train_size, len(full_dataset)))), batch_size=BATCH_SIZE, shuffle=False)

    final_model = TimeSeriesTransformer(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        d_model=best_d_model,
        nhead=best_nhead,
        num_layers=best_params['num_layers'],
        dim_feedforward=best_params['dim_feedforward'],
        dropout=best_params['dropout'],
        seq_len=best_seq_len
    ).to(device)

    optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    save_path = os.path.join(SAVE_DIR, 'best_transformer.pth')

    for epoch in range(FINAL_EPOCHS):
        final_model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = final_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Val
        final_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = final_model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{FINAL_EPOCHS} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(final_model.state_dict(), save_path)
            # print(f"  > New best model saved.")

    # Save Config
    config_to_save = {
        'input_dim': INPUT_DIM,
        'output_dim': OUTPUT_DIM,
        'd_model': best_d_model,
        'nhead': best_nhead,
        'num_layers': best_params['num_layers'],
        'dim_feedforward': best_params['dim_feedforward'],
        'dropout': best_params['dropout'],
        'seq_len': best_seq_len,
        'mean_val': MEAN_VAL,
        'std_val': STD_VAL,
        'feature_columns': ['year_sin','year_cos','day_sin','day_cos',"t2m","d2m","DNI","DHI","GHI","hr"],
        'target_columns': ["t2m", "DNI", "DHI", "GHI"]
    }
    
    with open(os.path.join(SAVE_DIR, 'transformer_config.pkl'), 'wb') as f:
        pickle.dump(config_to_save, f)


    print(f"\nTraining Complete. Model saved to {save_path}")
