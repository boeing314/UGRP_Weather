import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import optuna
import pickle
import os
from conversion_function import year_conv, time_conv, compute_humidity_ratio

# --- Configuration ---
DATA_FILE = 'dataset.csv'
SAVE_DIR = 'itransformer_models'
os.makedirs(SAVE_DIR, exist_ok=True)

# Tuning Settings
N_TRIALS = 100
EPOCHS_PER_TRIAL = 100
FINAL_EPOCHS = 200
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. Data Loading & Preprocessing ---
def load_data():
    df1 = pd.read_csv(DATA_FILE)
    
    # Scaling Params
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
    
    # Find which index in feature_columns corresponds to each target
    target_indices = [feature_columns.index(col) for col in target_columns]
    
    return raw_data, target_indices, len(feature_columns), len(target_columns), mean_val, std_val,feature_columns,target_columns

# Load data globally
RAW_DATA, TARGET_INDICES, NUM_ALL_FEATURES, NUM_TARGETS, MEAN_VAL, STD_VAL,feature_columns,target_columns = load_data()

# --- 2. Dataset Class ---
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, target_indices):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length
        self.target_indices = target_indices

    def __len__(self):
        return max(0, self.data.shape[0] - self.seq_length)

    def __getitem__(self, index):
        # x shape: [seq_length, num_features]
        x = self.data[index : index + self.seq_length]
        # y shape: [num_targets] (Predicting the NEXT step)
        y = self.data[index + self.seq_length, self.target_indices]
        return x, y

# --- 3. Inverse Transformer Architecture ---

class InverseTransformer(nn.Module):
    def __init__(self, seq_len, num_all_features, target_indices, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(InverseTransformer, self).__init__()
        
        self.num_all_features = num_all_features
        self.target_indices = target_indices
        self.d_model = d_model
        
        # 1. INVERSE EMBEDDING
        # Instead of embedding a single time step, we embed the entire time series of a feature.
        # Input to Linear: seq_len
        # Output of Linear: d_model
        # This layer learns the temporal characteristics implicitly.
        self.feature_embedding = nn.Linear(seq_len, d_model)
        
        # 2. TRANSFORMER ENCODER
        # Attention is calculated between FEATURES, not time steps.
        # It learns how "Temperature" affects "GHI", etc.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. PROJECTION (OUTPUT)
        # We project the latent d_model back to a scalar (size 1) for the prediction.
        # We will only apply this to the target tokens.
        self.projector = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, num_features]
        
        # 1. TRANSPOSE for Inverse approach
        # New shape: [batch_size, num_features, seq_len]
        # Now, "num_features" is the sequence length for the Transformer (the number of tokens)
        # And "seq_len" is the feature size for the embedding
        x = x.permute(0, 2, 1)
        
        # 2. Embed
        # Input: [batch, num_features, seq_len] -> Output: [batch, num_features, d_model]
        x_enc = self.feature_embedding(x)
        
        # 3. Encoder (Multivariate Correlation)
        # Output: [batch, num_features, d_model]
        enc_out = self.encoder(x_enc)
        
        # 4. Filter for Targets & Project
        # We have representations for ALL features, but we only want to predict the Targets.
        # We select the tokens corresponding to the target indices.
        
        # Select target tokens: [batch, num_targets, d_model]
        target_tokens = enc_out[:, self.target_indices, :]
        
        # Project to scalar: [batch, num_targets, 1]
        out = self.projector(target_tokens)
        
        # Remove last dim: [batch, num_targets]
        return out.squeeze(-1)

# --- 4. Optuna Objective ---

def objective(trial):
    # Hyperparameters
    # Note: seq_len is a structural parameter of the embedding now
    seq_length = trial.suggest_categorical('seq_length', [24, 48, 72, 96])
    
    # Model Params
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])
    d_model_mult = trial.suggest_int('d_model_mult', 8, 64, step=8)
    d_model = nhead * d_model_mult # Ensure divisibility
    
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dim_feedforward = trial.suggest_int('dim_feedforward', 128, 512)
    dropout = trial.suggest_float('dropout', 0.0, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    
    # Setup Data
    dataset = TimeSeriesDataset(RAW_DATA, seq_length, TARGET_INDICES)
    
    # Quick Train/Val Split
    train_size = int(len(dataset) * 0.85)
    indices = list(range(len(dataset)))
    train_loader = DataLoader(Subset(dataset, indices[:train_size]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, indices[train_size:]), batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    model = InverseTransformer(
        seq_len=seq_length,
        num_all_features=NUM_ALL_FEATURES,
        target_indices=TARGET_INDICES,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Tuning Loop
    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return avg_val_loss

# --- 5. Main Execution ---

if __name__ == "__main__":
    print("------------------------------------------------")
    print(" Step 1: Optuna Tuning for Inverse Transformer")
    print("------------------------------------------------")
    
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)

    print("\nBest Params:")
    bp = study.best_params
    for k, v in bp.items():
        print(f"  {k}: {v}")
        
    # Calculate derived params
    best_seq_len = bp['seq_length']
    best_d_model = bp['nhead'] * bp['d_model_mult']

    print("\n------------------------------------------------")
    print(" Step 2: Final Training")
    print("------------------------------------------------")
    
    # Full Dataset
    dataset = TimeSeriesDataset(RAW_DATA, best_seq_len, TARGET_INDICES)
    train_size = int(len(dataset) * 0.85)
    train_loader = DataLoader(Subset(dataset, list(range(train_size))), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, list(range(train_size, len(dataset)))), batch_size=BATCH_SIZE, shuffle=False)
    
    model = InverseTransformer(
        seq_len=best_seq_len,
        num_all_features=NUM_ALL_FEATURES,
        target_indices=TARGET_INDICES,
        d_model=best_d_model,
        nhead=bp['nhead'],
        num_layers=bp['num_layers'],
        dim_feedforward=bp['dim_feedforward'],
        dropout=bp['dropout']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=bp['lr'])
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    save_path = os.path.join(SAVE_DIR, 'best_itransformer.pth')
    
    for epoch in range(FINAL_EPOCHS):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out = model(batch_x)
                val_loss += criterion(out, batch_y).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1} | Train: {avg_train:.5f} | Val: {avg_val:.5f}")
            
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), save_path)
            
    # Save Config
    config = {
        'seq_len': best_seq_len,
        'd_model': best_d_model,
        'nhead': bp['nhead'],
        'num_layers': bp['num_layers'],
        'dim_feedforward': bp['dim_feedforward'],
        'dropout': bp['dropout'],
        'input_dim': NUM_ALL_FEATURES, # Used to init model
        'output_dim': NUM_TARGETS,
        'target_indices': TARGET_INDICES,
        'mean_val': MEAN_VAL,
        'std_val': STD_VAL,
        'feature_columns': feature_columns,
        'target_columns': target_columns
    }
    with open(os.path.join(SAVE_DIR, 'itransformer_config.pkl'), 'wb') as f:
        pickle.dump(config, f)
        

    print(f"Done. Model saved to {save_path}")
