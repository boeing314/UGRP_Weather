import pandas as pd
import random
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from conversion_function import year_conv, time_conv, compute_humidity_ratio
from nsag_functions import merge_file, reduce_size, new_child, update_file



# ---------- Load Data and Isolate Training Data ----------
print("Loading data from CSV...")

# 1. Load the CSV file using Pandas
try:
    df1 = pd.read_csv('dataset.csv')
except FileNotFoundError:
    print("Error: 'dataset.csv' not found.")
else:
    try:
        #date,time,d2m,t2m,ssr,strr,ssrd,strd,sp
        mean_val={"d2m":296.72937075746887,"t2m":301.567039019018,"DNI":210.59884837107128,"DHI":74.42870504330628,"GHI":219.5148092104472,"hr":0.01852766471396759}
        std_val={"d2m":2.14331552045601950,"t2m":2.7471734820877556,"DNI":276.0952387638112,"DHI":94.91021039014818,"GHI":297.0297968459997,"hr":0.0023766051788255255}
        df1['date']=df1['date'].apply(year_conv)
        df1['time']=df1['time'].apply(time_conv)
        df1['hr'] = compute_humidity_ratio(df1['d2m'], df1.get('sp', 101325))
        ll=["d2m","t2m","DNI","DHI","GHI","hr"]
        for hd in ll:
            df1[hd]=(df1[hd]-mean_val[hd])/std_val[hd]
        df1['year_sin']=np.sin(df1['date'])
        df1['year_cos']=np.cos(df1['date'])
        df1['day_sin']=np.sin(df1['time'])
        df1['day_cos']=np.cos(df1['time'])

        print(df1.head())  # View the updated dataframe
        feature_columns=['year_sin','year_cos','day_sin','day_cos',"t2m","d2m","DNI","DHI","GHI","hr"]
        target_columns = ["t2m", "DNI", "DHI", "GHI"]
        target_indices = [feature_columns.index(col) for col in target_columns]

        input_size = len(feature_columns) # This will be 10 now
        output_dim = len(target_columns) 

    except KeyError as e:
        print(f"Column not found: {e}")
        exit()

# ---------- Set Device and Seeds ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(0)
print(f"Using device: {device}")

# ---------- Define Dataset and Model Classes ----------
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, target_indices):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length
        self.target_indices = target_indices
    def __len__(self):
        return self.data.shape[0] - self.seq_length
    def __getitem__(self, index):
        x = self.data[index:index + self.seq_length]
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



child_count=0
df_1=pd.read_csv('gen1.csv')
df = pd.DataFrame(columns=['hidden_size','num_layers','dropout','seq_length','avg_loss'])
df.to_csv('parent1.csv', index=False)
for index, row in df_1.iterrows():

    hidden_size=int(row['hidden_size'])
    num_layers=int(row['num_layers'])
    dropout=float(row['dropout'])
    seq_length=int(row['seq_length'])
    learning_rate=0.001
    batch_sz=32

    raw_data = df1[feature_columns].values
    
    full_dataset = TimeSeriesDataset(raw_data, seq_length,target_indices)
    
    print(raw_data)
    print(f"Dataset created with {len(full_dataset)} sequences.")

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    val_split_point = int(np.floor(0.85 * dataset_size))
    val_indices= indices[:val_split_point]

#########################################################################################
    N_SPLITS=4
    # ---------- Time Series Cross-Validation ----------
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_validation_losses = []
    
    print(f"\n--- Starting {N_SPLITS}-Split Time Series Cross-Validation ---")
    
    for fold, (train_ids, val_ids) in enumerate(tscv.split(val_indices)):
        print(f"\n===== SPLIT {fold + 1}/{N_SPLITS} =====")
        print(f"Training on {len(train_ids)} samples, Validating on {len(val_ids)} samples.")
    
        train_subset = Subset(full_dataset, [val_indices[i] for i in train_ids])
        val_subset = Subset(full_dataset,[val_indices[i] for i in val_ids])
    
        train_loader = DataLoader(train_subset,batch_size=batch_sz,shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=batch_sz, shuffle=False)
    
        # Re-initialize a fresh model for each fold
        model = LSTMModel(input_size,hidden_size, num_layers,output_dim, dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=0)
    
        model.train()
        for epoch in range(40):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1} done for Split {fold + 1}")
    
        # Evaluation for the current fold
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                total_val_loss += loss.item() * batch_x.size(0)
    
        avg_val_loss = total_val_loss / len(val_ids)
        fold_validation_losses.append(avg_val_loss)
        print(f"Validation Loss for Split {fold + 1}: {avg_val_loss:.6f}")
    
    print("\n--- Cross-Validation Finished ---")
    avg_loss = np.mean(fold_validation_losses)
    std_loss = np.std(fold_validation_losses)
    print(f"Average Validation Loss across all splits: {avg_loss:.6f}")
    print(f"Standard Deviation of Validation Loss: {std_loss:.6f}")
###########################################################################################
    if avg_loss>0.0411:
        print('loss is high')
        continue
    child_count+=1
        
    with open('parent1.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([hidden_size,num_layers,dropout,seq_length,avg_loss])
        print([hidden_size,num_layers,dropout,seq_length,avg_loss])
        print("added data to file")
print('Initial LSTMS GENERATED')

