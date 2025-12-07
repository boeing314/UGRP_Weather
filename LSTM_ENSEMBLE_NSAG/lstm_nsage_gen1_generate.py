import pandas as pd
import numpy as np
from scipy.stats import qmc

n_samples = 50   # number of rows

# Parameter ranges
param_ranges = {
    "hidden_size": (32, 256),       # int
    "num_layers": (1, 6),           # int
    "dropout": (0.0, 0.5),          # float
    "seq_length": (5, 40)           # int
}

# Latin Hypercube sampler
sampler = qmc.LatinHypercube(d=len(param_ranges), seed=42)
sample = sampler.random(n=n_samples)

# Scale samples
scaled = qmc.scale(
    sample,
    l_bounds=[param_ranges[k][0] for k in param_ranges],
    u_bounds=[param_ranges[k][1] for k in param_ranges]
)

# Convert integer parameters
scaled[:, 0] = np.round(scaled[:, 0])  # hidden_size
scaled[:, 1] = np.round(scaled[:, 1])  # num_layers
scaled[:, 3] = np.round(scaled[:, 3])  # seq_length

# Create DataFrame
df = pd.DataFrame(scaled, columns=param_ranges.keys())

# FORCE integer dtype
df["hidden_size"] = df["hidden_size"].astype(int)
df["num_layers"] = df["num_layers"].astype(int)
df["seq_length"] = df["seq_length"].astype(int)

# Save CSV
df.to_csv("gen1.csv", index=False)
