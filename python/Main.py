import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

ticker = 'AAPL' 
df = yf.download(ticker, '2020-01-01')
df.Close.plot(figsize=(12, 8))

scaler = StandardScaler()

df['Close'] = scaler.fit_transform(df['Close'])
df.Close


seq_length = 30   
data = []

for i in range(len(df) - seq_length):
    data.append(df.Close[i:i+seq_length])


data = np.array(data)

train_size = int(0.8 * len(data))   # %80 eğitim, %20 test


X_train = torch.from_numpy(data[:train_size, :-1, :])
y_train = torch.from_numpy(data[:train_size, -1, :])
X_test  = torch.from_numpy(data[train_size:, :-1, :])
y_test  = torch.from_numpy(data[train_size:, -1, :])

class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
