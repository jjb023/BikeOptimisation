# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# Load data
df = pd.read_csv('january_data.csv', index_col='datetime', parse_dates=True)

# Select relevant columns (including weather data)
cols = ['0.1', 'temp', 'precip']  # Including weather data as features
df = df[cols]

# Normalize features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.values)

# Create sequences
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data)-sequence_length):
        x = data[i:(i+sequence_length), :-1]  # Use all but the last column as features
        y = data[i+sequence_length, 0]  # Use '0.1' column as target
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 100  # Define sequence length
X, y = create_sequences(df_scaled, sequence_length)

# Split data
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Convert to tensors
train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)


# Define the GRU model
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])  # Use only the last output
        return x

# Model parameters
input_dim = 3  # '0.1', 'temp', 'precip'
hidden_dim = 50
output_dim = 1  # Predicting one output
num_layers = 2

# Initialize model
model = GRUNet(input_dim, hidden_dim, output_dim, num_layers)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training the model
def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs=50):
    for epoch in range(n_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_loss = sum(criterion(model(x_val), y_val) for x_val, y_val in val_loader) / len(val_loader)
        print(f'Epoch {epoch+1}, Validation Loss: {valid_loss.item()}')

train_model(model, train_loader, val_loader, criterion, optimizer)
