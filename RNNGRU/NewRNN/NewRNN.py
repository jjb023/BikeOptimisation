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

# print("Column names in the DataFrame:", df.columns.tolist())

# Optional: Select a subset of columns if there are too many

cols = ['0.1']
columns = ['temp', 'precip'] + cols  # Adjust according to actual needs
df = df[columns]
#
# Normalize features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.values)

# Create sequences
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data)-sequence_length):
        x = data[i:(i+sequence_length), :-2]  # all columns except the last two are features
        y = data[i+sequence_length, -2:]  # the last two columns are the targets
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 100  # Using past 5 timestamps to predict the next one
X, y = create_sequences(df_scaled, sequence_length)

# SPLIT DATA

# Splitting data into train, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 25% of 80% = 20% of the total data

# Define GRU


# Convert to PyTorch tensors
train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# GRU model
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        return x

# Parameters
input_dim = X.shape[2]  # Number of features
hidden_dim = 50  # Number of features in hidden state
output_dim = y.shape[1]  # Number of prediction targets
num_layers = 2  # Number of GRU layers

# Model initialization
model = GRUNet(input_dim, hidden_dim, output_dim, num_layers)
print(model)



# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
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

# Call to train the model
train_model(model, train_loader, val_loader, criterion, optimizer)

# Assuming X_test is your new unseen data already preprocessed and shaped correctly:
# Convert X_test to a tensor if it's not already
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Set the model to evaluation mode
model.eval()

# No gradient is needed to be computed:
with torch.no_grad():
    predictions = model(X_test_tensor)

# Assuming you used a standard scaler, you might need to invert the scaling for actual use:
predictions_np = predictions.numpy()  # Convert to numpy array if further processing is needed outside PyTorch
actual_predictions = scaler.inverse_transform(predictions_np)  # Only if you've scaled your target variable

# Now, `actual_predictions` contains the actual predicted values in their original scale.
print("Predictions:", actual_predictions)

