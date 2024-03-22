import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv('your_new_file.csv')

# Convert 'datetime' column to datetime type
df['datetime'] = pd.to_datetime(df['datetime'])

# Feature extraction - considering 'temp', 'precip', and station changes as features
X = df[['temp', 'precip'] + ['station' + str(i) for i in range(1, 751)]].values
y = df[['station' + str(i) for i in range(1, 751)]].values

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define sequence length
sequence_length = 5  # You can adjust this as needed

# Create sequences
def create_sequences(X, y, sequence_length=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i:i+sequence_length])  # Select consecutive rows
        y_seq.append(y[i+sequence_length-1])  # Target is the station changes of the last hour in the sequence
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y, sequence_length)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define the GRU model
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=750, num_layers=2):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])  # Output from the last sequence step
        return x

# Initialize the model
model = GRUNet(input_dim=X_train_tensor.shape[2], hidden_dim=64)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Training loop
epochs = 250
train_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}')

# Plot training loss curve
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).numpy()

# Calculate performance metric, e.g., RMSE
test_rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'Test RMSE: {test_rmse}')

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test[0], label='Actual Station Changes', marker='o')  # Plotting changes for the first station
plt.plot(predictions[0], label='Predicted Station Changes', marker='x')
plt.xlabel('Time Step')
plt.ylabel('Station Changes')
plt.title('Actual vs. Predicted Station Changes')
plt.legend()
plt.show()