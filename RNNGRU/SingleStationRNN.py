import argparse
import pandas as pd
import numpy as np

import torch


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def load_data(sequence_length, column):
    df = pd.read_csv('2019BikeData/2019MergedBikeWeatherData.csv', index_col='Date', parse_dates=True)
    df = df.between_time('17:00', '17:30')  # replace with your specific time
    cols = [column, 'temp', 'precip']
    df = df[cols]
    df[column] = df[column].diff()  # calculate net change
    df = df.dropna()  # remove rows with NaN values
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.values)

    X, y = [], []
    for i in range(len(df_scaled) - sequence_length):
        X.append(df_scaled[i:i+sequence_length])
        y.append(df_scaled[i+sequence_length, 0])  # predict the net change of the selected column
    return np.array(X), np.array(y)
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.relu(x)
        x = self.fc(x[:, -1, :])

        return x

class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.best_loss = float('inf')
        self.early_stop_counter = 0
        self.train_losses = []  # store training losses
        self.val_losses = []  # store validation losses

    def train(self, train_loader, val_loader, n_epochs=50, patience=10000):
        for epoch in range(n_epochs):
            self.model.train()
            train_loss = 0
            for x_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(x_batch)
                y_batch = y_batch.unsqueeze(1)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            self.train_losses.append(train_loss / len(train_loader))  # average training loss

            self.model.eval()
            with torch.no_grad():
                valid_loss = sum(self.criterion(self.model(x_val), y_val) for x_val, y_val in val_loader) / len(val_loader)
            self.val_losses.append(valid_loss.item())  # store validation loss
            if valid_loss.item() < self.best_loss:
                self.best_loss = valid_loss.item()
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= patience:
                    print(f'Stopping early at epoch {epoch+1}')
                    break
            print(f'Epoch {epoch+1}, Validation Loss: {valid_loss.item()}')

def plot_losses(trainer):
    plt.plot(trainer.train_losses, label='Training loss')
    plt.plot(trainer.val_losses, label='Validation loss')
    plt.legend()
    plt.show()



def main(args):
    X, y = load_data(args.sequence_length, args.column)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    model = GRUNet(args.input_dim, args.hidden_dim//2, args.output_dim, args.num_layers//2)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    trainer = Trainer(model, criterion, optimizer)
    trainer.train(train_loader, val_loader, args.n_epochs)
    plot_losses(trainer)
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    evaluate(model, test_loader)

def plot_results(y_true, y_pred):
    plt.plot(y_true, label='True')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.show()

def evaluate(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x_test, y_test in test_loader:
            y_pred.append(model(x_test).numpy())
            y_true.append(y_test.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f'Mean Squared Error: {mse}')
    plot_results(y_true, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--column', type=str, default='1')
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=20)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--n_epochs', type=int, default=40)
    parser.add_argument('--sequence_length', type=int, default=50)
    args = parser.parse_args()
    main(args)





