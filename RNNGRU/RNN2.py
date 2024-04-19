import argparse
import pandas as pd
import numpy as np

import torch


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

def load_data(sequence_length, column):
    df = pd.read_csv('2019BikeData/2019MergedBikeWeatherData.csv', index_col='Date', parse_dates=True)
    print(df.columns)
    df = df.between_time('08:00', '09:00')  # replace with your specific time
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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        return x

class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, train_loader, val_loader, n_epochs=50):
        for epoch in range(n_epochs):
            self.model.train()
            for x_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(x_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                valid_loss = sum(self.criterion(self.model(x_val), y_val) for x_val, y_val in val_loader) / len(val_loader)
            print(f'Epoch {epoch+1}, Validation Loss: {valid_loss.item()}')

def main(args):
    X, y = load_data(args.sequence_length, args.column)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    model = GRUNet(args.input_dim, args.hidden_dim, args.output_dim, args.num_layers)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    trainer = Trainer(model, criterion, optimizer)
    trainer.train(train_loader, val_loader, args.n_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--column', type=str, default='1')
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=20)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--sequence_length', type=int, default=50)
    args = parser.parse_args()

    main(args)