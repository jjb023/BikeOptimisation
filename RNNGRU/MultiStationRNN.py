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
from datetime import datetime, timedelta

df = pd.read_csv('2019BikeData/2019MergedBikeWeatherData.csv', index_col='Date', parse_dates=True)

def load_data(sequence_length, day_of_week, time_start, time_end):
    df = pd.read_csv('2019BikeData/2019MergedBikeWeatherData.csv', index_col='Date', parse_dates=True)
    df = df[df.index.dayofweek == day_of_week]  # filter data for the same day of the week
    df = df.between_time(time_start, time_end)  # filter data for the same time frame
    df = df.diff()  # calculate net change for all columns
    df = df.dropna()  # remove rows with NaN values
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.values)

    X, y = [], []
    for i in range(len(df_scaled) - sequence_length):
        X.append(df_scaled[i:i+sequence_length])
        y.append(df_scaled[i+sequence_length])  # predict the net change of all columns
    return np.array(X), np.array(y)

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)  # output_dim now represents the number of columns
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
    # Create a list of 15-minute intervals in a 24-hour period
    time_intervals = [datetime.strftime(datetime.strptime(str(hour), "%H") + timedelta(minutes=15 * i), "%H:%M") for hour in range(24) for i in range(4)]



    for day_of_week in range(7):  # loop over each day of the week
        # Create a DataFrame to store the results for the current day
        results_df = pd.DataFrame(index=time_intervals, columns=df.columns)
        for time_interval in time_intervals:  # loop over each 15-minute interval
            # Load the data for the current day of the week and time interval
            X, y = load_data(args.sequence_length, day_of_week, time_interval, (datetime.strptime(time_interval, "%H:%M") + timedelta(minutes=15)).strftime("%H:%M"))
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

            test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
            test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
            y_pred = evaluate(model, test_loader)

            # Store the results in the DataFrame
            results_df.loc[time_interval] = y_pred
        # Write the results for the current day to a CSV file
        results_df.to_csv(f'WeekResults_Day{day_of_week}.csv')

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
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    presnext = 5*y_pred[-1]
    return presnext


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--column', type=str, default='1')
    parser.add_argument('--input_dim', type=int, default=811)
    parser.add_argument('--hidden_dim', type=int, default=20)
    parser.add_argument('--output_dim', type=int, default=811)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--n_epochs', type=int, default=40)
    parser.add_argument('--sequence_length', type=int, default=50)
    args = parser.parse_args()
    main(args)

# # Create a list of 15-minute intervals in a 24-hour period
# time_intervals = [datetime.strftime(datetime.strptime(str(hour), "%H") + timedelta(minutes=15 * i), "%H:%M") for hour in range(24) for i in range(4)]

# # Create a DataFrame to store the results
# results_df = pd.DataFrame(index=time_intervals, columns=df.columns)

# for time_interval in time_intervals:
#     # Filter the data for the current time interval
#     df = df.between_time(time_interval, (datetime.strptime(time_interval, "%H:%M") + timedelta(minutes=15)).strftime("%H:%M"))
    
#     # Train the model and predict the net change for the next time step
#     X, y = load_data(args.sequence_length)
#     X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

#     train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
#     val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
#     train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

#     model = GRUNet(args.input_dim, args.hidden_dim//2, args.output_dim, args.num_layers//2)
#     optimizer = optim.Adam(model.parameters(), lr=0.0001)
#     criterion = nn.MSELoss()

#     trainer = Trainer(model, criterion, optimizer)
#     trainer.train(train_loader, val_loader, args.n_epochs)

#     test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
#     test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
#     y_pred = evaluate(model, test_loader)

#     # Store the results in the DataFrame
#     results_df.loc[time_interval] = y_pred

# # Write the results to a CSV file
# results_df.to_csv('results.csv')
