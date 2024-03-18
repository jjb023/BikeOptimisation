
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load your preprocessed data (assuming it's in a pandas DataFrame)
# df = pd.read_csv('combined_10Jan-31Mar2016.csv')

# Define sequence length and number of features
sequence_length = 10  # Length of the input sequence
num_features = 1  # Number of features (e.g., demand for bikes)

# Convert DataFrame to numpy array
data = df['demand'].values.reshape(-1, 1)  # Assuming 'demand' is your target variable

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Function to create sequences for the RNN
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Create sequences for training
X, y = create_sequences(data_normalized, sequence_length)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define RNN architecture
class SimpleRNN(tf.keras.layers.Layer):
    def __init__(self, units):
        super(SimpleRNN, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.U = self.add_weight(shape=(self.units, self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        output = []
        state = tf.zeros(shape=(inputs.shape[0], self.units))
        for i in range(inputs.shape[1]):
            state = tf.tanh(tf.matmul(inputs[:, i, :], self.W) + tf.matmul(state, self.U) + self.b)
            output.append(state)
        return tf.stack(output, axis=1)

# Instantiate the model
model = SimpleRNN(units=50)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError()

# Train the model
for epoch in range(100):
    with tf.GradientTape() as tape:
        predictions = model(X_train)
        loss = loss_fn(y_train, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')

# Make predictions
train_predictions = model(X_train)
test_predictions = model(X_test)

# Inverse transform predictions
train_predictions = scaler.inverse_transform(train_predictions.numpy().reshape(-1, 1))
test_predictions = scaler.inverse_transform(test_predictions.numpy().reshape(-1, 1))

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(data[sequence_length:len(train_predictions) + sequence_length], train_predictions))
test_rmse = np.sqrt(mean_squared_error(data[-len(test_predictions):], test_predictions))
print(f'Training RMSE: {train_rmse}, Testing RMSE: {test_rmse}')
