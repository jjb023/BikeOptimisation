import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('results.csv')

# Choose the column to visualize
column = '4'  # replace with your column name

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df[column])
plt.title('Visualization of ' + column)
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.grid(True)
plt.show()