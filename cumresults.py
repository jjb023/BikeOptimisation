import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('results.csv')

# Cumulative sum of each column
df_cumulative = df.cumsum()

# Choose the column to visualize
column = '500'  # replace with your column name

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df_cumulative[column])
plt.title('Cumulative Visualization of ' + column)
plt.xlabel('Time Step')
plt.ylabel('Cumulative Value')
plt.grid(True)
plt.show()