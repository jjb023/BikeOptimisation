import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('predict_week_results.csv')


# Choose the column to visualize
column = '250'  # replace with your column name

# Plot the cumulative results for each day
plt.figure(figsize=(10, 6))
for day in df['Day'].unique():
    df_day = df[df['Day'] == day]
    df_day[column].cumsum().reset_index(drop=True).plot(label=day)

plt.title('Cumulative Results Over Each Day for ' + column)
plt.xlabel('Time Step')
plt.ylabel('Cumulative Value')
plt.grid(True)
plt.legend()  # Add a legend
plt.tight_layout()  # Adjust subplot parameters to give specified padding
plt.savefig('cumulative_results_plot.png')  # Save the plot to a file
plt.show()