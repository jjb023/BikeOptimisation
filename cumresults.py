import pandas as pd
import matplotlib.pyplot as plt

# Choose the column to visualize
column = '6'  # replace with your column name

plt.figure(figsize=(10, 6))

# Loop over each day of the week
for day_of_week in range(7):
    # Load the data
    df = pd.read_csv(f'WeekResults_Day{day_of_week}.csv')

    # Cumulative sum of the chosen column
    df_cumulative = df[column].cumsum()

    # Plot the data
    plt.plot(df_cumulative, label=f'Day {day_of_week}')

plt.title('Cumulative Visualization of ' + column)
plt.xlabel('Time Step')
plt.ylabel('Cumulative Value')
plt.grid(True)
plt.legend()  # Add a legend
plt.show()