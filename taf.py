import pandas as pd

# Load the data
df = pd.read_csv('2019BikeData/2019MergedBikeWeatherData.csv')

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Extract day of week and time
df['Day'] = df['Date'].dt.day_name()
df['Time'] = df['Date'].dt.strftime('%H:%M')

# Group by 'Time' and 'Day' and calculate the mean
df_mean = df.groupby(['Day', 'Time']).mean().reset_index()

# Sort the data by 'Day' and 'Time'
df_mean['Day'] = pd.Categorical(df_mean['Day'], categories=[
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
df_mean.sort_values(['Day', 'Time'], inplace=True)

# Write the results to a new CSV file
df_mean.to_csv('average_results.csv', index=False)