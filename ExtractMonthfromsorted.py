import pandas as pd

# Load data
df = pd.read_csv("WeatherData2023/combined_weather_bike_data.csv", parse_dates=['datetime'])

# Filter data for January
january_data = df[df['datetime'].dt.month == 1]

# Save the filtered data to a new CSV file
january_data.to_csv("january_data.csv", index=False)
