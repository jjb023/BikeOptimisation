import pandas as pd

# Load the weather data
weather_data = pd.read_csv('WeatherData2023/all2023weather.csv')
# Convert the datetime column to datetime objects
weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])

# Load the bike data
bike_data = pd.read_csv('CleaningNetBikeData/sorted_data_2023.csv')
# Convert the datetime column to datetime objects
bike_data.columns = ['datetime'] + list(bike_data.columns[1:])
bike_data['datetime'] = pd.to_datetime(bike_data['datetime'], format='%Y-%m-%d %H:%M:%S')

# Creating a new column 'hourly_datetime' in bike_data for merging
bike_data['hourly_datetime'] = bike_data['datetime'].dt.floor('H')  # Round down to the nearest hour

# Merge the two dataframes on the new 'hourly_datetime' column
combined_data = pd.merge(bike_data, weather_data, left_on='hourly_datetime', right_on='datetime', how='left', suffixes=('_bike', '_weather'))

# Drop the extra 'datetime_weather' and 'hourly_datetime' columns as they are no longer needed
combined_data.drop(columns=['datetime_weather', 'hourly_datetime'], inplace=True)
combined_data.rename(columns={'datetime_bike': 'datetime'}, inplace=True)

# Save the combined DataFrame to a new CSV file
combined_data.to_csv('WeatherData2023/combined_weather_bike_data.csv', index=False)

print("Combined data saved to 'combined_weather_bike_data.csv'.")
