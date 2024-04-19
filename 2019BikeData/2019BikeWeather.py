import pandas as pd

# Load the datasets
bike_data = pd.read_csv('2019BikeData/2019SortedNetBikeData.csv')
weather_data = pd.read_csv('2019WeatherData/2019combined_weather_data.csv')

# Strip any leading/trailing spaces in column names
bike_data.columns = bike_data.columns.str.strip()
weather_data.columns = weather_data.columns.str.strip()

# Convert the datetime columns
bike_data['Date'] = pd.to_datetime(bike_data['Date'])
weather_data.rename(columns={'datetime': 'Date'}, inplace=True)
weather_data['Date'] = pd.to_datetime(weather_data['Date'])

# Drop duplicates in the weather data to ensure the index for reindexing is unique
weather_data.drop_duplicates(subset='Date', keep='first', inplace=True)

# Set the Date column as the index for reindexing
weather_data.set_index('Date', inplace=True)

# Create a new index that includes all 15-minute intervals within the range of the existing data
new_index = pd.date_range(start=weather_data.index.min(), end=weather_data.index.max() + pd.Timedelta(hours=1), freq='15T')

# Reindex the DataFrame using forward fill to propagate the last valid observation forward
weather_data = weather_data.reindex(new_index, method='ffill').reset_index().rename(columns={'index': 'Date'})

# Merge the datasets on the 'Date' column
merged_data = pd.merge(bike_data, weather_data, on='Date', how='left')

# Fill any NA values
merged_data['temp'].fillna(method='ffill', inplace=True)
merged_data['precip'].fillna(method='ffill', inplace=True)

# Save the merged dataset
merged_data.to_csv('2019MergedBikeWeatherData.csv', index=False)

