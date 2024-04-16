import pandas as pd
from datetime import datetime

# Define the start and end months
months = ['1Jan', '2Feb', '3Mar', '4Apr', '5May', '6Jun', '7Jul', '8Aug', '9Sep', '10Oct', '11Nov', '12Dec']
year = '19'  # Suffix for the year

# Generate file names
file_names = [f"2019WeatherData/{month}{year}weatherdata.csv" for month in months]

# List to hold dataframes
dfs = []

# Read each file and append to the list
for file_name in file_names:
    try:
        df = pd.read_csv(file_name)
        dfs.append(df)
    except FileNotFoundError:
        print(f"File {file_name} not found. Skipping...")

# Concatenate all dataframes
final_df = pd.concat(dfs, ignore_index=True)

# Delete repeated rows
final_df.drop_duplicates(inplace=True)

# Save the concatenated DataFrame to a new CSV file
final_df.to_csv('2019WeatherData/2019combined_weather_data.csv', index=False)

print("All files have been concatenated and saved to combined_weather_data.csv.")