import csv
from datetime import datetime

# Function to read data from CSV file
def read_csv(filename):
    data = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header
        for row in csvreader:
            data.append(row)
    return data

# Function to combine and sort data
def combine_and_sort_data(file1, file2):
    data1 = read_csv(file1)
    data2 = read_csv(file2)
    
    # Combine data
    combined_data = data1 + data2
    
    # Sort combined data by datetime
    combined_data.sort(key=lambda x: datetime.strptime(x[0], '%Y-%m-%dT%H:%M:%S'))
    
    return combined_data

# Function to write data to CSV file
def write_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['datetime', 'temp', 'precip'])  # Header
        csvwriter.writerows(data)

# Combine and sort the data
combined_data = combine_and_sort_data("WeatherData/all_combined_weather_data.csv", "WeatherData/8augweatherdata.csv")

# Write the combined and sorted data to a new CSV file
write_csv(combined_data, "WeatherData/all_combined_weather_data.csv")

print("Combined and sorted data has been written to combined_weather_data.csv")
