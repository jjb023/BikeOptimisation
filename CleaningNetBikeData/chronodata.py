import pandas as pd
import csv

# Path to your data file
file_path = 'Net_bikes_per_time_step_full_year.csv'


# Read the data
with open(file_path, 'r') as file:
    lines = file.readlines()

# Create a DataFrame
data = {
    'timestamp': [],
    'values': []
}

for line in lines:
    parts = line.strip().split(',')
    if parts[0]:  # Check if the timestamp part is not empty
        data['timestamp'].append(parts[0])
        data['values'].append(parts[1:])
    else:
        continue  # Skip rows where the timestamp part is empty

df = pd.DataFrame(data)

# Handle cases where the timestamp might be malformed or empty
try:
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d: %H:%M')
except ValueError as e:
    print(f"Error parsing timestamp: {e}")
    # You could handle the error specifically here if needed

# Sort the DataFrame by timestamp
df_sorted = df.sort_values(by='timestamp')

# Save the sorted DataFrame to a new text file
sorted_file_path = 'sorted_data.csv'
with open(sorted_file_path, 'w') as file:
    for index, row in df_sorted.iterrows():
        timestamp_str = row['timestamp'].strftime('%Y-%m-%d: %H:%M')
        file.write(f"{timestamp_str}, {','.join(row['values'])}\n")

print(f"Sorted data has been saved to {sorted_file_path}")
