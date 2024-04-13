import pandas as pd

# Path to your data file
file_path = 'CleaningNetBikeData/sorted_data.csv'

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

# Filter out data not from 2023
df_2023 = df_sorted[df_sorted['timestamp'].dt.year == 2023]

# Prepare DataFrame for CSV output
# This involves converting the list of values back into separate columns
df_expanded = df_2023['values'].apply(pd.Series)
df_final = pd.concat([df_2023['timestamp'], df_expanded], axis=1)

# Save the filtered DataFrame to a new CSV file
sorted_file_path = 'sorted_data_2023.csv'
df_final.to_csv(sorted_file_path, index=False, header=False)

print(f"Data from 2023 has been saved to {sorted_file_path}")
