import pandas as pd

# Read the file
df = pd.read_csv('CleaningNetBikeData/Net_bikes_per_time_step_full_year.csv')  # replace 'your_file.csv' with your file path

# Get the number of columns
num_columns = df.shape[1]

print(f'The file has {num_columns} columns.')