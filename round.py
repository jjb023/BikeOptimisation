import pandas as pd

# Load the data
df = pd.read_csv('predict_week_results.csv')

# Round the values to the nearest integer
df = df.round(0)

# Save the rounded data back to the CSV file
df.to_csv('predict_week_results.csv', index=False)