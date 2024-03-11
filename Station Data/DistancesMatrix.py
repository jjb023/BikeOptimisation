import pandas as pd

# Step 1: Load the CSV data
df = pd.read_csv('BikeOptimisation\Station Data\distancesID.csv')

distance_matrix = df.pivot(index='Station 1', columns='Station 2', values='Distance (km)')

for station in distance_matrix.index:
    distance_matrix.loc[station, station] = 0.0
    for col in distance_matrix.columns:
        if pd.isnull(distance_matrix.loc[station, col]) and not pd.isnull(distance_matrix.loc[col, station]):
            distance_matrix.loc[station, col] = distance_matrix.loc[col, station]

# Fill NaN values with an appropriate value if necessary. For distances, NaN might be replaced with 0 or a large number
# depending on how you want to treat "no direct path" cases.
# distance_matrix.fillna(0, inplace=True)

# Print the resulting matrix
print(distance_matrix)
distance_matrix.to_csv('distance_matrix.csv', index=True)