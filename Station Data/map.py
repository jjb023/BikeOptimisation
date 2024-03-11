import folium
import pandas as pd

# Read the CSV file
df = pd.read_csv('Station Data/longlat_data.csv')

# Create a Folium map centered around London
map_london = folium.Map(location=[51.5074, -0.1278], zoom_start=10)

# Add markers for each location
for index, row in df.iterrows():
    folium.Marker(location=[row['Latitude'], row['Longitude']], 
                  popup=row['Name']).add_to(map_london)

# Save the map as an HTML file
map_london.save('Station Data/map_london.html')
