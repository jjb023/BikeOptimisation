import folium
import pandas as pd
from folium.plugins import BeautifyIcon

# Read the CSV file
df = pd.read_csv('Station Data/longlat_data.csv')

# Create a Folium map centered around London
map_london = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles='Stamen Terrain')

# Add markers for each location
for index, row in df.iterrows():
    # Customize marker icon
    icon = BeautifyIcon(
        icon='fas fa-map-marker',
        icon_shape='circle-dot',
        border_color='blue',
        border_width=2,
        text_color='blue',
        inner_icon_style='margin-top:0px;',
        background_color='transparent'
    )
    folium.Marker(location=[row['Latitude'], row['Longitude']], 
                  popup=row['Name'],
                  icon=icon).add_to(map_london)

# Save the map as an HTML file
map_london.save('Station Data/map_london.html')
