import networkx as nx
import numpy as np
import pandas as pd
import os

# Initialize the graph
G = nx.Graph()

# Number of stations and initial number of bikes
num_stations = 789
initial_bikes_per_station = 0

# Add stations as nodes with initial bikes
for station_id in range(1, num_stations + 1):
    station_str_id = str(station_id)  # Convert to string for consistency with CSV data
    G.add_node(station_str_id, bikes=initial_bikes_per_station)


timesteps = 48
station_ids = [str(station_id) for station_id in range(1, num_stations + 1)]
net_bikes = pd.DataFrame(index=range(timesteps), columns=station_ids, dtype=int)


def update_graph_from_csv(file_path, G, timestep):
    df = pd.read_csv(file_path, index_col=0)
    df.index = df.index.map(str) 
    df.columns = df.columns.map(str)  

    for i in df.index:
        for j in df.columns:
            if i in G.nodes and j in G.nodes:
                bikes_to_transfer = df.at[i, j]
                G.nodes[i]['bikes'] = G.nodes[i]['bikes'] - bikes_to_transfer
                if i != j:
                    G.nodes[j]['bikes'] += bikes_to_transfer

    # Update net_bikes for the current timestep
    for station in G.nodes:
        net_bikes.at[timestep, station] = G.nodes[station]['bikes']


csv_dir_path = 'BikeOptimisation/results/results'  
csv_files = sorted(os.listdir(csv_dir_path))  


for t, csv_file in enumerate(csv_files):
    if csv_file.endswith('.csv'):  
        file_path = os.path.join(csv_dir_path, csv_file)
        update_graph_from_csv(file_path, G, t)
        print(f"Timestep {t}:")
        print(net_bikes.iloc[t, :20].to_frame().T)  
