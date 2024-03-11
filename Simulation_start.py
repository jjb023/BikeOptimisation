import networkx as nx
import numpy as np
import pandas as pd

# Initialize the graph
G = nx.Graph()

# Station identifiers and initial bikes
stations = ['A', 'B', 'C', 'D']
initial_bikes = {'A': 10, 'B': 15, 'C': 20, 'D': 25}

# Add stations as nodes with initial bikes
for station, bikes in initial_bikes.items():
    G.add_node(station, bikes=bikes)

# Simulate journeys for simplicity (this would be based on your data)
timesteps = 100  # Total number of timesteps
np.random.seed(0)  # For reproducibility
checkouts = {station: np.random.randint(0, 47, size=timesteps) for station in stations}
returns = {station: np.random.randint(0, 40, size=timesteps) for station in stations}

# DataFrame to track the net number of bikes (optional, for visualization)
net_bikes = pd.DataFrame(index=range(timesteps), columns=stations, dtype=int)

# Initialize net_bikes DataFrame with initial values
for station in stations:
    net_bikes[station][0] = initial_bikes[station]

# Simulate bike movements
for t in range(1, timesteps):
    for station in stations:
        # Update bikes based on checkouts and returns
        bikes_checkout = checkouts[station][t]
        bikes_return = returns[station][t]
        current_bikes = G.nodes[station]['bikes'] - bikes_checkout + bikes_return
        
        # Ensure bikes don't fall below zero
        G.nodes[station]['bikes'] =  current_bikes

        net_bikes.at[t, station] = G.nodes[station]['bikes']

print(net_bikes)
