#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:13:36 2024

@author: edwardloughrey
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
checkouts = {station: np.random.randint(0, 40, size=timesteps) for station in stations}
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
        G.nodes[station]['bikes'] = max(current_bikes, 0)

        net_bikes.at[t, station] = G.nodes[station]['bikes']

print(net_bikes)

# Function to update the graph for each frame of the animation
def update(frame):
    plt.clf()  # Clear the previous plot
    # Draw the updated graph
    nx.draw(G, pos, with_labels=True, node_size=500, node_color=net_bikes.iloc[frame].values, cmap='coolwarm', vmin=0, vmax=40)
    # Add timestep text in the upper left corner
    plt.text(0.05, 0.95, f'Timestep: {frame}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    

# Create initial plot
pos = nx.spring_layout(G)
plt.figure(figsize=(8, 6))

# Create the animation
ani = FuncAnimation(plt.gcf(), update, frames=timesteps, interval=200)

# Save the animation as a GIF
ani.save('boris_bike_animation.gif', writer='pillow')

plt.show()  # Show the final plot

