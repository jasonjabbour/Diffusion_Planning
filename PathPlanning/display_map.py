import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

def display_map_and_paths(pair_id):
    # Load the map matrix from the JSON file
    with open(f"map_data/map_{pair_id}.json", 'r') as json_file:
        map_matrix = np.array(json.load(json_file))

    # Load the path data from the CSV file
    path_data = pd.read_csv("path_data/path_data.csv")
    rrt_path = path_data[(path_data['pair_id'] == pair_id) & (path_data['algo'] == 'RRT')][['x', 'y']].values
    astar_path = path_data[(path_data['pair_id'] == pair_id) & (path_data['algo'] == 'A*')][['x', 'y']].values

    # Extract the obstacle locations from the map matrix
    oy, ox = np.where(map_matrix == 1)

    # Start and goal locations can also be extracted from the map matrix
    sy, sx = np.where(map_matrix == 2)
    gy, gx = np.where(map_matrix == 3)

    # Create the plot
    plt.figure(figsize=(6, 6))
    plt.plot(ox, oy, ".k", markersize=5)  # Plot obstacles as black dots
    if len(rrt_path) > 0:
        plt.plot(rrt_path[:, 0], rrt_path[:, 1], '-r', label='RRT Path')  # RRT path
    if len(astar_path) > 0:
        plt.plot(astar_path[:, 0], astar_path[:, 1], '-b', label='A* Path')  # A* path
    plt.plot(sx, sy, "og", markersize=10, label='Start')  # Start position
    plt.plot(gx, gy, "xb", markersize=10, label='Goal')  # Goal position
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(0, map_matrix.shape[1])
    plt.ylim(0, map_matrix.shape[0])
    plt.show()

# Example usage
display_map_and_paths(pair_id=6)
