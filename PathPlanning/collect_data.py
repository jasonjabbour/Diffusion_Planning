import os
import csv
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from rrt import RRT  # Your RRT implementation
from a_star import AStarPlanner  # Your A* implementation

DISPLAY = False
COLLECT_DATA = True

def generate_obstacles(grid_max_x, grid_max_y):
    ox, oy = [], []

    # Basic border walls
    for i in range(0, grid_max_x):
        ox.append(i)
        oy.append(0.0)
    for i in range(0, grid_max_y):
        ox.append(grid_max_x - 1)
        oy.append(i)
    for i in range(0, grid_max_x):
        ox.append(i)
        oy.append(grid_max_y - 1)
    for i in range(0, grid_max_y):
        ox.append(0.0)
        oy.append(i)

    # Random obstacles
    num_obstacles = np.random.randint(10, 20)
    for _ in range(num_obstacles):
        shape_type = np.random.choice(['block', 'line'])
        x = np.random.randint(10, grid_max_x - 10)
        y = np.random.randint(10, grid_max_y - 10)

        if shape_type == 'block':
            max_block_size_x = min(grid_max_x - x, 20)
            max_block_size_y = min(grid_max_y - y, 20)
            block_size = np.random.randint(5, min(max_block_size_x, max_block_size_y))
            for bx in range(x, x + block_size):
                for by in range(y, y + block_size):
                    ox.append(bx)
                    oy.append(by)

        elif shape_type == 'line':
            if np.random.rand() > 0.5:  # horizontal line
                length = min(np.random.randint(10, 50), grid_max_x - x)
                for i in range(length):
                    ox.append(x + i)
                    oy.append(y)
            else:  # vertical line
                length = min(np.random.randint(10, 50), grid_max_y - y)
                for i in range(length):
                    ox.append(x)
                    oy.append(y + i)

    return ox, oy

def generate_random_point(min_value, max_value):
    return np.random.randint(min_value, max_value), np.random.randint(min_value, max_value)

def is_distance_sufficient(sx, sy, gx, gy, min_distance):
    return np.hypot(gx - sx, gy - sy) >= min_distance

def save_grid_image_and_data(ox, oy, sx, sy, gx, gy, pair_id, grid_size=100):
    # Ensure the directories exist
    os.makedirs("map_data", exist_ok=True)
    os.makedirs("map_images", exist_ok=True)

    # Initialize a matrix for the map
    map_matrix = np.zeros((grid_size, grid_size), dtype=int)

    # Mark obstacles in the matrix
    for x, y in zip(ox, oy):
        if 0 <= x < grid_size and 0 <= y < grid_size:
            map_matrix[int(y)][int(x)] = 1  # Mark obstacle

    # Mark start and goal positions
    map_matrix[int(sy)][int(sx)] = 2  # Mark start
    map_matrix[int(gy)][int(gx)] = 3  # Mark goal

    # Save the matrix as a JSON file
    with open(f"map_data/map_{pair_id}.json", 'w') as json_file:
        json.dump(map_matrix.tolist(), json_file)

    # Create and save the grid image
    plt.figure(figsize=(6, 6))
    # Plot the obstacles
    plt.plot(ox, oy, ".k")  # Obstacles are plotted as black dots
    # Plot the start and goal positions
    plt.plot(sx, sy, "og", label='Start')  # Start position
    plt.plot(gx, gy, "xb", label='Goal')  # Goal position
    # plt.legend()
    # plt.grid(True)
    plt.axis('equal')
    plt.xlim(0, grid_size)
    plt.ylim(0, grid_size)
    # Save the figure without displaying
    plt.savefig(f"map_images/map_{pair_id}.png")
    plt.close()


def save_paths(rrt_path, astar_path, sx, sy, ox, oy, gx, gy, grid_size):

    # Check for valid paths
    if rrt_path and astar_path and len(astar_path[0]) > 1:
        # Determine the next pair ID
        pair_id = 1
        csv_file_name = os.path.join('path_data','path_data.csv')
        write_header = not os.path.exists(csv_file_name)
        
        if not write_header:
            with open(csv_file_name, mode='r') as infile:
                reader = csv.reader(infile)
                existing_rows = list(reader)
                if existing_rows:
                    last_row = existing_rows[-1]
                    pair_id = int(last_row[0]) + 1

        with open(csv_file_name, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Write header if file is new
            if write_header:
                writer.writerow(["pair_id", "algo", "x", "y"])

            # Reverse RRT path if it starts with the goal
            if rrt_path[0] != [sx, sy]:
                rrt_path.reverse()

            # Reverse A* path if it ends with the start
            if astar_path[0][-1] == sx and astar_path[1][-1] == sy:
                astar_path = (astar_path[0][::-1], astar_path[1][::-1])

            # Write RRT paths
            for point in rrt_path:
                writer.writerow([pair_id, 'RRT', f"{point[0]:.2f}", f"{point[1]:.2f}"])

            # Write A* paths
            for x, y in zip(astar_path[0], astar_path[1]):
                writer.writerow([pair_id, 'A*', f"{x:.2f}", f"{y:.2f}"])

        save_grid_image_and_data(ox, oy, sx, sy, gx, gy, pair_id, grid_size)

def main():
    num_environments = 2000  # Number of environments to generate
    grid_max_x, grid_max_y = 100, 100  # Size of the grid area
    min_distance = 50.0  # Minimum distance between start and goal
    robot_radius = 1.0  # Robot radius for RRT

    for i in range(num_environments):

        print(f"**Sample: {i}")

        ox, oy = generate_obstacles(grid_max_x, grid_max_y)

        # Generate random start and goal positions with minimum distance
        while True:
            sx, sy = generate_random_point(1, grid_max_x - 1)
            gx, gy = generate_random_point(1, grid_max_y - 1)
            if is_distance_sufficient(sx, sy, gx, gy, min_distance):
                break

        # RRT Planning
        start_time = time.time()
        rrt = RRT(
            start=[sx, sy],
            goal=[gx, gy],
            rand_area=[0, grid_max_x],
            obstacle_list=list(zip(ox, oy, [robot_radius] * len(ox))),  # Convert to format required by RRT
            robot_radius=robot_radius, 
            max_iter=10000
        )
        rrt_path = rrt.planning(animation=False)
        end_time = time.time()
        rrt_planning_time = end_time - start_time
        
        # print(f"The rrt planning time: {rrt_planning_time}")

        # A* Planning
        grid_size = 1.0  # Grid resolution
        start_time = time.time()
        a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
        astar_path = a_star.planning(sx, sy, gx, gy)
        end_time = time.time()
        astar_planning_time = end_time - start_time

        # print(f"The A* planning time: {astar_planning_time}")

        if DISPLAY:
            # Plotting
            plt.figure()
            plt.plot(ox, oy, ".k")
            if rrt_path:
                plt.plot([x for (x, y) in rrt_path], [y for (x, y) in rrt_path], '-r', label='RRT Path')
            if astar_path:
                plt.plot(astar_path[0], astar_path[1], '-b', label='A* Path')
            plt.plot(sx, sy, "og", label='Start')
            plt.plot(gx, gy, "xb", label='Goal')
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            plt.title(f"RRT vs A* - Environment {i+1}")
            plt.pause(1)
        
        
        if COLLECT_DATA:
            save_paths(rrt_path, astar_path, sx, sy, ox, oy, gx, gy, grid_max_x)


    if DISPLAY:
        plt.show()

if __name__ == '__main__':
    main()

