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
COLLECT_PAIRS = False # Will collect both RRT and A* for a map
VERBOSE = False
DATASET_NUMBER = 2

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
    num_obstacles = np.random.randint(2, 6)  # Randomly decide how many obstacles to generate
    for _ in range(num_obstacles):
        # Randomly choose the type of obstacle: 'block', 'line', 'center', or 'jagged corridor'
        shape_type = np.random.choice(['block', 'line', 'center', 'jagged_corridor']) 

        # Randomly select the starting position of the obstacle
        # x and y can potentially place the obstacle near the edges of the grid
        x = np.random.randint(0, grid_max_x)
        y = np.random.randint(0, grid_max_y)

        if shape_type == 'block':
            # Generate a rectangular block of random size
            block_size_x = np.random.randint(20, 50)  # Random width
            block_size_y = np.random.randint(20, 50)  # Random height
            for bx in range(x, x + block_size_x):
                for by in range(y, y + block_size_y):
                    # Add points to the obstacle list, clipping to ensure they stay within bounds
                    if 0 <= bx < grid_max_x and 0 <= by < grid_max_y:
                        ox.append(bx)
                        oy.append(by)

        elif shape_type == 'line':
            # Generate a thick horizontal or vertical line
            thickness = np.random.randint(15, 40)  # Random line thickness

            if np.random.rand() > 0.5:  # Horizontal line
                line_length = np.random.randint(10, 50)  # Random line length
                for offset in range(-thickness // 2, thickness // 2 + 1):
                    for i in range(line_length):
                        # Add points to the obstacle list, clipping to ensure they stay within bounds
                        bx, by = x + i, y + offset
                        if 0 <= bx < grid_max_x and 0 <= by < grid_max_y:
                            ox.append(bx)
                            oy.append(by)
            else:  # Vertical line
                line_length = np.random.randint(10, 50)  # Random line length
                for offset in range(-thickness // 2, thickness // 2 + 1):
                    for i in range(line_length):
                        # Add points to the obstacle list, clipping to ensure they stay within bounds
                        bx, by = x + offset, y + i
                        if 0 <= bx < grid_max_x and 0 <= by < grid_max_y:
                            ox.append(bx)
                            oy.append(by)

        elif shape_type == 'center':
                # Generate 2-3 large blocks near the center of the map
                center_x, center_y = grid_max_x // 2, grid_max_y // 2
                num_blocks = np.random.randint(2, 4)  # 2 or 3 blocks
                for _ in range(num_blocks):
                    # Random offset around the center
                    offset_x = np.random.randint(-20, 20)
                    offset_y = np.random.randint(-20, 20)
                    block_size_x = np.random.randint(30, 50)  # Random block width
                    block_size_y = np.random.randint(30, 50)  # Random block height
                    start_x = center_x + offset_x
                    start_y = center_y + offset_y
                    for bx in range(start_x, start_x + block_size_x):
                        for by in range(start_y, start_y + block_size_y):
                            # Add points to the obstacle list, clipping to ensure they stay within bounds
                            if 0 <= bx < grid_max_x and 0 <= by < grid_max_y:
                                ox.append(bx)
                                oy.append(by)

        elif shape_type == 'jagged_corridor':
            # Generate a jagged corridor through the map
            corridor_length = np.random.randint(5, 10)  # Number of blocks creating corridor
            current_x, current_y = np.random.randint(0, grid_max_x), np.random.randint(0, grid_max_y)

            for _ in range(corridor_length):
                # Randomly shift direction to create jaggedness
                dx, dy = np.random.choice([-5, 0, 5]), np.random.choice([-5, 0, 5])
                current_x += dx
                current_y += dy

                # Create an obstacle block around the current point
                corridor_width = np.random.randint(5, 10)  # Width of each block in corridor
                for i in range(-corridor_width, corridor_width + 1):
                    for j in range(-corridor_width, corridor_width + 1):
                        bx, by = current_x + i, current_y + j
                        if 0 <= bx < grid_max_x and 0 <= by < grid_max_y:
                            ox.append(bx)
                            oy.append(by)

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
    with open(f"dataset_{DATASET_NUMBER}/map_data/map_{pair_id}.json", 'w') as json_file:
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
    plt.savefig(f"dataset_{DATASET_NUMBER}/map_images/map_{pair_id}.png")
    plt.close()


def save_paths(rrt_path, astar_path, sx, sy, ox, oy, gx, gy, grid_size, rrt_planning_time, astar_planning_time):


    # Validate paths based on collect_pairs flag
    valid_astar_path = astar_path and len(astar_path[0]) > 1
    valid_rrt_path = rrt_path is not None
    valid_paths = (COLLECT_PAIRS and valid_rrt_path and valid_astar_path) or (not COLLECT_PAIRS and valid_astar_path)

    if not valid_paths:
        print("NOT VALID PATH")
        return False # Skip saving if paths are not valid

    # Determine the next pair ID
    pair_id = 1
    csv_file_name = os.path.join(f'dataset_{DATASET_NUMBER}','path_data','path_data.csv')
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
            writer.writerow(["pair_id", "algo", "x", "y", "planning_time"])

        if COLLECT_PAIRS and valid_rrt_path:
            # Reverse RRT path if it starts with the goal
            if rrt_path[0] != [sx, sy]:
                rrt_path.reverse()

            # Write RRT paths
            for point in rrt_path:
                writer.writerow([pair_id, 'RRT', f"{point[0]:.2f}", f"{point[1]:.2f}", f"{rrt_planning_time:.4f}"])

        if valid_astar_path:
            # Reverse A* path if it ends with the start
            if astar_path[0][-1] == sx and astar_path[1][-1] == sy:
                astar_path = (astar_path[0][::-1], astar_path[1][::-1])

            # Write A* paths
            for x, y in zip(astar_path[0], astar_path[1]):
                writer.writerow([pair_id, 'A*', f"{x:.2f}", f"{y:.2f}", f"{astar_planning_time:.4f}"])

    save_grid_image_and_data(ox, oy, sx, sy, gx, gy, pair_id, grid_size)

    return True

def main():
    num_environments = 3000  # Number of environments to generate
    grid_max_x, grid_max_y = 100, 100  # Size of the grid area
    min_distance = 70.0  # Minimum distance between start and goal
    robot_radius = 5.0  # Robot radius
    env_count = 0 

    while env_count < num_environments:

        print(f"**Sample: {env_count}")

        ox, oy = generate_obstacles(grid_max_x, grid_max_y)

        # Generate random start and goal positions with minimum distance
        while True:
            sx, sy = generate_random_point(robot_radius+2, grid_max_x - robot_radius -1)
            gx, gy = generate_random_point(robot_radius+2, grid_max_y - robot_radius -1)
            if is_distance_sufficient(sx, sy, gx, gy, min_distance):
                break


        rrt_path = None
        rrt_planning_time = -1
        if COLLECT_PAIRS:
            # RRT Planning
            rrt = RRT(
                start=[sx, sy],
                goal=[gx, gy],
                rand_area=[0, grid_max_x],
                obstacle_list=list(zip(ox, oy, [robot_radius] * len(ox))),  # Convert to format required by RRT
                robot_radius=robot_radius, 
                max_iter=10000
            )
            start_time = time.time()
            rrt_path = rrt.planning(animation=False)
            end_time = time.time()
            rrt_planning_time = end_time - start_time
            
            if VERBOSE:
                print(f"The rrt planning time: {rrt_planning_time}")

        # A* Planning
        grid_size = 1  # Grid resolution
        a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
        start_time = time.time()
        astar_path = a_star.planning(sx, sy, gx, gy)
        end_time = time.time()
        astar_planning_time = end_time - start_time

        if VERBOSE:
            print(f"The A* planning time: {astar_planning_time}")

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
            plt.title(f"RRT vs A* - Environment {env_count+1}")
            plt.pause(1)
        
        
        if COLLECT_DATA:
            collected_env = save_paths(rrt_path, astar_path, sx, sy, ox, oy, gx, gy, grid_max_x, rrt_planning_time, astar_planning_time)
            if collected_env:
                env_count+=1
                print(f"Environment Count: {env_count}")

    if DISPLAY:
        plt.show()

if __name__ == '__main__':
    main()

