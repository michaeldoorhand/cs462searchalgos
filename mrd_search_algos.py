import time
from collections import deque
import heapq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def get_data():
    file1_name = "Adjacencies.txt"
    adj = {}
    try:
        with open(file1_name, 'r') as file1:
            for i in file1.readlines():
                line = i.strip()
                line = line.split(' ')    
                if line[0] in adj:
                    adj[line[0]].append(line[1])
                else:
                    adj[line[0]] = []
                    adj[line[0]].append(line[1])

                if line[1] in adj:
                    adj[line[1]].append(line[0])
                else:
                    adj[line[1]] = []
                    adj[line[1]].append(line[0])
    except FileNotFoundError:
        print(file1_name, "not found")

    sorted_adj = {}
    for items in adj.items():
        items[1].sort()


    file2_name = "Coordinates.csv"
    coords = {}
    try:
        with open(file2_name, 'r') as file2:
            for i in file2.readlines():
                line = i.strip()
                line = line.split(',')
                coords[line[0]] = [line[1],line[2]]
    except FileNotFoundError:
        print(file2_name, "not found")

    return([adj,coords])


def calculate_distance(coord1,coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    
    x1 = float(x1)
    x2 = float(x2)
    y1 = float(y1)
    y2 = float(y2)
    # Calculate Euclidean distance
    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    return distance

def calculate_total_distance(coords, path):
    total_distance = 0.0

    # Iterate through the path and calculate the distance between consecutive points
    for i in range(len(path) - 1):
        city1 = path[i]
        city2 = path[i + 1]

        # Extract coordinates for the two cities
        coord1 = coords[city1]
        coord2 = coords[city2]

        # Calculate the distance between the two cities and add it to the total
        distance = calculate_distance(coord1, coord2)
        total_distance += distance

    # Add the distance from the last city back to the starting city to complete the path
    first_city = path[0]
    last_city = path[-1]

    # Extract coordinates for the first and last cities
    first_coord = coords[first_city]
    last_coord = coords[last_city]

    # Calculate the distance from the last city back to the starting city and add it to the total
    distance = calculate_distance(first_coord, last_coord)
    total_distance += distance

    return total_distance


def plot_path(coords, adj, path, time_delay):
    # Extract x and y coordinates from the 'coords' dictionary
    x = [float(coord[0]) for coord in coords.values()]
    y = [float(coord[1]) for coord in coords.values()]

    # Initialize the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title('Path Plot', fontsize=20)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)

    # Create a scatter plot
    scatter = ax.scatter(x, y, marker='o')

    # Add labels
    for label, x_coord, y_coord in zip(coords.keys(), x, y):
        ax.annotate(label, xy=(x_coord, y_coord))

    # Create initial light grey lines connecting adjacent nodes
    for city, neighbors in adj.items():
        x_coord, y_coord = float(coords[city][0]), float(coords[city][1])
        for neighbor in neighbors:
            neighbor_x, neighbor_y = float(coords[neighbor][0]), float(coords[neighbor][1])
            plt.plot([x_coord, neighbor_x], [y_coord, neighbor_y], linestyle='-', color='lightgrey', linewidth=1)
        # Initialize an empty line for the path
        line, = ax.plot([], [], linestyle='-', color='red', linewidth=2)

    # Lists to store accumulated line data
    accumulated_x = []
    accumulated_y = []

    total_distance = 0
    # Function to update the plot with the next point in the path
    def update(frame):

            # Get the coordinates of the current city
            current_city = path[frame]
            current_coord = coords[current_city]
            x_coord, y_coord = float(current_coord[0]), float(current_coord[1])

            # Accumulate line data
            accumulated_x.append(x_coord)
            accumulated_y.append(y_coord)

            # Update the line data
            line.set_data(accumulated_x, accumulated_y)

            # Calculate the distance to the next city
            if current_city != path[-1]:
                next_city = path[frame + 1]
                next_coord = coords[next_city]
                next_x, next_y = float(next_coord[0]), float(next_coord[1])

            # Set the title to the current city name and the total distance
            ax.set_title(f'Path Plot: {current_city}')

    # Create an animation using FuncAnimation
    anim = FuncAnimation(fig, update, frames=len(path), repeat=False, interval=time_delay)

    # Show the final plot
    plt.grid(False)
    plt.show()

def brute_force(start, end, adj, coords, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []

    visited.add(start)
    path.append(start)

    if start == end:
        return True, path  # Path found

    for destination in adj[start]:
        if destination not in visited:
            found, found_path = brute_force(destination, end, adj, coords, visited, path)
            if found:
                return found, found_path

    path.pop()  # Remove the last city from the path
    return False, []
            

def breadth_first_search(start, end, adj, coords):
    # Create a queue for BFS
    queue = deque()
    # Initialize visited set and path dictionary
    visited = set()
    path = {}

    # Initialize the queue with the start node
    queue.append(start)
    visited.add(start)
    path[start] = None

    while queue:
        current_city = queue.popleft()  # Dequeue the current city

        # If we reach the end node, construct and return the path
        if current_city == end:
            # Reconstruct the path from end to start
            current = end
            result_path = []
            while current is not None:
                result_path.append(current)
                current = path[current]

            return list(reversed(result_path))  # Reverse the path to start from the start node

        for neighbor in adj[current_city]:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                path[neighbor] = current_city

    # If no path is found
    return None

def depth_first_search(start, end, adj, coords):
    # Initialize visited set and path dictionary
    visited = set()
    path = {}
    
    # Function to perform DFS recursively
    def dfs(current_city):
        visited.add(current_city)

        # If we reach the end node, construct and return the path
        if current_city == end:
            return [end]

        for neighbor in adj[current_city]:
            if neighbor not in visited:
                path[neighbor] = current_city
                result = dfs(neighbor)
                if result:
                    return [current_city] + result

        return None

    # Start DFS from the start node
    result_path = dfs(start)

    if result_path:
        return result_path  # Reverse the path to start from the start node
    else:
        return None  # If no path is found

def iterative_deepening_dfs(start, end, adj, coords):
    # Perform iterative deepening with increasing depth limits
    max_depth = len(adj)  # Maximum depth is the number of cities
    for depth_limit in range(1, max_depth + 1):
        visited = set()
        path = []
        found, found_path = iddfs_recursive(start, end, adj, coords, visited, path, depth_limit)
        if found:
            return found_path

    # If no path is found
    return None

def iddfs_recursive(current_city, end, adj, coords, visited, path, depth_limit):
    if len(path) > depth_limit:
        return False, []

    visited.add(current_city)
    path.append(current_city)

    if current_city == end:
        return True, path

    for neighbor in adj[current_city]:
        if neighbor not in visited:
            found, found_path = iddfs_recursive(neighbor, end, adj, coords, visited, path, depth_limit)
            if found:
                return found, found_path

    path.pop()
    return False, []


def best_first_search(start, end, adj, coords):
    # Initialize a priority queue (min-heap) for BFS
    priority_queue = [(calculate_distance(coords[start], coords[end]), start)]
    visited = set()
    path = {}

    while priority_queue:
        _, current_city = heapq.heappop(priority_queue)  # Dequeue the city with the lowest estimated distance

        if current_city in visited:
            continue

        visited.add(current_city)

        if current_city == end:
            # Reconstruct the path from end to start
            result_path = [end]
            while result_path[-1] != start:
                previous_city = path[result_path[-1]]
                result_path.append(previous_city)
            result_path.reverse()
            return result_path

        for neighbor in adj[current_city]:
            if neighbor not in visited:
                heapq.heappush(priority_queue, (calculate_distance(coords[neighbor], coords[end]), neighbor))
                path[neighbor] = current_city

    # If no path is found
    return None

def astar_search(start, end, adj, coords):
    def heuristic(city):
        # Calculate the Euclidean distance from 'city' to the 'end' city
        return calculate_distance(coords[city], coords[end])

    # Initialize the priority queue (min-heap) for A* search
    priority_queue = [(0 + heuristic(start), 0, start)]  # Initial cost, g(n), is 0
    visited = set()
    g_values = {city: float('inf') for city in adj}
    g_values[start] = 0
    path = {}

    while priority_queue:
        _, cost, current_city = heapq.heappop(priority_queue)  # Dequeue the city with the lowest cost

        if current_city in visited:
            continue

        visited.add(current_city)

        if current_city == end:
            # Reconstruct the path from end to start
            result_path = [end]
            while result_path[-1] != start:
                previous_city = path[result_path[-1]]
                result_path.append(previous_city)
            result_path.reverse()
            return result_path

        for neighbor in adj[current_city]:
            if neighbor not in visited:
                tentative_g_value = g_values[current_city] + calculate_distance(coords[current_city], coords[neighbor])

                if tentative_g_value < g_values[neighbor]:
                    g_values[neighbor] = tentative_g_value
                    heapq.heappush(priority_queue, (tentative_g_value + heuristic(neighbor), tentative_g_value, neighbor))
                    path[neighbor] = current_city

    # If no path is found
    return None


def main():
    data = get_data()
    adj = data[0]
    coords = data[1]
    locations = coords.keys()
    running = True
    while running:
        print('\n')
        print('Eligible locations: ')
        for location in coords.keys():
            print(location, end=' ')
        print('\n')

        print('Search Algorithms: ')
        print('0| Use all algorithms and compare results (no graphing)')
        print('1| Blind Brute Force')
        print('2| Breadth-first Search')
        print('3| Depth-first Search')
        print('4| ID-DFS Search')
        print('5| Best First Search')
        print('6| A* Search')
        print('')
        user_input = False
        while user_input == False:
            start = input('Please choose a starting location: ')
            if start not in adj.keys():
                print('Incorrect start destination inputted, please try again')
                continue
            end = input('Please choose an ending location: ')
            if end not in adj.keys():
                print('Incorrect end destination inputted, please try again')
                continue
            algo = int(input('Please choose a search algorithm (choose number): '))
            if algo < 0 or algo > 6:
                print('Incorrect algorithm selected, please try again')
                continue
            user_input = True
            print('')
 

        match algo:
            case 0:
                algorithms = [("Blind Brute Force", brute_force),("Breadth-first Search", breadth_first_search), ("Depth-first Search", depth_first_search),("ID-DFS Search", iterative_deepening_dfs), ("Best First Search", best_first_search),("A* Search", astar_search)]
                algos_results = {}
                for algo_name, algo_function in algorithms:
                    start_time = time.perf_counter()
                    results = algo_function(start, end, adj, coords)
                    end_time = time.perf_counter()
                    elapsed_time_ms = (end_time - start_time) * 1000
                    if algo_name == 'Blind Brute Force':
                        path = results[1]
                    else:
                        path = results
                    algos_results[algo_name] = [path,elapsed_time_ms,calculate_total_distance(coords,path)]
                    
            case 1:
                start_time = time.perf_counter()
                results = brute_force(start, end, adj, coords)
            case 2:
                start_time = time.perf_counter()
                results = breadth_first_search(start, end, adj, coords)
            case 3:
                start_time = time.perf_counter()
                results = depth_first_search(start, end, adj, coords)
            case 4:
                start_time = time.perf_counter()
                results = iterative_deepening_dfs(start, end, adj, coords)
            case 5:
                start_time = time.perf_counter()
                results = best_first_search(start, end, adj, coords)
            case 6:
                start_time = time.perf_counter()
                results = astar_search(start, end, adj, coords)
        
        if algo == 0:
            for algo_name, (path, elapsed_time_ms, total_distance) in algos_results.items():
                print(f'{algo_name} took {elapsed_time_ms:.4f}ms, Cities visited = {len(path)}, Distanced traveled = {total_distance:.2f}, and had the following path: ')
                print(path)
                print('________________________________________________________')

            sorted_results = sorted(algos_results.items(), key=lambda x: x[1][1])

            print('\n')
            print('')
            for algo_name, (path, elapsed_time_ms, total_distance) in sorted_results:
                print(f"{algo_name} took {elapsed_time_ms:.4f}ms to complete. Cities visited = {len(path)}, Distance traveled = {total_distance:.2f} ")
        else:
            if algo == 1:
                path = results[1]
            else:
                path = results

            print(path)
            #print(adj)
            end_time = time.perf_counter()
            
            # Calculate the elapsed time in milliseconds
            elapsed_time_ms = (end_time - start_time) * 1000      
            print(f"Algorithm took {elapsed_time_ms:.2f}ms to complete. Cities visited = {len(path)}, Distance traveled = {calculate_total_distance(coords, path):.2f} ")
            plot_path(coords,adj,path,1000)

        print('')
        print('Run again (Y)     Quit (X)')
        end_choice = input()


        if end_choice == 'X':
            running = False

main()
