import random
import sys
import time
import array
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

        print(city1 + ' to ' + city2)
        distance = calculate_distance(city1,city2,coords)

        total_distance += distance

    # Add the distance from the last city back to the starting city to complete the path
    first_city = path[0]
    last_city = path[-1]

    distance = calculate_distance(first_city, last_city, coords)

    total_distance += distance

    return total_distance


def plot_path(coords, path, time_delay):
    # Extract x and y coordinates from the 'coords' dictionary
    x = [float(coord[0]) for coord in coords.values()]
    y = [float(coord[1]) for coord in coords.values()]

    # Initialize the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('Path Plot', fontsize=20)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)

    # Create a scatter plot
    scatter = ax.scatter(x, y, marker='o')

    # Add labels
    for label, x_coord, y_coord in zip(coords.keys(), x, y):
        ax.annotate(label, xy=(x_coord, y_coord))

    # Initialize an empty line for the path
    line, = ax.plot([], [], linestyle='-', color='red', linewidth=2)

    # Lists to store accumulated line data
    accumulated_x = []
    accumulated_y = []

    total_distance = 0
    # Function to update the plot with the next point in the path
    def update(frame):
            nonlocal total_distance  # Declare total_distance as nonlocal to modify it within the function

            # Get the coordinates of the current city
            current_city = path[frame]
            print('current city is ' + current_city)

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
                distance = calculate_distance((x_coord, y_coord), (next_x, next_y))
                total_distance += distance

            # Set the title to the current city name and the total distance
            ax.set_title(f'Path Plot: {current_city} - Total Distance: {total_distance:.2f}')

    # Create an animation using FuncAnimation
    anim = FuncAnimation(fig, update, frames=len(path), repeat=False, interval=time_delay)

    # Show the final plot
    plt.grid(True)
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
            

def main():
    data = get_data()
    adj = data[0]
    coords = data[1]
    locations = coords.keys()
    print('\n')
    print('Eligible locations: ')
    for location in coords.keys():
        print(location, end=' ')
    print('\n')

    print('Search Algorithms: ')
    print('1| Blind Brute Force')
    print('2| Breadth-first Search')
    print('3| Depth-first Search')
    print('4| ID-DFS Search')
    print('5| Best First Search')
    print('6| A* Search')
    print('\n')

    start = input('Please choose a starting location: ')
    end = input('Please choose an ending location: ')
    algo = int(input('Please choose a search algorithm (choose number): '))

    match algo:
        case 1:
            print('yo')
            start_time = time.time()
            results = brute_force(start,end,adj,coords)
        case 2:
            print('x')
        case 3:
            print('x')
        case 4:
            print('x')
        case 5:
            print('x')
        case 6:
            print('x')

    path = results[1]
    print(path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    plot_path(coords,path,1000)



main()
