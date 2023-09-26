import random
import sys
import time
import array
import matplotlib.pyplot as plt

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

def calculate_distance(city1,city2,coords):
    x1, y1 = coords[city1]
    x2, y2 = coords[city2]
    
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


def plot_path(coords, path, time):
    x = []
    y = []

    for city in coords:
        city_coords = coords[city]
        x.append(city_coords[0])
        y.append(city_coords[1])
    
    data = {"x": [], "y": [], "label": []}
    for label, coord in coords.items():
        data["x"].append(float(coord[0]))
        data["y"].append(float(coord[1]))
        data["label"].append(label)

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.title('Scatter Plot', fontsize=20)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('y', fontsize=15)
    plt.scatter(data["x"], data["y"], marker='o')

    # Add labels
    for label, x, y in zip(data["label"], data["x"], data["y"]):
        plt.annotate(label, xy=(x, y))

    # Plot the path by connecting points in the order specified by 'path'
    path_x = [data["x"][data["label"].index(city)] for city in path]
    path_y = [data["y"][data["label"].index(city)] for city in path]
    plt.plot(path_x, path_y, linestyle='-', color='red', label='Path', linewidth=2)


    # Add text to the upper left of the figure
    formatted_time = "{:.10f}".format(time)
    plt.figtext(0.1, 0.95, 'Time Elapsed: ' + formatted_time + 's', fontsize=12, verticalalignment='top')

    # Add text to the upper right of the figure
    total = calculate_total_distance(coords, path)
    plt.figtext(0.9, 0.95, 'Distance traveled: ' + str(total), fontsize=12, verticalalignment='top', horizontalalignment='right')

    # Add legend
    plt.legend()

    # Show the plot
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
    #x = input('Please choose an algorithm ')

    start_time = time.time()


    a = brute_force('Hillsboro','Manhattan',adj,coords)

    end_time = time.time()
    elapsed_time = end_time - start_time
    formatted_time = "{:.10f}".format(elapsed_time)
    print(f"Elapsed Time: {formatted_time} seconds")


    path = a[1]
    x = calculate_distance('Leon','Mayfield',coords)
    total = calculate_total_distance(coords, path)
    print(total)
    plot_path(coords,path,elapsed_time)



main()
