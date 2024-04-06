import csv
from geopy.distance import geodesic

# Function to calculate distance between two coordinates
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).kilometers

# Function to read CSV file and extract coordinates
def read_csv(filename):
    coordinates = {}
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            id = row['ID']
            lat = float(row['Latitude'])
            lon = float(row['Longitude'])
            coordinates[id] = (lat, lon)
    return coordinates

# Function to calculate distances between all stations
def calculate_distances(coordinates):
    distances = {}
    stations = list(coordinates.keys())
    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            station1 = stations[i]
            station2 = stations[j]
            coord1 = coordinates[station1]
            coord2 = coordinates[station2]
            distance = calculate_distance(coord1, coord2)
            distances[(station1, station2)] = distance
    return distances

# Function to write distances to CSV file
def write_csv(distances, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Station 1', 'Station 2', 'Distance (km)'])
        for stations, distance in distances.items():
            writer.writerow([stations[0], stations[1], distance])

# Main function
def main(input_filename, output_filename):
    coordinates = read_csv(input_filename)
    distances = calculate_distances(coordinates)
    write_csv(distances, output_filename)
    print(f"Distances written to {output_filename}")

main('longlat_data.csv', 'distances.csv')
