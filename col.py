# open a file and read lines of csv data
import csv

places_of_drivers = {}

# Open the CSV file in read mode
with open('f1-20250907.csv', mode='r') as file:
    csv_reader = csv.reader(file)

    # Loop through the lines in the CSV file
    for row in csv_reader:
        try:
            i = int(row[0])
        except (TypeError, ValueError, IndexError):
            continue # next

        print(row)  # Each row will be a list of values from the CSV file
        driver = row[1].split()[-1]
        place = row[0]


        if driver not in places_of_drivers:
            places_of_drivers[driver] = []

        places_of_drivers[driver].append(place)

from pprint import pprint
pprint(places_of_drivers)
print(len(places_of_drivers))
