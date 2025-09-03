# the file contains multiple result sets, each starting with a line with the AI name or description
# followed by the header line and then the results


import csv
import logging

logging.basicConfig(level=logging.INFO)  # Change logging.INFO to logging.DEBUG to see debug messages

results = []
places_of_drivers = {}

class ResultSet:
    def __init__(self, description, places_of_drivers):
        self.description = description
        self.places_of_drivers = {}

    def __setitem__(self, driver, place):
        self.places_of_drivers[driver] = place

    def __str__(self):
        str = []
        str.append(f"{self.description:<20}")
        for driver, places in self.places_of_drivers.items():
            str.append(f"{driver}: {places}")
        return "\n".join(str)   

# Open the CSV file in read mode
csvfile = 'f1-20250907.csv'
with open(csvfile, mode='r') as file:
    csv_reader = csv.reader(file)

    # Loop through the lines in the CSV file
    for row in csv_reader:

        try:
            position = int(row[0])
        except ValueError:  # there is a column but it's not a number, is either tbe column header or the AI name
            first_value = row[0]

            if first_value != "Position": # the AI name or description
                ai_name = first_value
                places_of_drivers = {}
                resultSet = ResultSet(description=ai_name, places_of_drivers={})
                results.append(resultSet)
                continue
            else: 
                continue 

        except (TypeError, IndexError):
            continue # next

        logging.debug(f"csv file row: {row}")

        driver = row[1].split()[-1]
        place = row[0]
        resultSet[driver] = place


        if driver not in places_of_drivers:
            places_of_drivers[driver] = []

        places_of_drivers[driver].append(place)

logging.info(f"{len(results)} result sets found.")

# this is weird because the set of result sets should be a class
for result in results:
    logging.debug('-' * 40)
    logging.debug(result)

# compare results[4] with results[5], as if the latter were the real results and the former the predictions
diff321 = 0
for driver, place in results[4].places_of_drivers.items(): 
    # count how many drivers are in the same place in both results
    if driver in results[5].places_of_drivers:
        if place == results[5].places_of_drivers[driver]:
            logging.info(f"{driver} is in the same place {place} in both results.")
        else:
            logging.info(f"{driver} is in place {place} in predictions and in place {results[5].places_of_drivers[driver]} in results.")     

    diff = abs(int(place) - int(results[5].places_of_drivers.get(driver, place)))
    # score is 3 points for exact match, 2 points for 1 place difference, 1 point for 2 places difference, 0 points otherwise
    if diff == 0:
        score = 3
    elif diff == 1:         
        score = 2
    elif diff == 2:         
        score = 1
    else:
        score = 0      
    diff321 += score


# count how many drivers are in the same place in both results
same_place_count = sum(1 for driver, place in results[4].places_of_drivers.items() if driver in results[5].places_of_drivers and place == results[5].places_of_drivers[driver])
logging.info(f"{same_place_count} drivers are in the same place in both results.")
logging.info(f"Score (3 for exact match, 2 for 1 place difference, 1 for 2 places difference, 0 otherwise): {diff321}")

# getting the order of drivers right is more important than getting the exact places right
# so we will count how many drivers are in the same relative order in both results
# for example, if driver A is in place 1 and driver B is in place 2 in both results, that is a match
# but if driver A is in place 1 and driver B is in place 3 in predictions, that is not a match
# we will use a simple algorithm to count the number of pairs of drivers that are in the same relative order
same_order_count = 0
drivers = list(results[4].places_of_drivers.keys())
for i in range(len(drivers)):
    for j in range(i + 1, len(drivers)):
        driver_i = drivers[i]
        driver_j = drivers[j]
        if driver_i in results[5].places_of_drivers and driver_j in results[5].places_of_drivers:
            place_i_pred = int(results[4].places_of_drivers[driver_i])
            place_j_pred = int(results[4].places_of_drivers[driver_j])
            place_i_real = int(results[5].places_of_drivers[driver_i])
            place_j_real = int(results[5].places_of_drivers[driver_j])
            if (place_i_pred < place_j_pred and place_i_real < place_j_real) or (place_i_pred > place_j_pred and place_i_real > place_j_real):
                same_order_count += 1
logging.info(f"{same_order_count} pairs of drivers are in the same relative order in both results.")
