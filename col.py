# open a file and read lines of csv data
import csv

# return false if not int
def isint(i):
    

# Open the CSV file in read mode
with open('f1-20250907.csv', mode='r') as file:
    csv_reader = csv.reader(file)

    # Loop through the lines in the CSV file
    for row in csv_reader:
        print(row)  # Each row will be a list of values from the CSV file

