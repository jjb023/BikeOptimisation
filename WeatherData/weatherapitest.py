import requests
import sys
import csv
from io import StringIO
import csv

joshapikey = "8WFP3BTMF64JYW9Q4TM7E6MQE"
keeganapikey = "QBEDB6KFA2N6TT4XF6XTC63D6"


response = requests.request("GET", "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/london/2022-04-11/2022-04-20?unitGroup=metric&include=hours&elements=datetime,precip,temp&key=8WFP3BTMF64JYW9Q4TM7E6MQE&contentType=csv")

# Check if the request was successful
if response.status_code != 200:
    print('Unexpected Status code:', response.status_code)
    sys.exit()

# Parse the CSV text
csv_text = response.text

# Create a StringIO object to simulate a file-like object for csv.reader
csv_file = StringIO(csv_text)

# Use csv.reader to parse the CSV data
csv_reader = csv.reader(csv_file)

# Create a new CSV file
with open('WeatherData/weatherdata.csv', 'w', newline='') as file:
  writer = csv.writer(file)

  # Write each row of the CSV data to the new file
  for row in csv_reader:
    writer.writerow(row)


