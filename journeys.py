import pandas as pd
import numpy as np
from collections import defaultdict
import time


def validate(sample, metrics):

    metric_sum = metrics[0]
    metric_samples = metrics[1]
    returning_journeys = metrics[2]

    if metric_samples * 2 - returning_journeys == metric_sum:
        print(f'Validated {sample}')
        return True
    return False


def convert_minutes_to_hours(minutes):

    hours = str(minutes // 60)
    minutes = str(minutes % 60)
    if len(hours) == 1:
        hours = f'0{hours}'
    if len(minutes) == 1:
        minutes = f'0{minutes}'

    return f'{hours}:{minutes}'


def convert_hours_to_mins(hours):

    hour = int(hours.split(':')[0])
    minutes = int(hours.split(':')[-1])

    return hour * 60 + minutes


def journeys(stations, df, headers):

    length = len(stations)
    matrix = np.zeros((length, length), int)

    returning_journeys = 0

    for index in df.index:

        sample = df.loc[index]
        start_station = sample['StartStation Id']
        end_station = sample['EndStation Id']
        if start_station == end_station:
            returning_journeys += 1
            position_in_matrix_of_end_station = column_to_station_id[end_station]
            matrix[position_in_matrix_of_end_station][position_in_matrix_of_end_station] += 1
            continue
        position_in_matrix_of_start_station = column_to_station_id[start_station]
        position_in_matrix_of_end_station = column_to_station_id[end_station]

        matrix[position_in_matrix_of_start_station][position_in_matrix_of_end_station] += 1
        matrix[position_in_matrix_of_end_station][position_in_matrix_of_start_station] += 1

    use_metric = (np.sum(matrix), len(df.index), returning_journeys)

    return matrix, use_metric

def mean_time(stations, df, column_to_station_id):


    total_times = defaultdict(list)
    mean_times = {}
    length = len(stations)
    matrix = np.zeros((length, length), int)
    for index  in df.index:
        sample = df.loc[index]
        duration = sample['Duration']
        start_station, end_station = sample['StartStation Id'], sample['EndStation Id']
        total_times[(start_station, end_station)].append(duration)
    counter = 0
    for key in total_times.keys():
        mean_times[key] = np.mean(total_times[key])

    for connected_stations in mean_times.keys():
        start_station, end_station = int(connected_stations[0]), int(connected_stations[1])
        start_station, end_station = column_to_station_id[start_station], column_to_station_id[end_station]
        matrix[start_station][end_station] = mean_times[connected_stations]

    return matrix


def method_mean(times):
    matrix_list = []
    for sample in times.keys():
        df_to_append = pd.DataFrame(times[sample])
        print(df_to_append)
        matrix_to_append = mean_time(stations, df_to_append, column_to_station_id)
        matrix_list.append(matrix_to_append)

    mean_times_array = np.array(matrix_list)
    print(mean_times_array)
    np.save('combined_10Jan - 31Mar 2016_mean_journey_times.npy', mean_times_array)
    end = time.time()
    print(f'runtime: {end - start}')


def method_connections(times):

    store_results = []
    validation_metrics = {}
    for sample in times.keys():
        df_to_append = pd.DataFrame(times[sample])
        print(df_to_append)
        matrix_to_append, metrics = journeys(stations, df_to_append, headers_to_columns)
        store_results.append(matrix_to_append)
        validation_metrics[sample] = metrics

    print('Validating...')
    for sample in validation_metrics:
        time.sleep(0.025)
        if not validate(sample, validation_metrics[sample]):
            print(f'{sample} not Validated')
            break

    print('Validated')

    results = np.array(store_results)
    print(results)
    np.save('combined_10Jan - 31Mar 2016.npy', results)

    end = time.time()
    print(f'runtime: {end - start}')


# ----> Run

start = time.time()
df = pd.read_csv('combined_10Jan - 31Mar 2016.csv')
df.dropna(how='any', axis=0, inplace=True)

stations = set(df['EndStation Id'].astype(int).tolist() + df['StartStation Id'].astype(int).to_list())
headers_to_columns = {key: value for key, value in zip(list(range(len(df.columns))), df.columns)}

column_to_station_id = {value: i for i, value in enumerate(stations)}


times = {convert_minutes_to_hours(minutes): [] for minutes in range(0, 1440, 30)}
time_keys = list(times.keys())

for index in df.index:
    print(index)
    sample = df.loc[index]
    end_time = sample['End Date'].split(' ')[-1]

    end_time_hour, end_time_minutes = end_time.split(':')
    end_time_minutes = int(end_time_minutes)

    for i in time_keys:
        key_hour, key_minute_start = i.split(':')[0], int(i.split(':')[-1])

        if end_time_hour == key_hour:
            if key_minute_start < end_time_minutes < key_minute_start + 30:
                times[i].append(sample)
                
#uncomment method to run

# method_connections(times)
# method_mean(times)

    
