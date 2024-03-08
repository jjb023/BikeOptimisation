import numpy
import pandas as pd
import numpy as np
from collections import defaultdict
import time
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

        current_value = matrix[position_in_matrix_of_start_station][position_in_matrix_of_end_station]
        new_value = current_value + 1
        matrix[position_in_matrix_of_start_station][position_in_matrix_of_end_station] = new_value
        matrix[position_in_matrix_of_end_station][position_in_matrix_of_start_station] = new_value

    use_metric = (np.sum(matrix), len(df.index), returning_journeys)

    return matrix, use_metric

def validate(sample, metrics):

    metric_sum = metrics[0]
    metric_samples = metrics[1]
    returning_journeys = metrics[2]

    if metric_samples * 2 - returning_journeys == metric_sum:
        print(f'Validated {sample}.')
    time.sleep(0.2)


start = time.time()

df = pd.read_csv('combined_10Jan - 31Mar 2016.csv')
df.dropna(how='any', axis=0, inplace=True)

stations = set(df['EndStation Id'].astype(int).tolist() + df['StartStation Id'].astype(int).to_list())
headers_to_columns = {key: value for key, value in zip(list(range(len(df.columns))), df.columns)}

column_to_station_id = {value: i for i, value in enumerate(stations)}

#### -------

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

store_results = []
validation_metrics = {}
for sample in times.keys():
    df_to_append = pd.DataFrame(times[sample])
    print(df_to_append)
    matrix_to_append, metrics = journeys(stations, df_to_append, headers_to_columns)
    store_results.append(matrix_to_append)
    validation_metrics[sample] = metrics

for sample in validation_metrics:   
    validate(sample, validation_metrics[sample])

results = np.array(store_results)
print(results)

end = time.time()
print(f'runtime: {end-start}')











