import pandas as pd
from collections import defaultdict
import time
import numpy as np
import os
import shutil


def convert_time_to_minutes(time):

    hours, minutes = time.split(':')
    hours = int(hours)
    minutes = int(minutes)

    return int(hours*60 + minutes)

def convert_minutes_to_time(minutes):

    hours = str(minutes//60)
    if len(hours) == 1:
        hours = f'0{hours}'
    remainder = str(minutes % 60)
    if len(remainder) == 1:
        remainder = f'0{remainder}'

    return f'{hours}:{remainder}'


class Visualisation:

    def __init__(self):

        self.path = '/Users/keeganhill/PycharmProject/MDMPHASEC'
        self.mean_journey_times = defaultdict(list)
        self.station_numbers = {}
        self.stations = []
        self.mean_journey_time_dataframe = pd.DataFrame(0.0, columns=self.stations, index=self.stations)
        self.data = pd.read_csv('286JourneyDataExtract06Oct2021-12Oct2021.csv')

        self.data.dropna(how='all', axis=0, inplace=True)
        self.data.dropna(how='all', axis=1, inplace=True)
        self.data.drop('Rental Id', axis=1, inplace=True)
        self.data.drop('Bike Id', axis=1, inplace=True)

        for index in self.data.index:
            sample = self.data.iloc[index]
            end_station_id = sample['EndStation Id']
            start_station_id = sample['StartStation Id']
            if end_station_id not in self.stations:
                self.stations.append(end_station_id)
            if start_station_id not in self.stations:
                self.stations.append(start_station_id)

        self.stations = sorted(self.stations, reverse=False)

    def connections(self, copy_dataframe, filename):

        connections_dataframe = pd.DataFrame(0, columns=self.stations, index=self.stations)

        for index in copy_dataframe.index:
            sample = copy_dataframe.iloc[index]
            leaving_station = sample['StartStation Id']
            entering_station = sample['EndStation Id']

            current_connections = connections_dataframe.loc[leaving_station, entering_station]
            new_number_of_connections = current_connections + 1

            connections_dataframe.loc[leaving_station, entering_station] = new_number_of_connections
            connections_dataframe.loc[entering_station, leaving_station] = new_number_of_connections

        connections_dataframe.to_csv(filename)

    def mean_journey_time(self):

        for index in self.data.index:

            sample = self.data.iloc[index]
            leaving_station = sample['StartStation Id']
            entering_station = sample['EndStation Id']
            journey_duration = sample['Duration']
            unsorted_tuple = (leaving_station, entering_station)
            sorted_tuple = tuple(sorted(unsorted_tuple))
            self.mean_journey_times[sorted_tuple].append(journey_duration)

        for key in self.mean_journey_times.keys():
            cumulative_time = 0
            for time in self.mean_journey_times[key]:
                cumulative_time += time

            mean_time = cumulative_time / len(self.mean_journey_times[key])
            self.mean_journey_time_dataframe.loc[key[0], key[1]] = mean_time
            self.mean_journey_time_dataframe.loc[key[1], key[0]] = mean_time

        self.mean_journey_time_dataframe.to_csv('mean_journey_times.csv')

    def mean_time_within_timeframe(self):

        running = True
        new_dataframe = pd.DataFrame(columns=self.data.columns)
        date = '08/10/2021'
        start_time = '00:00'
        time_frame = int(30)
        start_time = convert_time_to_minutes(start_time)

        while running:
            date_for_writing_file = date.replace('/', '.')
            start_time_for_writing_file = convert_minutes_to_time(start_time)
            filename = f'connections: {date_for_writing_file} {start_time_for_writing_file}, {time_frame} minutes.csv'
            print(filename)

            for index in self.data.index:
                sample = self.data.iloc[index]
                sample_start_date, sample_start_time = sample['Start Date'].split(' ')
                sample_end_date, sample_end_time = sample['End Date'].split(' ')
                sample_start_time = convert_time_to_minutes(sample_start_time)
                sample_end_time = convert_time_to_minutes(sample_end_time)
                if (sample_start_date != date or sample_end_date != date or sample_end_time > start_time + time_frame
                        or sample_start_time < start_time):
                    continue
                new_dataframe = new_dataframe._append(sample)
            new_dataframe.reset_index(drop=True, inplace=True)
            self.connections(new_dataframe, filename)
            start_time += time_frame
            if start_time == 1440:
                running = False

    def sort(self):

        for filename in os.listdir(self.path):
            if 'connections' in filename:
                date = filename.split(' ')[1]
                path_to_folder = os.path.join(self.path, 'connections', date)
                if not os.path.exists(path_to_folder):
                    os.makedirs(path_to_folder)
                shutil.move(os.path.join(self.path, filename), path_to_folder)

        new_path = os.path.join(self.path, 'connections', '1')
        if not os.path.exists(new_path):
            os.makedirs(new_path)


instance = Visualisation()
instance.mean_time_within_timeframe()
instance.sort()
