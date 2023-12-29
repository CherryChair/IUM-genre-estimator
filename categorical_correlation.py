from sklearn.metrics import mutual_info_score as mi_score
from collections import Counter
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

def round_attrib(attrib_value, groups_number):
    return math.floor(attrib_value*groups_number)

def group_int_attrib(attrib_value, step):
    return math.floor(attrib_value/step)

def make_attrib_discrete(attrib_name, attrib_value):
    match attrib_name:
        case 'popularity':
            return group_int_attrib(attrib_value, 10)
        case 'danceability':
            return round_attrib(attrib_value, 10)
        case "energy":
            return round_attrib(attrib_value, 10)
        case "key":
            return attrib_value
        case "release_year":
            return group_int_attrib(attrib_value, 1)
        case "liveness":
            if attrib_value >= 0.5:
                return 11
            return round_attrib(attrib_value, 10)
        case "speechiness":
            if attrib_value >= 0.2:
                return 5
            return round_attrib(attrib_value, 40)
        case "tempo":
            return group_int_attrib(attrib_value, 50)
        case "time_signature":
            return attrib_value
        case "valence":
            return round_attrib(attrib_value, 10)
        case "duration_ms":
            return group_int_attrib(attrib_value, 10000)
        case "explicit":
            return attrib_value

attributes = [
    'popularity', 'release_year', 'danceability', 'energy', 'key',
    'speechiness', 'liveness', 'valence', 'tempo',
    'time_signature', 'duration_ms', 'explicit'
]
discrete_attributes = [
    'time_signature', 'release_year', 'key', 'tempo', 'duration_ms', 'explicit'
]
# data_tracks = pd.read_json('discretized_attribs_track_data_with_duration.jsonl', lines=True)

data_tracks = pd.read_json('prepared_track_data.jsonl', lines=True)
data_tracks['release_year'] = pd.to_datetime(data_tracks['release_date'], format='mixed').dt.year
data_tracks.drop('release_date', axis=1, inplace=True)
data_tracks.drop('acousticness', axis=1, inplace=True)
data_tracks.drop('instrumentalness', axis=1, inplace=True)
data_tracks.drop('mode', axis=1, inplace=True)
data_tracks.drop('loudness', axis=1, inplace=True)

print("MAKIN' ATTRIBS DISCRETE")
for index, row in data_tracks.iterrows():
    print(index)
    for attribute in attributes:
        data_tracks.loc[index, attribute] = make_attrib_discrete(attribute, row[attribute])
print("DONE MAKIN' ATTRIBS DISCRETE")

data_tracks.to_json('discretized_attribs_track.jsonl', orient='records', lines=True)

data_tracks_mi = pd.DataFrame([[mi_score(data_tracks[attribute].to_list(), data_tracks[sec_attribute].to_list()) for attribute in attributes] for sec_attribute in (attributes + ['genre'])], index = attributes + ['genre'], columns = attributes)
# data_tracks_mi = pd.DataFrame([[mi_score(data_tracks[attribute].to_list(), data_tracks[sec_attribute].to_list()) for attribute in attributes] for sec_attribute in (discrete_attributes + ['genre'])], index = discrete_attributes + ['genre'], columns = attributes)
# data_tracks_mi = pd.DataFrame([mi_score(data_tracks[attribute].to_list(), data_tracks['genre'].to_list()) for attribute in attributes], index = attributes, columns = ['genre'])
print(data_tracks_mi)
ax = sns.heatmap(data_tracks_mi, annot=True, square=True)
plt.show()