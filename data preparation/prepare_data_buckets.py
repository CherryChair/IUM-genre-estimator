import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import copy as cp
from classify_genre import classify_genre


def genre_in_bucket(genre):
    return genre in {'pop', 'classical', 'asian pop', 'latin', 'jazz/r&b', 'folk', 'rock',
                     'metal', 'country', 'electronic', 'world music', 'hip hop', 'classic pop', 'dance/disco',
                     'style rock', 'geographical rock'}


# Wczytanie danych
data_artists = pd.read_json('artists.jsonl', lines=True)
data_tracks = pd.read_json('tracks.jsonl', lines=True)
data_tracks['genre'] = ['' for r in range(len(data_tracks))]

data_artists.set_index('id', inplace=True)
# Przygotowanie danych
genres = data_artists['genres'].dropna().explode().tolist()
tracks = data_tracks['id_artist'].dropna().explode().tolist()
genre_counts = Counter(genres)
track_counts = Counter(tracks)
# Usuwanie poniżej pewnego progu wystąpień

for key, value in genre_counts.items():
    genre_counts[key] = ''


copy = list(genre_counts.items())

# Przygotowanie danych do wykresu

print('BUCKETING GENRES')
counter = 0
for key, value in copy:
    classified_genre = classify_genre(key)
    if (genre_in_bucket(classified_genre)):
        genre_counts[key] = classified_genre
    else:
        counter += 1
        # print(classified_genre)


print('FINISHED BUCKETING GENRES')
print(counter)

sum_tracks = 0
all_tracks = 0
for key, value in track_counts.items():
    artist_genres = data_artists.loc[key, "genres"]
    all_tracks += value
    for artist_genre in artist_genres:
        if artist_genre in genre_counts:
            sum_tracks += value
            break

print('ADDING GENRES TO TRACKS')

# copy_tracks = cp.deepcopy(data_tracks)
cnt = 0
for index, row in data_tracks.iterrows():
    artist_genres = data_artists.loc[row["id_artist"], "genres"]
    bucket_genres = []
    for artist_genre in artist_genres:
        bucket_genres.append(genre_counts[artist_genre])
    counter_bucket = Counter(bucket_genres)
    most_frequent, count = counter_bucket.most_common(1)[0]
    if genre_in_bucket(most_frequent):
        data_tracks.loc[index, "genre"] = most_frequent
    # i = True
    # for key, value in counter_bucket.items():
    #     print(cnt)
    #     cnt += 1
    #     if i and genre_in_bucket(key):
    #         i = False
    #         copy_tracks.loc[index, "genre"] = key
    #     elif genre_in_bucket(key):
    #         new_row = copy_tracks.loc[index].copy()
    #         new_row['genre'] = key
    #         copy_tracks.loc[len(copy_tracks.index)] = new_row

# Pozbywamy się niepotrzebnych kolumn
data_tracks.drop(['id_artist'], axis='columns', inplace=True)

data_tracks['release_year'] = pd.to_datetime(
    data_tracks['release_date']).dt.year
data_tracks.drop('release_date', axis=1, inplace=True)
data_tracks.drop('acousticness', axis=1, inplace=True)
data_tracks.drop('instrumentalness', axis=1, inplace=True)
data_tracks.drop('mode', axis=1, inplace=True)
data_tracks.drop('loudness', axis=1, inplace=True)
data_tracks.drop('tempo', axis=1, inplace=True)
data_tracks.drop('key', axis=1, inplace=True)
data_tracks.drop('liveness', axis=1, inplace=True)
data_tracks.drop('time_signature', axis=1, inplace=True)
# copy_tracks.drop(['id_artist'], axis='columns', inplace=True)
# data_tracks.drop(data_tracks[len(data_tracks['genres']) == 0].index, inplace = True)

data_tracks_to_save = data_tracks[data_tracks['genre'].map(
    lambda l: len(l)) > 0]
# copy_tracks_to_save = copy_tracks[copy_tracks['genre'].map(lambda l: len(l)) > 0]

data_tracks_to_save.to_json(
    'prepared_track_data.jsonl', orient='records', lines=True)
# copy_tracks_to_save.to_json('prepared_track_data_multiple_genres.jsonl', orient='records', lines=True)

bucketed_genres_to_count = data_tracks_to_save['genre'].dropna(
).explode().tolist()
# bucketed_genres_to_count = copy_tracks_to_save['genre'].dropna().explode().tolist()
counted_bucketed_genres = Counter(bucketed_genres_to_count)

labels, values = zip(*counted_bucketed_genres.items())

for key in genre_counts.keys():
    print(key)
# Tworzenie wykresu
plt.figure(figsize=(10, 5))
plt.bar(labels, values)
plt.xlabel('Gatunki muzyczne')
plt.ylabel('Liczba wystąpień')
plt.title('Liczność i rodzaje gatunków muzycznych')
plt.xticks(rotation=90)
plt.savefig('plot.png')
plt.show()
