import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# Wczytanie danych
data_artists = pd.read_json('artists.jsonl', lines=True)
# data_users = pd.read_json('users.jsonl', lines=True)
data_tracks = pd.read_json('tracks.jsonl', lines=True)
# data_sessions = pd.read_json('sessions.jsonl', lines=True)
data_tracks['genres'] = object()

data_artists.set_index('id', inplace=True)
print(data_artists.dtypes)
print(data_tracks.dtypes)
print(data_artists[0:3])
# print(data_artists.loc['4gdMJYnopf2nEUcanAwstx', 'genres'])
# exit()
# Przygotowanie danych
genres = data_artists['genres'].dropna().explode().tolist()
tracks = data_tracks['id_artist'].dropna().explode().tolist()
genre_counts = Counter(genres)
track_counts = Counter(tracks)
# Usuwanie poniżej pewnego progu wystąpień
copy = list(genre_counts.items())

sum = 0
sum_all = 0
# print(len(copy))
for key, value in copy:
    sum_all += value
    # if value < 100:
    #     sum += value
    #     del genre_counts[key]

# print(len(genre_counts.items()))
# print(sum)
# print(sum_all)

# copy = list(genre_counts.items())
# print(len(copy))
for key, value in genre_counts.items():
    genre_counts[key] = 0

for key,value in track_counts.items():
    artist_genres = data_artists.loc[key, "genres"]
    for artist_genre in artist_genres:
        if artist_genre in genre_counts:
            genre_counts[artist_genre] += value
            break


copy = list(genre_counts.items())
print(len(genre_counts.items()))

for key, value in copy:
    if value < 200:
        del genre_counts[key]


print(len(genre_counts.items()))
# Przygotowanie danych do wykresu


sum_tracks = 0
all_tracks = 0
for key,value in track_counts.items():
    artist_genres = data_artists.loc[key, "genres"]
    all_tracks += value
    for artist_genre in artist_genres:
        if artist_genre in genre_counts:
            sum_tracks += value
            break


print(all_tracks)
print(sum_tracks)

labels, values = zip(*genre_counts.items())


# Tworzenie wykresu
plt.figure(figsize=(10, 5))
plt.bar(labels, values)
plt.xlabel('Gatunki muzyczne')
plt.ylabel('Liczba wystąpień')
plt.title('Liczność i rodzaje gatunków muzycznych')
plt.xticks(rotation=90)
plt.show()