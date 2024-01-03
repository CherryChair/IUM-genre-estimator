import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# Wczytanie danych
data = pd.read_json('artists.jsonl', lines=True)
data2 = pd.read_json('tracks.jsonl', lines=True)

# Przygotowanie danych
artists = data['id'].dropna().explode().tolist()
tracks = data2['id_artist'].dropna().explode().tolist()
tracks_count = Counter(tracks)
artist_count = Counter(artists)

artists_without_track = len(artists)
tracks_without_artist = len(tracks)
# print(f'tracks_without_artist: {tracks_without_artist}')
# print(f'artists_without_track: {artists_without_track}')
number_of_tracks_artist = dict()

for label, value in tracks_count.items():
    if label in artist_count:
        tracks_without_artist -= value
        artists_without_track -= 1
        # if value == 114:
        #     print(label)
        if value not in number_of_tracks_artist:
            number_of_tracks_artist[value] = 1
        else:
            number_of_tracks_artist[value] += 1

i = 0

print(f'tracks without artist: {tracks_without_artist}')
print(f'artists without track: {artists_without_track}')
print(f'artists without track only counting artists without id -1: {artists_without_track - artist_count[-1]}')

print(f'track number: artists with that many tracks')
print(number_of_tracks_artist)

print("Artist count with over 100 tracks:")
copy = list(number_of_tracks_artist.items())
for key, value in copy:
    if key >=50:
        del number_of_tracks_artist[key]

labels, values = zip(*number_of_tracks_artist.items())


# Tworzenie wykresu
plt.figure(figsize=(10, 5))
plt.bar(labels, values)
plt.xlabel('Liczba utworów')
plt.ylabel('Liczba artystów o takiej liczbie utworów')
plt.title('Rozkład liczby utworów na artystę')
plt.xticks(rotation=90)
plt.show()