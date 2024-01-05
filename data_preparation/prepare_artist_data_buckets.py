import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import copy as cp
from classify_genre import classify_genre, genre_in_bucket


# Wczytanie danych
data_artists = pd.read_json('artists.jsonl', lines=True)
data_artists['genre'] = ['' for r in range(len(data_artists))]

data_artists.set_index('id', inplace=True)
# Przygotowanie danych
genres = data_artists['genres'].dropna().explode().tolist()
genre_counts = Counter(genres)
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
        print(classified_genre)


print('FINISHED BUCKETING GENRES')

print('ADDING GENRES TO artists')

# copy_artists = cp.deepcopy(data_artists)
cnt = 0
for index, row in data_artists.iterrows():
    artist_genres = row['genres']
    bucket_genres = []
    for artist_genre in artist_genres:
        bucket_genres.append(genre_counts[artist_genre])
    counter_bucket = Counter(bucket_genres)
    most_frequent, count = counter_bucket.most_common(1)[0]
    if genre_in_bucket(most_frequent):
        data_artists.loc[index, "genre"] = most_frequent


data_artists.drop(['genres'], axis='columns', inplace=True)

data_artists_to_save = data_artists[data_artists['genre'].map(
    lambda l: len(l)) > 0]

data_artists_to_save.to_json(
    'prepared_artist_data.jsonl', orient='records', lines=True)

bucketed_genres_to_count = data_artists_to_save['genre'].dropna(
).explode().tolist()
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
plt.show()
