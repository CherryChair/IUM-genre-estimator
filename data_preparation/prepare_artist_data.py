import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from classify_genre import classify_genre, genre_in_bucket
import sys


def load_data():
    data_artists = pd.read_json('artists.jsonl', lines=True)
    data_artists['genre'] = ['' for _ in range(len(data_artists))]
    data_artists.set_index('id', inplace=True)
    return data_artists


def bucket_genres(data_artists):
    genres = data_artists['genres'].dropna().explode().tolist()
    genre_counts = Counter(genres)
    for key in genre_counts:
        genre_counts[key] = ''
    print('BUCKETING GENRES')
    for key in list(genre_counts):
        classified_genre = classify_genre(key)
        if genre_in_bucket(classified_genre):
            genre_counts[key] = classified_genre
    print('FINISHED BUCKETING GENRES')
    return genre_counts


def add_genres_to_artists(data_artists, genre_counts):
    print('ADDING GENRES TO artists')
    for index, row in data_artists.iterrows():
        artist_genres = row['genres']
        bucket_genres = [genre_counts[artist_genre]
                         for artist_genre in artist_genres]
        counter_bucket = Counter(bucket_genres)
        most_frequent, _ = counter_bucket.most_common(1)[0]
        if genre_in_bucket(most_frequent):
            data_artists.at[index, "genre"] = most_frequent
    return data_artists


def add_genres_to_artists_complex(data_artists, genre_counts):
    print('ADDING GENRES TO artists')
    for index, row in data_artists.iterrows():
        artist_genres = row['genres']
        bucket_genres = [genre_counts[artist_genre]
                         for artist_genre in artist_genres]
        counter_bucket = Counter(bucket_genres)
        most_frequent, _ = counter_bucket.most_common(1)[0]
        if genre_in_bucket(most_frequent):
            data_artists.at[index, "genre"] = most_frequent
        data_artists.at[index, "genres"] = bucket_genres
    return data_artists


def save_prepared_data(data_artists, filename):
    data_artists_to_save = data_artists[data_artists['genre'].map(len) > 0]
    # Reset index to add 'id' field back
    data_artists_to_save.reset_index(inplace=True)
    data_artists_to_save.to_json(
        filename, orient='records', lines=True)
    return data_artists_to_save


def plot_genre_distribution(data_artists_to_save):
    bucketed_genres_to_count = data_artists_to_save['genre'].dropna(
    ).explode().tolist()
    counted_bucketed_genres = Counter(bucketed_genres_to_count)
    labels, values = zip(*counted_bucketed_genres.items())
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xlabel('Gatunki muzyczne')
    plt.ylabel('Liczba wystąpień')
    plt.title('Liczność i rodzaje gatunków muzycznych')
    plt.xticks(rotation=90)
    plt.savefig('plot.png')
    plt.show()


def main():
    data_artists = load_data()
    genre_counts = bucket_genres(data_artists)
    if len(sys.argv) > 1 and sys.argv[1] == "-c":
        data_artists = add_genres_to_artists_complex(
            data_artists, genre_counts)
        data_artists_to_save = save_prepared_data(
            data_artists, 'prepared_artist_data_complex.jsonl')
    else:
        data_artists = add_genres_to_artists(data_artists, genre_counts)
        data_artists_to_save = save_prepared_data(
            data_artists, 'prepared_artist_data.jsonl')
    plot_genre_distribution(data_artists_to_save)


if __name__ == '__main__':
    main()
