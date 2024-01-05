import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import copy as cp
from classify_genre import classify_genre, genre_in_bucket
import sys


def load_data():
    data_artists = pd.read_json('artists.jsonl', lines=True)
    data_tracks = pd.read_json('tracks.jsonl', lines=True)
    data_tracks['genre'] = ['' for r in range(len(data_tracks))]
    data_artists.set_index('id', inplace=True)
    return data_artists, data_tracks


def prepare_data(data_artists, data_tracks):
    genres = data_artists['genres'].dropna().explode().tolist()
    tracks = data_tracks['id_artist'].dropna().explode().tolist()
    genre_counts = Counter(genres)
    track_counts = Counter(tracks)
    for key, value in genre_counts.items():
        genre_counts[key] = ''
    return genre_counts, track_counts


def bucket_genres(genre_counts):
    print('BUCKETING GENRES')
    counter = 0
    copy = list(genre_counts.items())
    for key, value in copy:
        classified_genre = classify_genre(key)
        if (genre_in_bucket(classified_genre)):
            genre_counts[key] = classified_genre
        else:
            counter += 1
    print('FINISHED BUCKETING GENRES')
    print(counter)
    return genre_counts


def add_genres_to_tracks(data_artists, data_tracks, genre_counts, track_counts):
    print('ADDING GENRES TO TRACKS')
    sum_tracks = 0
    all_tracks = 0
    for key, value in track_counts.items():
        artist_genres = data_artists.loc[key, "genres"]
        all_tracks += value
        for artist_genre in artist_genres:
            if artist_genre in genre_counts:
                sum_tracks += value
                break
    for index, row in data_tracks.iterrows():
        artist_genres = data_artists.loc[row["id_artist"], "genres"]
        bucket_genres = []
        for artist_genre in artist_genres:
            bucket_genres.append(genre_counts[artist_genre])
        counter_bucket = Counter(bucket_genres)
        most_frequent, count = counter_bucket.most_common(1)[0]
        if genre_in_bucket(most_frequent):
            data_tracks.loc[index, "genre"] = most_frequent
    return data_tracks


def add_genres_to_tracks_complex(data_artists, data_tracks, genre_counts, track_counts):
    print('ADDING GENRES TO TRACKS')
    sum_tracks = 0
    all_tracks = 0
    for key, value in track_counts.items():
        artist_genres = data_artists.loc[key, "genres"]
        all_tracks += value
        for artist_genre in artist_genres:
            if artist_genre in genre_counts:
                sum_tracks += value
                break
    for index, row in data_tracks.iterrows():
        artist_genres = data_artists.loc[row["id_artist"], "genres"]
        bucket_genres = []
        for artist_genre in artist_genres:
            bucket_genres.append(genre_counts[artist_genre])
        i = True
        for genre in bucket_genres:
            if i and genre_in_bucket(genre):
                i = False
                data_tracks.loc[index, "genre"] = genre
            elif genre_in_bucket(genre):
                new_row = data_tracks.loc[index].copy()
                new_row['genre'] = genre
                data_tracks.loc[len(data_tracks.index)] = new_row
    return data_tracks

def clean_data(data_tracks):
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

def save_data(data_tracks, filename):
    data_tracks_to_save = data_tracks[data_tracks['genre'].map(
        lambda l: len(l)) > 0]
    data_tracks_to_save.to_json(
        filename, orient='records', lines=True)
    return data_tracks_to_save


def plot_data(saved_data_tracks):
    bucketed_genres_to_count = saved_data_tracks['genre'].dropna(
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
    complex = False
    if sys.argv[1] == "-c":
        complex = True
    data_artists, data_tracks = load_data()
    genre_counts, track_counts = prepare_data(data_artists, data_tracks)
    genre_counts = bucket_genres(genre_counts)
    data_tracks = clean_data(data_tracks)
    if complex:
        data_tracks = add_genres_to_tracks_complex(
            data_artists, data_tracks, genre_counts, track_counts)
        saved_data_tracks = save_data(data_tracks, 'prepared_track_data_complex.jsonl')
    else:
        data_tracks = add_genres_to_tracks(
            data_artists, data_tracks, genre_counts, track_counts)
        saved_data_tracks = save_data(data_tracks, 'prepared_track_data.jsonl')
    plot_data(saved_data_tracks)


if __name__ == "__main__":
    main()
