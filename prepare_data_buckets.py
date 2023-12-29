import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import copy as cp

def classify_genre(genre : str):
    if genre in {
        'argentine rock', 'austropop', 'australian rock', 'australian alternative rock', 
        'alternative dance', 'alternative rock', 'blues rock', 'alternative metal', 
        'new romantic', 'brazilian rock', 'album rock', 'classic rock', 
        'british invasion', 'art rock', 'britpop' 
    }:
        return 'rock'
    if genre in {
        'desi pop', 'opm', 'bulgarian pop', 'canadian pop', 'bubblegum dance', 'europop', 
        'slovak pop', 'classic uk pop', 'boy band', 'dance pop', 'pop', 'c-pop', 
        'classic italian pop', 'italian adult pop', 'art pop', 'dutch pop', 'classic norwegian pop', 
        'classic opm', 'classic polish pop', 'classic turkish pop', 'classic russian pop', 
        'classic swedish pop', 'finnish dance pop', 'disco' 
    }:
        return 'pop/dance'
    if genre in {
        'anime', 'anime score', 'classic j-rock', 'classic j-pop', 
        'j-pop', 'c-pop', 'mandopop', 'chinese indie', 
        'classic indo pop', 'k-pop', 'thai pop', 'classic thai pop', 
        'vintage taiwan pop', 'classic malaysian pop' 
    }:
        return 'asian pop'
    if genre in {
        'filmi', 'cumbia uruguaya', 'latin alternative', 'sertanejo', 
        'bossa nova', 'axe', 'latin jazz', 'latin christian', 'bachata', 'latin', 
        'corrido', 'banda', 'mariachi', 'grupera', 'bolero', 'tango'
    }:
        return 'latin'
    if genre in {
        'contemporary country', 'country', 'arkansas country', 'classic country pop'
    }:
        return 'country'
    if genre in {
        'kleine hoerspiel', 'hoerspiel', 'cantautor', 'cancion melodica', 'celtic', 
        'chanson', 'arab folk', 'classic tollywood', 'canzone d\'autore', 'anadolu rock', 
        'arabesk', 'canto popular uruguayo'
    }:
        return 'world music'
    if genre in {
        'bebop', 'cool jazz', 'avant-garde jazz', 'classic soul'
    }:
        return 'jazz/r&b'
    if genre == 'dub':
        return 'electronic'
    if genre in {
        'italian hip hop', 'french hip hop'
    }:
        return 'hip hop'
    if genre in {
        'classic soundtrack', 'adult standards' 
    }:
        return 'classical'
    if genre in {
        'american folk revival', 'classic finnish rock', 'classic finnish pop', 'classic greek pop', 
        'classic hungarian pop', 'classic israeli pop', 'beatlesque', 'classic kollywood', 
        'classic bollywood', 'classic icelandic pop', 'barnalog', 'classic italian pop', 
        'classic russian rock', 'big band', 'acoustic blues', 'czech folk',  'turkish folk' 
    }:
        return 'folk'
    if 'metal' in genre:
        return 'metal'
    if 'country' in genre:
        return 'country'
    if 'rock' in genre:
        return 'rock'
    if 'folk' in genre:
        return 'folk'
    if any([word in genre for word in ['rap', 'hip hop', 'hip-hop']]):
        return 'hip hop'
    if any([word in genre for word in ['techno', 'rave', 'electronica', 'dubstep', 'dark', 
            'house', 'industrial', 'dub', 'synth', 'new-age', 'deep', 'nu-disco', 'garage', 'EDM', 
            'breakbeat', 'trance', 'fusion', 'gabber', 'future', 'beats', 'effects', 'electronic', 
            'bass', 'FX', 'hardstyle', 'hard', 'kick']]):
        return 'electronic'
    if any([word in genre for word in [ 'jazz', 'r&b', 'funk', 'soul', 'swing', 'saxophone', 'bop']]):
        return 'jazz/r&b'
    if any([word in genre for word in ['latin', 'latino', 'salsa', 'merengue', 'reggae', 
        'reggaeton', 'cumbia', 'bolero', 'flamenco', 'tango', 'ranchera', 
        'mariachi', 'norteña', 'samba', 'bossa nova', 'trova', 'son', 'rumba', 
        'mambo', 'cha-cha-cha', 'fado', 'vallenato', 'pop latino', 'rock en español', 
        'jazz latino', 'mexico']]):
        return 'latin'
    if 'anime' in genre or ('pop' in genre and any([nation in genre for nation in ['afghan', 
        'armenian', 'azerbaijani', 'bahraini', 'bangladeshi', 'bhutanese', 
        'bruneian', 'burmese', 'cambodian', 'chinese', 'cypriot', 
        'emirati', 'filipino', 'georgian', 'indian', 'indonesian', 
        'iranian', 'iraqi', 'israeli', 'japanese', 'jordanian', 'kazakhstani',
        'kuwaiti', 'kyrgyzstani', 'laotian', 'lebanese', 'malaysian', 'maldivian', 'mongolian', 
        'nepalese', 'north korean', 'omani', 'pakistani', 'palestinian', 'saudi', 
        'singaporean', 'south korean', 'sri lankan', 'syrian', 'taiwanese', 'thai', 
        'timorese', 'turkish', 'turkmen', 'uzbekistani', 'vietnamese', 'yemeni']])):
        return 'asian pop'
    if ('pop' in genre or 'opm' in genre or 'disco' in genre):
        return 'pop/dance'
    if any([word in genre for word in ['classical', 'piano', 'soundtrack', 'instrumental']]):
        return 'classical'
    return genre

def genre_in_bucket(genre):
    return genre in {'pop/dance', 'classical', 'asian pop', 'latin', 'jazz/r&b', 'folk', 'rock',
        'metal', 'country', 'electronic', 'world music', 'hip hop'}

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
        print(classified_genre)


print('FINISHED BUCKETING GENRES')
print(counter)

sum_tracks = 0
all_tracks = 0
for key,value in track_counts.items():
    artist_genres = data_artists.loc[key, "genres"]
    all_tracks += value
    for artist_genre in artist_genres:
        if artist_genre in genre_counts:
            sum_tracks += value
            break

print('ADDING GENRES TO TRACKS')

# copy_tracks = cp.deepcopy(data_tracks)
cnt=  0
for index, row in data_tracks.iterrows():
    artist_genres = data_artists.loc[row["id_artist"], "genres"]
    bucket_genres = []
    for artist_genre in artist_genres:
        bucket_genres.append(genre_counts[artist_genre])
    counter_bucket = Counter(bucket_genres)
    most_frequent, count = counter_bucket.most_common(1)[0]
    if genre_in_bucket(most_frequent):
        data_tracks.loc[index, "genre"] = most_frequent
    i = True
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


data_tracks.drop(['id_artist'], axis='columns', inplace=True)
# copy_tracks.drop(['id_artist'], axis='columns', inplace=True)
# data_tracks.drop(data_tracks[len(data_tracks['genres']) == 0].index, inplace = True)

data_tracks_to_save = data_tracks[data_tracks['genre'].map(lambda l: len(l)) > 0]
# copy_tracks_to_save = copy_tracks[copy_tracks['genre'].map(lambda l: len(l)) > 0]

data_tracks_to_save.to_json('prepared_track_data.jsonl', orient='records', lines=True)
# copy_tracks_to_save.to_json('prepared_track_data_multiple_genres.jsonl', orient='records', lines=True)

bucketed_genres_to_count = data_tracks_to_save['genre'].dropna().explode().tolist()
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
plt.show()
