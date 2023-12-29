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
cnt=  0
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

data_artists_to_save = data_artists[data_artists['genre'].map(lambda l: len(l)) > 0]

data_artists_to_save.to_json('prepared_artist_data.jsonl', orient='records', lines=True)

bucketed_genres_to_count = data_artists_to_save['genre'].dropna().explode().tolist()
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
