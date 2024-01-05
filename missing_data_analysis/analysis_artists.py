import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# Wczytanie danych
data_artists = pd.read_json('artists.jsonl', lines=True)

# Przygotowanie danych
artists = data_artists['name'].dropna().explode().tolist()
artist_counts = Counter(artists)


# Przygotowanie danych do wykresu
labels, values = zip(*artist_counts.items())

name_dict = dict()
repeated_names = dict()

for label in labels:
    if label not in name_dict:
        name_dict[label] = 1
    else:
        repeated_names[label] = True
        name_dict[label] += 1
    
print(len(name_dict.items()))
print(len(repeated_names.items()))
for key, value in repeated_names:
    print(name_dict[label])

# # Tworzenie wykresu
# plt.figure(figsize=(10, 5))
# plt.bar(labels, values)
# plt.xlabel('Gatunki muzyczne')
# plt.ylabel('Liczba wystąpień')
# plt.title('Liczność i rodzaje gatunków muzycznych')
# plt.xticks(rotation=90)
# plt.show()