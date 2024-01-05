import matplotlib.pyplot as plt
import json
import pandas as pd

data = pd.read_json('tracks.jsonl', lines=True)

plt.hist(data['explicit'], bins=[-0.5, 0.5, 1.5], edgecolor='black')
plt.xticks([0, 1])
plt.xlabel('Explicit')
plt.ylabel('Liczba utworów')
plt.title('Rozkład paremetru explicit utworów (1 - jest, 0 - nie jest)')
plt.show()

# duration_ms <= 1000 sekund
data = data[data['duration_ms'] <= 1000000]

# Konwersja duration_ms na sekundy
data['duration_s'] = data['duration_ms'] / 1000

plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(10, 6))
plt.hist(data['duration_s'], bins=50, color='skyblue', edgecolor='black')
plt.title('Rozkład czasu trwania utworów (w sekundach)')
plt.xlabel('Czas trwania (s)')
plt.ylabel('Liczba utworów')
plt.show()