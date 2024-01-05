import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

data = pd.read_json('tracks.jsonl', lines=True)
data['release_year'] = pd.to_datetime(data['release_date']).dt.year
data.drop('release_date', axis=1, inplace=True)

sns.set(style="whitegrid")
fig, axes = plt.subplots(4, 3, figsize=(18, 16))

attributes = [
    'popularity', 'release_year', 'danceability', 'energy', 'key',
    'speechiness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'time_signature', 'mode'
]

for i, attr in enumerate(attributes):
    row, col = i // 3, i % 3
    sns.histplot(data[attr], ax=axes[row, col], kde=True, bins=30)
    axes[row, col].set_title(f'{attr}', fontsize=14)
    axes[row, col].set_xlabel('')
    axes[row, col].set_ylabel('')

plt.tight_layout(pad=3.5)
plt.show()