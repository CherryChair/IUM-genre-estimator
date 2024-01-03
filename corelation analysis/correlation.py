import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data_tracks = pd.read_json('prepared_track_data.jsonl', lines=True)

column_names = data_tracks.columns.to_list()
skipped_params = ['name', 'id', 'release_date', 'explicit', 'mode', 'key', 'genre', 'time_signature', 'duration_ms']

for skipped_param in skipped_params:
    column_names.remove(skipped_param) 

result_array = np.zeros((10,10))

for i in range(len(column_names)):
    result_array[i, i] = 1
    for j in range(i+1, len(column_names)):
        correlation = data_tracks[column_names[i]].corr(data_tracks[column_names[j]])
        print(f'corr({column_names[i]},{column_names[j]}): {correlation}')
        result_array[i, j] = correlation
        result_array[j, i] = correlation

ax = sns.heatmap(pd.DataFrame(result_array, index=column_names, columns=column_names), vmin=-1, vmax=1, annot=True, square=True)
plt.show()