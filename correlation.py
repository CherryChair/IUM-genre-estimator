import pandas as pd

data_tracks = pd.read_json('prepared_track_data.jsonl', lines=True)

column_names = data_tracks.columns.to_list()


for i in range(len(column_names)):
    for j in range(i+1, len(column_names)):
        skipped_params = ['name', 'id', 'release_date', 'explicit', 'mode', 'key', 'genres', 'time_signature', 'duration_ms']
        if column_names[i] in skipped_params or column_names[j] in skipped_params:
            continue
        print(f'corr({column_names[i]},{column_names[j]}): {data_tracks[column_names[i]].corr(data_tracks[column_names[j]])}')