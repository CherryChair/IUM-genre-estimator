import json

expected_ranges = {
    "acousticness": (0.0, 1.0),
    "danceability": (0.0, 1.0),
    "energy": (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
    "key": (-1, 11),
    "liveness": (0.0, 1.0),
    "loudness": (-60.0, 0.0),
    "mode": (0, 1),
    "speechiness": (0.0, 1.0),
    "tempo": (0.0, float('inf')),
    "time_signature": (3, 7),
    "valence": (0.0, 1.0)
}

required_attributes = [
    "id", "name", "popularity", "duration_ms", "explicit", "id_artist",
    "release_date", "danceability", "energy", "key", "mode", "loudness",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "time_signature"
]

def check_attributes(track, outfile, consider_mode=False):
    for attr in required_attributes:
        if attr not in track:
            outfile.write(f"Missing attribute: {attr} in track ID {track.get('id', 'unknown')}\n")
            continue
        if track[attr] is None:
            if consider_mode is True and attr == "mode":
                outfile.write(f"Null value in attribute: {attr} in track ID {track.get('id', 'unknown')}\n")
            continue

        if attr in expected_ranges:
            min_val, max_val = expected_ranges[attr]
            if not (min_val <= track[attr] <= max_val):
                outfile.write(f"Invalid value for {attr} in track ID {track['id']}: {track[attr]}\n")

def process_file(input_filename, output_filename):
    with open(output_filename, 'w') as outfile:
        with open(input_filename, 'r') as infile:
            for line in infile:
                track = json.loads(line)
                check_attributes(track, outfile)

if __name__ == '__main__':
    process_file('tracks.jsonl', 'results.txt')