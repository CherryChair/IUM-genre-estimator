from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)


def load_data():
    df_tracks = pd.read_json('prepared_track_data.jsonl', lines=True)
    df_artists = pd.read_json('prepared_artist_data.jsonl', lines=True)
    return df_tracks, df_artists


def prepare_data(df_tracks):
    features = ['popularity', 'release_year', 'danceability',
                'energy', 'speechiness', 'valence', 'duration_ms', 'explicit']
    X = df_tracks[features]
    y = df_tracks['genre']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test, features


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)


@app.route('/')
def index():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = [float(data['popularity']), int(data['release_year']), float(data['danceability']),
                  float(data['energy']), float(
                      data['speechiness']), float(data['valence']),
                  float(data['duration_ms']), int(data['explicit'])]
    prediction = model.predict([input_data])
    return jsonify({'genre': prediction.tolist()})


@app.route('/predict/artist/<id_artist>', methods=['GET'])
def predict_artist(id_artist):
    artist_songs = df_tracks[df_tracks['id_artist'] == id_artist]
    if not artist_songs.empty:
        song_features = artist_songs[features]
        genre_probabilities = model.predict_proba(song_features)
        average_probabilities = genre_probabilities.mean(axis=0)
        most_likely_genre = model.classes_[average_probabilities.argmax()]
        return jsonify({'genre': most_likely_genre})
    else:
        return jsonify({'error': 'No songs found for this artist'}), 404


@app.route('/artist/report', methods=['GET'])
def get_artists_report():
    df_artists_train, df_artists_test = train_test_split(
        df_artists, test_size=0.05)

    y_true = []
    y_pred = []
    for _, artist in df_artists_test.iterrows():
        artist_songs = df_tracks[df_tracks['id_artist'] == artist['id']]
        if not artist_songs.empty:
            song_features = artist_songs[features]
            genre_probabilities = model.predict_proba(song_features)
            average_probabilities = genre_probabilities.mean(axis=0)
            most_likely_genre = model.classes_[average_probabilities.argmax()]
            y_true.append(artist['genre'])
            y_pred.append(most_likely_genre)
    report = classification_report(y_true, y_pred)
    print(report)
    return jsonify({'classification_report': report})


if __name__ == '__main__':
    df_tracks, df_artists = load_data()
    X_train, X_test, y_train, y_test, features = prepare_data(df_tracks)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    app.run(debug=True)
