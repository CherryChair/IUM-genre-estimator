from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle

logging.basicConfig(filename='experiment.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')
app = Flask(__name__)


def load_data():
    df_tracks = pd.read_json('prepared_track_data.jsonl', lines=True)
    df_artists = pd.read_json('prepared_artist_data.jsonl', lines=True)
    df_tracks_complex = pd.read_json(
        'prepared_track_data_complex.jsonl', lines=True)
    df_artists_complex = pd.read_json(
        'prepared_artist_data_complex.jsonl', lines=True)
    return df_tracks, df_artists, df_tracks_complex, df_artists_complex


def prepare_data(df_tracks):
    features = ['popularity', 'release_year', 'danceability',
                'energy', 'speechiness', 'valence', 'duration_ms', 'explicit']
    X = df_tracks[features]
    y = df_tracks['genre']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return x_train, x_test, y_train, y_test, features


def train_model(x_train, y_train, n_estimators):
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    return classification_report(y_test, y_pred)


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
    logging.info(
        f'[ModelA_track] Input: {input_data}, Prediction: {prediction.tolist()}')
    return jsonify({'genre': prediction.tolist()})


@app.route('/predict-complex', methods=['POST'])
def predict_complex():
    data = request.get_json(force=True)
    input_data = [float(data['popularity']), int(data['release_year']), float(data['danceability']),
                  float(data['energy']), float(
                      data['speechiness']), float(data['valence']),
                  float(data['duration_ms']), int(data['explicit'])]
    prediction_probablities = model_complex.predict_proba([input_data])
    two_most_probable = np.argsort(prediction_probablities[0])[::-1][:2]
    genres = [model_complex.classes_[index] for index in two_most_probable]
    logging.info(f'[ModelB_track] Input: {input_data}, Prediction: {genres}')
    return jsonify({'genres': genres})


@app.route('/predict/artist/<id_artist>', methods=['GET'])
def predict_artist(id_artist):
    prediction = simple_predict_artist(model, id_artist)
    if prediction is not None:
        logging.info(
            f'[ModelA_artist] Input: {id_artist}, Prediction: {prediction}')
        return jsonify({'genre': prediction})
    else:
        logging.info(f'Artist ID: {id_artist}, No songs found')
        return jsonify({'error': 'No songs found for this artist'}), 404


def simple_predict_artist(model_simple, id_artist):
    artist_songs = df_tracks[df_tracks['id_artist'] == id_artist]
    if not artist_songs.empty:
        song_features = artist_songs[features]
        genre_probabilities = model_simple.predict_proba(song_features)
        average_probabilities = genre_probabilities.mean(axis=0)
        most_likely_genre = model_simple.classes_[
            average_probabilities.argmax()]
        return most_likely_genre
    else:
        return None


@app.route('/predict-complex/artist/<id_artist>', methods=['GET'])
def predict_artist_complex(id_artist):
    prediction = complex_predict_artist(model_complex, id_artist)
    if prediction is not None:
        logging.info(
            f'[ModelB_artist] Input: {id_artist}, Prediction: {prediction}')
        return jsonify({'genres': prediction})
    else:
        logging.info(f'Artist ID: {id_artist}, No songs found')
        return jsonify({'error': 'No songs found for this artist'}), 404


def complex_predict_artist(model_complex, id_artist):
    artist_songs = df_tracks[df_tracks['id_artist'] == id_artist]
    if not artist_songs.empty:
        song_features = artist_songs[features]
        genre_probabilities = model_complex.predict_proba(song_features)
        average_probabilities = genre_probabilities.mean(axis=0)
        two_most_probable = np.argsort(average_probabilities)[::-1][:2]
        genres = [model_complex.classes_[index] for index in two_most_probable]
        return genres
    else:
        return None


@app.route('/artist/report', methods=['GET'])
def get_artists_report():
    _, df_artists_test = train_test_split(
        df_artists, test_size=0.05)

    y_true = []
    y_pred = []
    for _, artist in df_artists_test.iterrows():
        prediction = simple_predict_artist(model, artist["id"])
        if prediction is not None:
            y_true.append(artist['genre'])
            y_pred.append(prediction)
    report = classification_report(y_true, y_pred)
    print(report)
    return jsonify({'classification_report': report})


@app.route('/artist/report-complex', methods=['GET'])
def get_artists_report_complex():
    _, df_artists_test = train_test_split(
        df_artists, test_size=0.05)

    y_true = []
    y_pred = []
    for _, artist in df_artists_test.iterrows():
        prediction = complex_predict_artist(model_complex, artist['id'])
        if prediction is not None:
            y_true.append(artist['genre'])
            y_pred.append(prediction)
    return my_classification_report_complex(y_true, y_pred)


def my_classification_report_complex(y_true, y_pred):
    genre_scores = dict()
    full_positives = 0
    partial_positives = 0
    full_negatives = 0
    for genres_true, genres_predicted in zip(y_true, y_pred):
        positives = 0
        for genre_predicted in genres_predicted:
            if genre_predicted in genres_true:
                positives += 1
                if genre_predicted not in genre_scores:
                    genre_scores[genre_predicted] = {
                        "true_positive": 1, "false_positive": 0}
                else:
                    genre_scores[genre_predicted]["true_positive"] += 1
            else:
                if genre_predicted not in genre_scores:
                    genre_scores[genre_predicted] = {
                        "true_positive": 0, "false_positive": 1}
                else:
                    genre_scores[genre_predicted]["false_positive"] += 1
        if positives == 2:
            full_positives += 1
        elif positives == 1:
            partial_positives += 1
        else:
            full_negatives += 1
    report = dict()
    macro_avg_precision = 0
    for key, item in genre_scores.items():
        precision = item["true_positive"] / \
            (item["true_positive"]+item["false_positive"])
        report[key] = {"precision": precision}
        macro_avg_precision += precision
    macro_avg_precision /= len(report.keys())
    report["macro_avg_precision"] = {"precision": macro_avg_precision}
    print(report)
    print(
        f"full_positives: {full_positives}, partial_positives: {partial_positives}, full_negatives: {full_negatives}")
    return jsonify({'precision_report': report, "full_positives": full_positives, "partial_positives": partial_positives, "full_negatives": full_negatives})


@app.route('/artist/report-compare', methods=['GET'])
def get_artists_report_compare():
    _, df_artists_test = train_test_split(
        df_artists, test_size=0.05)

    y_true = []
    y_pred_simple = []
    y_pred_complex = []
    for _, artist in df_artists_test.iterrows():
        simple_prediction = simple_predict_artist(model, artist["id"])
        complex_prediction = complex_predict_artist(
            model_complex, artist["id"])
        if simple_prediction is not None and complex_prediction is not None:
            y_true.append(artist['genre'])
            y_pred_simple.append(simple_prediction)
            y_pred_complex.append(complex_prediction)
    return compare_models(y_true, y_pred_simple, y_pred_complex)


def compare_models(y_true, y_pred_simple, y_pred_complex):
    report = {"true_complex": {"true_simple": 0, "false_simple": 0},
              "false_complex": {"true_simple": 0, "false_simple": 0}}
    for genre_true, genre_predicted_simple, genres_predicted_complex in zip(y_true, y_pred_simple, y_pred_complex):
        if genre_true == genre_predicted_simple:
            if genre_predicted_simple in genres_predicted_complex:
                report["true_complex"]["true_simple"] += 1
            else:
                report["false_complex"]["true_simple"] += 1
        else:
            if genre_true in genres_predicted_complex:
                report["true_complex"]["false_simple"] += 1
            else:
                report["false_complex"]["false_simple"] += 1
    print(report)
    return jsonify({"comparison_report": report})


def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_model_to_app():
    global model
    model = load_model('model.pkl')
    global model_complex
    model_complex = load_model('model_complex.pkl')


if __name__ == '__main__':
    load_model_to_app()
    df_tracks, df_artists, df_tracks_complex, df_artists_complex = load_data()
    X_train, X_test, y_train, y_test, features = prepare_data(df_tracks)
    X_train_complex, X_test_complex, y_train_complex, y_test_complex, features_complex = prepare_data(
        df_tracks_complex)
    app.run()
