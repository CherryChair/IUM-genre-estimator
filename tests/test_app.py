import pytest
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.insert(0, 'prediction_models')
from app import (
    prepare_data, train_model, evaluate_model, load_data)


def test_prepare_data():
    data = {
        'popularity': [10, 20, 30],
        'release_year': [2000, 2010, 2020],
        'danceability': [0.1, 0.2, 0.3],
        'energy': [0.1, 0.2, 0.3],
        'speechiness': [0.1, 0.2, 0.3],
        'valence': [0.1, 0.2, 0.3],
        'duration_ms': [100000, 200000, 300000],
        'explicit': [0, 1, 0],
        'genre': ['rock', 'pop', 'jazz']
    }
    df_tracks = pd.DataFrame(data)

    x_train, x_test, y_train, y_test, features = prepare_data(df_tracks)

    assert len(x_train) == 2
    assert len(x_test) == 1
    assert len(y_train) == 2
    assert len(y_test) == 1
    assert features == ['popularity', 'release_year', 'danceability',
                        'energy', 'speechiness', 'valence', 'duration_ms', 'explicit']


def test_train_model():
    x_train = [[10, 2000, 0.1, 0.1, 0.1, 0.1, 100000, 0],
               [20, 2010, 0.2, 0.2, 0.2, 0.2, 200000, 1]]
    y_train = ['rock', 'pop']

    model = train_model(x_train, y_train, 100)

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100


@pytest.mark.filterwarnings("ignore:Precision and F-score are ill-defined")
@pytest.mark.filterwarnings("ignore:Recall and F-score are ill-defined")
def test_evaluate_model():
    x_test = [[30, 2020, 0.3, 0.3, 0.3, 0.3, 300000, 0]]
    y_test = ['jazz']
    model = RandomForestClassifier(n_estimators=100)
    model.fit([[10, 2000, 0.1, 0.1, 0.1, 0.1, 100000, 0], [
              20, 2010, 0.2, 0.2, 0.2, 0.2, 200000, 1]], ['rock', 'pop'])

    report = evaluate_model(model, x_test, y_test)

    assert 'precision' in report
    assert 'recall' in report
    assert 'f1-score' in report


def test_load_data():
    df_tracks, df_artists, df_tracks_complex, df_artists_complex = load_data()
    assert isinstance(df_tracks, pd.DataFrame)
    assert isinstance(df_artists, pd.DataFrame)
    assert isinstance(df_tracks_complex, pd.DataFrame)
    assert isinstance(df_artists_complex, pd.DataFrame)

# Run the app server to test endpoints below


def test_index_endpoint():
    response = requests.get('http://localhost:5000/')
    assert response.status_code == 200


def test_get_artists_report_endpoint():
    response = requests.get('http://localhost:5000/artist/report')
    assert response.status_code == 200
    assert 'classification_report' in response.json()


def test_get_artists_report_complex_endpoint():
    response = requests.get('http://localhost:5000/artist/report-complex')
    assert response.status_code == 200
    assert 'precision_report' in response.json()


def test_get_artists_report_compare_endpoint():
    response = requests.get('http://localhost:5000/artist/report-compare')
    assert response.status_code == 200
    assert 'comparison_report' in response.json()


def test_predict_endpoint():
    response = requests.post('http://localhost:5000/predict', json={
        'popularity': 30,
        'release_year': 2020,
        'danceability': 0.3,
        'energy': 0.3,
        'speechiness': 0.3,
        'valence': 0.3,
        'duration_ms': 300000,
        'explicit': 0
    })

    assert response.status_code == 200
    assert 'genre' in response.json()


def test_predict_artist_endpoint():
    artist_id = "6EN9LJHqoZG0mgxLvedhcA"
    response = requests.get(
        f'http://localhost:5000/predict/artist/{artist_id}')

    assert response.status_code == 200
    assert 'genre' in response.json() or 'error' in response.json()


def test_predict_complex_endpoint():
    response = requests.post('http://localhost:5000/predict-complex', json={
        'popularity': 30,
        'release_year': 2020,
        'danceability': 0.3,
        'energy': 0.3,
        'speechiness': 0.3,
        'valence': 0.3,
        'duration_ms': 300000,
        'explicit': 0
    })
    assert response.status_code == 200
    assert 'genres' in response.json()


def test_predict_artist_complex_endpoint():
    artist_id = "6EN9LJHqoZG0mgxLvedhcA"
    response = requests.get(
        f'http://localhost:5000/predict-complex/artist/{artist_id}')
    assert response.status_code == 200
    assert 'genres' in response.json() or 'error' in response.json()
