from flask import Flask, request, jsonify, render_template
import json
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import flask

# Przygotowanie danych
df = pd.read_json('prepared_track_data.jsonl', lines=True)
features = ['popularity', 'release_year', 'danceability', 'energy', 'speechiness', 'valence', 'duration_ms', 'explicit']
X = df[features]
y = df['genre']

# Podział danych na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Trenowanie modelu KNN
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)

# Predykcja na danych testowych
y_pred = knn.predict(X_test)
print(f'Dokładność modelu: {accuracy_score(y_test, y_pred)}')
report = classification_report(y_test, y_pred)
print(report)

# Serwowanie predykcji
app = flask.Flask(__name__)

@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Convert all input data to float
    input_data = [float(data['popularity']), int(data['release_year']), float(data['danceability']), 
                  float(data['energy']), float(data['speechiness']), float(data['valence']), 
                  float(data['duration_ms']), int(data['explicit'])]
    prediction = knn.predict([input_data])
    return jsonify({'genre': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

# Logika eksperymentu A/B i zbieranie danych nie została tutaj zaimplementowana
# ze względu na złożoność i zakres tego zadania.