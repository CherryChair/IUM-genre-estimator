import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

data = pd.read_json('artists.jsonl', lines=True)

# 125 najpopularniejszych gatunków muzycznych
top_genres = data['genres'].explode().value_counts().head(125).index.tolist()

# gatunki, które należą do najpopularniejszych 125
data['genres'] = data['genres'].apply(lambda x: [genre for genre in x if genre in top_genres])

# najpopularniejszy gatunek dla każdego artysty
data['main_genre'] = data['genres'].apply(lambda x: max(x, key=x.count) if x else None)
data = data.dropna(subset=['main_genre'])

X = data['name']
y = data['main_genre']

clf = DummyClassifier(strategy="most_frequent")
clf.fit(X, y)
predictions = clf.predict(X)

print(classification_report(y, predictions))