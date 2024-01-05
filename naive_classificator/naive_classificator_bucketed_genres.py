import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

data = pd.read_json('prepared_artist_data.jsonl', lines=True)

# 125 najpopularniejszych gatunków muzycznych

# gatunki, które należą do najpopularniejszych 125

# najpopularniejszy gatunek dla każdego artysty
X = data['name']
y = data['genre']

clf = DummyClassifier(strategy="most_frequent")
clf.fit(X, y)
predictions = clf.predict(X)

print(classification_report(y, predictions))