from app import load_data, prepare_data
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint
from sklearn.model_selection import KFold

SEED = 789456123

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

df_tracks, df_artists, df_tracks_complex, df_artists_complex = load_data()
X_train, X_test, y_train, y_test, features = prepare_data(df_tracks)
X_train_complex, X_test_complex, y_train_complex, y_test_complex, features_complex = prepare_data(
    df_tracks_complex)

rf = RandomForestClassifier()

param_dist = {'n_estimators': randint(50,200), 'max_depth': randint(1,20)}

rand_search = RandomizedSearchCV(rf, param_distributions = param_dist, n_iter=50, cv=5, random_state=SEED)

rand_search.fit(X_train, y_train)

best_rf_1 = rand_search.best_estimator_

print('Model 1 best:',  rand_search.best_params_)

knn = KNeighborsClassifier(weights="distance", p=1)

param_grid = {'n_neighbors' : np.arange(39, 60, 1)}

knn_cv=KFold(n_splits=5,shuffle=True,random_state=SEED)

rand_search2 = GridSearchCV(knn, param_grid = param_grid, cv=knn_cv)

rand_search2.fit(X_train_complex, y_train_complex)


best_rf_2 = rand_search2.best_estimator_
print('Model 2 best:',  rand_search2.best_params_)
