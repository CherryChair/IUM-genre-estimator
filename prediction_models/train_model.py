from app import load_data, prepare_data, train_model, train_complex_model, evaluate_model
import pickle
from numpy.random import RandomState, SeedSequence

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

SEED = 789456123

if __name__ == '__main__':
    df_tracks, df_artists, df_tracks_complex, df_artists_complex = load_data()
    X_train, X_test, y_train, y_test, features = prepare_data(df_tracks)
    X_train_complex, X_test_complex, y_train_complex, y_test_complex, features_complex = prepare_data(
        df_tracks_complex)
    model = train_model(X_train, y_train, 173, 19, SEED)
    model_complex = train_complex_model(X_train_complex, y_train_complex, 29, "distance", 1)
    save_model(model, 'model.pkl')
    save_model(model_complex, 'model_complex.pkl')
    print("Base model evaluation: \n", evaluate_model(model, X_test, y_test))
    print("Complex model evaluation: \n", evaluate_model(
        model_complex, X_test_complex, y_test_complex))
