## Zrzuty ekranu z aplikacji
[![image1.png](https://i.postimg.cc/sDpXwsHx/image1.png)](https://postimg.cc/gn2m0FWb)
## Uruchamianie
Środowisko: `Python 3.9.2`
Wymagane wersje bibliotek znajdują się w requirements.txt. Instalacja:
```bash
pip install -r requirements.txt
```
Dane są przygotowywane za pomocą `prepare_data.py` i `prepare_artist_data.py`. Przetwarzają one dane z plików `artists.jsonl` i `tracks.jsonl`.
Do przygotowania danych do modelu podstawowego
```bash
python data_preparation/prepare_artist_data.py
python data_preparation/prepare_data.py
```
Do przygotowania danych do modelu rozszerzonego (wykonują się bardzo długo, czas wykonania`prepare_data.py` jest rzędu kilku godzin).
```bash
python data_preparation/prepare_artist_data -c
python data_preparation/prepare_data -c
```
Do wytrenowania modeli konieczne jest wykonanie
```bash
python prediction_models/train_model.py
```
Aplikację uruchamiamy z folderu w którym rozpakowane zostało archiwum, nadrzędnego do `prediction_models`. Do działania aplikacji konieczne są 
```bash
python prediction_models/app.py
```
Aby uruchomić testy należy wpisać przy włączonej aplikacji 
```bash
pytest
```
w folderze, w którym znajduje się plik test_app.py
