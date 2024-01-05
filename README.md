Środowisko: `Python 3.9.2`
Wymagane wersje bibliotek znajdują się w requirements.txt. Instalacja:
```bash
pip install -r requirements.txt
```

Dane są przygotowywane za pomocą `prepare_data_refactored.py` i `prepare_data_refactored.py`.
Do przygotowania danych do modelu podstawowego
```bash
python data_preparation/prepare_artist_data_refactored.py
python data_preparation/prepare_data_refactored.py
```
Do przygotowania danych do modelu rozszerzonego (wykonują się bardzo długo, czas wykonania`prepare_data_refactored.py` jest rzędu kilku godzin).
```bash
python data_preparation/prepare_artist_data_refactored.py -f
python data_preparation/prepare_data_refactored.py -f
```
Aplikację uruchamiamy z folderu w którym rozpakowane zostało archiwum, nadrzędnego do `prediction_models`.

```bash
python prediction_models/app.py
```