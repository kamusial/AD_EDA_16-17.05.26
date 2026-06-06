import pandas as pd
import numpy as np

# 1. Pobranie danych z internetu.
URL = "https://raw.githubusercontent.com/YueminLi/Airbnb_NYC_2019/master/AB_NYC_2019.csv"
df = pd.read_csv(URL)

# 2. Podstawowe rozpoznanie danych.
print("\n--- ROZMIAR DANYCH ---")
print(df.shape)

print("\n--- PIERWSZE 5 WIERSZY ---")
print(df.head())

print("\n--- INFORMACJE O KOLUMNACH ---")
df.info()

# 3. Analiza zmiennych liczbowych.
#    percentyle 1%, 5%, 95%, 99% pomagają zobaczyć wartości skrajne.

numeric_columns = ["price", "minimum_nights", "number_of_reviews",
                   "reviews_per_month", "availability_365"]

print("\n--- OPIS ZMIENNYCH LICZBOWYCH ---")
print(df[numeric_columns].describe(percentiles=[0.01, 0.05, 0.95, 0.99]).T.round(2).to_string())

# 4. Analiza braków.
missing = pd.DataFrame({
    "liczba_brakow": df.isna().sum(),
    "procent_brakow": (df.isna().mean() * 100).round(2)
})

print("\n--- BRAKI DANYCH ---")
print(missing.sort_values("procent_brakow", ascending=False))

# 5. Kopia robocza.
clean = df.copy()

# 6. Ujednolicenie nazw kolumn.
#    To dobra praktyka, szczególnie gdy pliki pochodzą z różnych źródeł.
clean.columns = clean.columns.str.strip().str.lower()
rows_before = len(clean)

# 7. Cena 0 jest nierealistyczna dla oferty noclegowej.
clean = clean[clean["price"] > 0].copy()

# 8. reviews_per_month:
#    Brak wartości często oznacza brak recenzji.
#    Dlatego uzupełniamy brak wartością 0.
clean["reviews_per_month"] = clean["reviews_per_month"].fillna(0)

# 9. last_review zamieniamy na prawdziwą datę.
#    errors="coerce" oznacza: jeśli pandas nie potrafi sparsować daty,
#    wstawi NaT, czyli brak daty.
clean["last_review"] = pd.to_datetime(clean["last_review"], errors="coerce")

# 10. Czyszczenie kolumn tekstowych.
#     Usuwamy nadmiarowe spacje i uzupełniamy braki tekstem technicznym.
text_columns = ["name", "host_name", "neighbourhood_group",
                "neighbourhood", "room_type"]

for col in text_columns:
    clean[col] = (
        clean[col]
        .fillna("brak informacji")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

# 11. Kontrola współrzędnych.
bad_geo = ~clean["latitude"].between(40.45, 40.95) | ~clean["longitude"].between(-74.30, -73.65)
print("\n--- LICZBA REKORDÓW Z PODEJRZANYMI WSPÓŁRZĘDNYMI ---")
print(bad_geo.sum())
clean = clean.loc[~bad_geo].copy()

# IQR, w pliku nr 3




print(clean.head(10).to_string())

