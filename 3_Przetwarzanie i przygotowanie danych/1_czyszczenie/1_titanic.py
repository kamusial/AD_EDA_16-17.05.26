import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Pobranie pliku CSV bezpośrednio z internetu.
# https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(URL)

# 2. Pierwsze rozpoznanie danych.
print("\n--- ROZMIAR DANYCH ---")
print(df.shape)
print("\n--- PIERWSZE 5 WIERSZY ---")
print(df.head().to_string())
print("\n--- INFORMACJE O KOLUMNACH ---")
df.info()
# describe
# 3. Analiza braków danych.
print("\n--- Analiza braków danych ---")
missing = pd.DataFrame({
    "liczba_brakow": df.isna().sum(),
    "procent_brakow": (df.isna().mean() * 100).round(2)
})
print(missing.sort_values("procent_brakow", ascending=False))
# sns.pairplot(df, hue='alive')
# plt.show()

# 4. Tworzymy kopię danych.
clean = df.copy()

# 5. Kolumna deck ma bardzo dużo braków.
clean = clean.drop(columns=["deck"])

# 6. mediana wieku w grupach płeć + klasa biletu
age_median_by_group = clean.groupby(["sex", "pclass"])["age"].transform("median")
clean["age"] = clean["age"].fillna(age_median_by_group)

# 7. Gdyby jakaś grupa nie miała mediany, uzupełniamy pozostałe braki medianą globalną.
clean["age"] = clean["age"].fillna(clean["age"].median())

# 8. Braki w porcie wejścia na statek uzupełniamy dominantą.
for col in ["embarked", "embark_town"]:
    most_common_value = clean[col].mode(dropna=True)[0]
    clean[col] = clean[col].fillna(most_common_value)

# 9. Dodajemy zmienną pochodną z NumPy.
clean["is_child"] = np.where(clean["age"] < 18, 1, 0)
