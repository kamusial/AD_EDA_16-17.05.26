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

# 10. Tworzymy przedziały wieku.
clean["age_group"] = pd.cut(
    clean["age"],
    bins=[0, 12, 18, 35, 60, np.inf],
    labels=["dziecko", "nastolatek", "mlody_dorosly", "dorosly", "senior"],
    right=False
)


# 11. Zmieniamy wybrane kolumny tekstowe/logiczne na typ category.
#     Typ category oszczędza pamięć i jasno pokazuje,
#     że kolumna ma ograniczony zestaw wartości.
categorical_columns = [
    "sex", "embarked", "embark_town", "class",
    "who", "adult_male", "alive", "alone", "age_group"
]
for col in categorical_columns:
    clean[col] = clean[col].astype("category")

print(clean.head().to_string())
# print(clean["sex"].dtype)
# # print(clean["sex"].cat.categories)

# 12. Sprawdzamy duplikaty.
duplicates_before = clean.duplicated().sum()
print(f'Liczba duplikatów przed czyszczeniem: {duplicates_before}')
clean = clean.drop_duplicates()

# 13. Kontrola po czyszczeniu.
print(f'Braki po czyszczeniu\n: {clean.isna().sum().sort_values(ascending=False)}')

print("\n--- TYPY DANYCH PO CZYSZCZENIU ---")
print(clean.dtypes)

print("\n--- PRZYKŁADOWE DANE PO CZYSZCZENIU ---")
print(clean.head().to_string())

# 14. Zapisujemy oczyszczony plik.
clean.to_csv("titanic_clean.csv", index=False)
print("\nZapisano plik: titanic_clean.csv")


