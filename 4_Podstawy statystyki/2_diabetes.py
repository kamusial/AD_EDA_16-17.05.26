import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("..\\2_EDA\\data\\diabetes.csv")
print(df.head())
kolumny_do_poprawy = ["glucose", "bloodpressure", "skinthickness", "insulin", "bmi"]

for kolumna in kolumny_do_poprawy:
    df[kolumna] = df[kolumna].replace(0, np.nan)

for kolumna in kolumny_do_poprawy:
    df[kolumna] = df[kolumna].fillna(df[kolumna].median())

print("\nLiczebność grup:")
print(df["outcome"].value_counts())
grupa_0 = df[df["outcome"] == 0]
grupa_1 = df[df["outcome"] == 1]

print("\nStatystyki dla glucose:")

print("\noutcome = 0")
print("Średnia:", grupa_0["glucose"].mean())
print("Mediana:", grupa_0["glucose"].median())
print("Odchylenie standardowe:", grupa_0["glucose"].std())

print("\noutcome = 1")
print("Średnia:", grupa_1["glucose"].mean())
print("Mediana:", grupa_1["glucose"].median())
print("Odchylenie standardowe:", grupa_1["glucose"].std())

plt.hist(grupa_0["glucose"], bins=30, alpha=0.5, label="outcome = 0")
plt.hist(grupa_1["glucose"], bins=30, alpha=0.5, label="outcome = 1")

plt.title("Rozkład glucose w grupach outcome = 0 i outcome = 1")
plt.xlabel("Glucose")
plt.ylabel("Liczba osób")
plt.legend()
plt.show()

# Test t-Welcha
# Hipoteza zerowa H0:
# średni poziom glucose w grupie outcome = 0
# jest taki sam jak w grupie outcome = 1.
# Hipoteza alternatywna H1:
# średni poziom glucose w obu grupach jest różny.
# Chcemy sprawdzić, czy różnica między grupami jest statystycznie istotna.

wynik = stats.ttest_ind(
    grupa_0["glucose"],
    grupa_1["glucose"],
    equal_var=False  # różna wariancja w grupach
)

print("\nTest t Welcha dla glucose:")
print("Statystyka testowa:", wynik.statistic)    # średnia glucose w grupa_0 jest dużo większa niż średnia glucose w grupa_1
print("p-value:", wynik.pvalue)  # jak bardzo nasze dane są zgodne z hipotezą zerową
if wynik.pvalue < 0.05:
    print("Różnica średnich glucose jest statystycznie istotna.")
else:
    print("Brak podstaw do stwierdzenia statystycznie istotnej różnicy.")

# Prawdopodobieństwo warunkowe.
# jakie jest prawdopodobieństwo jednego zdażenia, jeśli wiemy, że zaszło inne zdażenie?
# jaki odsetek osób ma outcome = 1, jeśli ich glukoza jest powyżej mediany
mediana_glucose = df["glucose"].median()
df["wysokie_glucose"] = df["glucose"] > mediana_glucose

tabela = pd.crosstab(
    df['wysokie_glucose'],
    df['outcome'],
    normalize="index"
)
print("\nPrawdopodobieństwo outcome pod warunkiem wysokiego lub niskiego glucose:")
print(tabela)