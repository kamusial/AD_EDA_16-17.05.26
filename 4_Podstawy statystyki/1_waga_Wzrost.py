# Cele zadania:
# 1. Przeliczyć wzrost i wagę na jednostki metryczne.
# 2. Porównać wzrost kobiet i mężczyzn.
# 3. Policzyć podstawowe statystyki opisowe.
# 4. Policzyć 95% przedział ufności dla średniego wzrostu.
# 5. Zinterpretować wyniki.

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("..\\2_EDA\\data\\weight-height.csv", sep=";")
print(df.head())
df["Height_cm"] = df["Height"] * 2.54
df["Weight_kg"] = df["Weight"] / 2.2

# porównanie grup
kobiety = df[df["Gender"] == "Female"]
mezczyzni = df[df["Gender"] == "Male"]
wzrost_kobiety = kobiety["Height_cm"]
wzrost_mezczyzni = mezczyzni["Height_cm"]

### STATYSTYKI OPISOWE
# ŚREDNIA:
# MEDIANA:
# WARIANCJA:
# ODCHYLENIE STANDARDOWE:

print("KOBIETY")
print("Liczba obserwacji:", len(wzrost_kobiety))
print("Średnia:", wzrost_kobiety.mean())
print("Mediana:", wzrost_kobiety.median())
print("Wariancja:", wzrost_kobiety.var())
print("Odchylenie standardowe:", wzrost_kobiety.std())

print("\nMĘŻCZYŹNI")
print("Liczba obserwacji:", len(wzrost_mezczyzni))
print("Średnia:", wzrost_mezczyzni.mean())
print("Mediana:", wzrost_mezczyzni.median())
print("Wariancja:", wzrost_mezczyzni.var())
print("Odchylenie standardowe:", wzrost_mezczyzni.std())

### PRZEDZIAŁ UFNOŚCI DLA ŚREDNIEJ

def przedzial_ufnosci_95(dane):
    srednia = dane.mean()
    odchylenie = dane.std()
    n = len(dane)

    blad_standardowy = odchylenie / (n ** 0.5)