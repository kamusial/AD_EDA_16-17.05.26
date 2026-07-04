# Cele zadania:
# 1. Przeliczyć wzrost i wagę na jednostki metryczne.
# 2. Porównać wzrost kobiet i mężczyzn.
# 3. Policzyć podstawowe statystyki opisowe.
# 4. Policzyć 95% przedział ufności dla średniego wzrostu.
# 5. Zinterpretować wyniki.

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

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
    t_krytyczne = stats.t.ppf(0.975, df=n - 1)
    dolna_granica = srednia - t_krytyczne * blad_standardowy
    gorna_granica = srednia + t_krytyczne * blad_standardowy
    return dolna_granica, gorna_granica

ci_kobiety = przedzial_ufnosci_95(wzrost_kobiety)
ci_mezczyzni = przedzial_ufnosci_95(wzrost_mezczyzni)

print("\n95% przedział ufności dla średniego wzrostu kobiet:")
print(ci_kobiety)

print("\n95% przedział ufności dla średniego wzrostu mężczyzn:")
print(ci_mezczyzni)

# Histogram pokazuje kształt rozkładu:
# - czy rozkład jest symetryczny,
# - gdzie znajduje się najwięcej obserwacji,
plt.hist(wzrost_kobiety, bins=30, alpha=0.5, label="Kobiety")
plt.hist(wzrost_mezczyzni, bins=30, alpha=0.5, label="Mężczyźni")
plt.title("Rozkład wzrostu kobiet i mężczyzn")
plt.xlabel("Wzrost [cm]")
plt.ylabel("Liczba osób")
plt.legend()
plt.show()

# Na podstawie wyników odpowiedz:
# 1. Która grupa ma większy średni wzrost?
# 2. Która grupa ma większe odchylenie standardowe?
# 3. Czy przedziały ufności dla kobiet i mężczyzn nachodzą na siebie?
# 4. Co oznacza 95% przedział ufności w tym przykładzie?

# Trening:
# Policz analogiczny 95% przedział ufności dla średniej wagi kobiet i mężczyzn.
# Sprawdź, jak zmieni się szerokość przedziału ufności, jeżeli użyjesz tylko losowych 100 obserwacji z każdej grupy. Wyjaśnij, dlaczego przedział zrobił się szerszy albo węższy.