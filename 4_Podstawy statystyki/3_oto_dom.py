import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("..\\2_EDA\\data\\otodom.csv")

print(df.head())
print(df.columns)

df = df[["cena", "powierzchnia"]]
df = df.dropna()
df = df[df["cena"] > 0]
df = df[df["powierzchnia"] > 0]

# Dodajemy nową zmienną: cena za metr kwadratowy.
df["cena_m2"] = df["cena"] / df["powierzchnia"]
print("\nOpis ceny za m2:")
print(df["cena_m2"].describe().round(2).T.to_string())

# plt.hist(df['cena_m2'], bins=30)
# plt.show()

# Liczymy kwartyle i IQR.
q1 = df["cena_m2"].quantile(0.25)
q3 = df["cena_m2"].quantile(0.75)
iqr = q3 - q1
print("\nQ1:", q1)
print("Q3:", q3)
print("IQR:", iqr)

# Wyznaczamy granice dla obserwacji odstających metodą IQR.
dolna_granica = q1 - 1.5 * iqr
gorna_granica = q3 + 1.5 * iqr

df_bez_outlierow = df[
    (df["cena_m2"] >= dolna_granica) &
    (df["cena_m2"] <= gorna_granica)
]
print("\nLiczba mieszkań przed usunięciem outlierów:", len(df))
print("Liczba mieszkań po usunięciu outlierów:", len(df_bez_outlierow))

# Porównujemy średnią i medianę przed oraz po usunięciu outlierów.
print("\nPrzed usunięciem outlierów:")
print("Średnia cena za m2:", df["cena_m2"].mean())
print("Mediana ceny za m2:", df["cena_m2"].median())

print("\nPo usunięciu outlierów:")
print("Średnia cena za m2:", df_bez_outlierow["cena_m2"].mean())
print("Mediana ceny za m2:", df_bez_outlierow["cena_m2"].median())

# Histogram ceny za m2.
plt.hist(df_bez_outlierow["cena_m2"], bins=30)
plt.title("Rozkład ceny za m2 po usunięciu outlierów")
plt.xlabel("Cena za m2")
plt.ylabel("Liczba mieszkań")
plt.show()

# Współczynnik korelacji Pearsona
korelacja = stats.pearsonr(
    df_bez_outlierow["powierzchnia"],
    df_bez_outlierow["cena"]
)
print("\nKorelacja Pearsona między powierzchnią a ceną:")
print("r:", korelacja.statistic)
print("p-value:", korelacja.pvalue)

# Regresja liniowa
regresja = stats.linregress(
    df_bez_outlierow["powierzchnia"],
    df_bez_outlierow["cena"]
)
print("\nRegresja liniowa:")
print("Nachylenie:", regresja.slope)
print("Wyraz wolny:", regresja.intercept)
print("R^2:", regresja.rvalue ** 2)
print("p-value:", regresja.pvalue)

# wykres punktowy
plt.scatter(
    df_bez_outlierow["powierzchnia"],
    df_bez_outlierow["cena"],
    alpha=0.4
)

plt.title("Zależność ceny od powierzchni")
plt.xlabel("Powierzchnia [m2]")
plt.ylabel("Cena")
plt.show()

# Policz korelację Spearmana między powierzchnia i cena.
# Porównaj ją z korelacją Pearsona i sprawdź różnicę między tymi miarami.

pearson = stats.pearsonr(
    df_bez_outlierow["powierzchnia"],
    df_bez_outlierow["cena"]
)

print("\nKORELACJA PEARSONA")
print("r:", pearson.statistic)
print("p-value:", pearson.pvalue)

spearman = stats.spearmanr(
    df_bez_outlierow["powierzchnia"],
    df_bez_outlierow["cena"]
)

print("\nKORELACJA SPEARMANA")
print("rho:", spearman.statistic)
print("p-value:", spearman.pvalue)

print("\nPORÓWNANIE KORELACJI")
print("Pearson r:", pearson.statistic)
print("Spearman rho:", spearman.statistic)

if abs(pearson.statistic - spearman.statistic) < 0.1:
    print("Wyniki Pearsona i Spearmana są podobne.")
    print("Może to sugerować, że zależność jest dość regularna i zbliżona do liniowej.")
elif abs(spearman.statistic) > abs(pearson.statistic):
    print("Korelacja Spearmana jest silniejsza niż Pearsona.")
    print("Może to sugerować zależność monotoniczną, ale nie idealnie liniową.")
else:
    print("Korelacja Pearsona jest silniejsza niż Spearmana.")
    print("Warto sprawdzić wykres punktowy i możliwy wpływ obserwacji odstających.")