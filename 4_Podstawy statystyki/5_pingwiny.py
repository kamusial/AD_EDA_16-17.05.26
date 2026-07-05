import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
df = pd.read_csv(url)

print("Pierwsze wiersze danych:")
print(df.head().to_string())

print("\nKolumny:")
print(df.columns)

def interpretuj_p_value(p_value, alpha=0.05):
    if p_value < alpha:
        return "Odrzucamy H0 — wynik jest statystycznie istotny."
    else:
        return "Brak podstaw do odrzucenia H0 — wynik nie jest statystycznie istotny."


print("\n" + "=" * 70)
print("CHI-KWADRAT — TEST ZGODNOŚCI")
print("=" * 70)

species_counts = df["species"].value_counts().sort_index()
print("\nObserwowane liczebności gatunków:")
print(species_counts)

expected_counts = [   species_counts.sum() / len(species_counts)   ]  *  len(species_counts)
print(expected_counts)

chi2_stat, p_value = stats.chisquare(f_obs=species_counts, f_exp=expected_counts)

df_chi = len(species_counts) - 1
print(f"Stopnie swobody: {df_chi}")
print(f"\nStatystyka chi-kwadrat: {chi2_stat:.4f}")
print(f"p-value: {p_value:.6f}")
print(interpretuj_p_value(p_value))

# 4. TEST t-STUDENTA — PORÓWNANIE DWÓCH ŚREDNICH - zrobione wczoraj

print("\n" + "=" * 70)
print("FISHER-SNEDECOR — ANOVA JEDNOCZYNNIKOWA")
print("=" * 70)

df_anova = df[["species", "body_mass_g"]].dropna()
print("\nLiczba obserwacji po usunięciu braków danych:")
print(len(df_anova))

adelie = df_anova[df_anova["species"] == "Adelie"]["body_mass_g"]
chinstrap = df_anova[df_anova["species"] == "Chinstrap"]["body_mass_g"]
gentoo = df_anova[df_anova["species"] == "Gentoo"]["body_mass_g"]

print("\nŚrednia masa ciała według gatunku:")

print(f"Adelie:    {adelie.mean():.2f} g")
print(f"Chinstrap: {chinstrap.mean():.2f} g")
print(f"Gentoo:    {gentoo.mean():.2f} g")

f_stat, p_value = stats.f_oneway(
    adelie,
    chinstrap,
    gentoo
)

k = 3
n = len(df_anova)

df_between = k - 1
df_within = n - k

print("\nWyniki ANOVA:")
print(f"Statystyka F: {f_stat:.4f}")
print(f"Stopnie swobody: F({df_between}, {df_within})")
print(f"p-value: {p_value:.10f}")