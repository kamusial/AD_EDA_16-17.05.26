import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
df = pd.read_csv(url)

print("Pierwsze wiersze danych:")
print(df.head().to_string())

print("\nKolumny:")
print(df.columns)

print("\n" + "=" * 70)
print("CHI-KWADRAT — TEST ZGODNOŚCI")
print("=" * 70)

species_counts = df["species"].value_counts().sort_index()
print("\nObserwowane liczebności gatunków:")
print(species_counts)

expected_counts = [   species_counts.sum() / len(species_counts)   ]  *  len(species_counts)
print(expected_counts)

chi2_stat, p_value = stats.chisquare(f_obs=species_counts, f_exp=expected_counts)
print(f"\nStatystyka chi-kwadrat: {chi2_stat:.4f}")
print(f"p-value: {p_value:.6f}")

