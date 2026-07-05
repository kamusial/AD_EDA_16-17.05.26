import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

matplotlib.use("Agg")  # zapisujemy wykresy do plików, nie otwieramy okienek
from pathlib import Path
from urllib.parse import urlencode

# 1. USTAWIENIA ANALIZY
print("\n" + "=" * 80)
print("1. USTAWIENIA ANALIZY")
print("=" * 80)

START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
ROW_LIMIT = 1000
OUTPUT_DIR = Path('wyniki_311')
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_FILE = OUTPUT_DIR / f"nyc_311_{START_DATE}_{END_DATE}_{ROW_LIMIT}.csv"

# 2. POBRANIE DANYCH
print("\n" + "=" * 80)
print("2. POBIERANIE DANYCH")
print("=" * 80)

columns = [
    "unique_key",
    "created_date",
    "closed_date",
    "agency",
    "complaint_type",
    "borough",
    "status",
]

where = (f"created_date between '{START_DATE}T00:00:00' and '{END_DATE}T23:59:59' "
         "AND closed_date IS NOT NULL "
         "AND borough IS NOT NULL"
)

params = {
    "$select": ",".join(columns),
    "$where": where,
    "$limit": ROW_LIMIT,
    "$order": "created_date ASC",
}

url = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv?" + urlencode(params)

if CACHE_FILE.exists():
    print(f"Wczytuję dane z cache: {CACHE_FILE}")
    df = pd.read_csv(CACHE_FILE)
else:
    print("Pobieram dane z publicznego API NYC Open Data...")
    print(f"Zakres dat: {START_DATE} — {END_DATE}")
    print(f"Limit wierszy: {ROW_LIMIT}")
    df = pd.read_csv(url)
    df.to_csv(CACHE_FILE, index=False)
    print(f"Zapisano cache: {CACHE_FILE}")

print(f'Liczba pobranych wierszy: {len(df):,}'.replace(',',' '))
print("Pierwsze 5 wierszy:")
print(df.head().to_string())

# 3. CZYSZCZENIE DANYCH
print("\n" + "=" * 80)
print("3. CZYSZCZENIE DANYCH")
print("=" * 80)

before = len(df)

df["created_date"] = pd.to_datetime(df["created_date"], errors='coerce')
df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce")

df["response_time_hours"] = (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600
df["created_day"] = df["created_date"].dt.date
df['borough'] = df['borough'].astype("string").str.strip().str.upper()
df["complaint_type"] = df["complaint_type"].astype("string").str.strip()
df["agency"] = df["agency"].astype("string").str.strip()
df = df[df["response_time_hours"] > 0]
df = df[df["response_time_hours"] <= 365 * 24]
df = df[~df["borough"].isin(["UNSPECIFIED", "0", "NAN", "<NA>"])]
print(df.head().to_string())

after = len(df)

print(f"Liczba wierszy przed czyszczeniem: {before:,}".replace(",", " "))
print(f"Liczba wierszy po czyszczeniu:    {after:,}".replace(",", " "))
print(f"Usunięto:                         {before - after:,}".replace(",", " "))

print("\nProcent braków danych po czyszczeniu:")
print((df.isna().mean() * 100).round(2).sort_values(ascending=False).to_string())

# 4. STATYSTYKI OPISOWE

print("\n" + "=" * 80)
print("4. STATYSTYKI OPISOWE")
print("=" * 80)

x = df["response_time_hours"]
print("Opis zmiennej response_time_hours, czyli czasu obsługi w godzinach:")
print(x.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]).round(2).to_string())
print("\nLiczba zgłoszeń według dzielnicy:")
print(df["borough"].value_counts().to_string())

n = x.count()
mean = x.mean()
std = x.std(ddof=1)
standard_error = std / np.sqrt(n)
t_critical = stats.t.ppf(0.975, df=n - 1)
ci_low = mean - t_critical * standard_error
ci_high = mean + t_critical * standard_error

print("\nŚrednia i 95% przedział ufności:")
print(f"średnia = {mean:.2f} godzin")
print(f"95% CI = ({ci_low:.2f}, {ci_high:.2f}) godzin")

# 5. OBSERWACJE ODSTAJĄCE METODĄ IQR

print("\n" + "=" * 80)
print("5. OBSERWACJE ODSTAJĄCE — METODA IQR")
print("=" * 80)

q1 = x.quantile(0.25)
q3 = x.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

print(f"Q1 = {q1:.2f} h")
print(f"Q3 = {q3:.2f} h")
print(f"IQR = {iqr:.2f} h")
print(f"Dolna granica = {lower:.2f} h")
print(f"Górna granica = {upper:.2f} h")

# porównanie
df_iqr = df[(df["response_time_hours"] >= lower) & (df["response_time_hours"] <= upper)].copy()
removed_percent = 100 * (1 - len(df_iqr) / len(df))

comparison = pd.DataFrame(
    {
        "pełne_dane": [len(df), df["response_time_hours"].mean(), df["response_time_hours"].median()],
        "bez_outlierow_IQR": [
            len(df_iqr),
            df_iqr["response_time_hours"].mean(),
            df_iqr["response_time_hours"].median(),
        ],
    },
    index=["liczba_obserwacji", "średnia_h", "mediana_h"],
)

print("\nPorównanie przed i po filtracji IQR:")
print(comparison.round(2).to_string())
print(f"\nUsunięto {removed_percent:.2f}% obserwacji jako odstające według reguły IQR.")

# 6. WYKRESY

print("\n" + "=" * 80)
print("6. WYKRESY")
print("=" * 80)

p95 = df["response_time_hours"].quantile(0.95)
df_hist = df[df["response_time_hours"] <= p95]
plt.figure(figsize=(10, 6))
plt.hist(df_hist["response_time_hours"], bins=50)
plt.title("Czas obsługi zgłoszeń 311 — do 95. percentyla")
plt.xlabel("Czas obsługi [godziny]")
plt.ylabel("Liczba zgłoszeń")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_histogram_czasu_obslugi.png", dpi=150)
plt.close()

# Wykres słupkowy najczęstszych typów zgłoszeń.
top_types = df["complaint_type"].value_counts().head(12).sort_values()

plt.figure(figsize=(10, 7))
plt.barh(top_types.index.astype(str), top_types.values)
plt.title("Top 12 typów zgłoszeń 311")
plt.xlabel("Liczba zgłoszeń")
plt.ylabel("Typ zgłoszenia")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_top_typy_zgloszen.png", dpi=150)
plt.close()

# Boxplot czasu obsługi według dzielnicy po usunięciu outlierów.
borough_order = df_iqr["borough"].value_counts().index.tolist()
box_data = [df_iqr[df_iqr["borough"] == b]["response_time_hours"] for b in borough_order]
plt.figure(figsize=(10, 6))
plt.boxplot(box_data, showfliers=False)
plt.xticks(range(1, len(borough_order) + 1), borough_order, rotation=30)
plt.title("Czas obsługi według dzielnicy — po filtracji IQR")
plt.xlabel("Dzielnica")
plt.ylabel("Czas obsługi [godziny]")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_boxplot_dzielnice.png", dpi=150)
plt.close()

# 7. PORÓWNANIE DWÓCH DZIELNIC — TEST T-WELCHA
print("\n" + "=" * 80)
print("6. TEST T-WELCHA — PORÓWNANIE DWÓCH DZIELNIC")
print("=" * 80)

borough_a = "BROOKLYN"
borough_b = "MANHATTAN"

available_boroughs = df_iqr["borough"].value_counts().index.tolist()
if borough_a not in available_boroughs or borough_b not in available_boroughs:
    borough_a, borough_b = available_boroughs[:2]

x_a = df_iqr[df_iqr["borough"] == borough_a]["response_time_hours"]
x_b = df_iqr[df_iqr["borough"] == borough_b]["response_time_hours"]

summary_groups = pd.DataFrame(
    {
        borough_a: [len(x_a), x_a.mean(), x_a.median(), x_a.std(ddof=1)],
        borough_b: [len(x_b), x_b.mean(), x_b.median(), x_b.std(ddof=1)],
    },
    index=["n", "średnia_h", "mediana_h", "odchylenie_std_h"],
)

print(summary_groups.round(2).to_string())

test_t = stats.ttest_ind(x_a, x_b, equal_var=False)
print("\nWynik testu t-Welcha:")
print(f"t = {test_t.statistic:.4f}")
print(f"p-value = {test_t.pvalue:.6g}")

if test_t.pvalue < 0.05:
    print("Wniosek: odrzucamy H0. Średnie czasy obsługi różnią się statystycznie istotnie.")
else:
    print("Wniosek: nie mamy podstaw do odrzucenia H0. Nie widać istotnej różnicy średnich.")