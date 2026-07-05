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

