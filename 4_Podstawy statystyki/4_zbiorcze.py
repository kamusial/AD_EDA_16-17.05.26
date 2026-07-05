import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

matplotlib.use("Agg")  # zapisujemy wykresy do plików, nie otwieramy okienek
from pathlib import Path
from urllib.parse import urlencode

# 1. USTAWIENIA ANALIZY

START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
ROW_LIMIT = 100
OUTPUT_DIR = Path('wyniki_311')
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_FILE = OUTPUT_DIR / f"nyc_311_{START_DATE}_{END_DATE}_{ROW_LIMIT}.csv"

# 2. POBRANIE DANYCH
print("\n" + "=" * 80)
print("1. POBIERANIE DANYCH")
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