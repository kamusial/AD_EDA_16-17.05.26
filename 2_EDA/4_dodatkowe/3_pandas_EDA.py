# ===================== SEKCJA 1: IMPORTY =====================
# Importujemy niezbędne biblioteki
import pandas as pd           # Główna biblioteka do analizy danych
import numpy as np            # Operacje matematyczne i tablice
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Skalowanie
import matplotlib.pyplot as plt  # Do szybkiej wizualizacji (opcjonalnie)
import warnings
warnings.filterwarnings('ignore')  # Ignorujemy ostrzeżenia dla czytelności

# ===================== SEKCJA 2: WCZYTANIE DANYCH Z INTERNETU =====================
# Używamy bezpośredniego URL do surowych danych CSV z GitHub (zbiór Melbourne Housing)
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

print("=== KROK 1: WCZYTANIE DANYCH Z INTERNETU ===")
try:
    # Wczytujemy dane bezpośrednio z URL – pandas obsługuje HTTP
    df = pd.read_csv(url)
    print("Dane zostały pomyślnie wczytane!")
except Exception as e:
    print(f"Błąd podczas wczytywania: {e}")
    exit()

# ===================== SEKCJA 3: WCZESNA EKSPLORACJA DANYCH (EDA) =====================
print("\n=== KROK 2: WCZESNA EKSPLORACJA DANYCH ===")

# Podgląd pierwszych 5 wierszy – co zawiera zbiór?
print("\n--- Pierwsze 5 wierszy danych: ---")
print(df.head())

# Informacje o strukturze danych: typy, niepuste wartości, rozmiar
print("\n--- Informacje o danych (dtypes, pamięć, brakujące): ---")
print(df.info())

# Statystyki opisowe dla kolumn numerycznych (min, max, mean, std, kwartyle)
print("\n--- Statystyki opisowe (kolumny numeryczne): ---")
print(df.describe())

# Sprawdzenie braków danych – ile brakujących wartości w każdej kolumnie?
print("\n--- Liczba brakujących wartości na kolumnę: ---")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])  # Pokazujemy tylko kolumny z brakami

# ===================== SEKCJA 4: CZYSZCZENIE DANYCH =====================
print("\n=== KROK 3: CZYSZCZENIE DANYCH ===")

# 4.1 Obsługa braków – uzupełniamy braki w 'total_bedrooms' medianą
#      (bardziej odporne na outlierki niż średnia)
print("\n--- Uzupełnianie brakujących wartości w kolumnie 'total_bedrooms' ---")
median_bedrooms = df['total_bedrooms'].median()
print(f"Mediana całkowitej liczby sypialni: {median_bedrooms}")
df['total_bedrooms'] = df['total_bedrooms'].fillna(median_bedrooms)

# 4.2 Sprawdzamy duplikaty i usuwamy je jeśli istnieją
print("\n--- Sprawdzanie duplikatów ---")
duplicates = df.duplicated().sum()
print(f"Liczba duplikatów w zbiorze: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplikaty zostały usunięte.")
else:
    print("Brak duplikatów.")

# 4.3 Identyfikacja i korekta outlierów (odstających wartości) w kolumnie 'median_house_value'
#      Używamy metody IQR (Interquartile Range) – wartości poza 1.5*IQR są outlierami
print("\n--- Identyfikacja outlierów w cenie domu ('median_house_value') ---")
Q1 = df['median_house_value'].quantile(0.25)
Q3 = df['median_house_value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['median_house_value'] < lower_bound) | (df['median_house_value'] > upper_bound)]
print(f"Liczba wykrytych outlierów w cenie: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
print(f"Dolna granica: {lower_bound}, Górna granica: {upper_bound}")

# Opcjonalnie: możemy je odciąć (w tym przykładzie pozostawiamy)
# df = df[(df['median_house_value'] >= lower_bound) & (df['median_house_value'] <= upper_bound)]

# 4.4 Zamiana kolumn z danymi tekstowymi na kategorie (typu 'object' -> 'category')
#      Zmniejsza to użycie pamięci i przyspiesza niektóre operacje.
print("\n--- Optymalizacja typów danych: zamiana 'object' na 'category' ---")
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category')
    print(f"Kolumna '{col}' zmieniona na typ category (kategorie: {df[col].nunique()})")

# ===================== SEKCJA 5: INŻYNIERIA CECH (FEATURE ENGINEERING) =====================
print("\n=== KROK 4: INŻYNIERIA CECH ===")

# 5.1 Tworzymy nową cechę: średnia liczba pokoi na gospodarstwo domowe
#      Uwaga: household to liczba gospodarstw domowych w bloku
print("\n--- Tworzenie cechy 'rooms_per_household' ---")
df['rooms_per_household'] = df['total_rooms'] / df['households']

# 5.2 Tworzymy cechę: średnia liczba sypialni na pokój
print("--- Tworzenie cechy 'bedrooms_per_room' ---")
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']

# 5.3 Tworzymy cechę: liczba osób na gospodarstwo domowe (populacja/gospodarstwa)
print("--- Tworzenie cechy 'population_per_household' ---")
df['population_per_household'] = df['population'] / df['households']

# 5.4 Dzielimy lokalizację geograficzną na przedziały (dyskretyzacja)
#      Dodajemy kolumnę z przedziałem długości geograficznej (longitude)
print("--- Dyskretyzacja długości geograficznej na przedziały ---")
df['longitude_bin'] = pd.cut(df['longitude'], bins=10, labels=False)

# 5.5 Kodowanie zmiennych kategorycznych (One-Hot Encoding) dla kolumny 'ocean_proximity'
#      Tworzymy osobne kolumny binarne dla każdej kategorii
print("\n--- One-Hot Encoding dla zmiennej 'ocean_proximity' ---")
ocean_dummies = pd.get_dummies(df['ocean_proximity'], prefix='ocean')
df = pd.concat([df, ocean_dummies], axis=1)
print(f"Dodano {ocean_dummies.shape[1]} nowych kolumn binarnych.")
print("Nowe kolumny:", list(ocean_dummies.columns))

# ===================== SEKCJA 6: SKALOWANIE DANYCH =====================
print("\n=== KROK 5: SKALOWANIE WYBRANYCH CECH ===")

# Wybieramy kolumny numeryczne do skalowania (pomijamy te już zakodowane oraz docelową)
features_to_scale = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                     'total_bedrooms', 'population', 'households', 'median_income',
                     'rooms_per_household', 'bedrooms_per_room', 'population_per_household']

# Sprawdzamy, czy wybrane kolumny istnieją
existing_features = [f for f in features_to_scale if f in df.columns]
print(f"Skalowane będą następujące cechy: {existing_features}")

# 6.1 StandardScaler: przekształca dane do rozkładu o średniej 0 i odchyleniu std = 1
print("\n--- StandardScaler (standaryzacja) ---")
scaler_std = StandardScaler()
df_scaled_std = df.copy()  # Tworzymy kopię, aby nie nadpisywać oryginału
df_scaled_std[existing_features] = scaler_std.fit_transform(df[existing_features])

# 6.2 MinMaxScaler: skaluje do zakresu [0, 1] (przydatne dla sieci neuronowych)
print("\n--- MinMaxScaler (normalizacja do przedziału [0,1]) ---")
scaler_minmax = MinMaxScaler()
df_scaled_minmax = df.copy()
df_scaled_minmax[existing_features] = scaler_minmax.fit_transform(df[existing_features])

# Prezentacja wyników skalowania – porównanie przed i po
print("\n--- Porównanie oryginalnych i przeskalowanych danych (pierwsze 5 wierszy dla 'median_income'): ---")
print("Oryginalne 'median_income':")
print(df['median_income'].head())
print("\nPo StandardScaler:")
print(df_scaled_std['median_income'].head())
print("\nPo MinMaxScaler:")
print(df_scaled_minmax['median_income'].head())

# ===================== SEKCJA 7: ZAAWANSOWANE OPERACJE GRUPOWANIA I AGREGACJI =====================
print("\n=== KROK 6: ZAAWANSOWANE GRUPOWANIE I AGREGACJE ===")

# 7.1 Grupowanie według bliskości oceanu i obliczanie wielu statystyk dla ceny domu
print("\n--- Średnia cena domu w zależności od odległości od oceanu ---")
ocean_group = df.groupby('ocean_proximity')['median_house_value'].agg(['mean', 'median', 'std', 'count'])
print(ocean_group)

# 7.2 Grupowanie po przedziałach długości geograficznej (utworzonych wcześniej)
print("\n--- Ceny i dochody według przedziałów długości geograficznej ---")
longitude_summary = df.groupby('longitude_bin').agg(
    avg_house_value=('median_house_value', 'mean'),
    avg_income=('median_income', 'mean'),
    count=('longitude_bin', 'count')
).round(2)
print(longitude_summary.head(10))

# 7.3 Zastosowanie funkcji transform() – normalizacja w obrębie grupy
#     Obliczamy cenę domu względem średniej w swojej grupie ocean_proximity
print("\n--- Cena względem średniej w swojej grupie ocean_proximity (transform) ---")
df['value_relative_to_ocean_mean'] = df.groupby('ocean_proximity')['median_house_value'].transform(
    lambda x: x / x.mean()
)
print("Nowa kolumna: 'value_relative_to_ocean_mean' (pierwsze 5 wartości):")
print(df['value_relative_to_ocean_mean'].head())

# ===================== SEKCJA 8: FILTROWANIE I KWERY ZA POMOCĄ .QUERY() =====================
print("\n=== KROK 7: ZAAWANSOWANE KWERY Z UŻYCIEM .query() ===")

# 8.1 Znajdź domy w przedziale cenowym powyżej 400k, z dochodem powyżej 5 i blisko oceanu (<1H OCEAN)
query1 = df.query("median_house_value > 400000 and median_income > 5 and ocean_proximity == '<1H OCEAN'")
print(f"Liczba domów spełniających kryteria: {len(query1)}")
if len(query1) > 0:
    print(query1[['median_house_value', 'median_income', 'ocean_proximity']].head())

# 8.2 Użycie zmiennych zewnętrznych w .query() – ceny wyższe niż 1.5 * średnia cena
avg_price = df['median_house_value'].mean()
query2 = df.query("median_house_value > @avg_price * 1.5")
print(f"\nLiczba domów z ceną > 1.5x średniej ({avg_price:.2f}): {len(query2)}")

# ===================== SEKCJA 9: ZAPIS PRZETWORZONYCH DANYCH =====================
print("\n=== KROK 8: ZAPIS PRZETWORZONEGO ZBIORU ===")

# Zapisujemy oczyszczony i wzbogacony zbiór do pliku CSV (bez skalowania, ale z nowymi cechami)
output_filename = "melbourne_housing_processed.csv"
df.to_csv(output_filename, index=False)
print(f"✅ Oczyszczony i wzbogacony zbiór zapisano jako: {output_filename}")

# Opcjonalnie: zapisujemy wersję po standaryzacji
df_scaled_std.to_csv("melbourne_housing_standardized.csv", index=False)
print(f"✅ Zbiór po standaryzacji zapisano jako: melbourne_housing_standardized.csv")

# ===================== SEKCJA 10: PODSUMOWANIE KOŃCOWE =====================
print("\n" + "="*60)
print("=== PODSUMOWANIE OPERACJI WYKONANYCH PRZEZ PROGRAM ===")
print("="*60)
print(f"1. Wczytano dane z internetu – liczba wierszy: {len(df)}")
print(f"2. Oczyszczono braki – uzupełniono medianą w 'total_bedrooms'")
print(f"3. Usunięto duplikaty (jeśli istniały)")
print(f"4. Wykryto outlierów w cenie: {len(outliers)}")
print(f"5. Dodano nowe cechy: rooms_per_household, bedrooms_per_room, population_per_household, longitude_bin")
print(f"6. Wykonano One-Hot Encoding dla kategorii ocean_proximity")
print(f"7. Przeskalowano {len(existing_features)} cech za pomocą StandardScaler i MinMaxScaler")
print(f"8. Wykonano zaawansowane grupowania i agregacje")
print(f"9. Zastosowano .query() do filtrowania")
print(f"10. Zapisano wyniki do plików CSV")
print("\n Program zakończony pomyślnie!")

# Opcjonalna wizualizacja (odkomentuj jeśli chcesz zobaczyć rozkład ceny)
# plt.hist(df['median_house_value'], bins=50, edgecolor='black')
# plt.title('Rozkład ceny domu (median_house_value)')
# plt.xlabel('Cena')
# plt.ylabel('Częstotliwość')
# plt.show()