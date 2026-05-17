import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. pobieranie danych

url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-03-31/beers.csv"

try:
    df = pd.read_csv(url)
    print('dane pobrane')
except Exception as e:
    print(f'Bląd {e}')
    print('Uzywam danych zapasowych')
    data = {
        'nazwa': ['IPA','IPA', 'Lager', 'Stout', 'Pilsner', 'Wheat', 'Porter', 'Ale', 'Bock'],
        'alkohol': [6.5, 6.5, 5.0, 7.2, 4.8, 5.2, 5.8, 5.5, 6.8],
        'goryczka': [65, 65, np.nan, 45, 30, 15, 40, 35, 25],
        'ocena': [4.2, 4.2, 3.8, 4.5, 3.9, 3.7, 4.1, 4.0, 4.3],
        'styl': ['IPA', 'IPA', 'Lager', 'Ciemne', 'Lager', np.nan, 'Ciemne', 'Jasne', 'Ciemne']
    }
    df = pd.DataFrame(data)

# 2. Podstawowe info
print('\n' + '='*50)
print(f'Wymiary danych: {df.shape}')
print(f'Liczba wierszy: {df.shape[0]}')
print(f'Liczba kolumn: {df.shape[1]}')

# 3. Podgląd danych
print('Pierwsze 5 piw')
print(df.head())
print('Ostatnie 3 piwa')
print(df.tail(3))

# 4. Typy danych
print(f'\n{df.info()}')

# 5. Statystyki numeryczne
kolumny_numeryczne = df.select_dtypes(include='number').columns   # nazwy kolumn numerycznych
if len(kolumny_numeryczne) > 0:
    print('Statystyki dla cech numerycznych:')
    print(df[kolumny_numeryczne].describe().T.to_string())
else:
    print('Brak kolumn numerycznych w danych')

# 6. Statystyki kategorycze

# 7. Brakujące wartości

# 8. Wizualizacje
print("\n" + "=" * 50)
print("TWORZENIE WYKRESÓW")
print("=" * 50)

# wykres 1, rozklad zawartości alkoholu
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1) # jeden wiersz, 2 kolumny, pierwszy z lewej
df['alkohol'].hist(bins=10, color='lightblue', edgecolor='black')
plt.title('Rozklad zawartosci alkoholu')
plt.xlabel('Zawartosc alko w (%)')
plt.ylabel('Liczba piw')
plt.subplot(1, 2, 2)   # z prawej
df.boxplot(column='alkohol', grid=False)
plt.title('Boxplot: Zawartość alkoholu')
plt.tight_layout()
plt.show()

