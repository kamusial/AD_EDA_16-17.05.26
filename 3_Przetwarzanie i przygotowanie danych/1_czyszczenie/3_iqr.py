import pandas as pd
import numpy as np

# Przykładowe dane
df = pd.DataFrame({
    "klient": ["A", "B", "C", "D", "E", "F", "G"],
    "wydatki": [100, 120, 130, 110, 115, 125, 900]
})

# Obliczamy pierwszy kwartyl
q1 = df["wydatki"].quantile(0.25)

# Obliczamy trzeci kwartyl
q3 = df["wydatki"].quantile(0.75)

# Obliczamy IQR
iqr = q3 - q1

# Wyznaczamy granice wartości typowych
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Oznaczamy wartości odstające
df["czy_odstajaca_iqr"] = np.where(
    (df["wydatki"] < lower_bound) | (df["wydatki"] > upper_bound),
    "tak",
    "nie"
)

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Dolna granica:", lower_bound)
print("Górna granica:", upper_bound)

print(df)