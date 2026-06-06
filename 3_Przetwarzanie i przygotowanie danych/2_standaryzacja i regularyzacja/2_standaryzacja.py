import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Załóżmy, że chcemy przewidzieć, czy klient kupi produkt.
# Mamy trzy cechy:
# - wiek
# - dochód
# - liczba wizyt na stronie
data = pd.DataFrame({
    "wiek": [22, 25, 47, 52, 46, 56, 23, 35, 40, 60],
    "dochod": [3000, 3500, 12000, 15000, 11000, 18000, 3200, 7000, 9000, 20000],
    "wizyty": [2, 3, 10, 12, 9, 15, 1, 6, 8, 16],
    "kupil": [0, 0, 1, 1, 1, 1, 0, 0, 1, 1]
})

print("\n--- DANE ORYGINALNE ---")
print(data)

# Oddzielamy cechy X od zmiennej docelowej y
# X = dane wejściowe, czyli cechy używane przez model
X = data[["wiek", "dochod", "wizyty"]]

# y = to, co model ma przewidywać
y = data["kupil"]

# Dzielimy dane na treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

scaler = StandardScaler()
# fit_transform robimy tylko na danych treningowych.
# scaler uczy się średniej i odchylenia standardowego z X_train.
X_train_scaled = scaler.fit_transform(X_train)

# transform robimy na danych testowych.
# Używamy tych samych średnich i odchyleń, które policzono na treningu.
X_test_scaled = scaler.transform(X_test)

print("\n--- DANE TRENINGOWE PO STANDARYZACJI ---")
print(X_train_scaled)

