import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

# 1. Tworzymy przykładowe dane

# Ustawiamy ziarno losowości, żeby wyniki były powtarzalne
np.random.seed(42)
# Tworzymy jedną cechę X
X = np.linspace(0, 10, 35).reshape(-1, 1)

# Tworzymy zmienną y z lekkim zakrzywieniem i szumem
y = 2 + 1.5 * X.ravel() - 0.15 * X.ravel() ** 2 + np.random.normal(0, 1.2, size=35)
# plt.scatter(X, y)
# plt.show()

# 2. Dzielimy dane na treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# 3. Tworzymy modele
# Używamy cech wielomianowych stopnia 15.
degree = 15

models = {
    "Bez regularyzacji": Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Ridge L2": Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "Lasso L1": Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.05, max_iter=20000))
    ])
}
results = []
# 4. Uczymy modele i liczymy błędy
for name, model in models.items():
    # Uczenie modelu
    model.fit(X_train, y_train)

    # Predykcja na danych treningowych
    y_train_pred = model.predict(X_train)

    # Predykcja na danych testowych
    y_test_pred = model.predict(X_test)

    # Liczymy błąd MSE
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    results.append({
        "model": name,
        "MSE_train": train_mse,
        "MSE_test": test_mse
    })

# Zamieniamy wyniki na tabelę
results_df = pd.DataFrame(results)

# 5. Wykres krzywych predykcji

X_grid = np.linspace(0, 10, 300).reshape(-1, 1)
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, label="Dane treningowe")
plt.scatter(X_test, y_test, label="Dane testowe")
for name, model in models.items():
    y_grid_pred = model.predict(X_grid)
    plt.plot(X_grid, y_grid_pred, label=name)

plt.title("Porównanie modeli: bez regularyzacji vs z regularyzacją")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
