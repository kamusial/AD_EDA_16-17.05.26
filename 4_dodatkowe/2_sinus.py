import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Określenie zakresu wartości x (od -2π do 2π z krokiem 0.1)
x = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
y = np.sin(x)

# Wykres
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
plt.title('Wykres funkcji sinus')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xticks(
    ticks=[-2*np.pi, -3*np.pi/2, -np.pi, -np.pi/2, 0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
    labels=['-2π', '-3π/2', '-π', '-π/2', '0', 'π/2', 'π', '3π/2', '2π']
)
plt.show()