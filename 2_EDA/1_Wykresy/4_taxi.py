# taxi 1 - 3zł/km + 5zł na start
# taxi 2 - 4zł/km + 1zł na start
# taxi 3 - 5zł/km + 0zł na start
# taxi 4 - 2zł/km + 10zł na start
# taxi 5 - 0zł/km + 30zł na start

import matplotlib.pyplot as plt

X = [i for i in range(0, 11)]
Y1 = [3*i + 5 for i in X]
Y2 = [4*i + 1 for i in X]
Y3 = [5*i + 0 for i in X]
Y4 = [2*i + 10 for i in X]
Y5 = [0*i + 30 for i in X]

plt.plot(X, Y1, 'ro--')
plt.plot(X, Y2, 'g^')
plt.plot(X, Y3, 'bD:')
plt.plot(X, Y4, 'o')
plt.plot(X, Y5, 'mp--')
plt.show()