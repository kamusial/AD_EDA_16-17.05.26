import matplotlib.pyplot as plt

# y = 5x - 2
# X = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
X = [i for i in range(0, 10)]
Y = [5*i - 2 for i in X]
print(X)
print(Y)
plt.plot(X, Y, 'ro--')
plt.show()