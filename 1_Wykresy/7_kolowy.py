import matplotlib.pyplot as plt

wydatki = ['mieszkanie', 'media', 'rozrywka', 'nauka', 'pokemony']
values = [2000, 400, 120, 700, 1230]
wytnij = [0 for i in wydatki]  # lista 5ciu zer
wytnij[2] = 0.3
plt.pie(values, labels=wydatki, explode=wytnij, shadow=True)
plt.show()