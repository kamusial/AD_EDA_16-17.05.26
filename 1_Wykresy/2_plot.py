import matplotlib.pyplot as plt

numbers1 = [5, 10, 15, 3, 20]
numbers2 = [10, 15, 3, 20, 3]
# plt.plot(numbers)
# plt.plot(numbers, 'o')   # niepołączone punkty
# plt.plot(numbers, '.')   # niepołączone kropki
plt.plot(numbers1, 's')   # niepołączone kwadraty
# plt.plot(numbers, 'ro')   # czerwone punkty
# plt.plot(numbers, 'g^')   # zielone trókąty
# plt.plot(numbers, 'r-')   # czerwona linia
# plt.plot(numbers, 'bD:')   # niebieskie diamenty połączone kropkami
plt.plot(numbers2, 'mp--')   # różowe pięciokąty połączone przerywanymi liniami

plt.show()