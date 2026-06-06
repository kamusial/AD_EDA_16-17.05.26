class Auto:
    def __init__(self, barwa, paliwo, wiek):
        self.kolor = barwa
        self.ilosc_paliwa = paliwo
        self.kondycja = 5
        self.tryb_ekonomiczny = False
        self.spalanie_na_100 = 14
        self.mandaty = []
        self.komentarze = []
        self.rocznik = 2026 - wiek

    def zasieg(self):
        zasieg = self.ilosc_paliwa / self.spalanie_na_100 * 100
        return round(zasieg * 0.9)

    def ustaw_tryb(self, tryb):
        if tryb == 'eco':
            self.spalanie_na_100 = 10
            self.tryb_ekonomiczny = True
            print('Tryb eco')
        elif tryb == 'normal':
            self.spalanie_na_100 = 14
            self.tryb_ekonomiczny = False
            print('Tryb normal')
        else:
            print('tryb nierozpoznany, brak zmian')

    def __str__(self):
        return (f'Rocznik: {self.rocznik}, mandaty: {self.mandaty}')

    def __


print('Program')
auto1 = Auto('black', 20, 4)
print(auto1.rocznik)
auto1.kondycja = 4
print(auto1.kondycja)
print(auto1.zasieg())
print(auto1)


