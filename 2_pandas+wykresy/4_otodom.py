import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('..\\data\\otodom.csv')
print(df)
print(df.describe().T.round(2).to_string())

sns.histplot(df.cena)
plt.show()
plt.scatter(df.powierzchnia, df.cena)
plt.show()