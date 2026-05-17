import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# pobrać penguins z sns.load_dataset
penguins = sns.load_dataset('penguins')
print(penguins)
print(type(penguins))

sns.pairplot(penguins, hue='species')
plt.show()

print(penguins.describe().T.to_string())
print(f'Nazwy kolumn: {penguins.columns}')
penguins_filtered = penguins.drop(columns=['island', 'sex']).dropna()
