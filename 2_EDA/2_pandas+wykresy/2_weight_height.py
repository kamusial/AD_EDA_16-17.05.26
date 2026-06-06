import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('weight-height.csv', sep=';')
print(df)

# df['Height2'] = df['Height'] * 2.54 # nowa kolumna
# df['Height'] = df['Height'] * 2.54 # nadpisanie kolumny
df['Height'] *= 2.54
df['Weight'] /= 2.2

print('\nDescribe:')
print(df.describe().T.round(2).to_string())
# plt.hist(df['Weight'], bins=30)
plt.hist(df.query("Gender=='Male'")['Weight'], bins=30)
plt.hist(df.query("Gender=='Female'")['Weight'], bins=30)
plt.show()

sns.histplot(df.query("Gender=='Male'")['Weight'])
sns.histplot(df.query("Gender=='Female'")['Weight'])
plt.show()