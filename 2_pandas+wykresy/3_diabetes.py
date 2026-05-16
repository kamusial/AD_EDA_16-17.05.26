import pandas as pd
import numpy as np
df = pd.read_csv('..\\data\\diabetes.csv')
print(f'Kształt danych: {df.shape}')
print(df.describe().T.round(2).to_string())

# wpisać wartosci srednie (bez zer) tam, gdzie 0 albo brak wartości
for col in ['glucose', 'bloodpressure', 'skinthickness', 'insulin',
       'bmi', 'diabetespedigreefunction', 'age']:
    # usunąć zera
    df[col] = df[col].replace(0, np.nan)
    # policzyć średnia
    mean_= df[col].mean()
    # wpisać średnia tam, gdzie brak wartości
    df[col] = df[col].replace(np.nan, mean_)

print('\nPo czyszczeniu')
print(df.describe().T.round(2).to_string())
df.to_csv('..\\data\\po_zmianach_cukrzyca.csv', index=False)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X = df.iloc[: , :-1]
y = df.outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test) ) ))

print(df.outcome.value_counts())
df1 = df.query("outcome==0").sample(n=500)   # 500 zdrowych
df2 = df.query("outcome==1").sample(n=500)   # 500 chorych
df3 = pd.concat([df1, df2])   # 500 zdrowych i 500 chorych

X = df3.iloc[: , :-1]
y = df3.outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test) ) ))
