import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.svm import SVC  #clasifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

df = pd.read_csv('iris.csv')
print(df)
print(df.describe().T.round().to_string())

X = df.iloc[: , :-1]
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

print('\nRegresja logistyczna')
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

sns.scatterplot(data=df, x='sepallength', y='sepalwidth', hue='class')
plt.scatter(5.6, 3.2, c='r')
plt.show()

sns.scatterplot(data=df, x='petallength', y='petalwidth', hue='class')
plt.scatter(5.2, 1.45, c='r')
plt.show()

sample = np.array([5.6, 3.2, 5.2, 1.45])
df["distance"] = (df.sepallength-sample[0])**2 + (df.sepalwidth-sample[1])**2 +\
                 (df.petallength-sample[2])**2 + (df.petalwidth-sample[3])**2


print('\nKNN')
model = KNeighborsClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))




print('\nDrzewo decyzyjne')
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

print('\nSVN')
model = SVC()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))