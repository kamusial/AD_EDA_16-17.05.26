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
penguin_features = penguins_filtered.drop(columns=['species'])
target = pd.get_dummies(penguins_filtered['species'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(penguin_features, target, test_size=0.2, random_state=0)

from tensorflow import keras
from numpy.random import seed
from tensorflow.random import set_seed
seed(1)
set_seed(2)

print(X_train.shape)

inputs = keras.Input(shape=[X_train.shape[1],])
hidden_layer = keras.layers.Dense(10, activation='relu')(inputs)
output_layer = keras.layers.Dense(3, activation='softmax')(hidden_layer)

model = keras.Model(inputs=inputs, outputs=output_layer)
model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())
history = model.fit(X_train, y_train, epochs=100)
sns.lineplot(x=history.epoch, y=history.history['loss'])
plt.show()

y_pred = model.predict(X_test)
prediction = pd.DataFrame(y_pred, columns=target.columns)
predicted_species = prediction.idxmax(axis='columns')
from sklearn.metrics import confusion_matrix
true_species = y_test.idxmax(axis='columns')
matrix = confusion_matrix(true_species, predicted_species)
confusion_df = pd.DataFrame(matrix, index=y_test.columns.values, columns=y_test.columns.values)
sns.heatmap(confusion_df, annot=True)
plt.show()