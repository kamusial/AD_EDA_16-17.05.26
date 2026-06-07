import pandas as pd
import numpy as np

df = pd.DataFrame({
    'key': ['A', 'B', 'C', 'A', 'A', 'B'], 
    'value': [1, 2, 3, 4, 5, 6],
    'value2': [10, 20, 30, 40, 50, 60]
})

group = df.groupby('key')

print(group)

for key, group in group:
    print(group)