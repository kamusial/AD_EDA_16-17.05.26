import pandas as pd
import numpy as np
from pandas import DataFrame

URL = "https://raw.githubusercontent.com/YueminLi/Airbnb_NYC_2019/master/AB_NYC_2019.csv"
df = pd.read_csv(URL)

print(type(df))
df.