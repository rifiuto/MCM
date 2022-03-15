import pandas as pd
import numpy as np
data = pd.read_csv("1.csv")
# print((data[data.columns[1]][74]))
# print((data[data.columns[1]] == "nan").sum())
print(pd.isna(data[data.columns[1]]).sum())
index = pd.isna(data[data.columns[1]])
print(data[index])
