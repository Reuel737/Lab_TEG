import pandas as pd

df = pd.read_pickle('ListaVars.pandas')

print(df.shape)        # (45, 4)
print(df.head())
print(df.dtypes)