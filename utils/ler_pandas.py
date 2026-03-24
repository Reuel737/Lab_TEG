# import pandas as pd

# df = pd.read_pickle('\\wsl.localhost\Ubuntu\home\Lab_TEG\projetos\IC-Reuel\dados_filtrados\pandas_regioes\transform-fp-head_cellcenter.pandas')

# print(df.shape)        # (45, 4)
# print(df.head())
# print(df.dtypes)

import pandas as pd
df = pd.read_pickle('dados_filtrados/pandas_regioes/transform-lrp-head_cellcenter.pandas')
print(df.columns.tolist())
print(df.shape)