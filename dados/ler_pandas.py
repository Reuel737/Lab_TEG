<<<<<<< HEAD:utils/ler_pandas.py
# import pandas as pd

# df = pd.read_pickle('\\wsl.localhost\Ubuntu\home\Lab_TEG\projetos\IC-Reuel\dados_filtrados\pandas_regioes\transform-fp-head_cellcenter.pandas')

# print(df.shape)        # (45, 4)
# print(df.head())
# print(df.dtypes)

import pandas as pd
df = pd.read_pickle('dados_filtrados/pandas_regioes/transform-fp-head_cellcenter.pandas')
print(df.columns.tolist())
=======
# import pandas as pd

# df = pd.read_pickle('\\wsl.localhost\Ubuntu\home\Lab_TEG\projetos\IC-Reuel\dados_filtrados\pandas_regioes\transform-fp-head_cellcenter.pandas')

# print(df.shape)        # (45, 4)
# print(df.head())
# print(df.dtypes)

import pandas as pd
df = pd.read_pickle('dados_filtrados/pandas_regioes/transform-lrp-head_cellcenter.pandas')
print(df.columns.tolist())
print(df.head())
>>>>>>> d61ff4ba3d93a406d63891e2350c8fdda3a4ec3c:dados/ler_pandas.py
print(df.shape)