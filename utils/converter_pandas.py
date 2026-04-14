import pandas as pd

df = pd.read_csv("/home/reuel_737/projetos/Lab_L2/regioes/csv_regioes/transform-rrp-head_cellcenter.csv")

df.to_pickle("/home/reuel_737/projetos/Lab_L2/regioes/pandas_regioes/rrp-head_cellcenter.pandas")


