import pandas as pd
import os

caminho_base = "for_pmv_prediction"
os.makedirs("csv_regioes", exist_ok=True)

regioes = {}

for caso in range(1, 46):

    pasta_caso = os.path.join(caminho_base, str(caso))

    for arquivo in os.listdir(pasta_caso):

        # pegar apenas arquivos _cellcenter.csv
        if arquivo.endswith("_cellcenter.csv"):

            caminho_arquivo = os.path.join(pasta_caso, arquivo)

            df = pd.read_csv(caminho_arquivo)

            df.insert(0, "caso", caso)

            nome_regiao = arquivo.replace(".csv", "")

            if nome_regiao not in regioes:
                regioes[nome_regiao] = []

            regioes[nome_regiao].append(df)

# concatena
dataframes_finais = {}

for nome_regiao, lista_dfs in regioes.items():
    dataframes_finais[nome_regiao] = pd.concat(lista_dfs, ignore_index=True)

# df_head = dataframes_finais[nome_regiao]

# for nome, df in dataframes_finais.items():
#     print(nome, df_head.head())
#     print(df_head.shape)
#     df.to_csv(f"csv_regioes/{nome}.csv", index=False)

# df_head = dataframes_finais["transform-fp-head_cellcenter"]
# print(df_head.head())
# print(df_head.shape)