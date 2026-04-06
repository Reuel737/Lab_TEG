# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# import os

# modelo_path = "result_fase5/exp15/exp15_fold14.keras"
# dataset_path = "result_fase5/exp15/exp15_fold14_dataset.npz"

# print(f"Carregando dataset: {dataset_path}")
# dados = np.load(dataset_path)
# xval, yval = dados['xval'], dados['yval']

# print(f"Carregando modelo: {modelo_path}")
# model = load_model(modelo_path)

# print("Realizando predições na validação...")
# pred_val = model.predict(xval)

# targets = ['pressure', 'x-velocity', 'y-velocity', 'z-velocity', 'temperature',
#            'incident-radiation', 'radiation-temperature', 'rad-heat-flux', 'vr']

# pasta_saida = "graficos/1_variavel_exp15"
# os.makedirs(pasta_saida, exist_ok=True)

# print(f"Gerando gráficos na pasta '{pasta_saida}'...")

# for i, target in enumerate(targets):
#     plt.figure(figsize=(8, 6))
    
#     reais = yval[:, i]
#     previstos = pred_val[:, i]
    
#     plt.scatter(reais, previstos, alpha=0.3, color='blue', label='Predição', s=10)
    
#     min_val = min(reais.min(), previstos.min())
#     max_val = max(reais.max(), previstos.max())
#     plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    
#     plt.title(f'Validação: {target}')
#     plt.xlabel('Real')
#     plt.ylabel('Previsto')
#     plt.legend()
#     plt.grid(True)
    
#     plt.savefig(os.path.join(pasta_saida, f"{target}_analise.png"))
#     plt.close()

# print("Concluído!")

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

modelo_path = "result_fase5/exp15/exp15_fold14.keras"
dataset_path = "result_fase5/exp15/exp15_fold14_dataset.npz"

dados = np.load(dataset_path)
xval, yval = dados['xval'], dados['yval']
model = load_model(modelo_path)
pred_val = model.predict(xval)

# targets = ['pressure', 'x-velocity', 'y-velocity', 'z-velocity', 'temperature',
#            'incident-radiation', 'radiation-temperature', 'rad-heat-flux', 'vr']
idx_var1 = 0 
idx_var2 = 8 
nome_var1 = 'Pressão'
nome_var2 = 'vr'

pasta_saida = "graficos/2_variavel_exp15"
os.makedirs(pasta_saida, exist_ok=True)

print("Gerando gráfico de acoplamento...")
plt.figure(figsize=(8, 6))

plt.scatter(yval[:, idx_var1], yval[:, idx_var2], 
            alpha=0.3, color='blue', label='CFD Real', s=5)

plt.scatter(pred_val[:, idx_var1], pred_val[:, idx_var2], 
            alpha=0.3, color='red', label='Rede Neural', s=5)

plt.title(f'{nome_var1} vs. {nome_var2}')
plt.xlabel(nome_var1)
plt.ylabel(nome_var2)
plt.legend()
plt.grid(True)

caminho_imagem = os.path.join(pasta_saida, f"relacao_{nome_var1}_vs_{nome_var2}.png")
plt.savefig(caminho_imagem)
plt.close()

print(f"Concluído! Salvo em '{caminho_imagem}'.")