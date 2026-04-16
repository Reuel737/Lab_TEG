"""
2 VARIÁVEIS
Gerar gráficos de acoplamento entre duas variáveis, comparando predição vs. real
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

regiao = "regioes/fp_head"
experimento = "exp15_L2_00001_fold21"
nome_experimento = "exp15_L2_00001"
modelo_path = f"{regiao}/results/{experimento}/{experimento}.keras"
dataset_path = f"{regiao}/results/{experimento}/{experimento}_fold21_dataset.npz"

dados = np.load(dataset_path)
xval, yval = dados['xval'], dados['yval']
model = load_model(modelo_path)
pred_val = model.predict(xval)

# targets = ['pressure', 'x-velocity', 'y-velocity', 'z-velocity', 'temperature',
#            'incident-radiation', 'radiation-temperature', 'rad-heat-flux', 'vr']
idx_var1 = 0
idx_var2 = 4 
nome_var1 = 'P'
nome_var2 = 'T'

pasta_saida = f"{regiao}/graficos/2_variavel/exp15_L2_00001"
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

caminho_imagem = os.path.join(pasta_saida, f"relacao_{nome_var1}_vs_{nome_var2}_{nome_experimento}.png")
plt.savefig(caminho_imagem)
plt.close()