"""
1 VARIÁVEL
Gerar gráficos de predição vs. real para cada variável individualmente, usando o modelo treinado
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

print(f"Carregando dataset: {dataset_path}")
dados = np.load(dataset_path)
xval, yval = dados['xval'], dados['yval']

print(f"Carregando modelo: {modelo_path}")
model = load_model(modelo_path)

print("Realizando predições na validação...")
pred_val = model.predict(xval)

targets = ['pressure', 'x-velocity', 'y-velocity', 'z-velocity', 'temperature',
           'incident-radiation', 'radiation-temperature', 'rad-heat-flux', 'vr']

pasta_saida = f"{regiao}/graficos/1_variavel/{experimento}"
os.makedirs(pasta_saida, exist_ok=True)

print(f"Gerando gráficos na pasta '{pasta_saida}'...")

for i, target in enumerate(targets):
    plt.figure(figsize=(8, 6))
    
    reais = yval[:, i]
    previstos = pred_val[:, i]
    
    plt.scatter(reais, previstos, alpha=0.3, color='blue', label='Predição', s=10)
    
    min_val = min(reais.min(), previstos.min())
    max_val = max(reais.max(), previstos.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    
    plt.title(f'Validação: {target}')
    plt.xlabel('Real')
    plt.ylabel('Previsto')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(pasta_saida, f"{target}.png"))
    plt.close()

print("Concluído!")