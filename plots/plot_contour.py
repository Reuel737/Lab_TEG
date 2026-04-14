import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.interpolate import griddata  # A biblioteca mágica para superfícies
import os

experimento = "exp15_L2_00001"
modelo_path = f"results/result_fp_head/{experimento}/{experimento}_fold21.keras"
dados_path = "dados_filtrados/pandas_regioes/transform-fp-head_cellcenter.pandas"

# targets = ['pressure', 'x-velocity', 'y-velocity', 'z-velocity', 'temperature',
#            'incident-radiation', 'radiation-temperature', 'rad-heat-flux', 'vr']
idx_alvo = 4
nome_alvo = 'T'

resolucao_base = 10
fator_escala = 40
nx = resolucao_base * fator_escala
ny = resolucao_base * fator_escala

print("Carregando dados da malha CFD...")
dados = pd.read_pickle(dados_path)

z_min = dados['z-coordinate'].min()
z_max = dados['z-coordinate'].max()

porcentagem_topo = 0.30
z_corte = z_max - porcentagem_topo * (z_max - z_min)

casca = dados[dados['z-coordinate'] >= z_corte]

print(f"Planificando {len(casca)} pontos da casca superior (Z >= {z_corte:.3f})...")

pts_xy_reais = casca[['x-coordinate', 'y-coordinate']].values
pts_z_reais = casca['z-coordinate'].values

vetor_x = np.linspace(casca['x-coordinate'].min(), casca['x-coordinate'].max(), nx)
vetor_y = np.linspace(casca['y-coordinate'].min(), casca['y-coordinate'].max(), ny)
Malha_X, Malha_Y = np.meshgrid(vetor_x, vetor_y)

# 'nearest' para garantir que os cantos do quadrado não fiquem vazios (NaN)
Malha_Z = griddata(pts_xy_reais, pts_z_reais, (Malha_X, Malha_Y), method='nearest')

x_flat = Malha_X.flatten()
y_flat = Malha_Y.flatten()
z_flat = Malha_Z.flatten() # <--- Z acompanha a curvatura

vel_flat = np.full_like(x_flat, casca['Vel'].iloc[0])
tin_flat = np.full_like(x_flat, casca['Tinsu'].iloc[0])
qin_flat = np.full_like(x_flat, casca['Qinsu'].iloc[0])

X_entrada = np.column_stack((x_flat, y_flat, z_flat, vel_flat, tin_flat, qin_flat))

print("Rede Neural prevendo na superfície da casca...")
modelo = load_model(modelo_path)
Y_previsto = modelo.predict(X_entrada)

temperatura_flat = Y_previsto[:, idx_alvo]
Malha_Temperatura = temperatura_flat.reshape(ny, nx)

print("Gerando gráfico...")
plt.figure(figsize=(10, 8))

contorno = plt.contourf(Malha_X, Malha_Y, Malha_Temperatura, levels=100, cmap='jet')
cbar = plt.colorbar(contorno)
cbar.set_label('Temperatura na Superfície (K)')  # rótulo conforme o índice alvo

plt.title(f'Planificação da Casca Superior (Top {porcentagem_topo*100}%) - Resolução {nx}x{ny}')
plt.xlabel('Coordenada X (m)')
plt.ylabel('Coordenada Y (m)')

pasta_saida = f"graficos/contour/fp_head/{experimento}"
os.makedirs(pasta_saida, exist_ok=True)
plt.savefig(os.path.join(pasta_saida, f"contour_{nome_alvo}_{experimento}.png"), dpi=300)

plt.show()