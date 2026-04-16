import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tensorflow.keras.models import load_model
import os

regiao = "regioes/fp_head"
experimento = "exp15_L2_00001_fold21"
nome_experimento = "exp15_L2_00001"
modelo_path = f"{regiao}/results/{experimento}/{experimento}.keras"
dados_path = "dados_filtrados/pandas_regioes/transform-fp-head_cellcenter.pandas"
idx_alvo = 0  # pressao
nome_alvo = 'P'

resolucao_base = 10
fator_escala = 40
nx = resolucao_base * fator_escala
ny = resolucao_base * fator_escala

print("Carregando dados originais do CFD...")
dados = pd.read_pickle(dados_path)

z_min = dados['z-coordinate'].min()
z_max = dados['z-coordinate'].max()
z_corte = z_max - 0.20 * (z_max - z_min) # Pega os 20% superiores
casca = dados[dados['z-coordinate'] >= z_corte]

pts_xy_reais = casca[['x-coordinate', 'y-coordinate']].values
pts_z_reais = casca['z-coordinate'].values

vetor_x = np.linspace(casca['x-coordinate'].min(), casca['x-coordinate'].max(), nx)
vetor_y = np.linspace(casca['y-coordinate'].min(), casca['y-coordinate'].max(), ny)
Malha_X, Malha_Y = np.meshgrid(vetor_x, vetor_y)

print(f"Criando malha plana de {nx}x{ny} pontos...")
Malha_Z_suave = griddata(pts_xy_reais, pts_z_reais, (Malha_X, Malha_Y), method='linear')
Malha_Z_borda = griddata(pts_xy_reais, pts_z_reais, (Malha_X, Malha_Y), method='nearest')
Malha_Z = np.where(np.isnan(Malha_Z_suave), Malha_Z_borda, Malha_Z_suave)

x_flat = Malha_X.flatten()
y_flat = Malha_Y.flatten()
z_flat = Malha_Z.flatten()

vel_flat = np.full_like(x_flat, casca['Vel'].iloc[0])
tin_flat = np.full_like(x_flat, casca['Tinsu'].iloc[0])
qin_flat = np.full_like(x_flat, casca['Qinsu'].iloc[0])

X_entrada = np.column_stack((x_flat, y_flat, z_flat, vel_flat, tin_flat, qin_flat))

print(f"Rede Neural calculando {len(x_flat)} pontos...")
modelo = load_model(modelo_path)
Y_previsto = modelo.predict(X_entrada, verbose=0)

temperatura_flat = Y_previsto[:, idx_alvo]
Malha_Temperatura = temperatura_flat.reshape(ny, nx)

print("Desenhando o Contour Plot...")
plt.figure(figsize=(10, 8))

contorno = plt.contourf(Malha_X, Malha_Y, Malha_Temperatura, levels=100, cmap='jet')
cbar = plt.colorbar(contorno)
cbar.set_label(nome_alvo)

plt.title(f'Previsão Térmica: Casca Superior (Resolução: {nx}x{ny})')
plt.xlabel('Coordenada X (m)')
plt.ylabel('Coordenada Y (m)')
plt.grid(False) # Sem grade para não poluir as cores

pasta_saida = f"{regiao}/graficos/contour/{experimento}"
os.makedirs(pasta_saida, exist_ok=True)
caminho_imagem = os.path.join(pasta_saida, f"contour_{nome_alvo}_{nome_experimento}.png")
plt.savefig(caminho_imagem, dpi=300)

plt.show()
print(f"Sucesso! Gráfico salvo em: {caminho_imagem}")