import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.interpolate import griddata  # A biblioteca mágica para superfícies
import os

# --- 1. CONFIGURAÇÕES ---
modelo_path = "result_fase5/exp15_L2_0001/exp15_L2_0001_fold23.keras" 
dados_path = "dados_filtrados/pandas_regioes/transform-fp-head_cellcenter.pandas"
idx_alvo = 0  # pressao

resolucao_base = 10
fator_escala = 40
nx = resolucao_base * fator_escala
ny = resolucao_base * fator_escala

print("Carregando dados da malha CFD...")
dados = pd.read_pickle(dados_path)

# --- 2. ISOLANDO A "CASCA SUPERIOR" ---
z_min = dados['z-coordinate'].min()
z_max = dados['z-coordinate'].max()

# Pega apenas do "Z de corte" pra cima (ex: os 20% superiores da cabeça)
porcentagem_topo = 0.30
z_corte = z_max - porcentagem_topo * (z_max - z_min)

# Filtra o dataframe para ter só a casca
casca = dados[dados['z-coordinate'] >= z_corte]

print(f"Planificando {len(casca)} pontos da casca superior (Z >= {z_corte:.3f})...")

# --- 3. CRIANDO A MALHA 2D E MAPEANDO A CURVATURA (Z) ---
# Extrai as coordenadas reais (nuvem de pontos irregular)
pts_xy_reais = casca[['x-coordinate', 'y-coordinate']].values
pts_z_reais = casca['z-coordinate'].values

# Cria o grid 2D perfeitamente regular (planificação no XY)
vetor_x = np.linspace(casca['x-coordinate'].min(), casca['x-coordinate'].max(), nx)
vetor_y = np.linspace(casca['y-coordinate'].min(), casca['y-coordinate'].max(), ny)
Malha_X, Malha_Y = np.meshgrid(vetor_x, vetor_y)

# 'nearest' para garantir que os cantos do quadrado não fiquem vazios (NaN)
Malha_Z = griddata(pts_xy_reais, pts_z_reais, (Malha_X, Malha_Y), method='nearest')

# Planifica tudo para o formato que a Rede Neural gosta (listas 1D)
x_flat = Malha_X.flatten()
y_flat = Malha_Y.flatten()
z_flat = Malha_Z.flatten() # <--- Z acompanha a curvatura

# Condições de contorno
vel_flat = np.full_like(x_flat, casca['Vel'].iloc[0])
tin_flat = np.full_like(x_flat, casca['Tinsu'].iloc[0])
qin_flat = np.full_like(x_flat, casca['Qinsu'].iloc[0])

# Monta o X (Entrada)
X_entrada = np.column_stack((x_flat, y_flat, z_flat, vel_flat, tin_flat, qin_flat))

# --- 4. PREDIÇÃO NA SUPERFÍCIE CURVA ---
print("Rede Neural prevendo na superfície da casca...")
modelo = load_model(modelo_path)
Y_previsto = modelo.predict(X_entrada)

temperatura_flat = Y_previsto[:, idx_alvo]
# Remonta a matriz visual 2D
Malha_Temperatura = temperatura_flat.reshape(ny, nx)

# --- 5. GRÁFICO (VISTO DE CIMA) ---
print("Gerando gráfico...")
plt.figure(figsize=(10, 8))

contorno = plt.contourf(Malha_X, Malha_Y, Malha_Temperatura, levels=100, cmap='jet')
cbar = plt.colorbar(contorno)
cbar.set_label('Pressão (Pa)')  # Ajuste o rótulo conforme o índice alvo

plt.title(f'Planificação da Casca Superior (Top {porcentagem_topo*100}%) - Resolução {nx}x{ny}')
plt.xlabel('Coordenada X (m)')
plt.ylabel('Coordenada Y (m)')

pasta_saida = "graficos/contour"
os.makedirs(pasta_saida, exist_ok=True)
plt.savefig(os.path.join(pasta_saida, f"contour_{idx_alvo}.png"), dpi=300)

plt.show()