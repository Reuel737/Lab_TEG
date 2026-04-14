import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 1. Caminho do modelo já treinado
modelo_path = "result_fase5/exp15/exp15_fold14.keras"

DATAFILE_NOVA_REGIAO = "dados_filtrados/pandas_regioes/transform-motorist-head_cellcenter.pandas" 

print(f"Carregando dados da nova região: {DATAFILE_NOVA_REGIAO}")
dados = pd.read_pickle(DATAFILE_NOVA_REGIAO)  

colunas_x = ['x-coordinate', 'y-coordinate', 'z-coordinate', 'Vel', 'Tinsu', 'Qinsu']
colunas_y = ['pressure', 'x-velocity', 'y-velocity', 'z-velocity', 'temperature',
             'incident-radiation', 'radiation-temperature', 'rad-heat-flux', 'vr']

X_novo = dados[colunas_x].to_numpy()
Y_real_novo = dados[colunas_y].to_numpy()

print("Carregando modelo e prevendo para a nova região...")
model = load_model(modelo_path)
Y_prev_novo = model.predict(X_novo)

idx_x, idx_y, idx_z = 0, 1, 2
idx_alvo = 4
nome_alvo = 'Temperatura'

print(f"Abrindo janela 3D interativa para a nova região...")

fig = plt.figure(figsize=(18, 6))

coord_x = X_novo[:, idx_x]
coord_y = X_novo[:, idx_y]
coord_z = X_novo[:, idx_z]

val_real = Y_real_novo[:, idx_alvo]
val_prev = Y_prev_novo[:, idx_alvo]
erro_absoluto = np.abs(val_real - val_prev)

vmin = min(val_real.min(), val_prev.min())
vmax = max(val_real.max(), val_prev.max())

tamanho_ponto = 2
alfa = 0.2 

ax1 = fig.add_subplot(131, projection='3d')
sc1 = ax1.scatter(coord_x, coord_y, coord_z, c=val_real, cmap='jet', 
                  vmin=vmin, vmax=vmax, s=tamanho_ponto, alpha=alfa)
ax1.set_title('1. CFD Real (Nova Região)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax2 = fig.add_subplot(132, projection='3d')
sc2 = ax2.scatter(coord_x, coord_y, coord_z, c=val_prev, cmap='jet', 
                  vmin=vmin, vmax=vmax, s=tamanho_ponto, alpha=alfa)
ax2.set_title('2. Rede Neural (Previsão)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

cbar1 = fig.colorbar(sc1, ax=[ax1, ax2], fraction=0.02, pad=0.05)
cbar1.set_label(nome_alvo)

ax3 = fig.add_subplot(133, projection='3d')
sc3 = ax3.scatter(coord_x, coord_y, coord_z, c=erro_absoluto, cmap='Reds', 
                  s=tamanho_ponto, alpha=alfa)
ax3.set_title('3. Erro Absoluto')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

cbar2 = fig.colorbar(sc3, ax=ax3, fraction=0.02, pad=0.05)
cbar2.set_label('Erro Absoluto')

plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05, wspace=0.1)
plt.show()