import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

modelo_path = "result_fp_head/exp15/exp15_fold04.keras"
dataset_path = "result_fp_head/exp15/exp15_fold04_dataset.npz"

print("Carregando arquivos...")
dados = np.load(dataset_path)
xval, yval = dados['xval'], dados['yval']
model = load_model(modelo_path)
pred_val = model.predict(xval)

# Índices das coordenadas X, Y, Z no array xval
idx_x, idx_y, idx_z = 0, 1, 2

idx_alvo = 0
nome_alvo = 'P'

pasta_saida = "graficos/heatmap"
os.makedirs(pasta_saida, exist_ok=True)

fig = plt.figure(figsize=(20, 6))

coord_x = xval[:, idx_x]
coord_y = xval[:, idx_y]
coord_z = xval[:, idx_z]

val_real = yval[:, idx_alvo]
val_prev = pred_val[:, idx_alvo]

# Erro absoluto
erro_absoluto = np.abs(val_real - val_prev)

vmin = min(val_real.min(), val_prev.min())
vmax = max(val_real.max(), val_prev.max())

tamanho_ponto = 2
alfa = 0.2 

# PAINEL 1: REAL
ax1 = fig.add_subplot(131, projection='3d')
sc1 = ax1.scatter(coord_x, coord_y, coord_z, c=val_real, cmap='jet', 
                  vmin=vmin, vmax=vmax, s=tamanho_ponto, alpha=alfa)
ax1.set_title(f'1. CFD Real ({nome_alvo})')
ax1.set_xlabel('Coord X')
ax1.set_ylabel('Coord Y')
ax1.set_zlabel('Coord Z')

# PAINEL 2: PREVISTO
ax2 = fig.add_subplot(132, projection='3d')
sc2 = ax2.scatter(coord_x, coord_y, coord_z, c=val_prev, cmap='jet', 
                  vmin=vmin, vmax=vmax, s=tamanho_ponto, alpha=alfa)
ax2.set_title(f'2. Rede Neural ({nome_alvo})')
ax2.set_xlabel('Coord X')
ax2.set_ylabel('Coord Y')
ax2.set_zlabel('Coord Z')

cbar1 = fig.colorbar(sc1, ax=[ax1, ax2], fraction=0.03, pad=0.05)
cbar1.set_label(f'{nome_alvo}')

# PAINEL 3: ERRO
ax3 = fig.add_subplot(133, projection='3d')
sc3 = ax3.scatter(coord_x, coord_y, coord_z, c=erro_absoluto, cmap='Reds', 
                  s=tamanho_ponto, alpha=alfa)
ax3.set_title('3. Erro Absoluto (3D)')
ax3.set_xlabel('Coord X')
ax3.set_ylabel('Coord Y')
ax3.set_zlabel('Coord Z')

cbar2 = fig.colorbar(sc3, ax=ax3, fraction=0.03, pad=0.05)
cbar2.set_label('Margem de Erro Absoluto')

plt.tight_layout()
caminho_imagem = os.path.join(pasta_saida, f"heatmap_{nome_alvo}_exp15.png")
plt.savefig(caminho_imagem)

print(f"Gráfico salvo em '{caminho_imagem}'.")

plt.show()
plt.close()
