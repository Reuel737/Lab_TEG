import numpy as np
import matplotlib.pyplot as plt
import os

arquivo_erros = "result_fase5/exp15/exp15_fold04_erros.npz"

Target = ['pressure', 'x-velocity', 'y-velocity', 'z-velocity', 'temperature', 
          'incident-radiation', 'radiation-temperature', 'rad-heat-flux', 'vr']

dados = np.load(arquivo_erros)
erros_matriz = dados['erros']

fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for i, var_name in enumerate(Target):
    ax = axes[i]
    
    erro_var = erros_matriz[:, i]
    erros_ordenados = np.sort(erro_var)
    y_cumulativo = np.arange(1, len(erros_ordenados) + 1) / len(erros_ordenados)
    
    ax.hist(erro_var, bins=50, density=True, cumulative=True, 
            histtype='step', color='darkorange', linewidth=1.5, 
            label='Cumulative histogram')

    ax.plot(erros_ordenados, y_cumulativo, color='#1f77b4', linewidth=2, 
            label='CDF')
    
    ax.axhline(y=0.90, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5)
    
    p99 = np.percentile(erros_ordenados, 99)
    limite_x = p99 * 1.5 if p99 > 0 else 0.1 
    ax.set_xlim(0, limite_x)
    ax.set_ylim(0, 1.05)
    
    ax.set_title(f'{var_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Erro Absoluto', fontsize=10)
    ax.set_ylabel('Probability of occurrence', fontsize=10)
    ax.grid(True, linestyle='-', alpha=0.7)
    
    if i == 0:
        ax.legend(loc='lower right')

plt.tight_layout()

pasta_saida = "graficos/hist_cumulativo"
os.makedirs(pasta_saida, exist_ok=True)
caminho_imagem = os.path.join(pasta_saida, "hist_cumulativo_exp15.png")
plt.savefig(caminho_imagem, dpi=300)

plt.show()