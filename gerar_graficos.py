#!/usr/bin/env python3
"""
Gera todos os gráficos para cada região registrada em regioes/melhores_modelos.csv.

Tipos de gráfico:
  - hist_cumulativo : CDF dos erros absolutos (todos os folds disponíveis agregados)
  - 1_variavel      : scatter pred vs real por variável (melhor fold)
  - 2_variavel      : scatter acoplamento P vs T e P vs Y-vel (melhor fold)
  - heatmap         : scatter 3D real/previsto/erro para P e T (melhor fold)
  - contour         : contorno 2D interpolado para P e T (melhor fold + dados pandas)

Uso:
  python3 gerar_graficos.py                   # todas as regiões no CSV
  python3 gerar_graficos.py fp_head mt_head   # regiões específicas
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.interpolate import griddata

TARGET_NAMES = [
    'pressure', 'x-velocity', 'y-velocity', 'z-velocity', 'temperature',
    'incident-radiation', 'radiation-temperature', 'rad-heat-flux', 'vr'
]

# Caminhos dos dados brutos — completo para todas as regiões (incluindo as sem treino ainda)
DATA_FILES = {
    'fp_head':   'dados_filtrados/pandas_regioes/transform-fp-head_cellcenter.pandas',
    'lrp_head':  'dados_filtrados/pandas_regioes/transform-lrp-head_cellcenter.pandas',
    'rrp_head':  'dados_filtrados/pandas_regioes/transform-rrp-head_cellcenter.pandas',
    'mt_head':   'dados_filtrados/pandas_regioes/transform-motorist-head_cellcenter.pandas',
    'mt_core':   'dados_filtrados/pandas_regioes/transform-motorist-core_cellcenter.pandas',
    'mt_l_foot': 'dados_filtrados/pandas_regioes/transform-motorist-left-foot_cellcenter.pandas',
    'mt_r_foot': 'dados_filtrados/pandas_regioes/transform-motorist-right-foot_cellcenter.pandas',
    'outlet':    'dados_filtrados/pandas_regioes/transform-outlet_cellcenter.pandas',
}

MELHORES_CSV = 'regioes/melhores_modelos.csv'


# ---------------------------------------------------------------------------
# Parse de melhores_modelos.csv
# ---------------------------------------------------------------------------

def parse_melhores(filtro=None):
    """
    Retorna lista de dicts com tudo necessário para gerar gráficos.
    filtro: lista de nomes de diretório de região (e.g. ['fp_head']); None = todos.
    """
    df = pd.read_csv(MELHORES_CSV)
    entradas = []
    for _, row in df.iterrows():
        caminho = os.path.normpath(row['Caminho_Modelo'])
        partes = caminho.split(os.sep)
        regiao_dir = partes[1]                          # ex: 'fp_head', 'mt_head'

        if filtro and regiao_dir not in filtro:
            continue

        fold = int(str(row['Melhor_Fold']).split()[-1])
        results_dir = os.path.dirname(caminho)
        basename = os.path.basename(caminho)            # ex: '512x4_fold21.keras'
        prefix = basename.replace(f'_fold{fold:02d}.keras', '')  # ex: '512x4'
        experimento = prefix

        entradas.append({
            'regiao_dir':  regiao_dir,
            'regiao_path': os.path.join('regioes', regiao_dir),
            'results_dir': results_dir,
            'prefix':      prefix,
            'experimento': experimento,
            'fold':        fold,
            'val_loss':    float(row['Val_Loss']),
            'data_file':   DATA_FILES.get(regiao_dir, ''),
        })
    return entradas


# ---------------------------------------------------------------------------
# Helpers de caminho
# ---------------------------------------------------------------------------

def _erros_path(results_dir, prefix, fold):
    return os.path.join(results_dir, f'{prefix}_fold{fold:02d}_erros.npz')

def _dataset_path(results_dir, prefix, fold):
    return os.path.join(results_dir, f'{prefix}_fold{fold:02d}_dataset.npz')

def _model_path(results_dir, prefix, fold):
    return os.path.join(results_dir, f'{prefix}_fold{fold:02d}.keras')

def get_all_folds(results_dir, prefix):
    return [f for f in range(1, 31) if os.path.isfile(_erros_path(results_dir, prefix, f))]


# ---------------------------------------------------------------------------
# Funções de plot
# ---------------------------------------------------------------------------

def plot_hist_cumulativo(erros_todos, regiao_path, experimento):
    pasta = os.path.join(regiao_path, 'graficos', 'hist_cumulativo')
    os.makedirs(pasta, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    for i, (ax, var) in enumerate(zip(axes.flatten(), TARGET_NAMES)):
        err = erros_todos[:, i]
        err_ord = np.sort(err)
        y_cum = np.arange(1, len(err_ord) + 1) / len(err_ord)

        ax.hist(err, bins=50, density=True, cumulative=True,
                histtype='step', color='darkorange', linewidth=1.5,
                label='Histograma cumulativo')
        ax.plot(err_ord, y_cum, color='#1f77b4', linewidth=2, label='CDF')
        ax.axhline(0.90, color='red',   linestyle='--', alpha=0.5, label='90%')
        ax.axhline(0.95, color='green', linestyle='--', alpha=0.5, label='95%')

        p99 = np.percentile(err_ord, 99)
        ax.set_xlim(0, max(p99 * 1.5, 0.1))
        ax.set_ylim(0, 1.05)
        ax.set_title(var, fontsize=12, fontweight='bold')
        ax.set_xlabel('Erro Absoluto')
        ax.set_ylabel('Probabilidade acumulada')
        ax.grid(True, linestyle='-', alpha=0.7)
        if i == 0:
            ax.legend(loc='lower right', fontsize=8)

    plt.suptitle(f'Distribuição de Erros (todos os folds) — {experimento}', fontsize=14)
    plt.tight_layout()
    caminho = os.path.join(pasta, f'hist_cumulativo_{experimento}.png')
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f'    [hist_cumulativo] {caminho}')


def plot_1_variavel(yval, pred_val, regiao_path, experimento):
    pasta = os.path.join(regiao_path, 'graficos', '1_variavel', experimento)
    os.makedirs(pasta, exist_ok=True)

    for i, target in enumerate(TARGET_NAMES):
        fig, ax = plt.subplots(figsize=(8, 6))
        reais = yval[:, i]
        prev  = pred_val[:, i]
        ax.scatter(reais, prev, alpha=0.3, color='blue', s=10, label='Predição')
        lo, hi = min(reais.min(), prev.min()), max(reais.max(), prev.max())
        ax.plot([lo, hi], [lo, hi], 'r--', label='Ideal')
        ax.set_title(f'Validação: {target}')
        ax.set_xlabel('Real')
        ax.set_ylabel('Previsto')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta, f'{target}.png'), dpi=150)
        plt.close()

    print(f'    [1_variavel] {pasta}')


def plot_2_variavel(yval, pred_val, regiao_path, experimento):
    pasta = os.path.join(regiao_path, 'graficos', '2_variavel', experimento)
    os.makedirs(pasta, exist_ok=True)

    pares = [(0, 4, 'P', 'T'), (0, 2, 'P', 'Yvel')]
    for idx1, idx2, n1, n2 in pares:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(yval[:, idx1],     yval[:, idx2],     alpha=0.3, color='blue', s=5, label='CFD Real')
        ax.scatter(pred_val[:, idx1], pred_val[:, idx2], alpha=0.3, color='red',  s=5, label='Rede Neural')
        ax.set_title(f'{n1} vs. {n2}')
        ax.set_xlabel(n1)
        ax.set_ylabel(n2)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta, f'relacao_{n1}_vs_{n2}_{experimento}.png'), dpi=150)
        plt.close()

    print(f'    [2_variavel] {pasta}')


def plot_heatmap(xval, yval, pred_val, regiao_path, experimento):
    pasta = os.path.join(regiao_path, 'graficos', 'heatmap', experimento)
    os.makedirs(pasta, exist_ok=True)

    cx, cy, cz = xval[:, 0], xval[:, 1], xval[:, 2]

    for idx_alvo, nome_alvo in [(0, 'P'), (4, 'T')]:
        real = yval[:, idx_alvo]
        prev = pred_val[:, idx_alvo]
        erro = np.abs(real - prev)
        vmin, vmax = min(real.min(), prev.min()), max(real.max(), prev.max())

        fig = plt.figure(figsize=(20, 6))

        ax1 = fig.add_subplot(131, projection='3d')
        sc1 = ax1.scatter(cx, cy, cz, c=real, cmap='jet', vmin=vmin, vmax=vmax, s=2, alpha=0.2)
        ax1.set_title(f'1. CFD Real ({nome_alvo})')
        ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(cx, cy, cz, c=prev, cmap='jet', vmin=vmin, vmax=vmax, s=2, alpha=0.2)
        ax2.set_title(f'2. Rede Neural ({nome_alvo})')
        ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

        fig.colorbar(sc1, ax=[ax1, ax2], fraction=0.03, pad=0.05).set_label(nome_alvo)

        ax3 = fig.add_subplot(133, projection='3d')
        sc3 = ax3.scatter(cx, cy, cz, c=erro, cmap='Reds', s=2, alpha=0.2)
        ax3.set_title('3. Erro Absoluto')
        ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
        fig.colorbar(sc3, ax=ax3, fraction=0.03, pad=0.05).set_label('Erro Absoluto')

        plt.suptitle(f'{experimento} — {nome_alvo}', fontsize=13)
        caminho = os.path.join(pasta, f'heatmap_{nome_alvo}_{experimento}.png')
        plt.savefig(caminho, dpi=150)
        plt.close()

    print(f'    [heatmap] {pasta}')


def plot_contour(dados_pandas, model, regiao_path, experimento):
    pasta = os.path.join(regiao_path, 'graficos', 'contour', experimento)
    os.makedirs(pasta, exist_ok=True)

    nx = ny = 400
    z_min = dados_pandas['z-coordinate'].min()
    z_max = dados_pandas['z-coordinate'].max()
    casca = dados_pandas[dados_pandas['z-coordinate'] >= z_max - 0.20 * (z_max - z_min)]

    pts_xy = casca[['x-coordinate', 'y-coordinate']].values
    pts_z  = casca['z-coordinate'].values
    vx = np.linspace(casca['x-coordinate'].min(), casca['x-coordinate'].max(), nx)
    vy = np.linspace(casca['y-coordinate'].min(), casca['y-coordinate'].max(), ny)
    MX, MY = np.meshgrid(vx, vy)

    MZ = griddata(pts_xy, pts_z, (MX, MY), method='linear')
    MZ = np.where(np.isnan(MZ), griddata(pts_xy, pts_z, (MX, MY), method='nearest'), MZ)

    X_ent = np.column_stack((
        MX.flatten(), MY.flatten(), MZ.flatten(),
        np.full(nx * ny, casca['Vel'].iloc[0]),
        np.full(nx * ny, casca['Tinsu'].iloc[0]),
        np.full(nx * ny, casca['Qinsu'].iloc[0]),
    ))
    Y_prev = model.predict(X_ent, verbose=0, batch_size=2000)

    for idx_alvo, nome_alvo in [(0, 'P'), (4, 'T')]:
        plt.figure(figsize=(10, 8))
        ct = plt.contourf(MX, MY, Y_prev[:, idx_alvo].reshape(ny, nx), levels=100, cmap='jet')
        plt.colorbar(ct).set_label(nome_alvo)
        plt.title(f'Previsão — Casca Superior | {nome_alvo} ({experimento})')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.tight_layout()
        caminho = os.path.join(pasta, f'contour_{nome_alvo}_{experimento}.png')
        plt.savefig(caminho, dpi=150)
        plt.close()

    print(f'    [contour] {pasta}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    filtro = sys.argv[1:] if len(sys.argv) > 1 else None
    entradas = parse_melhores(filtro)

    if not entradas:
        print('Nenhuma região encontrada. Verifique os argumentos ou o CSV.')
        return

    for cfg in entradas:
        print(f'\n{"="*60}')
        print(f'  Região : {cfg["regiao_dir"]}')
        print(f'  Exp    : {cfg["experimento"]}  |  Melhor fold: {cfg["fold"]:02d}  |  val_loss: {cfg["val_loss"]:.6f}')
        print(f'{"="*60}')

        results_dir = cfg['results_dir']
        prefix      = cfg['prefix']
        regiao_path = cfg['regiao_path']
        experimento = cfg['experimento']
        best_fold   = cfg['fold']

        # hist_cumulativo — agrega erros de todos os folds disponíveis
        all_folds = get_all_folds(results_dir, prefix)
        print(f'  Folds disponíveis: {len(all_folds)}')
        print('  Gerando hist_cumulativo...')
        erros = np.vstack([np.load(_erros_path(results_dir, prefix, f))['erros'] for f in all_folds])
        plot_hist_cumulativo(erros, regiao_path, experimento)

        # demais gráficos usam o melhor fold
        print(f'  Carregando fold {best_fold:02d}...')
        d        = np.load(_dataset_path(results_dir, prefix, best_fold))
        xval     = d['xval']
        yval     = d['yval']
        model    = load_model(_model_path(results_dir, prefix, best_fold))
        pred_val = model.predict(xval, verbose=0, batch_size=2000)

        print('  Gerando 1_variavel...')
        plot_1_variavel(yval, pred_val, regiao_path, experimento)

        print('  Gerando 2_variavel...')
        plot_2_variavel(yval, pred_val, regiao_path, experimento)

        print('  Gerando heatmap...')
        plot_heatmap(xval, yval, pred_val, regiao_path, experimento)

        print('  Gerando contour...')
        if os.path.isfile(cfg['data_file']):
            dados_pandas = pd.read_pickle(cfg['data_file'])
            plot_contour(dados_pandas, model, regiao_path, experimento)
        else:
            print(f'    [contour] SKIP — não encontrado: {cfg["data_file"]}')

    print(f'\n{"="*60}')
    print('  Concluído!')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
