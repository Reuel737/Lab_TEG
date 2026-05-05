#!/usr/bin/env python3
"""
Análise comparativa entre regiões: R², MAE e MSE por variável.

Lê regioes/melhores_modelos.csv, carrega o melhor modelo de cada região
e gera:
  - regioes/analise_regioes_metricas.csv  : tabela completa de métricas
  - regioes/graficos/heatmap_r2.png       : heatmap região × variável (R²)
  - regioes/graficos/heatmap_mae.png      : heatmap região × variável (MAE)
  - regioes/graficos/bar_valloss.png      : val_loss do melhor fold por região

Uso:
  python3 analise/analise_regioes.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

MELHORES_CSV = 'regioes/melhores_modelos.csv'
OUT_DIR      = 'regioes/graficos'

TARGETS = [
    'pressure', 'x-velocity', 'y-velocity', 'z-velocity', 'temperature',
    'incident-radiation', 'radiation-temperature', 'rad-heat-flux', 'vr',
]

TARGETS_LABEL = [
    'Pressão', 'Vel-X', 'Vel-Y', 'Vel-Z', 'Temperatura',
    'Rad. Incidente', 'Temp. Radiante', 'Flux. Rad.', 'VR',
]


def _metricas(yreal, ypred):
    mse  = float(np.mean((yreal - ypred) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(yreal - ypred)))
    ss_r = np.sum((yreal - ypred) ** 2)
    ss_t = np.sum((yreal - yreal.mean()) ** 2)
    r2   = float(1.0 - ss_r / ss_t) if ss_t > 0 else float('nan')
    return mse, rmse, mae, r2


def _parse_melhores():
    df = pd.read_csv(MELHORES_CSV)
    entradas = []
    for _, row in df.iterrows():
        caminho     = os.path.normpath(row['Caminho_Modelo'])
        partes      = caminho.split(os.sep)
        regiao_dir  = partes[1]
        fold        = int(str(row['Melhor_Fold']).split()[-1])
        results_dir = os.path.dirname(caminho)
        prefix      = os.path.basename(caminho).replace(f'_fold{fold:02d}.keras', '')
        entradas.append({
            'regiao_dir':  regiao_dir,
            'results_dir': results_dir,
            'prefix':      prefix,
            'fold':        fold,
            'val_loss':    float(row['Val_Loss']),
            'experimento': str(row['Experimento']),
        })
    return entradas


def _label_regiao(nome):
    mapa = {
        'fp_head':   'FP Cabeça',
        'lrp_head':  'LRP Cabeça',
        'rrp_head':  'RRP Cabeça',
        'mt_head':   'MT Cabeça',
        'mt_core':   'MT Tronco',
        'mt_l_foot': 'MT Pé Esq.',
        'mt_r_foot': 'MT Pé Dir.',
        'outlet':    'Outlet',
    }
    return mapa.get(nome, nome)


def calcular_metricas(entradas):
    registros = []
    matrizes  = {'r2': {}, 'mae': {}, 'mse': {}}

    for cfg in entradas:
        rd, prefix, fold = cfg['results_dir'], cfg['prefix'], cfg['fold']
        dataset_path = os.path.join(rd, f'{prefix}_fold{fold:02d}_dataset.npz')
        model_path   = os.path.join(rd, f'{prefix}_fold{fold:02d}.keras')

        if not os.path.isfile(dataset_path) or not os.path.isfile(model_path):
            print(f'  [SKIP] arquivos não encontrados para {cfg["regiao_dir"]}')
            continue

        print(f'  Carregando {cfg["regiao_dir"]} (fold {fold:02d})...')
        d        = np.load(dataset_path)
        xval     = d['xval']
        yval     = d['yval']
        model    = tf.keras.models.load_model(model_path)
        pred_val = model.predict(xval, batch_size=2000, verbose=0)

        r2_row  = {}
        mae_row = {}
        mse_row = {}
        for i, t in enumerate(TARGETS):
            mse, rmse, mae, r2 = _metricas(yval[:, i], pred_val[:, i])
            registros.append({
                'regiao':      cfg['regiao_dir'],
                'experimento': cfg['experimento'],
                'fold':        fold,
                'variavel':    t,
                'mse':         round(mse,  6),
                'rmse':        round(rmse, 6),
                'mae':         round(mae,  6),
                'r2':          round(r2,   4),
            })
            r2_row[t]  = r2
            mae_row[t] = mae
            mse_row[t] = mse

        label = _label_regiao(cfg['regiao_dir'])
        matrizes['r2'][label]  = r2_row
        matrizes['mae'][label] = mae_row
        matrizes['mse'][label] = mse_row

    return pd.DataFrame(registros), matrizes


def plot_heatmap(matriz_dict, metrica, cmap, vmin, vmax, titulo, arquivo):
    regioes = list(matriz_dict.keys())
    dados   = np.array([[matriz_dict[r].get(t, float('nan')) for t in TARGETS]
                        for r in regioes])

    fig, ax = plt.subplots(figsize=(14, max(4, len(regioes) * 0.9 + 1.5)))
    im = ax.imshow(dados, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax, label=metrica)

    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels(TARGETS_LABEL, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(regioes)))
    ax.set_yticklabels(regioes, fontsize=10)

    for i in range(len(regioes)):
        for j in range(len(TARGETS)):
            val = dados[i, j]
            if not np.isnan(val):
                cor = 'white' if (metrica == 'R²' and val < 0.6) or (metrica != 'R²' and val > (vmax * 0.6)) else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7.5, color=cor)

    ax.set_title(titulo, fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(arquivo, dpi=150)
    plt.close()
    print(f'  Salvo: {arquivo}')


def plot_bar_valloss(entradas, arquivo):
    regioes  = [_label_regiao(e['regiao_dir']) for e in entradas]
    val_loss = [e['val_loss'] for e in entradas]
    cores    = plt.cm.tab10(np.linspace(0, 1, len(regioes)))

    fig, ax = plt.subplots(figsize=(max(8, len(regioes) * 1.4), 5))
    bars = ax.bar(regioes, val_loss, color=cores, edgecolor='black', linewidth=0.6)
    ax.bar_label(bars, fmt='%.4f', fontsize=9, padding=3)
    ax.set_ylabel('Val Loss (MSE) — melhor fold', fontsize=11)
    ax.set_title('Val Loss do Melhor Fold por Região (512×4)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(val_loss) * 1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(arquivo, dpi=150)
    plt.close()
    print(f'  Salvo: {arquivo}')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    entradas = _parse_melhores()
    if not entradas:
        print('Nenhuma região em melhores_modelos.csv.')
        sys.exit(1)

    print(f'\nRegiões encontradas: {[e["regiao_dir"] for e in entradas]}')
    print('\nCalculando métricas...')
    df_metricas, matrizes = calcular_metricas(entradas)

    if df_metricas.empty:
        print('Nenhum dado carregado. Verifique os caminhos em melhores_modelos.csv.')
        sys.exit(1)

    csv_path = os.path.join('regioes', 'analise_regioes_metricas.csv')
    df_metricas.to_csv(csv_path, index=False)
    print(f'\n  Tabela completa salva em: {csv_path}')

    # Imprime tabela resumo de R² no terminal
    print('\n' + '='*70)
    print('  RESUMO R² por Região × Variável')
    print('='*70)
    regioes = list(matrizes['r2'].keys())
    header  = f'  {"Região":18s}' + ''.join(f'{t[:7]:>8s}' for t in TARGETS_LABEL)
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for r in regioes:
        linha = f'  {r:18s}' + ''.join(
            f'{matrizes["r2"][r].get(t, float("nan")):>8.3f}' for t in TARGETS
        )
        print(linha)
    print('='*70)

    print('\nGerando gráficos...')

    plot_heatmap(
        matrizes['r2'], 'R²', 'RdYlGn', 0.0, 1.0,
        'R² por Região e Variável (Melhor Fold — 512×4)',
        os.path.join(OUT_DIR, 'heatmap_r2.png'),
    )
    plot_heatmap(
        matrizes['mae'], 'MAE', 'YlOrRd_r',
        0.0, max(v for r in matrizes['mae'].values() for v in r.values() if not np.isnan(v)),
        'MAE por Região e Variável (Melhor Fold — 512×4)',
        os.path.join(OUT_DIR, 'heatmap_mae.png'),
    )
    plot_bar_valloss(entradas, os.path.join(OUT_DIR, 'bar_valloss.png'))

    print('\nConcluído!')


if __name__ == '__main__':
    main()
