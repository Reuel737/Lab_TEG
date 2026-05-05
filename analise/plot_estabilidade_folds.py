#!/usr/bin/env python3
"""
Estabilidade dos 30 folds de K-Fold CV por região.

Para cada região em regioes/melhores_modelos.csv, lê o CSV de histórico
do experimento correspondente e plota:
  - regioes/graficos/estabilidade_folds.png : boxplot val_loss × região
  - regioes/graficos/folds_scatter.png      : val_loss por fold (linha) × região

Uso:
  python3 analise/plot_estabilidade_folds.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MELHORES_CSV = 'regioes/melhores_modelos.csv'
OUT_DIR      = 'regioes/graficos'


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


def _parse_melhores():
    df = pd.read_csv(MELHORES_CSV)
    entradas = []
    for _, row in df.iterrows():
        caminho     = os.path.normpath(row['Caminho_Modelo'])
        partes      = caminho.split(os.sep)
        regiao_dir  = partes[1]
        experimento = str(row['Experimento'])
        best_fold   = int(str(row['Melhor_Fold']).split()[-1])
        entradas.append({
            'regiao_dir':  regiao_dir,
            'experimento': experimento,
            'best_fold':   best_fold,
        })
    return entradas


def carregar_historico(regiao_dir, experimento):
    hist_path = os.path.join('regioes', regiao_dir, 'historico',
                             f'hist_kfold_{experimento}.csv')
    if not os.path.isfile(hist_path):
        print(f'  [SKIP] historico não encontrado: {hist_path}')
        return None
    df = pd.read_csv(hist_path)
    df['val_loss_final'] = pd.to_numeric(df['val_loss_final'], errors='coerce')
    return df.dropna(subset=['val_loss_final'])


def plot_boxplot(dados_regioes, arquivo):
    """
    dados_regioes: dict {label: array de val_loss por fold}
    """
    labels = list(dados_regioes.keys())
    vals   = [dados_regioes[l] for l in labels]
    n_reg  = len(labels)

    fig, ax = plt.subplots(figsize=(max(8, n_reg * 1.6), 6))
    bp = ax.boxplot(vals, labels=labels, patch_artist=True, notch=False,
                    medianprops={'color': 'red', 'linewidth': 2})

    cores = plt.cm.tab10(np.linspace(0, 1, n_reg))
    for patch, cor in zip(bp['boxes'], cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.7)

    ax.set_ylabel('Val Loss (MSE)', fontsize=12)
    ax.set_title('Estabilidade dos 30 Folds de K-Fold CV por Região (512×4)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(arquivo, dpi=150)
    plt.close()
    print(f'  Salvo: {arquivo}')


def plot_scatter_folds(dados_regioes, best_folds, arquivo):
    """
    Plota val_loss por fold como linhas, destacando o melhor fold de cada região.
    """
    labels = list(dados_regioes.keys())
    n_reg  = len(labels)
    cores  = plt.cm.tab10(np.linspace(0, 1, n_reg))

    fig, ax = plt.subplots(figsize=(13, 6))
    for label, cor in zip(labels, cores):
        vals     = dados_regioes[label]
        n_folds  = len(vals)
        folds    = np.arange(1, n_folds + 1)
        ax.plot(folds, vals, marker='o', markersize=4, linewidth=1.2,
                color=cor, alpha=0.8, label=label)
        if label in best_folds:
            bf  = best_folds[label]
            idx = bf - 1
            if 0 <= idx < n_folds:
                ax.scatter([bf], [vals[idx]], color=cor, s=80, zorder=5,
                           edgecolors='black', linewidths=1.2)

    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Val Loss (MSE)', fontsize=12)
    ax.set_title('Val Loss por Fold — K-Fold CV (512×4)\n(marcador sólido = melhor fold registrado)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.85)
    ax.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(arquivo, dpi=150)
    plt.close()
    print(f'  Salvo: {arquivo}')


def imprimir_resumo(dados_regioes):
    print('\n' + '='*60)
    print('  ESTATÍSTICAS DOS 30 FOLDS POR REGIÃO')
    print('='*60)
    print(f'  {"Região":18s} {"Média":>8} {"Std":>8} {"Min":>8} {"Max":>8} {"N":>4}')
    print('  ' + '-'*56)
    for label, vals in dados_regioes.items():
        a = np.array(vals)
        print(f'  {label:18s} {a.mean():>8.4f} {a.std():>8.4f} {a.min():>8.4f} {a.max():>8.4f} {len(a):>4}')
    print('='*60)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    entradas = _parse_melhores()
    if not entradas:
        print('Nenhuma região em melhores_modelos.csv.')
        sys.exit(1)

    dados_regioes = {}
    best_folds    = {}

    for cfg in entradas:
        df = carregar_historico(cfg['regiao_dir'], cfg['experimento'])
        if df is None:
            continue
        label = _label_regiao(cfg['regiao_dir'])
        vals  = df.sort_values('fold')['val_loss_final'].tolist()
        dados_regioes[label] = vals
        best_folds[label]    = cfg['best_fold']

    if not dados_regioes:
        print('Nenhum dado de histórico encontrado.')
        sys.exit(1)

    imprimir_resumo(dados_regioes)

    plot_boxplot(dados_regioes,
                 os.path.join(OUT_DIR, 'estabilidade_folds.png'))
    plot_scatter_folds(dados_regioes, best_folds,
                       os.path.join(OUT_DIR, 'folds_scatter.png'))

    print('\nConcluído!')


if __name__ == '__main__':
    main()
