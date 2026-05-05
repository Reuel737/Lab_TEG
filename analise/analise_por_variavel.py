#!/usr/bin/env python3
"""
Análise por variável (MSE, MAE, R²) para cada região registrada em
regioes/melhores_modelos.csv.

Uso:
  python3 analise/analise_por_variavel.py                 # todas as regiões
  python3 analise/analise_por_variavel.py fp_head mt_head # regiões específicas
"""

import sys
import os
import csv
import numpy as np
import tensorflow as tf

MELHORES_CSV = 'regioes/melhores_modelos.csv'

TARGETS = [
    'pressure', 'x-velocity', 'y-velocity', 'z-velocity', 'temperature',
    'incident-radiation', 'radiation-temperature', 'rad-heat-flux', 'vr',
]


def _parse_melhores(filtro=None):
    import pandas as pd
    df = pd.read_csv(MELHORES_CSV)
    entradas = []
    for _, row in df.iterrows():
        caminho = os.path.normpath(row['Caminho_Modelo'])
        partes   = caminho.split(os.sep)
        regiao_dir = partes[1]
        if filtro and regiao_dir not in filtro:
            continue
        fold = int(str(row['Melhor_Fold']).split()[-1])
        results_dir = os.path.dirname(caminho)
        basename    = os.path.basename(caminho)
        prefix      = basename.replace(f'_fold{fold:02d}.keras', '')
        entradas.append({
            'regiao_dir':  regiao_dir,
            'results_dir': results_dir,
            'prefix':      prefix,
            'fold':        fold,
            'experimento': prefix,
        })
    return entradas


def _metricas(yreal, ypred):
    mse  = np.mean((yreal - ypred) ** 2)
    mae  = np.mean(np.abs(yreal - ypred))
    ss_r = np.sum((yreal - ypred) ** 2)
    ss_t = np.sum((yreal - yreal.mean()) ** 2)
    r2   = 1.0 - ss_r / ss_t if ss_t > 0 else float('nan')
    return mse, mae, r2


def analisar(cfg):
    rd, prefix, fold = cfg['results_dir'], cfg['prefix'], cfg['fold']
    dataset_path = os.path.join(rd, f'{prefix}_fold{fold:02d}_dataset.npz')
    model_path   = os.path.join(rd, f'{prefix}_fold{fold:02d}.keras')

    if not os.path.isfile(dataset_path):
        print(f'  [SKIP] dataset não encontrado: {dataset_path}')
        return
    if not os.path.isfile(model_path):
        print(f'  [SKIP] modelo não encontrado: {model_path}')
        return

    d    = np.load(dataset_path)
    xval = d['xval']
    yval = d['yval']

    model    = tf.keras.models.load_model(model_path)
    pred_val = model.predict(xval, batch_size=2000, verbose=0)

    print(f'\n{"="*75}')
    print(f'  Região : {cfg["regiao_dir"]}  |  Exp: {cfg["experimento"]}  |  Fold {fold:02d}')
    print(f'{"="*75}')
    print(f'  {"Target":25s} | {"MSE val":>10} | {"MAE val":>10} | {"R² val":>8}')
    print(f'  {"-"*63}')

    rows = []
    for i, t in enumerate(TARGETS):
        mse, mae, r2 = _metricas(yval[:, i], pred_val[:, i])
        print(f'  {t:25s} | {mse:>10.4f} | {mae:>10.4f} | {r2:>8.4f}')
        rows.append([t, round(mse, 6), round(mae, 6), round(r2, 6)])

    out_csv = os.path.join('regioes', cfg['regiao_dir'],
                           f'analise_por_variavel_{cfg["experimento"]}.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['target', 'mse_val', 'mae_val', 'r2_val'])
        writer.writerows(rows)
    print(f'\n  Salvo em: {out_csv}')


def main():
    filtro  = sys.argv[1:] if len(sys.argv) > 1 else None
    entradas = _parse_melhores(filtro)
    if not entradas:
        print('Nenhuma região encontrada. Verifique os argumentos ou o CSV.')
        return
    for cfg in entradas:
        analisar(cfg)


if __name__ == '__main__':
    main()
