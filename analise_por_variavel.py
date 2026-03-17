import numpy as np
import tensorflow as tf

# Carrega dados e modelo
data = np.load('result_fase2/exp16_run1/exp16_run1dataset.npz')
xtrain = data['arr_0']
xval   = data['arr_1']
ytrain = data['arr_2']
yval   = data['arr_3']

model = tf.keras.models.load_model('result_fase2/exp16_run1/exp16_run1.keras')

targets = ['pressure', 'x-velocity', 'y-velocity', 'z-velocity', 'temperature',
           'incident-radiation', 'radiation-temperature', 'rad-heat-flux', 'vr']

# Predicoes
pred_train = model.predict(xtrain, batch_size=500, verbose=0)
pred_val   = model.predict(xval,   batch_size=500, verbose=0)

print('\n' + '='*75)
print('ANÁLISE POR VARIÁVEL — exp15_run1')
print('='*75)
print(f'{"Target":25s} | {"MSE train":>10} | {"MSE val":>10} | {"MAE val":>10} | {"R² val":>8}')
print('-'*75)

for i, t in enumerate(targets):
    mse_t = np.mean((ytrain[:,i] - pred_train[:,i])**2)
    mse_v = np.mean((yval[:,i]   - pred_val[:,i]  )**2)
    mae_v = np.mean(np.abs(yval[:,i] - pred_val[:,i]))
    
    # R²
    ss_res = np.sum((yval[:,i] - pred_val[:,i])**2)
    ss_tot = np.sum((yval[:,i] - yval[:,i].mean())**2)
    r2 = 1 - ss_res/ss_tot
    
    print(f'{t:25s} | {mse_t:>10.4f} | {mse_v:>10.4f} | {mae_v:>10.4f} | {r2:>8.4f}')

print('='*75)
print('\nSalvando em analise_por_variavel.csv...')

import csv
with open('analise_por_variavel16.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['target', 'mse_train', 'mse_val', 'mae_val', 'r2_val'])
    for i, t in enumerate(targets):
        mse_t = np.mean((ytrain[:,i] - pred_train[:,i])**2)
        mse_v = np.mean((yval[:,i]   - pred_val[:,i]  )**2)
        mae_v = np.mean(np.abs(yval[:,i] - pred_val[:,i]))
        ss_res = np.sum((yval[:,i] - pred_val[:,i])**2)
        ss_tot = np.sum((yval[:,i] - yval[:,i].mean())**2)
        r2 = 1 - ss_res/ss_tot
        writer.writerow([t, round(mse_t,6), round(mse_v,6), round(mae_v,6), round(r2,6)])

print('Salvo!')
