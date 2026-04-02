import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Normalization
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os, csv, datetime, argparse

log_dir_def = "logs/fase5/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument("--Layers",     default="10 10")
parser.add_argument("--DataFile",   default='dados_filtrados/pandas_regioes/transform-fp-head_cellcenter.pandas', type=str)
parser.add_argument("--FileOutPut", default="result_fase5/exp", type=str)
parser.add_argument("--MaxIter",    default=5000, type=int)
parser.add_argument("--BatchSize",  default=500,  type=int)
parser.add_argument("--LogDir",     default=log_dir_def, type=str)
parser.add_argument("--HistCSV",    default="historico/hist_kfold.csv", type=str)
parser.add_argument("--KFolds",     default=30, type=int)
parser.add_argument("--L2",         default=0.0, type=float)
parser.add_argument("--Acum",       default=None, type=int)

args = parser.parse_args()
Layers     = [int(x) for x in args.Layers.split()]
DataFile   = args.DataFile
filename   = args.FileOutPut
maxiter    = args.MaxIter
BatchSize  = args.BatchSize
log_dir    = args.LogDir
hist_csv   = args.HistCSV
k          = args.KFolds
l2_factor  = args.L2
acum       = args.Acum

os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
os.makedirs(os.path.dirname(hist_csv)  if os.path.dirname(hist_csv)  else '.', exist_ok=True)

Vars   = ['x-coordinate','y-coordinate','z-coordinate','Vel','Tinsu','Qinsu']
Target = ['pressure','x-velocity','y-velocity','z-velocity','temperature',
          'incident-radiation','radiation-temperature','rad-heat-flux','vr']

print(f'Carregando dados: {DataFile}')
Dados = pd.read_pickle(DataFile)

X = Dados[Vars].to_numpy()
Y = Dados[Target].to_numpy()

print(f'Total de pontos: {len(X)} | K={k} folds')

# Normalizacao global
normaliza = Normalization(axis=-1)
normaliza.adapt(X)

# KFold scikit-learn
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# CSV
hist_exists = os.path.isfile(hist_csv)
csv_file = open(hist_csv, 'a', newline='')
writer = csv.writer(csv_file)
if not hist_exists:
    writer.writerow(['timestamp','experimento','fold','layers','maxiter','batchsize',
                     'l2','train_loss_final','val_loss_final','melhor_epoca','total_epocas'])

fold_val_losses = []

# Variáveis para rastrear o melhor modelo globalmente
best_global_val_loss = float('inf')
best_fold = -1

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    xtrain, xval = X[train_idx], X[val_idx]
    ytrain, yval = Y[train_idx], Y[val_idx]

    print(f'\n[Fold {fold_idx:02d}/{k}] treino={len(xtrain)} pts | val={len(xval)} pts')

    reg = L2(l2_factor) if l2_factor > 0 else None
    activation = len(Layers) * ['relu']

    hiddenlayers = [normaliza]
    for neurons, act in zip(Layers, activation):
        hiddenlayers += [Dense(neurons, activation=act, kernel_regularizer=reg)]
    hiddenlayers += [Dense(len(Target), activation='linear', kernel_regularizer=reg)]

    model = Sequential(hiddenlayers)
    model.compile(
        optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                       epsilon=1e-07, gradient_accumulation_steps=acum, name='adam'),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )

    fold_log_dir = os.path.join(log_dir, f'fold{fold_idx:02d}')
    os.makedirs(fold_log_dir, exist_ok=True)

    callbacks = [
        TensorBoard(log_dir=fold_log_dir, histogram_freq=0,
                    write_graph=False, update_freq='epoch'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=50,
                          min_delta=0.0001, mode='min', verbose=0,
                          cooldown=20, min_lr=1e-5),
        EarlyStopping(monitor='val_loss', patience=200, min_delta=1e-8,
                      verbose=0, restore_best_weights=True),
    ]

    history = model.fit(
        xtrain, ytrain,
        epochs=maxiter,
        batch_size=BatchSize,
        verbose=0,
        callbacks=callbacks,
        validation_data=(xval, yval)
    )

    val_loss_final   = min(history.history['val_loss'])
    train_loss_final = history.history['loss'][-1]
    melhor_epoca     = int(np.argmin(history.history['val_loss'])) + 1
    total_epocas     = len(history.history['loss'])

    fold_val_losses.append(val_loss_final)

    # Identifica se este é o melhor fold até o momento
    if val_loss_final < best_global_val_loss:
        best_global_val_loss = val_loss_final
        best_fold = fold_idx

    # Salva o modelo e o dataset com o número do fold atual
    model_filename = f"{filename}_fold{fold_idx:02d}.keras"
    dataset_filename = f"{filename}_fold{fold_idx:02d}_dataset.npz"
    
    model.save(model_filename)
    np.savez(dataset_filename, xtrain=xtrain, xval=xval, ytrain=ytrain, yval=yval)

    writer.writerow([
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        os.path.basename(filename),
        fold_idx,
        str(Layers),
        maxiter,
        BatchSize,
        l2_factor,
        round(train_loss_final, 6),
        round(val_loss_final, 6),
        melhor_epoca,
        total_epocas
    ])
    csv_file.flush()

    print(f'  val_loss={val_loss_final:.4f} | train_loss={train_loss_final:.4f} | melhor_epoca={melhor_epoca}/{total_epocas}')

csv_file.close()

media = np.mean(fold_val_losses)
std   = np.std(fold_val_losses)
print('\n' + '='*52)
print('  RESUMO K-FOLD')
print('='*52)
print(f'  Experimento : {os.path.basename(filename)}')
print(f'  Layers      : {Layers}')
print(f'  K folds     : {k}')
print(f'  val_loss médio : {media:.4f}')
print(f'  val_loss std   : {std:.4f}')
print(f'  val_loss min   : {min(fold_val_losses):.4f}')
print(f'  val_loss max   : {max(fold_val_losses):.4f}')
print('-'*52)
print(f'  MELHOR FOLD    : {best_fold:02d} (val_loss: {best_global_val_loss:.6f})')
print(f'  Melhor Modelo  : {filename}_fold{best_fold:02d}.keras')
print(f'  Melhor Dataset : {filename}_fold{best_fold:02d}_dataset.npz')
print('='*52)