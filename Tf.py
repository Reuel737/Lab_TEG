import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Normalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import mse
from sklearn.model_selection import train_test_split
import numpy as np
 
 
import datetime
log_dir_def = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
 
import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument("--Layers", default="10 10", help='maximum depth')
parser.add_argument("--DataFile", default='dados_filtrados/pandas_regioes/transform-fp-head_cellcenter.pandas', type=str, help='maximum depth')
parser.add_argument("--FileOutPut", default="untitle", type=str, help='maximum depth')
parser.add_argument("--MaxIter", default=1000, type=int, help='maximum depth')
parser.add_argument("--BatchSize", default=500, type=int, help='maximum depth')
parser.add_argument("--Acum", default=None, type=int, help='maximum depth')
parser.add_argument("--LogDir", default=log_dir_def, type=str, help='maximum depth')
parser.add_argument("--HistCSV", default="hist_exp.csv", type=str, help='CSV file to save experiment summary')
parser.add_argument("--Plot", action='store_true', help='Enable plot mode (default: False)')
parser.add_argument("--L2", default=0.0, type=float, help='L2 regularization factor (0 = disabled)')
 
 
Layers    = [int(x) for x in str.split(parser.parse_args().Layers,' ')]
DataFile  = parser.parse_args().DataFile
filename  = parser.parse_args().FileOutPut
import os
os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
maxiter   = parser.parse_args().MaxIter
BatchSize = parser.parse_args().BatchSize
acum = parser.parse_args().Acum
log_dir   = parser.parse_args().LogDir
hist_csv  = parser.parse_args().HistCSV
plot      = parser.parse_args().Plot
l2_factor = parser.parse_args().L2
 
Dados = pd.read_pickle(DataFile)
#Vars = ['x','y','z','Vel','Tinsu','Qinsu']
#Target = ['u','v','w','p','t']
Vars = ['x-coordinate','y-coordinate','z-coordinate','Vel','Tinsu','Qinsu']
Target = ['pressure','x-velocity','y-velocity','z-velocity','temperature','incident-radiation','radiation-temperature','rad-heat-flux','vr']
 
xtrain,xval,ytrain,yval = train_test_split(Dados[Vars].to_numpy(),Dados[Target].to_numpy(),test_size=0.1)
 
np.savez(filename+'dataset.npz',xtrain,xval,ytrain,yval)
 
# TensorBoard progress Monitor
from tensorflow.keras.callbacks import TensorBoard
 
#if log_dir != log_dir_def:
#    import os
#    os.makedirs(log_dir, exist_ok=True)
 
print('Log folder '+log_dir)
 
# Set up TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=log_dir,
                         histogram_freq=30,
                         write_graph=True,
                         write_images=True,
                         update_freq='epoch',
#                         profile_batch=2,
                         embeddings_freq=100,
#                         embeddings_metadata=None
                                   )
 
ninput = len(Vars)
 
normaliza = Normalization(axis=-1)
normaliza.adapt( Dados[Vars].to_numpy())
 
layers=Layers
activation=len(layers) * ['relu']
noutput = len(Target)
actoutput = 'linear'
 
 
reg = L2(l2_factor) if l2_factor > 0 else None
 
#hiddenlayers  = [ Input(shape=[ninput])]
hiddenlayers  = [ normaliza]
hiddenlayers += [ Dense(neurons, activation=act,kernel_regularizer=reg) for neurons,act in zip(layers,activation)]
hiddenlayers += [ Dense(noutput, activation=actoutput,kernel_regularizer=reg) ]
 
model = Sequential(hiddenlayers)
 
early_stopping = EarlyStopping(
    monitor='loss',
    patience=200,
    min_delta=1e-8,
    #mode='min',
    verbose=1,
    restore_best_weights=True
)
 
from tensorflow.keras.optimizers.schedules import ExponentialDecay
#initial_learning_rate = 0.001
#lr_schedule = ExponentialDecay(
#    initial_learning_rate,
#    decay_steps=10000,
#    decay_rate=0.96,
#    staircase=True)
 
# Pass the schedule to the optimizer
#from tensorflow.keras.optimizers import SGD
##optimizer = SGD(learning_rate=lr_schedule)
#optimizer = SGD(
#    learning_rate=0.0001, 
#    decay=0.00001/1000, 
#    momentum=0.9, 
#    nesterov=True # often works slightly better than standard momentum
#)
 
#from tensorflow.keras.optimizers import Adadelta
#optimizer =Adadelta(
#    learning_rate=0.001,
#    rho=0.95,
#    epsilon=1e-07,
#    weight_decay=None,
#    clipnorm=None,
#    clipvalue=None,
#    global_clipnorm=None,
#    use_ema=False,
#    ema_momentum=0.99,
#    ema_overwrite_frequency=None,
#    loss_scale_factor=None,
#    gradient_accumulation_steps=None,
#    name='adadelta',
#)
 
#from tensorflow.keras.optimizers import Adagrad
#optimizer = Adagrad(
#    learning_rate=0.001,
#    initial_accumulator_value=0.1,
#    epsilon=1e-07,
#    weight_decay=None,
#    clipnorm=None,
#    clipvalue=None,
#    global_clipnorm=None,
#    use_ema=False,
#    ema_momentum=0.99,
#    ema_overwrite_frequency=None,
#    loss_scale_factor=None,
#    gradient_accumulation_steps=None,
#    name='adagrad',
#)
 
from tensorflow.keras.optimizers import Adam
optimizer = Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    loss_scale_factor=None,
    gradient_accumulation_steps=acum,
    name='adam',
)
 
from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Metric to monitor (e.g., validation loss)
    factor=0.9,          # Factor by which the learning rate will be reduced. new_lr = lr * factor
    patience=50,          # Number of epochs with no improvement after which learning rate will be reduced
    min_delta=0.0001,    # Threshold for measuring the new optimum, to focus only on significant changes
    mode='min',          # In 'min' mode, lr is reduced when the monitored quantity stops decreasing
    verbose=1,           # 0: quiet, 1: update messages
    cooldown = 20,
    min_lr=1e-5          # Lower bound on the learning rate
)
 
model.compile(
    optimizer = optimizer,
    #optimizer='adam',
    #optimizer='SGD',
    #optimizer=sgd_optimizer,
    #loss='mean_absolute_error',  # Standard loss for regression
    #loss='mean_squared_logarithmic_error',  # Standard loss for regression
    loss='mean_squared_error',  # Standard loss for regression
    metrics=['mean_absolute_error'] # Additional metrics for evaluation
)
 
callbk = [tensorboard_callback,
          reduce_lr,
          early_stopping,
          ]
 
 
history = model.fit( #normaliza(Dados[Vars].to_numpy()),
    xtrain,
    ytrain,
    epochs=maxiter,  # Number of training iterations
    batch_size=BatchSize,
    verbose=1,
    callbacks = callbk,
    validation_data=(xval,yval)
)
 
model.save(filename+'.keras')
 
ylims = [(minimo,maximo) for minimo,maximo in zip( Dados[Target].min(axis=0),Dados[Target].max(axis=0) )]
 
#Pred = model(normaliza(Dados[Vars].to_numpy()))
 
Predt = model(xtrain[0:BatchSize,:]).numpy()
c = BatchSize
while c+BatchSize < len(xtrain):
    Predt = np.row_stack((Predt,model(xtrain[c:c+BatchSize,:]).numpy()))
    c += BatchSize
 
Predt = np.row_stack((Predt,model(xtrain[c:,:]).numpy()))
 
Predv = model(xval[0:BatchSize,:]).numpy()
c = BatchSize
while c+BatchSize < len(xval):
    Predv = np.row_stack((Predv,model(xval[c:c+BatchSize,:]).numpy()))
    c += BatchSize
 
Predv = np.row_stack((Predv,model(xval[c:,:]).numpy()))
 
from sklearn.linear_model import LinearRegression
LinRegErro = []
for i,t in enumerate(Target): LinRegErro.append(LinearRegression().fit(yval[:,i].reshape(-1,1),Predv[:,i].reshape(-1,1)))
 
print('Mean Squared Error: train ', mse(ytrain, Predt),' - validation ',mse(yval, Predv))
 
#import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
 
#fig, axs = plt.subplots(len(Targets),1)
for i,t in enumerate(Target):
    plt.clf()
    #title = 'Target '+Target[i]+' - Rsquare ' + str(LinRegErro[i].score( yval[:,i].reshape(-1,1),Predv[:,i].reshape(-1,1))) 
    plt.plot(ytrain[:,i],Predt[:,i],'.k',label='Traing Set')
    plt.plot(yval[:,i],Predv[:,i],'.b',label='Validation Set')
    plt.xlabel('Target - '+t)
    plt.ylabel('Prediction - '+t)
    #plt.title(title)
    plt.xlim(ylims[i])
    plt.ylim(ylims[i])
    plt.plot(ylims[i],ylims[i],'k')
    plt.grid('on')
    plt.savefig(filename+'-'+t+'-erro.eps')
    if plot: plt.show()
 
errot = np.abs(ytrain - Predt)
errov = np.abs(yval - Predv)
 
for i,t in enumerate(Target):
    plt.clf()
    plt.hist(errot[:,i], bins=30, color='skyblue', edgecolor='black')
    plt.hist(errov[:,i], bins=30, edgecolor='black')
    plt.xlabel('Abs Error')
    plt.ylabel('Frequency')
    plt.xlim((0,np.append(errov[:,i],errot[:,i]).max() ))
    plt.ylim((0,500000))
    plt.title('Error Distribution - '+t)
    plt.grid(True) # Optional: adds grid line
    plt.savefig(filename+'-'+t+'-hist.eps')
    if plot: plt.show()
    plt.clf()
 
# RESUMO DO EXPERIMENTO
 
val_loss_final   = history.history['val_loss'][-1]
train_loss_final = history.history['loss'][-1]
melhor_epoca     = int(np.argmin(history.history['val_loss'])) + 1
total_epocas     = len(history.history['loss'])
 
print("\n")
print("=" * 52)
print("  RESUMO DO EXPERIMENTO")
print("=" * 52)
print(f"  Experimento   : {filename}")
print(f"  Layers        : {Layers}")
print(f"  MaxIter       : {maxiter}  |  BatchSize: {BatchSize}")
print(f"  L2 factor     : {l2_factor}")
print("-" * 52)
print(f"  train_loss_final : {train_loss_final:.6f}")
print(f"  val_loss_final   : {val_loss_final:.6f}")
print(f"  melhor_epoca     : {melhor_epoca} / {total_epocas}")
print("=" * 52)
print()
 
# Salvar resumo em CSV acumulativo
import csv, datetime
hist_exists = os.path.isfile(hist_csv)
with open(hist_csv, 'a', newline='') as f:
    writer = csv.writer(f)
    if not hist_exists:
        writer.writerow(["timestamp","experimento","layers","maxiter","batchsize","l2","train_loss_final","val_loss_final","melhor_epoca","total_epocas"])
    writer.writerow([
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        os.path.basename(filename),
        str(Layers),
        maxiter,
        BatchSize,
        l2_factor,
        round(train_loss_final, 6),
        round(val_loss_final, 6),
        melhor_epoca,
        total_epocas
    ])
print(f"  Resumo salvo em: {hist_csv}")