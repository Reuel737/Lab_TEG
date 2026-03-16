@echo off
set DATAFILE=dados_filtrados/pandas_regioes/transform-fp-head_cellcenter.pandas
set MAXITER=5000
set BATCHSIZE=500
set RESULTS_DIR=result_fase3
set LOGS_DIR=logs/fase3
set HISTCSV=historico/hist_exp_fase3.csv

mkdir %RESULTS_DIR% 2>nul
mkdir %LOGS_DIR% 2>nul
mkdir historico 2>nul

:: exp15: 512x4
call :run_exp exp15 "512 512 512 512" 1 0.0001
call :run_exp exp15 "512 512 512 512" 2 0.0001
call :run_exp exp15 "512 512 512 512" 3 0.0001

call :run_exp exp15 "512 512 512 512" 1 0.00005
call :run_exp exp15 "512 512 512 512" 2 0.00005
call :run_exp exp15 "512 512 512 512" 3 0.00005

call :run_exp exp15 "512 512 512 512" 1 0.00001
call :run_exp exp15 "512 512 512 512" 2 0.00001
call :run_exp exp15 "512 512 512 512" 3 0.00001

:: exp16: 512x5
call :run_exp exp16 "512 512 512 512 512" 1 0.0001
call :run_exp exp16 "512 512 512 512 512" 2 0.0001
call :run_exp exp16 "512 512 512 512 512" 3 0.0001

call :run_exp exp16 "512 512 512 512 512" 1 0.00005
call :run_exp exp16 "512 512 512 512 512" 2 0.00005
call :run_exp exp16 "512 512 512 512 512" 3 0.00005

call :run_exp exp16 "512 512 512 512 512" 1 0.00001
call :run_exp exp16 "512 512 512 512 512" 2 0.00001
call :run_exp exp16 "512 512 512 512 512" 3 0.00001

echo.
echo ======================================================
echo  Fase 3 concluida!
echo  Para visualizar no TensorBoard:
echo    tensorboard --logdir %LOGS_DIR%
echo ======================================================
goto :eof

:run_exp
set EXP=%1
set LAYERS=%2
set RUN=%3
set L2=%4
set L2_TAG=%L2:.=p%
set LOG_TAG=%EXP%_L2-%L2_TAG%_run%RUN%
set LOG_DIR=%LOGS_DIR%/%LOG_TAG%
set OUTPUT=%RESULTS_DIR%/%LOG_TAG%/%LOG_TAG%

mkdir %RESULTS_DIR%\%LOG_TAG% 2>nul

echo.
echo ======================================================
echo  Rodando: %EXP% ^| L2: %L2% ^| Run: %RUN%
echo  Layers:  [%LAYERS%]
echo  Log:     %LOG_DIR%
echo  Output:  %RESULTS_DIR%/%LOG_TAG%/
echo ======================================================

python Tf.py ^
    --Layers %LAYERS% ^
    --DataFile %DATAFILE% ^
    --MaxIter %MAXITER% ^
    --BatchSize %BATCHSIZE% ^
    --L2 %L2% ^
    --LogDir %LOG_DIR% ^
    --FileOutPut %OUTPUT% ^
    --HistCSV %HISTCSV%

echo  Concluido: %LOG_TAG%
goto :eof
