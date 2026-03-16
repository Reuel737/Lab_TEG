@echo off
set DATAFILE=dados_filtrados/pandas_regioes/transform-fp-head_cellcenter.pandas
set MAXITER=5000
set BATCHSIZE=500
set RESULTS_DIR=result_fase2
set LOGS_DIR=logs/fase2
set HISTCSV=historico/hist_exp_fase2.csv

mkdir %RESULTS_DIR% 2>nul
mkdir %LOGS_DIR% 2>nul
mkdir historico 2>nul

:: exp11 run2 e run3 (run1 ja rodou no notebook)
:: call :run_exp exp11 "256 256 256 256" 2
:: call :run_exp exp11 "256 256 256 256" 3

:: exp12: 256x5
:: call :run_exp exp12 "256 256 256 256 256" 1
:: call :run_exp exp12 "256 256 256 256 256" 2
:: call :run_exp exp12 "256 256 256 256 256" 3

:: exp15: 512x4
call :run_exp exp15 "512 512 512 512" 1
call :run_exp exp15 "512 512 512 512" 2
call :run_exp exp15 "512 512 512 512" 3

:: exp16: 512x5
call :run_exp exp16 "512 512 512 512 512" 1
call :run_exp exp16 "512 512 512 512 512" 2
call :run_exp exp16 "512 512 512 512 512" 3

echo.
echo ======================================================
echo  Fase 2 concluida!
echo  Para visualizar no TensorBoard:
echo    tensorboard --logdir %LOGS_DIR%
echo ======================================================
goto :eof

:run_exp
set EXP=%1
set LAYERS=%2
set RUN=%3
set LOG_TAG=%EXP%_run%RUN%
set LOG_DIR=%LOGS_DIR%/%LOG_TAG%
set OUTPUT=%RESULTS_DIR%/%LOG_TAG%/%LOG_TAG%

mkdir %RESULTS_DIR%\%LOG_TAG% 2>nul

echo.
echo ======================================================
echo  Rodando: %EXP% ^| Run: %RUN% ^| Layers: [%LAYERS%]
echo  Log:     %LOG_DIR%
echo  Output:  %RESULTS_DIR%/%LOG_TAG%/
echo ======================================================

python Tf.py ^
    --Layers %LAYERS% ^
    --DataFile %DATAFILE% ^
    --MaxIter %MAXITER% ^
    --BatchSize %BATCHSIZE% ^
    --LogDir %LOG_DIR% ^
    --FileOutPut %OUTPUT% ^
    --HistCSV %HISTCSV%

echo  Concluido: %EXP% run%RUN%
goto :eof
