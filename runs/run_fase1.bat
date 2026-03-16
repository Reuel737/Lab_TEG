@echo off
set DATAFILE=dados_filtrados/pandas_regioes/transform-fp-head_cellcenter.pandas
set MAXITER=1000
set BATCHSIZE=500
set RESULTS_DIR=result_fase1
set LOGS_DIR=logs/fase1
set HISTCSV=historico/hist_exp_fase1.csv

mkdir %RESULTS_DIR% 2>nul
mkdir %LOGS_DIR% 2>nul
mkdir historico 2>nul

:: Todos os experimentos comentados estao abaixo
:: Descomenta os que quiser rodar

:: call :run_exp exp01 "64 64"
:: call :run_exp exp02 "64 64 64"
:: call :run_exp exp03 "64 64 64 64"
:: call :run_exp exp04 "64 64 64 64 64"

:: call :run_exp exp05 "128 128"
:: call :run_exp exp06 "128 128 128"
:: call :run_exp exp07 "128 128 128 128"
:: call :run_exp exp08 "128 128 128 128 128"

call :run_exp exp09 "256 256"
call :run_exp exp10 "256 256 256"
call :run_exp exp11 "256 256 256 256"
call :run_exp exp12 "256 256 256 256 256"

call :run_exp exp13 "512 512"
call :run_exp exp14 "512 512 512"
call :run_exp exp15 "512 512 512 512"
call :run_exp exp16 "512 512 512 512 512"

echo.
echo ======================================================
echo  Todos os experimentos concluidos!
echo  Para visualizar no TensorBoard:
echo    tensorboard --logdir %LOGS_DIR%
echo ======================================================
goto :eof

:run_exp
set EXP=%1
set LAYERS=%2
set LAYERS_TAG=%LAYERS: =-%
set LOG_TAG=%EXP%_L%LAYERS_TAG%
set LOG_DIR=%LOGS_DIR%/%LOG_TAG%
set OUTPUT=%RESULTS_DIR%/%LOG_TAG%/%LOG_TAG%

mkdir %RESULTS_DIR%\%LOG_TAG% 2>nul

echo.
echo ======================================================
echo  Rodando: %EXP% ^| Layers: [%LAYERS%]
echo  Log:     %LOG_DIR%
echo  Output:  %OUTPUT%
echo ======================================================

python Tf.py ^
    --Layers %LAYERS% ^
    --DataFile %DATAFILE% ^
    --MaxIter %MAXITER% ^
    --BatchSize %BATCHSIZE% ^
    --LogDir %LOG_DIR% ^
    --FileOutPut %OUTPUT% ^
    --HistCSV %HISTCSV%

echo  Concluido: %EXP%
goto :eof
