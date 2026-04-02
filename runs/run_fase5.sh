#!/bin/bash

DATAFILE="dados_filtrados/pandas_regioes/transform-fp-head_cellcenter.pandas"
MAXITER=5000
BATCHSIZE=500
KFOLDS=30
RESULTS_DIR="result_fase5"
LOGS_DIR="logs/fase5"

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "historico"

run_kfold() {
    local EXP=$1
    local LAYERS=$2

    local LOG_DIR="${LOGS_DIR}/${EXP}"
    local OUTPUT="${RESULTS_DIR}/${EXP}/${EXP}"
    local HISTCSV="historico/hist_kfold_${EXP}.csv"

    mkdir -p "$LOG_DIR"
    mkdir -p "$OUTPUT"

    echo ""
    echo "======================================================"
    echo " K-Fold: $EXP | Layers: [$LAYERS]"
    echo " K=$KFOLDS folds | Split por CaseId"
    echo " Log:    $LOG_DIR"
    echo " Output: $OUTPUT"
    echo " CSV:    $HISTCSV"
    echo "======================================================"

    python3 Tf_kfold.py \
        --Layers "$LAYERS" \
        --DataFile "$DATAFILE" \
        --MaxIter $MAXITER \
        --BatchSize $BATCHSIZE \
        --KFolds $KFOLDS \
        --LogDir "$LOG_DIR" \
        --FileOutPut "$OUTPUT" \
        --HistCSV "$HISTCSV"

    echo " Concluído: $EXP"
}

# run_kfold "exp08" "128 128 128 128 128"
# run_kfold "exp12" "256 256 256 256 256"
# run_kfold "exp13" "512 512"
# run_kfold "exp14" "512 512 512"
run_kfold "exp15" "512 512 512 512"
# run_kfold "exp16" "512 512 512 512 512"

echo ""
echo "======================================================"
echo " Fase 5 concluída!"
echo " Para visualizar no TensorBoard:"
echo "   tensorboard --logdir $LOGS_DIR"
echo "======================================================"
