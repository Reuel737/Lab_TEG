#!/bin/bash

DATAFILE="dados_filtrados/pandas_regioes/transform-fp-head_cellcenter.pandas"
MAXITER=5000
BATCHSIZE=500
KFOLDS=30
REGIAO="fp_head"
RESULTS_DIR="regioes/${REGIAO}/results"
LOGS_DIR="regioes/${REGIAO}/logs"

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

run_kfold() {
    local EXP=$1
    local LAYERS=$2

    local LOG_DIR="${LOGS_DIR}/${EXP}"
    local OUTPUT="${RESULTS_DIR}/${EXP}/${EXP}"
    local HISTCSV="regioes/${REGIAO}/historico/hist_kfold_${EXP}.csv"

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

run_kfold "512x4" "512 512 512 512"

echo ""
echo "======================================================"
echo " Para visualizar no TensorBoard:"
echo "   tensorboard --logdir $LOGS_DIR"
echo "======================================================"
