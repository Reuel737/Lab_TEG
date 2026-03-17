#!/bin/bash

DATAFILE="dados_filtrados/pandas_regioes/transform-fp-head_cellcenter.pandas"
MAXITER=5000
BATCHSIZE=500
RESULTS_DIR="result_fase3"
LOGS_DIR="logs/fase3"
HISTCSV="historico/hist_exp_fase3.csv"

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "historico"

run_exp() {
    local EXP=$1
    local LAYERS=$2
    local RUN=$3
    local L2=$4
    local L2_TAG="${L2//./_}"
    local LOG_TAG="${EXP}_L2-${L2_TAG}_run${RUN}"
    local LOG_DIR="${LOGS_DIR}/${LOG_TAG}"
    local OUTPUT="${RESULTS_DIR}/${LOG_TAG}/${LOG_TAG}"
    mkdir -p "${RESULTS_DIR}/${LOG_TAG}"

    echo ""
    echo "======================================================"
    echo " Rodando: $EXP | L2: $L2 | Run: $RUN"
    echo " Layers:  [$LAYERS]"
    echo " Log:     $LOG_DIR"
    echo " Output:  ${RESULTS_DIR}/${LOG_TAG}/"
    echo "======================================================"

    python3 Tf.py \
        --Layers "$LAYERS" \
        --DataFile "$DATAFILE" \
        --MaxIter $MAXITER \
        --BatchSize $BATCHSIZE \
        --L2 $L2 \
        --LogDir "$LOG_DIR" \
        --FileOutPut "$OUTPUT" \
        --HistCSV "$HISTCSV"

    echo " Concluído: $LOG_TAG"
}

# exp15: 512x4
run_exp "exp15" "512 512 512 512" 1 0.0001
run_exp "exp15" "512 512 512 512" 2 0.0001
run_exp "exp15" "512 512 512 512" 3 0.0001

run_exp "exp15" "512 512 512 512" 1 0.00005
run_exp "exp15" "512 512 512 512" 2 0.00005
run_exp "exp15" "512 512 512 512" 3 0.00005

run_exp "exp15" "512 512 512 512" 1 0.00001
run_exp "exp15" "512 512 512 512" 2 0.00001
run_exp "exp15" "512 512 512 512" 3 0.00001

# exp16: 512x5
run_exp "exp16" "512 512 512 512 512" 1 0.0001
run_exp "exp16" "512 512 512 512 512" 2 0.0001
run_exp "exp16" "512 512 512 512 512" 3 0.0001

run_exp "exp16" "512 512 512 512 512" 1 0.00005
run_exp "exp16" "512 512 512 512 512" 2 0.00005
run_exp "exp16" "512 512 512 512 512" 3 0.00005

run_exp "exp16" "512 512 512 512 512" 1 0.00001
run_exp "exp16" "512 512 512 512 512" 2 0.00001
run_exp "exp16" "512 512 512 512 512" 3 0.00001

echo ""
echo "======================================================"
echo " Fase 3 concluída!"
echo " Para visualizar no TensorBoard:"
echo "   tensorboard --logdir $LOGS_DIR"
echo "======================================================"
