!/bin/bash

DATAFILE="dados_filtrados/pandas_regioes/transform-fp-head_cellcenter.pandas"
MAXITER=5000
BATCHSIZE=500
RESULTS_DIR="result_fase2"
LOGS_DIR="logs/fase2"
HISTCSV="historico/hist_exp_fase2.csv"

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

run_exp() {
    local EXP=$1
    local LAYERS=$2
    local RUN=$3

    local LOG_TAG="${EXP}_run${RUN}"
    local LOG_DIR="${LOGS_DIR}/${LOG_TAG}"
    local OUTPUT="${RESULTS_DIR}/${LOG_TAG}/${LOG_TAG}"

    mkdir -p "${RESULTS_DIR}/${LOG_TAG}"

    echo ""
    echo "======================================================"
    echo " Rodando: $EXP | Run: $RUN | Layers: [$LAYERS]"
    echo " Log:     $LOG_DIR"
    echo " Output:  ${RESULTS_DIR}/${LOG_TAG}/"
    echo "======================================================"

    python3 Tf.py \
        --Layers "$LAYERS" \
        --DataFile "$DATAFILE" \
        --MaxIter $MAXITER \
        --BatchSize $BATCHSIZE \
        --LogDir "$LOG_DIR" \
        --FileOutPut "$OUTPUT" \
        --HistCSV "$HISTCSV"

    echo " Concluído: $EXP run$RUN"
}

# EXPERIMENTOS

# exp11: 256x4
# run_exp "exp11" "256 256 256 256" 1 --já rodado
run_exp "exp11" "256 256 256 256" 2
run_exp "exp11" "256 256 256 256" 3

# exp12: 256x5
run_exp "exp12" "256 256 256 256 256" 1
run_exp "exp12" "256 256 256 256 256" 2
run_exp "exp12" "256 256 256 256 256" 3

# exp15: 512x4
run_exp "exp15" "512 512 512 512" 1
run_exp "exp15" "512 512 512 512" 2
run_exp "exp15" "512 512 512 512" 3

# exp16: 512x5
run_exp "exp16" "512 512 512 512 512" 1
run_exp "exp16" "512 512 512 512 512" 2
run_exp "exp16" "512 512 512 512 512" 3

echo ""
echo "======================================================"
echo " Fase 2 concluída!"
echo " Para visualizar no TensorBoard:"
echo "   tensorboard --logdir $LOGS_DIR"
echo "======================================================"
