!/bin/bash

DATAFILE="dados_filtrados/pandas_regioes/transform-fp-head_cellcenter.pandas"
MAXITER=1000
BATCHSIZE=500
RESULTS_DIR="result_fase1"
LOGS_DIR="logs/fit"

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

run_exp() {
    local EXP=$1
    local LAYERS=$2

    local LOG_TAG="${EXP}_L${LAYERS// /-}"
    local LOG_DIR="${LOGS_DIR}/${LOG_TAG}"
    local OUTPUT="${RESULTS_DIR}/${LOG_TAG}/${LOG_TAG}"
    mkdir -p "${RESULTS_DIR}/${LOG_TAG}"

    echo ""
    echo "======================================================"
    echo " Rodando: $EXP | Layers: [$LAYERS]"
    echo " Log:     $LOG_DIR"
    echo " Output:  $OUTPUT"
    echo "======================================================"

    python Tf.py \
        --Layers "$LAYERS" \
        --DataFile "$DATAFILE" \
        --MaxIter $MAXITER \
        --BatchSize $BATCHSIZE \
        --LogDir "$LOG_DIR" \
        --FileOutPut "$OUTPUT"

    echo " Concluído: $EXP"
}

# EXPERIMENTOS

# run_exp "exp01" "64 64"
# run_exp "exp02" "64 64 64"
# run_exp "exp03" "64 64 64 64"
# run_exp "exp04" "64 64 64 64 64"

# run_exp "exp05" "128 128"
# run_exp "exp06" "128 128 128"
# run_exp "exp07" "128 128 128 128"
# run_exp "exp08" "128 128 128 128 128"

run_exp "exp09" "256 256"
run_exp "exp10" "256 256 256"
run_exp "exp11" "256 256 256 256"
run_exp "exp12" "256 256 256 256 256"

run_exp "exp13" "512 512"
run_exp "exp14" "512 512 512"
run_exp "exp15" "512 512 512 512"
run_exp "exp16" "512 512 512 512 512"

echo ""
echo "======================================================"
echo " Todos os experimentos concluídos!"
echo " Para visualizar no TensorBoard:"
echo "   tensorboard --logdir $LOGS_DIR"
echo "======================================================"