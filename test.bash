#!/usr/bin/bash

set -e
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATA_DIR=/home/daizhirui/Data/erl_neural_sddf/datasets
OUTPUT_DIR=/home/daizhirui/results/erl_neural_sddf/baselines/RaDe-GS
COMMON_ARGS="--skip_train"

for dataset in $(ls $DATA_DIR); do
    if [[ $dataset != *"rgbd" ]]; then
            continue
    fi
    echo $dataset

    python3 $SCRIPT_DIR/render.py -s $DATA_DIR/$dataset \
        -m $OUTPUT_DIR/$dataset \
        $COMMON_ARGS

done
