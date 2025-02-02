#!/usr/bin/bash

set -e
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATA_DIR=/home/daizhirui/Data/erl_neural_sddf/datasets
OUTPUT_DIR=/home/daizhirui/results/erl_neural_sddf/RaDe-GS-depth-only
ITERATION=30000
COMMON_ARGS="--skip_train --iteration $ITERATION -r 1"

for dataset in $(ls $DATA_DIR); do
    if [[ $dataset != *"rgbd" ]]; then
            continue
    fi
    echo $dataset
    if [[ $dataset != "replica-office-4-rgbd" ]]; then
        continue
    fi
    mkdir -p $OUTPUT_DIR/$dataset/test
    python3 $SCRIPT_DIR/render.py -s $DATA_DIR/$dataset \
        -m $OUTPUT_DIR/$dataset \
        $COMMON_ARGS | tee $OUTPUT_DIR/$dataset/test/log-$ITERATION.txt
done
