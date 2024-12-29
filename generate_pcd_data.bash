#!/usr/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

set -e
set -x

DATA_DIR=/home/daizhirui/Data/erl_neural_sddf/datasets
for dataset in $(ls $DATA_DIR); do
    if [[ $dataset != *"rgbd" ]]; then
        continue
    fi
    output_file=$DATA_DIR/$dataset/points3d.ply
    if [ -f $output_file ]; then
        continue
    fi
    pcd_path=$DATA_DIR/$dataset/scans/train/pcd
    num_files=$(ls $pcd_path | wc -l)
    downsample_ratio=0.05
    if [[ $dataset == "gibson-allensville-rgbd" ]]; then
        downsample_ratio=0.03
    fi
    python3 $SCRIPT_DIR/merge_point_cloud.py $pcd_path $num_files --output_file $output_file --downsample_ratio $downsample_ratio
done
