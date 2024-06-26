#!/bin/bash

for index in {0..59}; do
    start_index=$((index + 0))
    end_index=$((start_index + 1))
    while true; do
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7," accelerate launch utils_data/lowlevel_prompt_train.py --start_gpu $start_index --end_gpu $end_index
        exit_status=$?
        if [ $exit_status -eq 0 ]; then
            echo "程序已自然终止, 停止脚本执行。start_gpu=$start_index, end_gpu=$end_index"
            break
        else
            echo "程序中途中止，重新启动... start_gpu=$start_index, end_gpu=$end_index"
        fi
    done
done