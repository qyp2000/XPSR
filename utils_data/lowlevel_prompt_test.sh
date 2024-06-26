#!/bin/bash

while true; do
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7," accelerate launch utils_data/lowlevel_prompt_test.py
    exit_status=$?
    if [ $exit_status -eq 0 ]; then
        echo "程序已自然终止，停止脚本执行。"
        break
    else
        echo "程序中途中止，重新启动..."
    fi
done