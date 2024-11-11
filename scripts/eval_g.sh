#!/bin/bash

cd ..

# MODEL=LLaVA-ov-72b-chat
MODEL=$1
# MODEL=MiniCPMV2.0

[[ -d logs/generate/$MODEL ]] || mkdir -p logs/generate/$MODEL

# for MODEL in LLaVA-ov-0.5b LLaVA-ov-7b LLaVA-ov-72b-sft LLaVA-ov-72b-chat
# do
for DATA in InfoVQA SlideVQA ArxivQA ChartQA MP-DocVQA PlotQA
# for DATA in InfoVQA
do
for ORACLE in 1
do

    python -u scripts/generate/generate.py \
        --model_name $MODEL \
        --dataset_name $DATA \
        --rank 0 \
        --world_size 1 \
        --use_positive_sample $ORACLE \
        --topk 3 \
        --results_root_dir ./data/checkpoints/eval-2024-11-06-174337-maxq-512-maxp-2048-bsz-16-pooling-wmean-attention-causal-gpus-per-node-8 \
        --task_type multi_image \
        --concatenate_type horizontal \
        --ocr_type '' \
        2>&1 | tee logs/generate/$MODEL/eval_g_${DATA}_${ORACLE}.log

done
done
# done

cd -