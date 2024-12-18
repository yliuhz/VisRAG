#!/bin/bash

cd ..

# MODEL=LLaVA-ov-72b-chat
MODEL=$1
# MODEL=MiniCPMV2.0

[[ -d logs/generate/sample/layer_attn/${MODEL} ]] || mkdir -p logs/generate/sample/layer_attn/${MODEL}

# for MODEL in LLaVA-ov-0.5b LLaVA-ov-7b LLaVA-ov-72b-sft LLaVA-ov-72b-chat
# do
# for DATA in InfoVQA ChartQA ArxivQA MP-DocVQA PlotQA SlideVQA
for DATA in InfoVQA MP-DocVQA
do
for ORACLE in 0
do
for TOPK in 1 3 5 7 9
do
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -u scripts/generate/generate_layer_attn.py \
        --model_name $MODEL \
        --dataset_name $DATA \
        --rank 0 \
        --world_size 1 \
        --use_positive_sample $ORACLE \
        --topk $TOPK \
        --results_root_dir ./data/checkpoints/eval-2024-11-06-174337-maxq-512-maxp-2048-bsz-16-pooling-wmean-attention-causal-gpus-per-node-8 \
        --task_type multi_image \
        --concatenate_type horizontal \
        --ocr_type '' \
        2>&1 | tee logs/generate/sample/layer_attn/${MODEL}/eval_g_${DATA}_${ORACLE}_${TOPK}.log
done
done
done
# done

cd -