#!/bin/bash

cd ..

[[ -d logs ]] || mkdir logs

# bash scripts/eval_retriever/eval_on_exist.sh 512 2048 16 8 wmean causal ArxivQA,ChartQA,MP-DocVQA,InfoVQA,PlotQA,SlideVQA /ssddata/liuyue/github/VisRAG/pretrained_models/VisRAG-Ret 2>&1 | tee logs/eval_r_on_exist.log

bash scripts/eval_retriever/eval_on_exist.sh 512 2048 16 8 wmean causal MP-DocVQA /ssddata/liuyue/github/VisRAG/pretrained_models/VisRAG-Ret 2>&1 | tee logs/eval_r_on_exist.log

cd -