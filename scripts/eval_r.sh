#!/bin/bash

cd ..

[[ -d logs ]] || mkdir logs

bash scripts/eval_retriever/eval.sh 512 2048 16 8 wmean causal ArxivQA,ChartQA,MP-DocVQA,InfoVQA,PlotQA,SlideVQA /ssddata/liuyue/github/VisRAG/pretrained_models/VisRAG-Ret 2>&1 | tee logs/eval_r.log

cd -