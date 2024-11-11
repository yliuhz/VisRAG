#!/bin/bash


cd ..

[[ -d logs ]] || mkdir logs

bash scripts/train_retriever/train.sh 2048 16 8 0.02 1 true false config/deepspeed.json 1e-5 false wmean causal 1 true 2 false pretrained_models/MiniCPM-V-2 openbmb/VisRAG-Ret-Train-In-domain-data 2>&1 | tee logs/train.log

cd -