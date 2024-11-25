
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# https://stackoverflow.com/a/1894296/28353583
import ast

if __name__ == '__main__':

    model = f'LLaVA-ov-7b'
    dataset = f'SlideVQA'

    topks = [1, 3, 5, 7]

    # fig, axes = plt.subplots(1, len(topks), figsize=(5*len(topks), 5), dpi=300)
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.gca()

    data_for_hist = {}
    for tidx, topk in enumerate(topks):
        # ax = axes[tidx]

        path = f'/ssddata/liuyue/github/VisRAG/data/checkpoints/generator/which_important_by_attn/{model}/eval-2024-11-06-174337-maxq-512-maxp-2048-bsz-16-pooling-wmean-attention-causal-gpus-per-node-8/{model}_eval-2024-11-06-174337-maxq-512-maxp-2048-bsz-16-pooling-wmean-attention-causal-gpus-per-node-8_{dataset}_multi_image_top{topk}_attn.csv'

        df = pd.read_csv(path) ## query-id,gt_doc_index,prefer_doc_index,correct
        df_out = pd.DataFrame(df, columns=list(df.columns) + ['label'])

        total = len(df)
        
        prefer_doc_index = df['prefer_doc_index'].tolist()
        data_for_hist[topk] = prefer_doc_index
    
    sns.histplot(data=data_for_hist, ax=ax)
    fig.savefig(f'/ssddata/liuyue/github/VisRAG/data/checkpoints/generator/which_important_by_attn/{model}/eval-2024-11-06-174337-maxq-512-maxp-2048-bsz-16-pooling-wmean-attention-causal-gpus-per-node-8/attn.png', bbox_inches='tight', pad_inches=0.1)


