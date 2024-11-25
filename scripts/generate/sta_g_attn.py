
import os
import pandas as pd

# https://stackoverflow.com/a/1894296/28353583
import ast

if __name__ == '__main__':

    model = f'LLaVA-ov-7b'
    dataset = f'SlideVQA'
    topk = 7

    path = f'/ssddata/liuyue/github/VisRAG/data/checkpoints/generator/which_important_by_attn/{model}/eval-2024-11-06-174337-maxq-512-maxp-2048-bsz-16-pooling-wmean-attention-causal-gpus-per-node-8/{model}_eval-2024-11-06-174337-maxq-512-maxp-2048-bsz-16-pooling-wmean-attention-causal-gpus-per-node-8_{dataset}_multi_image_top{topk}_attn.csv'

    df = pd.read_csv(path) ## query-id,gt_doc_index,prefer_doc_index,correct
    df_out = pd.DataFrame(df, columns=list(df.columns) + ['label'])

    total = len(df)
    T_attn_T_ret_T_ans = 0.0 ## 1
    T_attn_F_ret_T_ans = 0.0 ## 2
    T_attn_T_ret_F_ans = 0.0 ## 3
    T_attn_F_ret_F_ans = 0.0 ## 4
    F_attn_T_ret_T_ans = 0.0 ## 5
    F_attn_F_ret_T_ans = 0.0 ## 6
    F_attn_T_ret_F_ans = 0.0 ## 7
    F_attn_F_ret_F_ans = 0.0 ## 8
    

    for i in range(total):
        row = df.loc[i,:]
        gt_doc_index = ast.literal_eval(row['gt_doc_index'])
        if row['correct'] == 1 and row['prefer_doc_index'] in gt_doc_index:
            T_attn_T_ret_T_ans += 1
            df_out.loc[i, 'label'] = 1
        elif row['correct'] == 1 and row['prefer_doc_index'] not in gt_doc_index and len(gt_doc_index) > 0:
            F_attn_T_ret_T_ans += 1
            df_out.loc[i, 'label'] = 5
        elif row['correct'] == 1 and row['prefer_doc_index'] not in gt_doc_index and len(gt_doc_index) == 0:
            F_attn_F_ret_T_ans += 1
            df_out.loc[i, 'label'] = 6

        elif row['correct'] == 0 and row['prefer_doc_index'] not in gt_doc_index and len(gt_doc_index) > 0:
            F_attn_T_ret_F_ans += 1
            df_out.loc[i, 'label'] = 7
        elif row['correct'] == 0 and row['prefer_doc_index'] not in gt_doc_index and len(gt_doc_index) == 0:
            F_attn_F_ret_F_ans += 1
            df_out.loc[i, 'label'] = 8
        elif row['correct'] == 0 and row['prefer_doc_index'] in gt_doc_index: 
            T_attn_T_ret_F_ans += 1
            df_out.loc[i, 'label'] = 3
        else:
            print(row)
            print(row['correct'] == 0, row['prefer_doc_index'] not in gt_doc_index, len(gt_doc_index))
            raise ValueError('Error')

    df_show = pd.DataFrame(data={
        'TG,TR': [T_attn_T_ret_T_ans/total, F_attn_T_ret_T_ans/total],
        'TG,FR': [T_attn_F_ret_F_ans/total, F_attn_F_ret_F_ans/total],
        'FG,TR': [T_attn_T_ret_F_ans/total, F_attn_T_ret_F_ans/total],
        'FG,FR': [T_attn_F_ret_F_ans/total, F_attn_F_ret_F_ans/total]
    }, index=['T_attn', 'F_attn'])

    print(df_show)

    df_out.to_csv(path.replace('.csv', '_label.csv'), index=False)