import json
import os
import argparse
import glob
from openmatch.utils import load_from_trec
import torch
from PIL import Image
import base64
from io import BytesIO
from transformers import AutoTokenizer as Tokenizer_class
from openmatch.generation_utils import get_flatten_table, preprocess_text, is_numeric_data, is_within_5_percent, horizontal_concat, vertical_concat
from openai import OpenAI
import pandas as pd
import polars as pl
import numpy as np

def load_parquet(root=None, return_type='pd'):
    """
    Load data from Parquet files with optional filtering on date_id, time_id, and selected columns.

    Parameters:
    - date_id_range (tuple, optional): Range of date_id to filter (start, end). Default is None, which means all dates.
    - time_id_range (tuple, optional): Range of time_id to filter (start, end). Default is None, which means all times.
    - columns (list, optional): List of columns to load. Default is None, which means all columns.
    - return_type (str, optional): Type of data to return ('pl' for Polars DataFrame or 'pd' for Pandas DataFrame). Default is 'pl'.

    Returns:a
    - pl.DataFrame or pd.DataFrame: The filtered data as a Polars or Pandas DataFrame.
    """
    # Load data using Polars lazy loading (scan_parquet)
    data = pl.scan_parquet(f"{root}")

    # Collect the data to execute the lazy operations
    if return_type == 'pd':
        return data.collect().to_pandas()
    else:
        return data.collect()


def images_to_base64_list(image_list):
    base64_list = []
    for img in image_list:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        base64_list.append(img_base64)
    return base64_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True) # option: MiniCPM, MiniCPMV2.0, MiniCPMV2.6, gpt4o
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world_size', type=int, required=True)

    parser.add_argument('--use_positive_sample', type=int, required=True)
    parser.add_argument('--topk', type=int)
    parser.add_argument('--results_root_dir', type=str)
    
    parser.add_argument('--task_type', type=str, required=True) # option: text, page_concatenation, weighted_selection, multi_image
    parser.add_argument('--concatenate_type', type=str) # option: horizontal, vertical
    parser.add_argument('--ocr_type', type=str)
    args = parser.parse_args()
    return args

def main():
    # args = parse_args()
    model_name = "MiniCPMV2.0"
    task_type = "page_concatenation"

    image_path = f"examples/poster2.jpg"
    query = "Who was the youngest player in the 2015 ICC Cricket World Cup?"
    max_new_tokens = 20 ## 最多回答多少token?

    rank = 0
        
    #加载模型
    if (task_type == 'weighted_selection'):
        if (model_name == 'MiniCPMV2.0'):
            from openmatch.modeling.weighted_selection.MiniCPMV20.modeling_minicpmv import \
                MiniCPMV as ModelForCausalLM_class
    else:
        if (model_name == 'gpt4o'):
            client = OpenAI(api_key=None)  # Write your OpenAI API key here
        else:
            from transformers import AutoModel as Model_class
            from transformers import AutoModelForCausalLM as ModelForCausalLM_class

            print(f'I am here')
        
    if (model_name == 'MiniCPM'):
        model_name_or_path = None # Write your model path here
        tokenizer = Tokenizer_class.from_pretrained(model_name_or_path)
        model = ModelForCausalLM_class.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

    elif (model_name == 'MiniCPMV2.0'):
        model_name_or_path = "/ssddata/liuyue/github/PruneRAG/VisRAG/pretrained_models/MiniCPM-V-2" # Write your model path here
        tokenizer = Tokenizer_class.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = ModelForCausalLM_class.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        model = model.to(device='cuda', dtype=torch.bfloat16)
        model.eval()

    elif (model_name == 'MiniCPMV2.6'):
        model_name_or_path = "/ssddata/liuyue/github/PruneRAG/VisRAG/pretrained_models/MiniCPM-V-2_6" # Write your model path here
        model = Model_class.from_pretrained(model_name_or_path, trust_remote_code=True,
            attn_implementation='sdpa', torch_dtype=torch.bfloat16)
        model = model.eval().cuda()
        tokenizer = Tokenizer_class.from_pretrained(model_name_or_path, trust_remote_code=True)

    if (model_name != 'gpt4o'):
        model.to(rank)


    ### process input
    image_list = [Image.open(image_path).convert('RGB')]
    print(image_list[0])

    if (model_name == 'MiniCPMV2.0'):
        input = [{'role': 'user', 'content': f"Answer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"}]
    elif (model_name == 'MiniCPMV2.6'):
        input = [{'role': 'user', 'content': image_list + [input[0]['content']]}]



    ### chat with MLLM
    if (model_name == 'MiniCPMV2.0'):
        responds, context, _ = model.chat(
                        image=image_list[0], # image_list only has one element
                        msgs=input,
                        context=None,
                        tokenizer=tokenizer,
                        sampling=False,
                        max_new_tokens=max_new_tokens
                    )

    elif (model_name == 'MiniCPMV2.6'):
                    input = [{'role': 'user', 'content': image_list + [input[0]['content']]}]
                    responds = model.chat(
                        image=None,
                        msgs=input,
                        tokenizer=tokenizer,
                        sampling=False,
                        max_new_tokens=max_new_tokens
                    )
    

    print(responds)
    # print(type(model).__name__, model.__class__.__name__)
if __name__ == '__main__':

    main()