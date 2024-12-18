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
from datasets import load_dataset

import numpy as np
import copy
import warnings
warnings.filterwarnings("ignore")

import polars as pl
from attn_utils import (
    load_image, 
    aggregate_llm_attention, aggregate_vit_attention,
    heterogenous_stack,
    show_mask_on_image
)

def load_parquet(data_dir=None, return_type='pd'):
    """
    Load data from Parquet files with optional filtering on date_id, time_id, and selected columns.

    Parameters:
    - date_id_range (tuple, optional): Range of date_id to filter (start, end). Default is None, which means all dates.
    - time_id_range (tuple, optional): Range of time_id to filter (start, end). Default is None, which means all times.
    - columns (list, optional): List of columns to load. Default is None, which means all columns.
    - return_type (str, optional): Type of data to return ('pl' for Polars DataFrame or 'pd' for Pandas DataFrame). Default is 'pl'.

    Returns:
    - pl.DataFrame or pd.DataFrame: The filtered data as a Polars or Pandas DataFrame.
    """
    # data_dir = '../input/jane-street-real-time-market-data-forecasting'
    # Load data using Polars lazy loading (scan_parquet)
    # data = pl.scan_parquet(f"{data_dir}/train.parquet")
    data = pl.scan_parquet(f"{data_dir}")

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
    parser.add_argument('--model_name', type=str, required=True, choices=['MiniCPM', 'MiniCPMV2.0', 'MiniCPMV2.6', 'gpt4o', 'LLaVA-ov-0.5b', 'LLaVA-ov-7b', 'LLaVA-ov-72b-sft', 'LLaVA-ov-72b-chat', 'llava-v1.5-7b'])
    parser.add_argument('--dataset_name', type=str, choices=['ArxivQA', 'ChartQA', 'PlotQA', 'MP-DocVQA', 'SlideVQA', 'InfoVQA'], required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world_size', type=int, required=True)

    parser.add_argument('--use_positive_sample', type=int, required=True)
    parser.add_argument('--topk', type=int)
    parser.add_argument('--results_root_dir', type=str)
    
    parser.add_argument('--task_type', type=str, required=True, choices=['text', 'page_concatenation', 'weighted_selection', 'multi_image'])
    parser.add_argument('--concatenate_type', type=str, choices=['horizontal', 'vertical'])
    parser.add_argument('--ocr_type', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model_name = args.model_name

    dataset_name = args.dataset_name
    if (dataset_name == 'ArxivQA'):
        max_new_tokens = 2
    else:
        max_new_tokens = 20

    rank = args.rank
    world_size = args.world_size
    use_positive_sample = True if args.use_positive_sample else False
    if (not use_positive_sample):
        if (args.topk == None):
            raise Exception("topk is None!")
        if (args.results_root_dir == None):
            raise Exception("results_root_dir is None!")
        topk = args.topk
        results_root_dir = args.results_root_dir
        results_dir = os.path.join(results_root_dir, dataset_name)

        # load trec results
        partitions = glob.glob(os.path.join(results_dir, "test.*.trec"))
        run = {}
        for part in partitions:
            print("loading", part)
            run.update(load_from_trec(part))
    
    task_type = args.task_type
    if (task_type == 'page_concatenation'):
        if (args.concatenate_type == None):
            raise Exception("concatenate_type is None!")
        concatenate_type = args.concatenate_type
    elif (task_type == 'text'):
        if (args.ocr_type == None):
            raise Exception("ocr_type is None!")
        ocr_type = args.ocr_type
    
    input_dir = "/ssddata/liuyue/github/PruneRAG/VisRAG/qa_datasets" # Write your input path here
    input_dir_sample = "/ssddata/liuyue/github/PruneRAG/VisRAG/qa_datasets_sample"
    # if (task_type == 'text'):
    #     input_dir = os.path.join(input_dir, 'ocr', f'ocr_{ocr_type}', dataset_name)
    # else:
    #     input_dir = os.path.join(input_dir, 'image', dataset_name)

    # query_path = os.path.join(input_dir, f'{dataset_name}-eval-queries.parquet')
    # corpus_path = os.path.join(input_dir, f'{dataset_name}-eval-corpus.parquet')

    # query_path = os.path.join(input_dir, f'VisRAG-Ret-Test-{dataset_name}', 'queries')
    query_path = os.path.join(input_dir_sample, f'VisRAG-Ret-Test-{dataset_name}', 'queries')
    corpus_path = os.path.join(input_dir, f'VisRAG-Ret-Test-{dataset_name}', 'corpus')

    # build docid->content
    content = {}
    if (task_type == 'text'):
        corpus_ds = load_dataset(f"openbmb/VisRAG-Ret-Test-{dataset_name}", name="corpus", split="train")
        for i in range(len(corpus_ds)):
            corpus_id = corpus_ds[i]['corpus-id']
            text = corpus_ds[i]['text']
            content[corpus_id] = text
    else:
        # corpus_ds = load_dataset(f"openbmb/VisRAG-Ret-Test-{dataset_name}", name="corpus", split="train")
        corpus_ds = load_dataset(f"{input_dir}/VisRAG-Ret-Test-{dataset_name}", name="corpus", split="train")
        
        for i in range(len(corpus_ds)):
            corpus_id = corpus_ds[i]['corpus-id']
            image = corpus_ds[i]['image'].convert('RGB')
            content[corpus_id] = image

    #加载模型
    if (task_type == 'weighted_selection'):
        if (model_name == 'MiniCPMV2.0'):
            from openmatch.modeling.weighted_selection.MiniCPMV20.modeling_minicpmv import MiniCPMV as ModelForCausalLM_class
    else:
        if (model_name == 'gpt4o'):
            client = OpenAI(api_key=None) # Write your OpenAI API key here
        else:
            from transformers import AutoModel as Model_class
            from transformers import AutoModelForCausalLM as ModelForCausalLM_class
        
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
        model_name_or_path = '/ssddata/liuyue/github/PruneRAG/VisRAG/pretrained_models/MiniCPM-V-2_6' # Write your model path here
        model = Model_class.from_pretrained(model_name_or_path, trust_remote_code=True,
            attn_implementation='sdpa', torch_dtype=torch.bfloat16)
        model = model.eval().cuda()
        tokenizer = Tokenizer_class.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    # elif model_name == "LLaVA-ov-0.5b":
    elif 'LLaVA-ov' in model_name:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
        from llava.conversation import conv_templates, SeparatorStyle
        from accelerate import Accelerator
        accelerator = Accelerator()

        model2path = {
            'LLaVA-ov-0.5b': '/ssddata/liuyue/github/PruneRAG/VisRAG/pretrained_models/llava-onevision-qwen2-0.5b-ov',
            'LLaVA-ov-7b': '/ssddata/liuyue/github/PruneRAG/VisRAG/pretrained_models/llava-onevision-qwen2-7b-ov',
            'LLaVA-ov-72b-sft': '/ssddata/liuyue/github/PruneRAG/VisRAG/pretrained_models/llava-onevision-qwen2-72b-ov-sft',
            'LLaVA-ov-72b-chat': '/ssddata/liuyue/github/PruneRAG/VisRAG/pretrained_models/llava-onevision-qwen2-72b-ov-chat',
        }

        model_name_or_path = model2path[model_name]
        model_name0 = "llava_qwen"
        device = "cuda"
        device_map = "auto"
        # device_map = accelerator.device
        llava_model_args = {
                "multimodal": True,
            }
        overwrite_config = {}
        overwrite_config["image_aspect_ratio"] = "pad"
        llava_model_args["overwrite_config"] = overwrite_config
        tokenizer, model, image_processor, max_length = load_pretrained_model(model_name_or_path, None, model_name0, device_map=device_map, torch_dtype='bfloat16', attn_implementation='eager', **llava_model_args)

        model.eval()

    elif 'llava'  in model_name:
        import sys
        sys.path.append("/ssddata/liuyue/github/VLM-Visualizer/models")
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

        model2path = {
            'llava-v1.5-7b': 'liuhaotian/llava-v1.5-7b',
            'llava-v1.5-13b': 'liuhaotian/llava-v1.5-13b',
        }

        model_path = model2path[model_name] ## "liuhaotian/llava-v1.5-7b"
        load_8bit = False
        load_4bit = False
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name0 = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, 
            None, # model_base
            model_name0, 
            load_8bit, 
            load_4bit, 
            device=device,
            torch_dtype='bfloat16',
            attn_implementation='eager',
        )

        model.eval()

    if (model_name != 'gpt4o' and 'LLaVA-ov' not in model_name and 'llava' not in model_name):
        model.to(rank)
    
    qids, gt_doc_indexes, prefer_doc_indexes, corrects = [], [], [], []
    history_datas = []
    correct = 0
    total_num = 0
    # query_df = pd.read_parquet(query_path)
    query_df = load_parquet(query_path)
    df_attn = pd.DataFrame(columns=['qid', 'layer', 'attn_system', 'attn_image', 'attn_user',  'attn_newline', 'attn_output', 'token_system', 'token_image', 'token_user', 'token_newline', 'token_output'])
    for cnt, row in query_df.iterrows():
        if (cnt % world_size != rank):
            continue
        history_data = {}
        query = row['query']
        history_data['query'] = query
        qid = row['query-id']
        history_data['qid'] = qid
        answer = row['answer']
        history_data['original_answer'] = answer
        if (answer is None):
            raise Exception
        if (use_positive_sample):
            if (dataset_name == 'SlideVQA'):
                # due to the special format of SlideVQA, we need to split the qid to get the docid
                docid = qid.split('query_number')[0]
                docid = docid.split('tcy6')
            else:
                docid = [qid[:-1 - len(qid.split('-')[-1])]]
        else:
            # get top-k docid
            docid = []
            doc_scores = []
            doc_cnt = 0
            for key, value in sorted(run[qid].items(), key=lambda item: item[1], reverse=True):
                if (doc_cnt < topk):
                    docid.append(key)
                    doc_scores.append(value)
                    doc_cnt += 1
                else:
                    break
            if (len(docid) < topk):
                raise Exception("len(docid) < topk!")
        history_data['docid'] = docid

        ## store the ground truth docid
        gt_docid = None
        if (dataset_name == 'SlideVQA'):
            # due to the special format of SlideVQA, we need to split the qid to get the docid
            gt_docid = qid.split('query_number')[0]
            gt_docid = gt_docid.split('tcy6')
        else:
            gt_docid = [qid[:-1 - len(qid.split('-')[-1])]]
        ## get the index of the ground truth docid
        gt_doc_index = []
        for gt_docid_item in gt_docid:
            if gt_docid_item in docid:
                gt_doc_index.append(docid.index(gt_docid_item)) ## start from 0

        if (task_type == 'text'):
            if (dataset_name == 'ChartQA'):
                # get table 
                table_dir = None # Write your table path here
                csv_file_path = [os.path.join(table_dir, f"{docid_item.split('.')[0]}.csv") for docid_item in docid]
                doc_list = [get_flatten_table(csv_file_path_item) for csv_file_path_item in csv_file_path]
                doc = '\n'.join(doc_list)
                input = f"Image:{doc}\nAnswer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"
            elif (dataset_name == 'ArxivQA'):
                prompt = ''
                doc_list = [content[docid_item] for docid_item in docid]
                doc = '\n'.join(doc_list)
                options = row['options']
                options_prompt = 'Options:\n'
                # if A, B, C, D is not at the beginning
                flag = 0
                for i, option in enumerate(options):
                    if not option.startswith(f"{chr(65 + i)}"):
                        flag = 1
                        break
                if flag:
                    # pre-process
                    for i, option in enumerate(options):
                        options[i] = f"{chr(65 + i)}. {option.strip()}"
                for item in options:
                    options_prompt += f'{item}\n'
                prompt += f'Hint: {doc}\n'
                prompt += f'Question: {query}\n'
                prompt += options_prompt
                prompt += '''Answer directly with the letter of the correct option as the first character.'''
                input = prompt
            elif (dataset_name == 'PlotQA'):
                doc_list = [content[docid_item] for docid_item in docid]
                doc = '\n'.join(doc_list)
                input = f"Image:{doc}\nAnswer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"
            elif (dataset_name == 'MP-DocVQA'):
                doc_list = [content[docid_item] for docid_item in docid]
                doc = '\n'.join(doc_list)
                input = f"Image:{doc}\nAnswer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"
            elif (dataset_name == 'SlideVQA'):
                doc_list = [content[docid_item] for docid_item in docid]
                doc = '\n'.join(doc_list)
                input = f"Image:{doc}\nAnswer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"
            elif (dataset_name == 'InfoVQA'):
                doc_list = [content[docid_item] for docid_item in docid]
                doc = '\n'.join(doc_list)
                input = f"Image:{doc}\nAnswer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"
            
            history_data['prompt'] = input
            
            if (model_name == 'MiniCPM'):
                responds, history = model.chat(tokenizer, input, temperature=0.8, top_p=0.8, max_new_tokens=max_new_tokens)
            elif (model_name == 'gpt4o'):
                max_retries = 10
                retries = 0
                while retries < max_retries:
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": f"{input}"
                                        }
                                    ],
                                }
                            ],
                            max_tokens=max_new_tokens,
                        )
                        responds = response.choices[0].message.content
                        break
                    except Exception as e:
                        retries += 1
                        print(f"retry times: {retries}/{max_retries}")
                        if retries >= max_retries:
                            print("Unable to call the API, skipping this call.")
                            responds = None
                if (retries >= max_retries):
                    continue
                    
        else:
            image_list = [content[docid_item] for docid_item in docid]
            
            if (task_type == 'page_concatenation'):
                if (concatenate_type not in ['horizontal', 'vertical']):
                    raise Exception("concatenate_type error!")
                elif (concatenate_type == 'horizontal'):
                    image_list = [horizontal_concat(image_list)]
                elif (concatenate_type == 'vertical'):
                    image_list = [vertical_concat(image_list)]
                
            if (dataset_name == 'ChartQA'):
                input = [{'role': 'user', 'content': f"Answer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"}]
            elif (dataset_name == 'ArxivQA'):
                prompt = ''
                options = row['options']
                options_prompt = 'Options:\n'
                # if A, B, C, D is not at the beginning
                flag = 0
                for i, option in enumerate(options):
                    if not option.startswith(f"{chr(65 + i)}"):
                        flag = 1
                        break
                if flag:
                    # pre-process
                    for i, option in enumerate(options):
                        options[i] = f"{chr(65 + i)}. {option.strip()}"
                for item in options:
                    options_prompt += f'{item}\n'
                prompt += f'Question: {query}\n'
                prompt += options_prompt
                prompt += '''Answer directly with the letter of the correct option as the first character.'''
                input = [{'role': 'user', 'content': prompt}]
            elif (dataset_name == 'PlotQA'):
                input = [{'role': 'user', 'content': f"Answer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"}]
            elif (dataset_name == 'MP-DocVQA'):
                input = [{'role': 'user', 'content': f"Answer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"}]
            elif (dataset_name == 'SlideVQA'):
                input = [{'role': 'user', 'content': f"Answer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"}]
            elif (dataset_name == 'InfoVQA'):
                input = [{'role': 'user', 'content': f"Answer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"}]
            
            history_data['prompt'] = input[0]['content']

            if (task_type == 'page_concatenation'):
                if (model_name == 'MiniCPMV2.0'):
                    responds, context, _ = model.chat(
                        image=image_list[0], # image_list only has one element
                        msgs=input,
                        context=None,
                        tokenizer=tokenizer,
                        sampling=False,
                        max_new_tokens=max_new_tokens
                    )
                elif 'llava' in model_name:
                    if "llama-2" in model_name.lower():
                        conv_mode = "llava_llama_2"
                    elif "mistral" in model_name.lower():
                        conv_mode = "mistral_instruct"
                    elif "v1.6-34b" in model_name.lower():
                        conv_mode = "chatml_direct"
                    elif "v1" in model_name.lower():
                        conv_mode = "llava_v1"
                    elif "mpt" in model_name.lower():
                        conv_mode = "mpt"
                    else:
                        conv_mode = "llava_v0"

                    print(f'model_name:\t{model_name}\t conv_mode:\t{conv_mode}')

                    conv = conv_templates[conv_mode].copy()
                    if "mpt" in model_name.lower():
                        roles = ('user', 'assistant')
                    else:
                        roles = conv.roles

                    conv = conv_templates[conv_mode].copy()
                    if "mpt" in model_name.lower():
                        roles = ('user', 'assistant')
                    else:
                        roles = conv.roles

                    image_tensor, images = process_images(image_list, image_processor, model.config)
                    image = images[0]
                    image_size = image.size
                    if type(image_tensor) is list:
                        image_tensor = [image.to(model.device, dtype=torch.bfloat16) for image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

                    input_prompt = input[0]['content'].replace('Answer:', '')
                    if model.config.mm_use_im_start_end:
                        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + input_prompt
                    else:
                        inp = DEFAULT_IMAGE_TOKEN + '\n' + input_prompt

                    conv.append_message(conv.roles[0], inp)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
                    with torch.inference_mode():
                        outputs = model.generate(
                            input_ids,
                            images=image_tensor,
                            image_sizes=[image_size],
                            do_sample=False,
                            max_new_tokens=512,
                            use_cache=True,
                            return_dict_in_generate=True,
                            output_attentions=True,
                        )

                    responds = tokenizer.decode(outputs["sequences"][0]).strip().replace('</s>','')
                    
            elif (task_type == 'multi_image'):
                if (model_name == 'MiniCPMV2.6'):
                    input = [{'role': 'user', 'content': image_list + [input[0]['content']]}]
                    responds = model.chat(
                        image=None,
                        msgs=input,
                        tokenizer=tokenizer,
                        sampling=False,
                        max_new_tokens=max_new_tokens
                    )
                
                # elif model_name == "LLaVA-ov-0.5b":
                elif 'LLaVA-ov' in model_name:
                    # input = [{'role': 'user', 'content': image_list + [input[0]['content']]}]
                    image_tensors = process_images(image_list, image_processor, model.config)
                    image_tensors = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensors]
                    conv_template = "qwen_1_5"

                    n_images = len(image_tensors)
                    question = ''
                    for iidx in range(n_images):
                        question += f'{DEFAULT_IMAGE_TOKEN}\n'
                    question += input[0]['content']

                    conv = copy.deepcopy(conv_templates[conv_template])
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()

                    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                    image_sizes = [image.size for image in image_list]

                    outputs = model.generate(
                        input_ids,
                        images=image_tensors,
                        image_sizes=image_sizes,
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=max_new_tokens,
                        return_dict_in_generate=True,
                        output_attentions=True,
                    )
                    text_outputs = tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=True)
                    # print(text_outputs[0])
                    responds = text_outputs[0]

                    input_embeds, embed_sizes = model.prepare_inputs_labels_for_multimodal(input_ids, None, None, None, None, image_tensors, ["image"], image_sizes=image_sizes, return_embed_size=True)
                    assert sum(embed_sizes[0]) == input_embeds.shape[1]
                    embed_sizes_cumsum = np.cumsum(embed_sizes[0])
                    system_idx = torch.arange(0, embed_sizes_cumsum[0]).int()
                    image_idx = torch.tensor([]).int()
                    newline_idx = torch.tensor([]).int()
                    for iidx in range(len(image_tensors)):
                        image_idx = torch.cat([image_idx, torch.arange(embed_sizes_cumsum[iidx*2], embed_sizes_cumsum[iidx*2+1]).int()])
                        if iidx < len(image_tensors)-1:
                            newline_idx = torch.cat([newline_idx, torch.arange(embed_sizes_cumsum[iidx*2+1], embed_sizes_cumsum[iidx*2+2]).int()]) ## there is a '\n' between two image tokens
                    user_idx = torch.arange(embed_sizes_cumsum[-2], embed_sizes_cumsum[-1]).int()

                    nlayers = len(outputs["attentions"][0])
                    nouttokens = len(outputs["sequences"][0])

                    

                    for i, layer in enumerate(outputs["attentions"][0]): ## 第一维=output token, 第二维=LLM layer id
                        newline = [system_idx.shape[0], image_idx.shape[0], user_idx.shape[0], newline_idx.shape[0], len(outputs['sequences'][0])]
                        layeri = []
                        for j in range(len(outputs['attentions'])):
                            layeri.append(outputs['attentions'][j][i])
                        
                        llm_attn_matrix = heterogenous_stack(
                            list(map(aggregate_llm_attention, layeri))
                        )

                        a_system, a_image, a_user, a_output, a_newline = 0.0, 0.0, 0.0, 0.0, 0.0

                        n_output_tokens = len(outputs['sequences'][0])
                        output_token_attns = llm_attn_matrix

                        a_system = output_token_attns[:, system_idx].sum() ## sum of attention on system tokens
                        a_image = output_token_attns[:, image_idx].sum()
                        a_user = output_token_attns[:, user_idx].sum()
                        a_newline = output_token_attns[:, newline_idx].sum()
                        a_output = output_token_attns[:, -n_output_tokens:].sum()
                        
                        newline = [a_system, a_image, a_user, a_newline, a_output] + newline
                        newline = [qid, i] + newline

                        df_attn.loc[len(df_attn)] = newline

                    del outputs
                    del image_tensors

                elif (model_name == 'gpt4o'):
                    max_retries = 10
                    retries = 0
                    while retries < max_retries:
                        try:
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": f"{input[0]['content']}"
                                        }
                                    ],
                                }
                            ]
                            for base64_string_item in images_to_base64_list(image_list):
                                messages[0]["content"].append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{base64_string_item}"}
                                })

                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=messages,
                                max_tokens=max_new_tokens,
                            )
                            responds = response.choices[0].message.content
                            break
                        except Exception as e:
                            retries += 1
                            print(f"retry times: {retries}/{max_retries}")
                            if retries >= max_retries:
                                print("Unable to call the API, skipping this call.")
                                responds = None
                    if (retries >= max_retries):
                        continue
            elif (task_type == 'weighted_selection'):
                if (model_name == 'MiniCPMV2.0'):
                    responds = model.replug_chat(
                    image_list=image_list,
                    msgs=input,
                    doc_scores=doc_scores,
                    context=None,
                    tokenizer=tokenizer,
                    sampling=False,
                    max_new_tokens=max_new_tokens
                    )
            
        total_num += 1
        
        responds_backup = responds
            
        #pre-process
        correct_sig = False
        if (dataset_name == 'ChartQA'):
            responds = preprocess_text(responds)
            answer = preprocess_text(answer)
            if ('%' in responds and '%' not in answer):
                responds = responds.replace('%', '')
            if ('%' not in responds and '%' in answer):
                answer = answer.replace('%', '')
            print(f"query: {query}")
            print(f"responds:{responds}")
            print(f"answer:{answer}")
            print(f'gt_doc_index:{gt_doc_index}')
            print('---------------')
            if (responds == answer):
                correct += 1
                correct_sig = True
            elif(is_numeric_data(responds) and is_numeric_data(answer) and answer != '0' and is_within_5_percent(responds, answer)):
                correct += 1
                correct_sig = True
        elif (dataset_name == 'ArxivQA'):
            responds = responds[0].upper()
            answer = answer[0].upper()
            print(f"query: {query}")
            print(f"responds:{responds}")
            print(f"answer:{answer}")
            print(f'gt_doc_index:{gt_doc_index}')
            print('---------------')
            if (responds == answer):
                correct += 1
                correct_sig = True
        elif (dataset_name == 'PlotQA'):
            responds = preprocess_text(responds)
            is_str = 1
            if (type(answer) != str):
                is_str = 0
                answer = str(answer)
            answer = preprocess_text(answer)
            if ('%' in responds and '%' not in answer):
                responds = responds.replace('%', '')
            if ('%' not in responds and '%' in answer):
                answer = answer.replace('%', '')
            print(f"query: {query}")
            print(f"responds:{responds}")
            print(f"answer:{answer}")
            print(f'gt_doc_index:{gt_doc_index}')
            print('---------------')
            if (responds == answer):
                correct += 1
                correct_sig = True
            elif(is_numeric_data(responds) and (not is_str) and float(answer) != 0.0 and is_within_5_percent(responds, answer)):
                correct += 1
                correct_sig = True
        elif (dataset_name == 'MP-DocVQA'):
            responds = preprocess_text(responds)
            if (not isinstance(answer, list)):
                # answer = [answer]
                answer = answer.tolist()
            for i, answer_item in enumerate(answer):
                answer[i] = preprocess_text(answer_item)
            if ('%' in responds and '%' not in answer[0]):
                responds = responds.replace('%', '')
            if ('%' not in responds and '%' in answer[0]):
                answer = [answer_item.replace('%', '') for answer_item in answer]
            print(f"query: {query}")
            print(f"responds:{responds}")
            print(f"answer:{answer}")
            print(f'gt_doc_index:{gt_doc_index}')
            print('---------------')
            for answer_item in answer:
                if (responds == answer_item):
                    correct += 1
                    correct_sig = True
                    break
        elif (dataset_name == 'SlideVQA'):
            responds = preprocess_text(responds)
            answer = preprocess_text(answer)
            if ('%' in responds and '%' not in answer):
                responds = responds.replace('%', '')
            if ('%' not in responds and '%' in answer):
                answer = answer.replace('%', '')
            print(f"query: {query}")
            print(f"responds:{responds}")
            print(f"answer:{answer}")
            print(f'gt_doc_index:{gt_doc_index}')
            print('---------------')
            if (responds == answer):
                correct += 1
                correct_sig = True
        elif (dataset_name == 'InfoVQA'):
            responds = preprocess_text(responds)
            if (not isinstance(answer, list)):
                # answer = [answer]
                answer = answer.tolist()
            for i, answer_item in enumerate(answer):
                answer[i] = preprocess_text(answer_item)
            if ('%' in responds and '%' not in answer[0]):
                responds = responds.replace('%', '')
            if ('%' not in responds and '%' in answer[0]):
                answer = [answer_item.replace('%', '') for answer_item in answer]
            print(f"query: {query}")
            print(f"responds:{responds}")
            print(f"answer:{answer}")
            print(f'gt_doc_index:{gt_doc_index}')
            print('---------------')
            for answer_item in answer:
                if (responds == answer_item):
                    correct += 1
                    correct_sig = True
                    break
        
        history_data['preprocessed_responds'] = responds
        history_data['preprocessed_answer'] = answer
        history_data['original_responds'] = responds_backup
        
        
        # calculate accuracy
        if (dataset_name == 'ChartQA'):     
            print(f"{dataset_name}:{total_num}_Accuracy:{float(correct) / total_num}, {correct_sig}")
        elif (dataset_name == 'ArxivQA'):
            print(f"{dataset_name}:{total_num}_Accuracy:{float(correct) / total_num}, {correct_sig}")
        elif (dataset_name == 'PlotQA'):
            print(f"{dataset_name}:{total_num}_Accuracy:{float(correct) / total_num}, {correct_sig}")
        elif (dataset_name == 'MP-DocVQA'):
            print(f"{dataset_name}:{total_num}_Accuracy:{float(correct) / total_num}, {correct_sig}")
        elif (dataset_name == "SlideVQA"):
            print(f"{dataset_name}:{total_num}_Accuracy:{float(correct) / total_num}, {correct_sig}")
        elif (dataset_name == 'InfoVQA'):
            print(f"{dataset_name}:{total_num}_Accuracy:{float(correct) / total_num}, {correct_sig}")

        for k,v in history_data.items():
            if type(v) == np.ndarray:
                history_data[k] = v.tolist()
        history_datas.append(json.dumps(history_data))

        qids.append(qid)
        gt_doc_indexes.append(gt_doc_index)
        corrects.append(correct_sig)
                
    output_dir = "/ssddata/liuyue/github/PruneRAG/VisRAG/data/checkpoints/generator/layer_attn" # Write your output path here

    

    prefix = model_name
    output_dir = os.path.join(output_dir, prefix)
    prefix += '_'
    if (use_positive_sample):
        output_dir = os.path.join(output_dir, 'upper_bound')
        prefix += 'upper_bound'
    else:
        output_dir = os.path.join(output_dir, os.path.basename(results_root_dir))
        prefix += str(os.path.basename(results_root_dir))

    if (use_positive_sample):
        prefix = f"{prefix}_{dataset_name}_oracle"
    else:
        prefix = f"{prefix}_{dataset_name}_{task_type}_top{topk}"

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{prefix}_result.jsonl")
    print(f"writing to {result_path}")
    with open(result_path, 'w') as file:
        acc = float(correct) / total_num
        if (dataset_name == 'ArxivQA'):
            data = {'Accuracy':acc}
        elif (dataset_name == 'PlotQA'):
            data = {'Accuracy':acc}
        elif (dataset_name == 'ChartQA'):
            data = {'Accuracy':acc}
        elif (dataset_name == 'SlideVQA'):
            data = {'Accuracy':acc}
        elif (dataset_name == 'MP-DocVQA'):
            data = {'Accuracy':acc}
        elif (dataset_name == 'InfoVQA'):
            data = {'Accuracy':acc}
        file.write(json.dumps(data)+'\n')
        
    history_path = os.path.join(output_dir, f"{prefix}_history.jsonl")
    print(f"writing to {history_path}")
    with open(history_path, 'w') as file:
        for history_data in history_datas:
            file.write(history_data + '\n')
   
    df_attn.to_csv(os.path.join(output_dir, f"{prefix}_attn.csv"), index=False)
    
if __name__ == '__main__':

    main()