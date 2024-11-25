
### OCR image to text
### LLM extract entities
### Match text with entities

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np
import fastdeploy as fd
from PIL import Image
import io
from io import BytesIO
import polars as pl
import pandas as pd

from prompts.prompt import SYSTEM_PROMPT, USER_PROMPT
from openai import OpenAI
import re
from json_repair import repair_json
from tqdm import tqdm

import logging

# logging.basicConfig(
#     filename='a.log',
#     level=logging.DEBUG,
#     format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
#     # datefmt='%Y-%m-%d %H:%M:%S',
#     handlers=[
#         logging.StreamHandler(),
#     ]
# )
logger = logging.getLogger(__file__)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)
logger.propagate = False



# ----------------------- Hardcoded Constants -----------------------

# Model file paths
# DET_MODEL_DIR='/mnt/data/user/tc_agi/user/xubokai/mmtrain/document-convert-models-en/en_PP-OCRv3_det_infer'
# REC_MODEL_DIR='/mnt/data/user/tc_agi/user/xubokai/mmtrain/document-convert-models-en/en_PP-OCRv3_rec_infer'
# CLS_MODEL_DIR='/mnt/data/user/tc_agi/user/xubokai/mmtrain/document-convert-models-en/ch_ppocr_mobile_v2.0_cls_infer'
# REC_LABEL_FILE='/mnt/data/user/tc_agi/user/xubokai/mmtrain/document-convert-models-en/en_PP-OCRv3_rec_infer/en_dict.txt'

DET_MODEL_DIR='/ssddata/liuyue/github/VisRAG/scripts/demo/ocr_pipeline/document-convert-models-en/en_PP-OCRv3_det_infer'
REC_MODEL_DIR='/ssddata/liuyue/github/VisRAG/scripts/demo/ocr_pipeline/document-convert-models-en/en_PP-OCRv3_rec_infer'
CLS_MODEL_DIR='/ssddata/liuyue/github/VisRAG/scripts/demo/ocr_pipeline/document-convert-models-en/ch_ppocr_mobile_v2.0_cls_infer'
REC_LABEL_FILE='/ssddata/liuyue/github/VisRAG/scripts/demo/ocr_pipeline/document-convert-models-en/en_PP-OCRv3_rec_infer/en_dict.txt'

# Input image path
# IMAGE_PATH = "./examples/form.png"  # Write your image path here

# Inference device configuration
BACKEND = "gpu"  # Options: "gpu" or "cpu"
DEVICE_ID = 0     # Set GPU device ID if using GPU

# Other parameters
MIN_SCORE = 0.6  # Recognition score threshold

# ----------------------- Function Definitions -----------------------

def decode_image(image_path):
    """
    Decode the image from the given path.
    """
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def decode_byte_image(bytes):
    image_data = BytesIO(bytes)
    image = Image.open(image_data).convert('RGB')

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def calculate_spaces_and_newlines(current_box, previous_box, space_threshold=45, line_threshold=15):
    """Calculate the number of spaces and newlines between two text boxes."""
    spaces = 0
    newlines = 0
    
    # Check if the text boxes are on the same line
    if abs(current_box[1] - previous_box[1]) < line_threshold:
        spaces = max(1, int(abs(current_box[0] - previous_box[0]) / space_threshold))
    else:
        newlines = max(1, int(abs(current_box[1] - previous_box[1]) / line_threshold))
    
    return spaces, newlines

def tostr_layout_preserving(result):
    """Convert OCR results into a layout-preserving merged string."""
    text_boxes = []
    for box, text, score in zip(result.boxes, result.text, result.rec_scores):
        if score >= MIN_SCORE:  # Only include text boxes with score >= 0.6
            coords = [(box[i], box[i + 1]) for i in range(0, len(box), 2)]
            center_x = (coords[0][0] + coords[2][0]) / 2
            center_y = (coords[0][1] + coords[2][1]) / 2
            text_boxes.append((center_x, center_y, text, coords))

    # Sort text boxes from top to bottom and left to right
    text_boxes = sorted(text_boxes, key=lambda x: (x[1], x[0]))
    
    # Merge text boxes
    merged_text = []
    previous_box = None
    for box in text_boxes:
        if previous_box is not None:
            spaces, newlines = calculate_spaces_and_newlines(box, previous_box)
            merged_text.append('\n' * newlines + ' ' * spaces)
        merged_text.append(box[2])
        previous_box = box

    res_text = ''.join(merged_text)
    return res_text

def build_option():
    """Build FastDeploy runtime options based on backend and device."""
    det_option = fd.RuntimeOption()
    cls_option = fd.RuntimeOption()
    rec_option = fd.RuntimeOption()

    if BACKEND.lower() == "gpu":
        det_option.use_gpu(DEVICE_ID)
        cls_option.use_gpu(DEVICE_ID)
        rec_option.use_gpu(DEVICE_ID)
    else:
        det_option.use_cpu()
        cls_option.use_cpu()
        rec_option.use_cpu()

    return det_option, cls_option, rec_option

# ----------------------- Main Function -----------------------

### transform image bytes into texts
def ocr(bytes):
    # Build model file paths
    det_model_file = os.path.join(DET_MODEL_DIR, "inference.pdmodel")
    det_params_file = os.path.join(DET_MODEL_DIR, "inference.pdiparams")

    cls_model_file = os.path.join(CLS_MODEL_DIR, "inference.pdmodel")
    cls_params_file = os.path.join(CLS_MODEL_DIR, "inference.pdiparams")

    rec_model_file = os.path.join(REC_MODEL_DIR, "inference.pdmodel")
    rec_params_file = os.path.join(REC_MODEL_DIR, "inference.pdiparams")

    # Build runtime options
    det_option, cls_option, rec_option = build_option()

    # Initialize models
    det_model = fd.vision.ocr.DBDetector(
        det_model_file, det_params_file, runtime_option=det_option
    )

    cls_model = fd.vision.ocr.Classifier(
        cls_model_file, cls_params_file, runtime_option=cls_option
    )

    rec_model = fd.vision.ocr.Recognizer(
        rec_model_file, rec_params_file, REC_LABEL_FILE, runtime_option=rec_option
    )

    # Set preprocessor and postprocessor parameters for the Det model
    det_model.preprocessor.max_side_len = 960
    det_model.postprocessor.det_db_thresh = 0.3
    det_model.postprocessor.det_db_box_thresh = 0.6
    det_model.postprocessor.det_db_unclip_ratio = 1.5
    det_model.postprocessor.det_db_score_mode = "slow"
    det_model.postprocessor.use_dilation = False

    # Set postprocessor parameters for the Cls model
    cls_model.postprocessor.cls_thresh = 0.9

    # Create PP-OCRv3 instance
    ppocr_v3 = fd.vision.ocr.PPOCRv3(
        det_model=det_model, cls_model=cls_model, rec_model=rec_model
    )

    # Read and process the image
    # image = decode_image(IMAGE_PATH)
    image = decode_byte_image(bytes['bytes'])
    
    # print("-------> Performing OCR prediction")
    result = ppocr_v3.predict(image)
    # print("-------> OCR prediction completed")

    # Generate the result
    text = tostr_layout_preserving(result)

    # print("-------> OCR result:")
    # print(text)

    return text

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

def try_parse_json_object(input: str) -> tuple[str, dict]:
    """JSON cleaning and formatting utilities."""
    # Sometimes, the LLM returns a json string with some extra description, this function will clean it up.

    result = None
    try:
        # Try parse first
        result = json.loads(input)
    except json.JSONDecodeError:
        logger.warning("Warning: Error decoding faulty json, attempting repair")

    if result:
        return input, result

    _pattern = r"\{(.*)\}"
    _match = re.search(_pattern, input, re.DOTALL)
    input = "{" + _match.group(1) + "}" if _match else input

    # Clean up json string.
    input = (
        input.replace("{{", "{")
        .replace("}}", "}")
        .replace('"[{', "[{")
        .replace('}]"', "}]")
        .replace("\\", " ")
        .replace("\\n", " ")
        .replace("\n", " ")
        .replace("\r", "")
        .strip()
    )

    # Remove JSON Markdown Frame
    if input.startswith("```json"):
        input = input[len("```json") :]
    if input.endswith("```"):
        input = input[: len(input) - len("```")]
    if input.startswith('"'):
        input = input[1:]
    if input.endswith('"'):
        input = input[: len(input) - 1]

    try:
        result = json.loads(input)
    except json.JSONDecodeError:
        # Fixup potentially malformed json string using json_repair.
        input = str(repair_json(json_str=input, return_objects=False))

        # Generate JSON-string output using best-attempt prompting & parsing techniques.
        try:
            result = json.loads(input)
        except json.JSONDecodeError:
            logger.warning("error loading json, json=%s", input)
            return input, {}
        else:
            if not isinstance(result, dict):
                logger.warning("not expected dict type. type=%s:", type(result))
                return input, {}
            return input, result
    else:
        return input, result

def strIn(stra, strb):
    return strb in stra


if __name__ == "__main__":

    datasets = [
        "ArxivQA",
        "ChartQA",
        "MP-DocVQA",
        "InfoVQA",
        "PlotQA",
        "SlideVQA",
    ]

    root = "/ssddata/liuyue/github/VisRAG/qa_datasets"
    file_temp = "VisRAG-Ret-Test-{}"
    outdir = "/ssddata/liuyue/github/VisRAG/qa_datasets2"

    for dataset in datasets:
        logger.info(f'Data: {dataset}')
        os.makedirs(f'{outdir}/{file_temp.format(dataset)}/temp', exist_ok=True)

        corpus_df = load_parquet(f"{root}/{file_temp.format(dataset)}/corpus/*.parquet")
        qrel_df = load_parquet(f"{root}/{file_temp.format(dataset)}/qrels/*.parquet")
        query_df = load_parquet(f"{root}/{file_temp.format(dataset)}/queries/*.parquet")

        if not os.path.exists(f'{outdir}/{file_temp.format(dataset)}/temp/ocr_texts.parquet'):
            ocr_texts = corpus_df['image'].apply(ocr)

            logger.warning(f'Using dataframe.to_parquet')
            pd.DataFrame({'corpus-id': corpus_df['corpus-id'], 'ocr_texts': ocr_texts}, columns=['corpus-id', 'ocr_texts']).to_parquet(f'{outdir}/{file_temp.format(dataset)}/temp/ocr_texts.parquet', compression='zstd')
        else:
            ocr_texts = load_parquet(f'{outdir}/{file_temp.format(dataset)}/temp/ocr*')

        logger.info(f'{type(ocr_texts)}, {len(ocr_texts)}')

        client = OpenAI(base_url='https://api.siliconflow.cn/v1', api_key='sk-yxarqolvbcqiuluirnggghmkokbxnumgmanfzcgbvcaastxh')   

        kept_query_ids = []
        entity_df = pd.DataFrame({'query': query_df['query']}, columns=['query', 'entities', 'llm_res', 'parsed_res', 'first_entity'])

        if not os.path.exists(f'{outdir}/{file_temp.format(dataset)}/temp/entity.parquet'):
            # for i in tqdm(range(len(query_df))):
            for i in tqdm(range(50)):
                query = query_df['query'].iloc[i]
            # for query in query_df['query']:
                completion = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT.format(query)}
                    ],
                    response_format={ "type": "json_object" }
                )

                llm_res = completion.choices[0].message.content
                _, res = try_parse_json_object(llm_res)

                if isinstance(res, dict):
                    entities = res.get('named_entities', [])
                    
                    maxTimes = 0
                    if entities is not None:
                        entity_df.loc[i, 'entities'] = entities
                        entity_df.loc[i, 'llm_res'] = llm_res
                        entity_df.loc[i, 'parsed_res'] = res
                        entity_df.loc[i, 'first_entity'] = entities[0] if len(entities) > 0 else None
                        # for e in entities:
                        #     if type(e) == str:
                        #         checkse = ocr_texts.apply(strIn, strb=e)
                        #         maxTimes = max(maxTimes, checkse.sum())
                    
                    if maxTimes <= 1:
                        kept_query_ids.append(i)
            
            query_df = query_df.iloc[kept_query_ids, :]
            print(f'Kept {len(kept_query_ids)} queries')


            entity_df.to_csv(f'{outdir}/{file_temp.format(dataset)}/temp/entity.csv', index=False)

            logger.warning(f'Using dataframe.to_parquet')
            entity_df.to_parquet(f'{outdir}/{file_temp.format(dataset)}/temp/entity.parquet', compression='zstd')

            ### write filtered query df to new directory
            os.makedirs(f'{outdir}/{file_temp.format(dataset)}/queries', exist_ok=True)

            logger.warning(f'Using dataframe.to_parquet')
            query_df.to_parquet(f'{outdir}/{file_temp.format(dataset)}/queries/train-00000-of-00001.parquet', compression='zstd')


        try:
            load_parquet(f'{outdir}/{file_temp.format(dataset)}/queries/')
        except:
            logger.error('Error loading saved parquet !!')

        exit(0)
    pass
