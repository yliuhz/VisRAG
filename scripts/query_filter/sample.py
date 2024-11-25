
### Randomly sample 500 queries

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from PIL import Image
import io
from io import BytesIO
import polars as pl
import pandas as pd

from prompts.prompt import SYSTEM_PROMPT, USER_PROMPT
from openai import OpenAI
import re
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


if __name__ == "__main__":
    nsamples = 500

    datasets = [
        "ArxivQA",
        "ChartQA",
        "MP-DocVQA",
        "InfoVQA",
        "PlotQA",
        "SlideVQA",
    ]

    root = "/ssddata/liuyue/github/PruneRAG/VisRAG/qa_datasets"
    file_temp = "VisRAG-Ret-Test-{}"
    outdir = "/ssddata/liuyue/github/PruneRAG/VisRAG/qa_datasets_sample"

    for dataset in datasets:
        logger.info(f'Data: {dataset}')

        corpus_df = load_parquet(f"{root}/{file_temp.format(dataset)}/corpus/*.parquet")
        qrel_df = load_parquet(f"{root}/{file_temp.format(dataset)}/qrels/*.parquet")
        query_df = load_parquet(f"{root}/{file_temp.format(dataset)}/queries/*.parquet")

        # Sample queries
        query_df = query_df.sample(frac=1).reset_index(drop=True).loc[:nsamples]

        # Save sampled queries
        os.makedirs(f"{outdir}/{file_temp.format(dataset)}/queries/", exist_ok=True)
        query_df.to_parquet(f"{outdir}/{file_temp.format(dataset)}/queries/queries.parquet")
