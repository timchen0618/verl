# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import tempfile

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from verl.utils.hdfs_io import copy, makedirs

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."
DEFAULT_USER_CONTENT_PREFIX_LEGACY = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)

DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can use the search tool to look up relevant information. "
    "You can search as many times as you want. If you find no further external knowledge "
    "needed, you can directly provide the answer inside <answer> and </answer>, without "
    "detailed illustrations. For example, <answer> Beijing </answer>. Question: "
)
  


def process_row(row, split_name, row_index, system_content, user_content_prefix):
    """Process a single row into SearchR1-like format."""
    question = row.get("question", "")
    user_content = user_content_prefix.rstrip("\n") + question
    prompt = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]

    reward_model_data = row.get("reward_model")
    if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
        ground_truth = (reward_model_data.get("ground_truth"))
    else:
        ground_truth = (row.get("golden_answers"))
        reward_model_data = {'ground_truth': {'target': ground_truth}} 

    data_source_tagged = "searchR1_" + str(row.get("data_source", ""))
    tools_kwargs = {
        "search": {
            "create_kwargs": {"ground_truth": ground_truth, "question": question, "data_source": data_source_tagged}
        }
    }
    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "question": question,
        "split": split_name,
        "tools_kwargs": tools_kwargs,
    }

    return pd.Series(
        {
            "data_source": data_source_tagged,
            "prompt": prompt,
            "ability": row.get("ability"),
            "reward_model": reward_model_data,
            "extra_info": extra_info,
            "metadata": row.get("metadata"),
        }
    )


def iter_splits_and_rows(args):
    """Yield (split_name, rows) for each data split. rows are dict-like."""
    if args.dataset_source == "flashrag":
        if load_dataset is None:
            raise ImportError("FlashRAG requires 'datasets'. Install with: pip install datasets")
        subset = args.flashrag_subset
        logger.info(f"Loading RUC-NLPIR/FlashRAG_datasets ({subset})...")
        dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", subset)
        for split_name in dataset.keys():
            rows = []
            for i, ex in enumerate(dataset[split_name]):
                rows.append({
                    "question": ex["question"],
                    "golden_answers": (ex.get("golden_answers")),
                    "data_source": f"flashrag_{subset}",
                    "ability": ex.get("ability"),
                    "metadata": ex.get("metadata"),
                })
            yield split_name, rows
    else:
        with tempfile.TemporaryDirectory() as tmp:
            for split in ["train", "test"]:
                parquet_path = f"{split}.parquet"
                try:
                    logger.info(f"Downloading {parquet_path} from {args.hf_repo_id}")
                    local_path = hf_hub_download(
                        repo_id=args.hf_repo_id,
                        filename=parquet_path,
                        repo_type="dataset",
                        local_dir=tmp,
                        local_dir_use_symlinks=False,
                    )
                    df = pd.read_parquet(local_path)
                    logger.info(f"Loaded {len(df)} rows from {parquet_path}")
                    yield split, [df.loc[i] for i in range(len(df))]
                except EntryNotFoundError:
                    logger.warning(f"{parquet_path} not found in {args.hf_repo_id}")
                except Exception as e:
                    logger.error(f"Error processing {split}: {e}")


def main(args, system_content, user_content_prefix):
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    processed_files = []
    for split_name, rows in iter_splits_and_rows(args):
        df = pd.DataFrame(
            process_row(row, split_name, i, system_content, user_content_prefix) for i, row in enumerate(rows)
        )
        out_path = os.path.join(local_save_dir, f"{split_name}.parquet")
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved {len(df)} rows to {out_path}")
        processed_files.append(out_path)

    if not processed_files:
        logger.warning("No data was processed")
        return

    logger.info(f"Processed {len(processed_files)} files to {local_save_dir}")
    if args.hdfs_dir:
        try:
            makedirs(args.hdfs_dir)
            copy(src=local_save_dir, dst=args.hdfs_dir)
            logger.info(f"Copied to HDFS: {args.hdfs_dir}")
        except Exception as e:
            logger.error(f"Error copying to HDFS: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Search-R1 from HuggingFace, process, and save to Parquet.")
    parser.add_argument(
        "--dataset_source",
        choices=["hf_parquet", "flashrag"],
        default="hf_parquet",
        help="'hf_parquet': train/test.parquet from --hf_repo_id; 'flashrag': RUC-NLPIR/FlashRAG_datasets (--flashrag_subset).",
    )
    parser.add_argument("--flashrag_subset", default="nq", help="Subset when dataset_source=flashrag (e.g. nq, hotpotqa).")
    parser.add_argument("--hf_repo_id", default="PeterJinGo/nq_hotpotqa_train", help="HuggingFace dataset repo ID.")
    parser.add_argument("--local_dir", default="~/data/searchR1_processed_direct", help="Output directory.")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS copy destination.")

    args = parser.parse_args()
    main(args, DEFAULT_SYSTEM_CONTENT, DEFAULT_USER_CONTENT_PREFIX)
