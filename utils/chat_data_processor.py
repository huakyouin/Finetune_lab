"""
用于将各种数据集转化为对话message数据集
Author: huakyouin
"""
import torch
import os
import json
from datasets import load_dataset, Dataset, concatenate_datasets
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='数据集处理工具')
    parser.add_argument('--data_path', type=str, required=True, help='数据集路径')
    parser.add_argument('--filter', type=str, help='过滤条件，格式为col_name1:val1,col_name2:val2')
    parser.add_argument('--col_mappings', type=str, help='列名转换字典，格式为old_col:new_col,old_col2:new_col2')
    parser.add_argument('--save_path', type=str, required=True, help='保存路径')

    args = parser.parse_args()
    print(args)
    # 加载数据集
    data = load_dataset(args.data_path)
    data = concatenate_datasets([d for key, d in data.items() if isinstance(d, Dataset)])

    # 处理过滤条件
    if args.filter:
        filters = [f.split(':') for f in args.filter.split(',')]
        for col_name, val in filters:
            data = data.filter(lambda example, col_name=col_name, val=val: example[col_name] == val)

    # 处理列名转换
    if args.col_mappings:
        col_mappings = dict(mapping.split(':') for mapping in args.col_mappings.split(','))
        data = data.rename_columns(col_mappings)

    # 构建chat
    data = data.map(lambda example: {"message": [{"role": "system", "content": example["instruction"]},
                                                 {"role": "user", "content": example['input']},
                                                 {"role": "assistant", "content": example['output']}]})

    # 保存数据集到JSONL格式
    data.save_to_disk(args.save_path)

## usage:
##    python chat_data_processor.py --data_path ../data/raw/findata --filter task:FINNA --save_path ../data/processed/SFT/FinCUGE




# 采样
#     num_samples = min(len(dataset), sample_size)
#     proportion = weight / weights  # 计算当前数据集应该抽取的比例
#     samples.append(dataset.select(range(int(proportion * num_samples))))  # 抽取样本

# 合并样本
#     combined_dataset = concatenate_datasets(samples)
