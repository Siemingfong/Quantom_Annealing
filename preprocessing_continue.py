# Copyright (C) 2023 Ming-Fong Sie <seansie07@gmail.com> & Yu-Jing Lin <elvisyjlin@gmail.com>

# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
# Import all used packages

import argparse
import json
import os
import pandas as pd
import shutil
import glob
from tqdm import tqdm

from data_loader import TxLoader
from summary import TxNCounter
from utils import run_from_ipython
# Parse arguments

def parse(args=None):
    parser = argparse.ArgumentParser(
        prog='Preprocessing',
        description='Create the label-class mapping and a table containing in/out sizes of each transaction.'
    )
    parser.add_argument('--data', '-d', help='data path', type=str, default='./data')
    parser.add_argument('--out', '-o', help='output path', type=str, default='./data_p')
    return parser.parse_args() if args is None else parser.parse_args(args)
args = parse([]) if run_from_ipython() else parse()
print(args)

# 設定數據和輸出路徑
data_path = args.data
out_path = args.out

# Load the Bitcoin dataset
dataset = pd.read_csv(os.path.join(data_path, 'dataset_mymerge.csv'))
print('# of addresses:', len(dataset['address']))
dataset.head(5)
# Checkpoint the mapping between labels and classes (categories)

total_classes = list(sorted(set(dataset['class'])))
class2label = {}
label2class = total_classes
for l, c in enumerate(total_classes):
    class2label[c] = l

json.dump(class2label, open(os.path.join(out_path, 'class2label.json'), 'w', encoding='utf-8'))
json.dump(label2class, open(os.path.join(out_path, 'label2class.json'), 'w', encoding='utf-8'))
print('===> class2label.json')
print(class2label)
print('===> label2class.json')
print(label2class)

# 加載交易資料
loader = TxLoader(root=os.path.join(data_path, 'transactions'), max_size=100000)
counter = TxNCounter(loader)
success, failed = 0, 0
print(loader.index)

# Set the batch size for splitting files
batch_size = 100000
# file_counter = 1
tx_in_out_sizes = {}
keys = list(loader.index.keys())  # Get the keys from the loader index

# 確定最新的 tx_in_out_sizes 檔案
file_pattern = os.path.join(out_path, "tx_in_out_sizes*.json")
file_list = sorted(glob.glob(file_pattern), key=lambda x: int(x.split('sizes')[-1].split('.json')[0]))

if file_list:
    # 讀取最新的文件
    last_file_path = file_list[-1]
    with open(last_file_path, 'r') as f:
        last_data = json.load(f)

    # 獲取最後一個處理過的交易哈希
    last_tx_hash = list(last_data.keys())[-1]

    # 在 keys 列表中找到這個交易哈希的位置
    last_index = keys.index(last_tx_hash) + 1

    # 確定檔案計數器的起始值
    file_counter = int(last_file_path.split('sizes')[-1].split('.json')[0]) + 1

else:
    # 如果沒有找到任何文件，從頭開始
    last_index = 0
    file_counter = 1

# 從最後一個已處理的交易開始繼續處理
index = last_index + 1  # 從最後一個已處理的交易後一個開始
for tx_hash in tqdm(keys[last_index:]):
    in_size, out_size = counter.get(tx_hash)
    
    # Check if in_size and out_size are retrieved successfully
    if in_size is None or out_size is None:
        failed += 1
    else:
        success += 1
        # Store the input and output sizes in the dictionary
        tx_in_out_sizes[tx_hash] = [in_size, out_size]

    # Check if it's time to write to a new file
    if index % batch_size == 0 or index == len(keys):
        output_file_path = os.path.join(out_path, f"tx_in_out_sizes{file_counter}.json")
        with open(output_file_path, 'w') as outfile:
            json.dump(tx_in_out_sizes, outfile)
        tx_in_out_sizes = {}  # Reset the dictionary for the next batch
        file_counter += 1

    index += 1  # Increment index outside the loop

# Save the final overall transaction data to a separate file using counter.save()
#counter.save(os.path.join(out_path, 'tx_in_out_sizes.json'))

print('# of success / failed:', success, '/', failed)
