# Copyright (C) 2023 Sean Ming-Fong Sie <seansie07@gmail.com> Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import json
import math
import os
import numpy as np
import pickle
import scipy.stats
from json import JSONDecodeError

from data_loader import load_btc2usd, TxLoader
from utils import divide, mean, parse_date

class TxNCounter():
    def __init__(self, tx_loader=None):
        # assert tx_loader is not None, 'Tx_loader is not specified.'
        self.tx_loader = tx_loader
        self.content = {}
        self.not_found = set()
    
    def __len__(self):
        return len(self.content)
    
    def get(self, tx_hash):
        n_in, n_out = None, None

        if self.tx_loader is None:
            # 如果 tx_loader 為空，則處理此情況
            # 這裡將 n_in 和 n_out 設置為默認值（或者執行其他適當的操作）
            return n_in, n_out

        if tx_hash in self.content:
            n_in, n_out = self.content[tx_hash]
        elif tx_hash not in self.not_found:
            tx_data = self.tx_loader.get(tx_hash)
            if tx_data is not None and isinstance(tx_data, dict):
                try:
                    n_in = tx_data['vin_sz']
                    n_out = tx_data['vout_sz']
                    self.content[tx_hash] = (n_in, n_out)
                except Exception as e:
                    pass
            else:
                self.not_found.add(tx_hash)
        return n_in, n_out

    def load(self, file):
        self.content = json.load(open(file, 'r', encoding='utf-8'))
        print('Loaded TxNCounter file', file)
    
    def save(self, file):
        json.dump(self.content, open(file, 'w', encoding='utf-8'))
        print('Saved as', file)


class TxHistory():
    def __init__(self, root='data/', ignore_error=False, to_date=None, balance=0, data_p=100, txn_file='data_p/tx_in_out_sizes1.json'):
        self.ignore_error = ignore_error
        self.to_date = to_date
        self.root = root
        self.txn_file = txn_file  #確保初始化了 self.txn_file

        # 獲取文件數量並在那裡減去 2
        data_folder = 'data_p'  # 您的文件夾名稱
        num_files = len([name for name in os.listdir(os.path.join(data_folder)) if os.path.isfile(os.path.join(data_folder, name))])
        self.data_p = num_files - 2 if num_files > 2 else 0  #確定要讀取的文件數

        self.txn_files = [f'tx_in_out_sizes{i}.json' for i in range(1, self.data_p + 1)]  # 建立檔案名稱清單
        self.init()
        self.reset(balance)
    
    def reset(self, balance=0):
        self.txs = {}
        self.balance = balance
    
    def init(self):
        ### Load yahoo finance data ###
        self.btc2usd = load_btc2usd(os.path.join(self.root, 'BTC-USD.csv'))
        
        # print('Loaded BTC-USD.csv.')
        ### self.btc2usd = load_btc2usd(os.path.join(self.root, 'market-price.csv')) ###
        ### print('market-price.csv.') ###
        self.coinbase = pickle.load(open(os.path.join(self.root, 'coinbase.pkl'), 'rb'))
        print('Loaded coinbase.pkl. Some coinbase transaction ids:')
        # print('coinbase pkl:', self.coinbase)
        print(list(self.coinbase['tx2blk'].keys())[:5])

        # Load the 'tx_in_out_sizes{i}.json' files
        for i in range(1, self.data_p + 1):
            file_path = os.path.join('data_p/', f"tx_in_out_sizes{i}.json")
            self.load_file(file_path)
            self.txn = TxNCounter()
            self.txn.load(file_path)
            print('Loaded TxN file', file_path)

    def load_file(self, file_path):
        # Code to process file content
        print(f"Processing file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read and process the file content here
            data = json.load(file)
            # Process 'data' as needed

    def add_balance(self, value):
        assert value >= 0
        self.balance += value

    def add(self, tx):
        ### important variables ###
        tx_hash = tx['tx_hash']
        tx_input_n = tx['tx_input_n']
        tx_output_n = tx['tx_output_n']
        block_height = tx['block_height']
        ref_balance = tx['ref_balance']
        date = parse_date(tx['confirmed'])
        
        ### filter out invalid transactions ###
        if self.to_date is not None and date > self.to_date:
            # print('Filtered date', date, 'from tx', tx_hash)
            return
        if date not in self.btc2usd:
            # print('Invalid datetime', date, 'in tx', tx_hash)
            # print(tx)
            return
        
        ### new entity ###
        if tx_hash not in self.txs:
            n_in, n_out = self.txn.get(tx_hash)
            self.txs[tx_hash] = {
                'input_n': [],
                'output_n': [],
                'input_v_btc': [],
                'output_v_btc': [],
                'input_v_usd': [],
                'output_v_usd': [],
                'date': date,
                'height': block_height,
                'balance_btc': ref_balance,
                'balance_usd': ref_balance * self.btc2usd[date],
                'num_inputs': n_in,
                'num_outputs': n_out
            }
        
        ### convert BTC (satoshi) to USD ###
        # value_btc = float(tx['value']) / 100000000
        # value_usd = value_btc * self.btc2usd[date]
        value_btc = ref_balance
        value_usd = ref_balance * self.btc2usd[date]
        
        if tx_input_n != -1:
            if tx_input_n in self.txs[tx_hash]['input_n']:
                if self.ignore_error: return
                assert False, 'Ah!'
            self.txs[tx_hash]['input_n'].append(tx_input_n)
            self.txs[tx_hash]['input_v_btc'].append(value_btc)
            self.txs[tx_hash]['input_v_usd'].append(value_usd)
        if tx_output_n != -1:
            if tx_output_n in self.txs[tx_hash]['output_n']:
                if self.ignore_error: return
                assert False, 'Oh no!'
            self.txs[tx_hash]['output_n'].append(tx_output_n)
            self.txs[tx_hash]['output_v_btc'].append(value_btc)
            self.txs[tx_hash]['output_v_usd'].append(value_usd)
        # if tx_input_n != -1 and tx_output_n != -1:
        #     # assert False, 'Oops!'
    def __len__(self):
        return len(self.txs)
    
    @staticmethod
    def feature_names():
        return np.array([
            'n_tx', 'total_days',
            'total_spent_btc', 'total_received_btc',
            'total_spent_usd', 'total_received_usd',
            'mean_balance_btc', 'std_balance_btc',
            'mean_balance_usd', 'std_balance_usd',
            'n_received', 'n_spent', 'n_coinbase', 'n_payback',
            'n_received_inclusive', 'n_spent_inclusive', 'n_coinbase_inclusive', 'n_payback_inclusive',
            'f_tx', 'f_received', 'f_coinbase',
            'f_spent_digits_-3', 'f_spent_digits_-2', 'f_spent_digits_-1', 'f_spent_digits_0',
            'f_spent_digits_1', 'f_spent_digits_2', 'f_spent_digits_3', 'f_spent_digits_4',
            'f_spent_digits_5', 'f_spent_digits_6', 'f_received_digits_-3', 'f_received_digits_-2',
            'f_received_digits_-1', 'f_received_digits_0', 'f_received_digits_1', 'f_received_digits_2',
            'f_received_digits_3', 'f_received_digits_4', 'f_received_digits_5', 'f_received_digits_6',
            'r_payback', 'n_inputs_in_spent', 'n_outputs_in_spent',
            'interval_1st_moment', 'interval_2nd_moment', 'interval_3rd_moment', 'interval_4th_moment',
            'dist_total_1st_moment', 'dist_total_2nd_moment', 'dist_total_3rd_moment', 'dist_total_4th_moment',
            'dist_coinbase_1st_moment', 'dist_coinbase_2nd_moment', 'dist_coinbase_3rd_moment', 'dist_coinbase_4th_moment',
            'dist_spend_1st_moment', 'dist_spend_2nd_moment', 'dist_spend_3rd_moment', 'dist_spend_4th_moment',
            'dist_receive_1st_moment', 'dist_receive_2nd_moment', 'dist_receive_3rd_moment', 'dist_receive_4th_moment',
            'dist_payback_1st_moment', 'dist_payback_2nd_moment', 'dist_payback_3rd_moment', 'dist_payback_4th_moment',
            'n_multi_in', 'n_multi_out', 'n_multi_in_out'
        ])
    
    def summarize(self, max_num=None):
        balances_btc = []
        balances_usd = []
        total_received_btc = 0
        total_spent_btc = 0
        total_balance_btc = 0
        total_spent_usd = 0
        total_received_usd = 0
        n_received = 0
        n_spent = 0
        n_coinbase = 0
        n_payback = 0
        n_received_inclusive = 0
        n_spent_inclusive = 0
        n_coinbase_inclusive = 0
        n_payback_inclusive = 0
        n_spent_digits = dict([(d, 0) for d in range(-3, 7)])
        n_received_digits = dict([(d, 0) for d in range(-3, 7)])
        n_inputs_in_spent = []
        n_outputs_in_spent = []
        prev_date = None
        interval_days = []
        max_date = None
        min_date = None
        dist_total = []
        dist_coinbase = []
        dist_spend = []
        dist_receive = []
        dist_payback = []
        n_multi_in = 0  
        n_multi_out = 0
        n_multi_in_out = 0
        count = 0
        
        for tx_hash, tx in sorted(self.txs.items(), key=lambda x: x[1]['date'], reverse=True):
            if max_num is not None and count == max_num:
                break
            count += 1
            
            balances_btc.append(tx['balance_btc'])
            balances_usd.append(tx['balance_usd'])
            
            # dist_total[tx['height']] += 1
            dist_total.append(tx['height'])
            if tx_hash in self.coinbase['tx2blk']:
                # dist_coinbase[tx['height']] += 1
                dist_coinbase.append(tx['height'])
                n_coinbase_inclusive += 1
            if len(tx['input_n']) > 0:
                # dist_spend[tx['height']] += 1
                dist_spend.append(tx['height'])
                n_spent_inclusive += 1

                # if the address is one of the spenders, count the n_in and n_out
                if tx_hash not in self.coinbase['tx2blk']:
                    if tx['num_inputs'] is not None:
                        n_inputs_in_spent.append(tx['num_inputs'])
                    if tx['num_outputs'] is not None:
                        n_outputs_in_spent.append(tx['num_outputs'])

            if len(tx['output_n']) > 0:
                # dist_receive[tx['height']] += 1
                dist_receive.append(tx['height'])
                n_received_inclusive += 1
            if len(tx['input_n']) > 0 and len(tx['output_n']) > 0:
                # dist_payback[tx['height']] += 1
                dist_payback.append(tx['height'])
                n_payback_inclusive += 1

            # Calculate the number of transactions with multiple inputs and outputs
            if isinstance(tx['output_n'], list) and len(tx['output_n']) > 0:
                n_multi_out = tx['output_n'][0]
            if isinstance(tx['input_n'], list) and len(tx['input_n']) > 0:
                n_multi_in = tx['input_n'][0]
            if n_multi_out > 0 and n_multi_in > 0:
                n_multi_in_out = 1


            total_spent_btc += sum(tx['input_v_btc'])
            total_received_btc += sum(tx['output_v_btc'])
            total_spent_usd += sum(tx['input_v_usd'])
            total_received_usd += sum(tx['output_v_usd'])
            value = sum(tx['input_v_usd']) - sum(tx['output_v_usd'])
            if tx_hash in self.coinbase['tx2blk']:
                n_coinbase += 1
            elif value > 0:
                n_spent += 1
            else:
                n_received += 1
            if len(tx['input_n']) > 0 and len(tx['output_n']) > 0:
                n_payback += 1
            
            # total_spent_btc += sum(tx['input_v_btc'])
            # total_received_btc += sum(tx['output_v_btc'])
            
            if value > 0:
                digit = math.floor(math.log(value, 10))
                if digit >= -3 and digit <= 6:
                    n_spent_digits[digit] += 1
            elif value < 0:
                digit = math.floor(math.log(-value, 10))
                if digit >= -3 and digit <= 6:
                    n_received_digits[digit] += 1
            else:  # value is zero
                n_received_digits[0] += 1
            
            if max_date is None:
                max_date = tx['date']
            else:
                max_date = max(max_date, tx['date'])
            if min_date is None:
                min_date = tx['date']
            else:
                min_date = min(min_date, tx['date'])
            if prev_date is not None:
                interval_days.append((prev_date - tx['date']).days)
            prev_date = tx['date']
        
        total_days = 1
        if max_date is not None and min_date is not None and max_date != min_date:
            total_days = (max_date - min_date).days
        # if min_date is not None and min_date != self.to_date:
        #     total_days = (self.to_date - min_date).days
            
        # total_balance_btc = total_received_btc - total_spent_btc
        
        total_spent_digits = sum(n_spent_digits[d] for d in n_spent_digits)
        total_received_digits = sum(n_received_digits[d] for d in n_received_digits)
        
        n_tx = len(self.txs)
        if n_tx == 0:
            return None
        
        f_tx = divide(n_tx, total_days)
        p_tx = divide(total_days, n_tx)
        f_received = divide(n_received, n_tx)
        f_coinbase = divide(n_coinbase, n_tx)
        f_spent_digits = {d: divide(n_spent_digits[d], total_spent_digits) for d in n_spent_digits}
        f_received_digits = {d: divide(n_received_digits[d], total_received_digits) for d in n_received_digits}
        r_payback = divide(n_payback, n_tx)
        
        if len(dist_total) == 0: dist_total = [0]
        if len(dist_coinbase) == 0: dist_coinbase = [0]
        if len(dist_spend) == 0: dist_spend = [0]
        if len(dist_receive) == 0: dist_receive = [0]
        if len(dist_payback) == 0: dist_payback = [0]
        dist_total = np.array(dist_total)
        dist_coinbase = np.array(dist_coinbase)
        dist_spend = np.array(dist_spend)
        dist_receive = np.array(dist_receive)
        dist_payback = np.array(dist_payback)
        dist_total -= dist_total.min()
        dist_coinbase -= dist_coinbase.min()
        dist_spend -= dist_spend.min()
        dist_receive -= dist_receive.min()
        dist_payback -= dist_payback.min()
        
        return np.array([
            n_tx,
            total_days,
            total_spent_btc,
            total_received_btc,
            total_spent_usd,
            total_received_usd,
            np.mean(balances_btc),
            np.var(balances_btc),
            np.mean(balances_usd),
            np.var(balances_usd),
            n_received,
            n_spent,
            n_coinbase,
            n_payback,
            n_received_inclusive,
            n_spent_inclusive,
            n_coinbase_inclusive,
            n_payback_inclusive,
            f_tx,
            f_received,
            f_coinbase,
            f_spent_digits[-3],
            f_spent_digits[-2],
            f_spent_digits[-1],
            f_spent_digits[0],
            f_spent_digits[1],
            f_spent_digits[2],
            f_spent_digits[3],
            f_spent_digits[4],
            f_spent_digits[5],
            f_spent_digits[6],
            f_received_digits[-3],
            f_received_digits[-2],
            f_received_digits[-1],
            f_received_digits[0],
            f_received_digits[1],
            f_received_digits[2],
            f_received_digits[3],
            f_received_digits[4],
            f_received_digits[5],
            f_received_digits[6],
            r_payback,
            mean(n_inputs_in_spent),
            mean(n_outputs_in_spent),
            # np.mean(n_inputs_in_spent),
            # np.mean(n_outputs_in_spent),
            np.mean(divide(np.array(interval_days), p_tx)),
            np.var(divide(np.array(interval_days), p_tx)),
            scipy.stats.skew(divide(np.array(interval_days), p_tx)),
            scipy.stats.kurtosis(divide(np.array(interval_days), p_tx), fisher=False),
            np.mean(dist_total),
            np.var(dist_total),
            scipy.stats.skew(dist_total),
            scipy.stats.kurtosis(dist_total, fisher=False),
            np.mean(dist_coinbase),
            np.var(dist_coinbase),
            scipy.stats.skew(dist_coinbase),
            scipy.stats.kurtosis(dist_coinbase, fisher=False),
            np.mean(dist_spend),
            np.var(dist_spend),
            scipy.stats.skew(dist_spend),
            scipy.stats.kurtosis(dist_spend, fisher=False),
            np.mean(dist_receive),
            np.var(dist_receive),
            scipy.stats.skew(dist_receive),
            scipy.stats.kurtosis(dist_receive, fisher=False),
            np.mean(dist_payback),
            np.var(dist_payback),
            scipy.stats.skew(dist_payback),
            scipy.stats.kurtosis(dist_payback, fisher=False),
            n_multi_in,
            n_multi_out,
            n_multi_in_out
        ])