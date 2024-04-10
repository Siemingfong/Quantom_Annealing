import os
import re
import json
import pickle
from datetime import datetime
from bitcoinrpc.authproxy import AuthServiceProxy, JSONRPCException
from tqdm import tqdm
import functools
import time
import sys

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_index(filename):
    match = re.search(r'index(\d+).json', filename)
    if match:
        return int(match.group(1))
    return -1

def batch_write_transactions(transactions_cache, address_directory, directory_count, index_directory, existing_index_data):
    new_index_data = existing_index_data.copy()
    # print("new_index_data: ", new_index_data)

    for transaction_data in transactions_cache:
        transaction_hash = transaction_data['tx_hash']
        # print("transaction_hash: ", transaction_hash)
        addresses = transaction_data['addresses']
        # print("addresses: ", addresses)
        directory = transaction_data['directory']
        # print("directory: ", directory)
        balancees = transaction_data['balancees']
        # print("balancees: ", balancees)
        formatted_date = transaction_data['formatted_date']
        block_height = transaction_data['block_height']
        vin_sz = transaction_data['vin_sz']
        vout_sz = transaction_data['vout_sz']
        # txrefs = transaction_data['txrefs']
        # input_tx_id = transaction_data['input_tx_id']

        create_directory_if_not_exists(directory)

        transaction_file_path = os.path.join(directory, f'{transaction_hash}.json')
        with open(transaction_file_path, 'w') as file:
            json.dump(transaction_data, file, indent=4)

        if addresses:
            address_filename = f'{addresses}.json'
            print("address_filename: ", address_filename)
            address_directory_path = os.path.join(address_directory, address_filename)
            # print("address_directory_path: ", address_directory_path)

            if os.path.exists(address_directory_path):
                with open(address_directory_path, 'r') as address_file:
                    existing_data = json.load(address_file)
                    # print("existing_data: ", existing_data)
            else:
                existing_data = {'balance': 0, 'txrefs': []}

            txref_dict = {
                'tx_hash': transaction_hash,
                'tx_input_n': vin_sz,
                'block_height': block_height,
                'tx_output_n': vout_sz,
                'ref_balance': float(balancees),
                'confirmed': formatted_date
            }
            # existing_data['txrefs'].append(txref_dict)
            # existing_data['balance'] = float(balancees)
            
            # Ensure the number of txrefs doesn't exceed 1000
            if len(existing_data) < 1000:
                existing_data['txrefs'].append(txref_dict)
                existing_data['balance'] = float(balancees)
            else:
                existing_data = existing_data[:1000]  # 只保留前面 1000 個交易參考
                existing_data['txrefs'] = existing_data
                
            with open(address_directory_path, 'w') as address_file:
                json.dump(existing_data, address_file, indent=4)

        new_index_data[transaction_hash] = directory

    index_filename = f'index{directory_count}.json'
    index_filepath = os.path.join(index_directory, index_filename)
    with open(index_filepath, 'w') as index_file:
        json.dump(new_index_data, index_file, indent=4)

def retry_on_connection_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3  # 設定最大重試次數
        retry_delay = 5  # 設定重試延遲時間（秒）

        for attempt in range(1, max_retries + 1):
            try:
                return func(*args, **kwargs)
            except ConnectionError as e:
                print(f"ConnectionError occurred: {e}")
                print(f"Retry attempt {attempt} of {max_retries}")
                if attempt < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Maximum retries exceeded. Exiting.")
                    sys.exit(1)

    return wrapper

def save_transaction_data_to_json():
    rpc_user = 'your_rpc_username'
    rpc_password = 'your_rpc_password'
    rpc_host = 'localhost'
    rpc_port = 8332

    transaction_count = 0
    directory_count = 1
    index_directory = '../transactions/index'
    address_directory = '../address'
    BATCH_SIZE = 1000
    block_hash = ''  # 或者賦予一個合適的預設值
    transactions_cache = []
    coinbase_data = {}  # 初始化coinbase数据

    create_directory_if_not_exists(address_directory)
    create_directory_if_not_exists(index_directory)

    index_files = os.listdir(index_directory)
    index_files.sort(key=extract_index)
    # print("index_files: ", index_files)

    if index_files:
        last_file = index_files[-1]
        if last_file.endswith('.json'):
            with open(os.path.join(index_directory, last_file), 'r') as json_file:
                index_data = json.load(json_file)
                block_hash = list(index_data.keys())[-1]
                directory_count = int(index_data[block_hash])
                transaction_count = len(index_data) % 10000

    start_block_height = -1
    end_block_height = 800000

    try:
        rpc_connection = AuthServiceProxy(f"http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}")
        raw_transaction = rpc_connection.getrawtransaction(block_hash, True)
        if "blockhash" in raw_transaction:
            block_hash = raw_transaction["blockhash"]
            block_info = rpc_connection.getblock(block_hash)
            block_height = block_info["height"]
            start_block_height = block_height

        for block_height in tqdm(range(start_block_height, end_block_height + 1)):
            block_hash = rpc_connection.getblockhash(block_height)
            block = rpc_connection.getblock(block_hash)

            for tx_id in block['tx']:
                transaction = rpc_connection.getrawtransaction(tx_id, True)
                transaction_hash = transaction['txid']
                directory = str(directory_count).zfill(8)

                vout_sz = 0
                addresses = []
                balancees = []
                # txrefs = []
                for vout in transaction['vout']:
                    balance = vout.get('value')
                    if balance is not None and balance > 0:
                        balancees = [balancees] if isinstance(balancees, float) else balancees
                        balancees.append(balance)
                    else:
                        balancees.append(0)
                    # print("Balance:", balance)
                    # txref = transaction_hash
                    # txrefs.append(txref)
                    if 'scriptPubKey' in vout and 'address' in vout['scriptPubKey']:
                        address = vout['scriptPubKey']['address']
                        addresses.append(address)
                        # print("addresses: ", addresses)
                        vout_sz += 1
                    else:
                        addresses.append('0')

                vin_sz = 0
                # input_tx_id = []
                for vin in transaction['vin']:
                    vin_sz += 1
                    if 'coinbase' in vin:
                        coinbase_data[transaction_hash] = directory
                        input_tx_id = 0
                    elif 'txid' in vin:
                        input_tx_id = vin['txid']

                timestamp = block['time']
                utc_datetime = datetime.utcfromtimestamp(timestamp)
                formatted_date = utc_datetime.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

                transaction_data = {
                    'tx_hash': transaction_hash,
                    # 'input_tx_id': input_tx_id,
                    'vin_sz': vin_sz,
                    # 'txrefs': txrefs,
                    'vout_sz': vout_sz,
                    'block_height': block_height,
                    'balancees': float(balancees[0]) if balancees and len(balancees) > 0 else 0,
                    'formatted_date': formatted_date,
                    'addresses': addresses[0],
                    'directory': directory
                }

                transactions_cache.append(transaction_data)
                # print("input_tx_id: ", input_tx_id)
                # print("vin_sz: ", vin_sz)
                # print("vout_sz: ", vout_sz)
                # print("block_height: ", block_height)
                # print("balancees: ", float(balancees[0]))
                # print("formatted_date: ", formatted_date)
                # print("addresses: ", addresses[0])
                # print("directory: ", directory)
                # print("transactions_cache: ", transactions_cache)

                transaction_count += 1
                print("transaction_count: ", transaction_count)
                if transaction_count % 1000 == 0:
                    batch_write_transactions(transactions_cache, address_directory, directory_count, index_directory, index_data)
                    index_data = {transaction['tx_hash']: directory for transaction in transactions_cache}
                    transactions_cache.clear()
                    # 索引文件寫入後，更新index_data
                    transaction_count = 0
                    directory_count += 1
                    
                    # 最后，将coinbase数据保存到文件
                    with open('../coinbase.pkl', 'wb') as coinbase_file:
                        pickle.dump({"tx2blk": coinbase_data}, coinbase_file)

        # if transactions_cache:
        #     batch_write_transactions(transactions_cache, address_directory, directory_count, index_directory, index_data)
        #     transactions_cache.clear()



    except JSONRPCException as e:
        print("RPC request error:", e.error)
    except Exception as e:
        print("Error:", str(e))

save_transaction_data_to_json()