import os
import re
import json
import pickle
from datetime import datetime
from bitcoinrpc.authproxy import AuthServiceProxy, JSONRPCException
from tqdm import tqdm

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 使用正規表達式從文件名中提取索引數字
def extract_index(filename):
    match = re.search(r'index(\d+).json', filename)
    if match:
        return int(match.group(1))
    return -1

def save_transaction_data_to_json():

    # Setting Bitcoin Core RPC
    rpc_user = 'your_rpc_username'
    rpc_password = 'your_rpc_password'
    rpc_host = 'localhost'
    rpc_port = 8332

    # Setting init parameters
    transaction_count = 0
    directory_count = 1
    index_data = {}
    addresses =[]
    coinbase_data = {}
    block_height = 1

    address_directory = '../address'
    # Create address directory if not exists
    if not os.path.exists(address_directory):
        os.makedirs(address_directory)

    # Load index file get block_hash, directory_count, transaction_count
    index_directory = '../transactions/index'  # 設置索引文件夾路徑
    index_files = os.listdir(index_directory)  # 使用os.listdir()獲取文件夾中的文件列表並存儲在index_files變量中
    print("index_files: ", index_files)
    index_files.sort(key=extract_index) # 使用extract_index函數將文件名轉換為索引數字並對文件列表進行排序

    # 確保文件列表非空
    if index_files:
        # 獲取最後一個文件的文件名
        last_file = index_files[-1]

        # 檢查文件是否為JSON文件（可選）
        if last_file.endswith('.json'):
            json_filepath = os.path.join(index_directory, last_file)

            with open(json_filepath, 'r') as json_file:
                index_data = json.load(json_file)
                block_hash = list(index_data.keys())[-1]
                directory_count = int(index_data[block_hash])
                transaction_count = len(index_data) % 10000
                # 這裡您可以處理JSON數據，data變量包含了JSON文件中的內容
                print(f"成功讀取最後一個文件: {last_file}")
        else:
            print(f"最後一個文件 '{last_file}' 不是JSON文件")
    else:
        print(f"文件夾 '{index_directory}' 為空")

    print(f"Last block_hash: {block_hash}")
    print(f"Last directory_count: {directory_count}")
    print(f"Last transaction_count: {transaction_count}")

    # Use the block heights for the start and end times
    start_block_height = -1
    end_block_height = 800000

    try:
        rpc_connection = AuthServiceProxy(f"http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}")
        
        # Get raw tx data
        raw_transaction = rpc_connection.getrawtransaction(block_hash, True)
        # Initial start block height
        while start_block_height == -1:
            if block_height != -1:
                # 獲取區塊訊息
                block_info = rpc_connection.getblock(raw_transaction["blockhash"])
                # print(f"Transaction {block_hash} is in block height {block_height}")
            else:
                print(f"Transaction {block_hash} is not yet confirmed in a block.")

            if "blockhash" in raw_transaction:
                block_hash = raw_transaction["blockhash"]

                # Use block_hash lookup block height
                block_info = rpc_connection.getblock(block_hash)
                block_height = block_info["height"]
                start_block_height = block_height
                # print("start_block_height: ", start_block_height)
                # print(f"Transaction with txid '{block_hash}' is in block height {block_height}")
                break
            else:
                print(f"Transaction with txid '{block_hash}' is not yet confirmed in a block.")
        
        previous_block_height = start_block_height  # 初始化先前區塊高度為起始高度

        for block_height in tqdm(range(start_block_height, end_block_height + 1)):
            
            # 確認區塊高度的變化
            if block_height != previous_block_height:
                # 更新先前區塊高度為目前處理的區塊高度
                previous_block_height = block_height

                block_hash = rpc_connection.getblockhash(block_height)
                # print("block height: ",block_height)
                block = rpc_connection.getblock(block_hash)

                for tx_id in block['tx']:
                    # Check block hash in transaction
                    if "blockhash" in raw_transaction:
                        block_hash = raw_transaction["blockhash"]
                        transaction = rpc_connection.getrawtransaction(tx_id, True)
                        transaction_hash = transaction['txid']
                    else:
                        break
                    # print("TXID: ", transaction_hash)
                    directory = str(directory_count).zfill(8)
                    print("File Numbers:", directory)

                    # Add this code to check if the directory exists, and create it if not
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    # Check transaction and address exist or not
                    file_path = os.path.join(directory, f'{transaction_hash}.json')

                    # Get output addresses and balancees and output txids
                    vout_sz = 0
                    addresses = []
                    balancees = []
                    txrefs = []
                    vout_list = transaction['vout']
                    for vout in vout_list:
                        # Get output balance
                        balance = vout['value']

                        if balance is not None and balance > 0:  # Check balance > 0
                            balancees.append(balance)
                            # print("Balance:", balance)
                        else:
                            balancees.append(0)
                            print("No or empty output balance found for TXID:", transaction_hash)

                        # Get output txid
                        txref = transaction_hash
                        txrefs.append(txref)

                        if 'scriptPubKey' in vout and 'address' in vout['scriptPubKey']:
                            # Get output address
                            address = vout['scriptPubKey']['address']
                            addresses.append(address)

                        vout_sz += 1

                    # Check if there are output addresses before accessing addresses[0]
                    if len(addresses) > 0:
                        print("Output Addresses:", addresses)
                    else:
                        print("No output addresses found for TXID:", transaction_hash)
                    # print("txrefs", txrefs)
                    
                    vin_sz = 0
                    # Get input txid
                    input_tx_id = []
                    vin_list = transaction['vin']

                    for vin in vin_list:
                        vin_sz +=1
                        if 'coinbase' not in vin and 'txid' in vin:
                            input_tx_id = vin['txid']
                            # print("Input TXID:", input_tx_id)

                    # Get block time
                    timestamp = block['time']
                    # 使用datetime.utcfromtimestamp將Unix時間戳轉換為UTC日期時間對象
                    utc_datetime = datetime.utcfromtimestamp(timestamp)
                    # 使用strftime將日期時間對象格式化為'%Y-%m-%dT%H:%M:%S.%fZ'格式
                    formatted_date = utc_datetime.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                    # print(formatted_date)

                    # Save addresses to a JSON file
                    if addresses:
                        address_filename = f'{addresses[0]}.json'
                        address_directory_path = os.path.join(address_directory, address_filename)
                        
                        txrefs_list = []  # 用於存儲交易參考的列表

                        # Check if the file already exists
                        if os.path.exists(address_directory_path):
                            # If the file exists, load the existing data
                            with open(address_directory_path, 'r') as address_file:
                                existing_data = json.load(address_file)
                                if 'balance' in existing_data:
                                    existing_balance = existing_data['balance']
                                    # Update the balance with the new value, if available
                                    if balancees and float(balancees[0]) > 0:
                                        existing_data['balance'] = float(balancees[0])
                                    # Append the new transaction reference to the existing list
                                    txref_dict = {
                                        'tx_hash': transaction_hash,
                                        'tx_input_n': vin_sz,
                                        'block_height': block_height,
                                        'tx_output_n': vout_sz,
                                        'ref_balance': float(balancees[0]),
                                        'confirmed': formatted_date
                                    }
                                    txrefs_list = existing_data.get('txrefs', [])
                                    # Ensure the number of txrefs doesn't exceed 1000
                                    if len(txrefs_list) < 1000:
                                        txrefs_list.append(txref_dict)
                                    else:
                                        txrefs_list = txrefs_list[:1000]  # 只保留前面 1000 個交易參考
                                    existing_data['txrefs'] = txrefs_list

                            # Save the updated data back to the file
                            with open(address_directory_path, 'w') as address_file:
                                json.dump(existing_data, address_file, indent=4, default=str)
                        else:
                            # If the file does not exist, create a new JSON file with the initial data
                            txref_dict = {
                                'tx_hash': transaction_hash,
                                'tx_input_n': vin_sz,
                                'block_height': block_height,
                                'tx_output_n': vout_sz,
                                'ref_balance': float(balancees[0]),
                                'confirmed': formatted_date
                            }
                            txrefs_list.append(txref_dict)
                            with open(address_directory_path, 'w') as address_file:
                                json.dump({
                                    'balance': float(balancees[0]) if balancees else 0,
                                    'txrefs': txrefs_list
                                }, address_file, indent=4, default=str)
                    else:
                        # Handle the case where there are no output addresses
                        # You can skip this transaction or provide a default value
                        print("No output addresses found for TXID:", transaction_hash)

                    # Add this code to check if the directory exists, and create it if not
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    # Save transaction data to a JSON file
                    file_path = os.path.join(directory, f'{transaction_hash}.json')
                    with open(file_path, 'w') as file:
                        json.dump({
                            'tx_hash': transaction_hash,
                            'tx_input_n': input_tx_id,
                            'vin_sz': vin_sz,
                            'tx_output_n': txrefs,
                            'vout_sz': vout_sz,
                            'block_height': block_height,
                            'ref_balance': float(balancees[0]),
                            'confirmed': datetime.utcfromtimestamp(block['time']).strftime('%Y-%m-%d %H:%M:%S')
                        }, file, indent=4, default=str)

                    # Index file name
                    index_filename = f'index{directory_count}.json'
                    index_filepath = os.path.join(index_directory, index_filename)

                    # Update index data
                    index_data[transaction_hash] = directory

                    with open(index_filepath, 'w') as json_file:
                        json.dump(index_data, json_file)

                    transaction_count += 1
                    if transaction_count % 10000 == 0:
                        transaction_count = 0
                        index_data = {}
                        directory_count += 1

                    # Append the "coinbase" field to the coinbase_data
                    coinbase_data = {"tx2blk": index_data}

                # Save coinbase data to coinbase.pkl
                with open('../coinbase.pkl', 'wb') as coinbase_file:
                    pickle.dump(coinbase_data, coinbase_file)

    except JSONRPCException as e:
        print("RPC request error:", e.error)

    except Exception as e:
        print("Error:", str(e))

save_transaction_data_to_json()