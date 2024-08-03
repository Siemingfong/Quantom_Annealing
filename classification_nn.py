# Import all used packages

import argparse
import collections
import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.utils import class_weight

from tqdm import tqdm

from learn import get_model, get_params
from utils import run_from_ipython, np2df
from viz import show_cm_list, show_rp_list

if run_from_ipython():
    import matplotlib
    %matplotlib inline
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context('notebook')  # 'notebook', 'paper', 'talk', 'poster'
    # sns.set_style('dark')  # None, 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'
    
# Parse Arguments

def parse(args=None):
    parser = argparse.ArgumentParser(
        prog='Classification',
        description='Train and test a machine learning classification method on the extracted features.'
    )
    parser.add_argument('--n_folds', help='n folds cross validation', type=int, default=10)
    parser.add_argument('--feature_type', '-f',
                        help='feature type ("b" | "e" | "m" or "if4", "if5", "if10", "if13", "if20", "if64")',
                        type=str, default='bem')
    parser.add_argument('--scheme', '-s', help='data scheme', type=str,
                        choices=['address', 'entity'], default='address')
    parser.add_argument('--gpu', help='use GPU', action='store_true')
    parser.add_argument('--output', '-o', help='output path', type=str, default='./data_p')
    parser.add_argument('--result', '-r', help='result path', type=str, default='./result')
    parser.add_argument('--temp', '-t', help='temp path', type=str, default='./temp')
    return parser.parse_args() if args is None else parser.parse_args(args)
args = parse([]) if run_from_ipython() else parse()
print(args)


# Define the experiment setting

n_folds = args.n_folds                       # 10
feature_type = args.feature_type             # 'b' | 'e' | 'm' or 'if4', 'if5', 'if10', 'if13', 'if20', 'if64'
scheme = args.scheme                         # 'address', 'entity'
gpu = args.gpu
output_path = args.output
result_path = args.result
temp_path = args.temp

# Check the experiment setting

assert not feature_type.startswith('if') and len(feature_type) > 0 or \
       feature_type.startswith('if') and feature_type[2:].isdigit()
assert scheme in ['address', 'entity']

# Number of epochs to train
# The specified epochs are enough to converge for each scheme
if scheme == 'address':
    epochs = 4000
elif scheme == 'entity':
    epochs = 1500

# Show the experiment setting

print('Experiment Setting')
print('===> Feature Types:  ', feature_type)
print('===> Data Scheme:    ', scheme)
print('===> Use GPU:        ', gpu)
print('===> Training epochs:', epochs)


# 定義一個函數來加載和抽樣數據
def load_sample(file_path, sample_fraction=0.1):
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=100000):
        chunks.append(chunk.sample(frac=sample_fraction))
    return pd.concat(chunks)


# Load transaction history summarization data

# 設定樣本比例
sample_fraction = 0.1  # 取 10% 的數據樣本

# data_file = 'data.{}.csv'.format(scheme)
data_file = 'nanzero_normalization_data.{}.csv'.format(scheme)
# data_file = 'quantum_qubo_data.{}.csv'.format(scheme)
# data_file = 'all_selected_features_quantum_qubo_data.{}.csv'.format(scheme)
file_path = os.path.join(output_path, data_file)
data = load_sample(file_path, sample_fraction)
# data = pd.read_csv(os.path.join(output_path, data_file))
print (data)
if run_from_ipython():
    data.head(4)
else:
    print(data.head(4))
    
    
# Define 4 types of features (basic statistics, extra statistics, moments and patterns)

basic = [
    'f_tx', 'f_received', 'f_coinbase',
    'f_spent_digits_-3', 'f_spent_digits_-2', 'f_spent_digits_-1', 'f_spent_digits_0',
    'f_spent_digits_1', 'f_spent_digits_2', 'f_spent_digits_3', 'f_spent_digits_4',
    'f_spent_digits_5', 'f_spent_digits_6', 'f_received_digits_-3', 'f_received_digits_-2',
    'f_received_digits_-1', 'f_received_digits_0', 'f_received_digits_1', 'f_received_digits_2',
    'f_received_digits_3', 'f_received_digits_4', 'f_received_digits_5', 'f_received_digits_6',
    'r_payback', 'n_inputs_in_spent', 'n_outputs_in_spent'
]
extra = [
    'n_tx', 'total_days', 'n_spent', 'n_received', 'n_coinbase', 'n_payback',
    'total_spent_btc', 'total_received_btc',
    'total_spent_usd', 'total_received_usd',
    'mean_balance_btc', 'std_balance_btc',
    'mean_balance_usd', 'std_balance_usd'
]
moments = [
    'interval_1st_moment', 'interval_2nd_moment', 'interval_3rd_moment', 'interval_4th_moment',
    'dist_total_1st_moment', 'dist_total_2nd_moment', 'dist_total_3rd_moment', 'dist_total_4th_moment',
    'dist_coinbase_1st_moment', 'dist_coinbase_2nd_moment', 'dist_coinbase_3rd_moment', 'dist_coinbase_4th_moment',
    'dist_spend_1st_moment', 'dist_spend_2nd_moment', 'dist_spend_3rd_moment', 'dist_spend_4th_moment',
    'dist_receive_1st_moment', 'dist_receive_2nd_moment', 'dist_receive_3rd_moment', 'dist_receive_4th_moment',
    'dist_payback_1st_moment', 'dist_payback_2nd_moment', 'dist_payback_3rd_moment', 'dist_payback_4th_moment'
]
patterns =[
    'tx_input', 'tx_output',
    'n_multi_in', 'n_multi_out', 'n_multi_in_out'
]

features = []
if not feature_type.startswith('if') and len(feature_type) > 0:
    if 'b' in feature_type:
        features += basic
    if 'e' in feature_type:
        features += extra
    if 'm' in feature_type:
        features += moments
    if 'p' in feature_type:
        features += patterns
        print("Patterns included:", patterns)
elif feature_type.startswith('if') and feature_type[2:].isdigit():
    """
    Important features from LightGBM with BEM
    [ 0 25 24 29 40 37 27 23 56 36  1 28 26 57 32 38 44 45 33 18 39 60 53 35
     34 52 41 17 14 15 16 19 42  5  6 47  7 46  2 54  4 43  8 59 58 55  9 13
     61 48  3 31 10 62 20 21 63 30 49 11 51 50 22 12]
    """
    all_features = basic + extra + moments + patterns
    if_indices = [
        0, 25, 24, 29, 40, 37, 27, 23, 56, 36,
        1, 28, 26, 57, 32, 38, 44, 45, 33, 18,
        39, 60, 53, 35, 34, 52, 41, 17, 14, 15,
        16, 19, 42, 5, 6, 47, 7, 46, 2, 54,
        4, 43, 8, 59, 58, 55, 9, 13, 61, 48,
        3, 31, 10, 62, 20, 21, 63, 30, 49, 11,
        51, 50, 22, 12
    ]
    if_features = [all_features[i] for i in if_indices]
    n_if = int(feature_type[2:])
    features = if_features[:n_if]
else:
    raise Exception('Invalid feature types: {:s}'.format(feature_type))

invalid_features = [feature for feature in features if feature not in data.columns]
assert len(invalid_features) == 0, 'Invalid features: ' + ', '.join(invalid_features)

X = data.get(features).values
y = data['class'].values
print (features)
print (feature_type)
print (X)
print (y)

class2label = json.loads(open(os.path.join(output_path, 'class2label.json'), 'r').read())
label2class = json.loads(open(os.path.join(output_path, 'label2class.json'), 'r').read())
class_names = np.array([label2class[i] for i in range(6)])
print (class_names)
y_names = class_names
# y_names = class_names[y]
# y_names = np.array(class_names)[y.astype(int)]

print(len(X), len(y), len(features))

os.makedirs(result_path, exist_ok=True)


import os
import pickle
import shutil
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model

# 如果不想使用 GPU，可以設置環境變量
if not gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# GPU 配置 for TensorFlow 2.x
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 為每個 GPU 設置允許記憶體增長
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def build_model(input_size, num_classes, summary=False):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_size,)),
        BatchNormalization(momentum=0.0, epsilon=1e-5),
        # Dropout(0.2),
        # Dense(512, activation='relu'),
        # BatchNormalization(momentum=0.0, epsilon=1e-5),
        # Dropout(0.2),
        # Dense(512, activation='relu'),
        # BatchNormalization(momentum=0.0, epsilon=1e-5),
        Dropout(0.2),
        Dense(512, activation='relu'),
        BatchNormalization(momentum=0.0, epsilon=1e-5),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    if summary:
        model.summary()
    return model


import time
import numpy as np
import os
import shutil
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from keras.models import load_model
import keras

# 加入時間模組

# 訓練過程
train_cm_list = []
train_rp_list = []
valid_cm_list = []
valid_rp_list = []
fi_list = []

# Add these lists to store AUC scores
train_auc_list = []
valid_auc_list = []

# 宣告 K-Fold
n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

# # 標準化資料
# print('Normalizing data...')
# X = np.nan_to_num((X - np.mean(X, axis=0)) / np.std(X, axis=0))
# X = np.clip(X, np.percentile(X, 1, axis=0), np.percentile(X, 99, axis=0))

# 開始計算整體訓練時間
total_start_time = time.time()  # 全部訓練開始時間

# 開始交叉驗證
for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    # 取得分割的訓練集和驗證集
    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
    y_train_ = keras.utils.to_categorical(y_train, num_classes=len(class2label))
    y_valid_ = keras.utils.to_categorical(y_valid, num_classes=len(class2label))
    
    # 建立模型
    model = build_model(X.shape[1], len(label2class), summary=(i==0))
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 訓練模型
    acc = -1
    for epoch in range(epochs):
        model.train_on_batch(X_train, y_train_)
        accs = model.test_on_batch(X_valid, y_valid_)
        if accs[1] > acc:
            acc = accs[1]
            print(acc, epoch)
            model.save(os.path.join(temp_path, 'model.h5'))
    model = load_model(os.path.join(temp_path, 'model.h5'))
    
    # 在訓練集上評估
    y_pred = np.argmax(model.predict(X_train), axis=1)
    cm = confusion_matrix(y_train, y_pred)
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_sum[cm_sum == 0] = 1  # 防止除零
    cm = cm / cm_sum
    train_cm_list.append(cm)
    
    # 在驗證集上評估
    y_pred = np.argmax(model.predict(X_valid), axis=1)
    cm = confusion_matrix(y_valid, y_pred)
    cm = cm / cm.sum(axis=1, keepdims=True)
    valid_cm_list.append(cm)
    rp = classification_report(y_valid, y_pred, target_names=class_names, output_dict=True)
    valid_rp_list.append(rp)
    
    # Calculate AUC for the training set
    y_train_prob = model.predict(X_train)
    train_auc = roc_auc_score(y_train, y_train_prob, multi_class="ovr", average="macro")
    train_auc_list.append(train_auc)

    # Calculate AUC for the validation set
    y_valid_prob = model.predict(X_valid)
    valid_auc = roc_auc_score(y_valid, y_valid_prob, multi_class="ovr", average="macro")
    valid_auc_list.append(valid_auc)

# 全部訓練結束時間
total_end_time = time.time()
total_training_time = total_end_time - total_start_time  # 計算總訓練時間
print(f"Total training time: {total_training_time:.2f} seconds")  # 輸出總訓練時間

# 清除臨時檔案
shutil.rmtree(temp_path)


import os
import pickle

# 假設 result_path 和其他變數都已經定義

# Create the result path if it does not exist
if not os.path.exists(result_path):
    os.makedirs(result_path)

# Save training results

# 確保 experiment_name 是一個有效的文件名
experiment_name = os.path.join(result_path, '{}.{}.{}'.format('nn', feature_type, scheme))

# 確保 valid_cm_list 不為空
if not valid_cm_list:
    raise ValueError("valid_cm_list is empty during saving process")

results = {
    'train_cm_list': train_cm_list,
    'valid_cm_list': valid_cm_list,
    'train_rp_list': train_rp_list,
    'valid_rp_list': valid_rp_list,
    'fi_list': fi_list,
    'train_auc_list': train_auc_list,
    'valid_auc_list': valid_auc_list,
    'y_valid': y_valid,  # 添加真實標籤
    'y_valid_prob': y_valid_prob  # 添加預測概率
}
pickle.dump(results, open(experiment_name + '.pkl', 'wb'))

# Save model
model_save_path = '{}_model.pkl'.format(experiment_name)
with open(model_save_path, 'wb') as model_file:
    pickle.dump(model, model_file)

print(f"Results and model saved to {result_path}")


# Average confusion matrix of training set in K-fold

print('Average confusion matrix of training set in {:d}-fold'.format(n_folds))
show_cm_list(train_cm_list, class_names)