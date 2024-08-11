import numpy as np
from pyqubo import Array
import neal
import matplotlib.pyplot as plt
import requests
import seaborn as sns

import pandas as pd
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.embedding.chain_strength import uniform_torque_compensation
from dwave.embedding.chain_breaks import majority_vote

class FeatureSelection(object):
    def __init__(self, num_features, dependence_coefficients, influence_coefficients):
        # Number of features
        self.num_features = num_features
        self.dependence_coefficients = dependence_coefficients
        self.influence_coefficients = influence_coefficients
        
        # Create binary variables for the features
        self.qubo_linear = {i: -influence_coefficients[i] for i in range(num_features)}
        self.qubo_quadratic = {(i, j): dependence_coefficients[i][j]
                       for i in range(num_features) for j in range(i + 1, num_features)
                       if not np.isnan(dependence_coefficients[i][j]) and dependence_coefficients[i][j] != 0}

    def compile(self):
        # Combine linear and quadratic terms
        return dimod.BinaryQuadraticModel(self.qubo_linear, self.qubo_quadratic, 0.0, vartype=dimod.BINARY)
    
# 初始化一個集合來儲存所有選中的特徵
all_selected_features = set()

# 初始化一個字典來儲存每個 class 的 selected_features
class_selected_features = {}

# Load the CSV file
for i in range(0, 6):
    # Load the class 0~6 CSV file
    file_path = f'../data_p/quantum_data.address_class{i}.csv'
    df = pd.read_csv(file_path)

    # Extracting each column as an array
    columns = df.columns
    features = df[columns[:-1]]  # All columns except the last one
    result = df[columns[-1]]    # The last column
    n_features = features.shape[1]

    # Calculate the correlation matrix for features
    feature_correlation = features.corr(method='spearman')

    # Calculate the correlation of each feature with the result
    result_correlation = features.apply(lambda x: x.corr(result, method='spearman'))

    feature_qubo = FeatureSelection(n_features, feature_correlation.values, result_correlation.values)
    bqm = feature_qubo.compile()

    # 使用 D-Wave 量子計算機來解 QUBO 問題
    qpu_advantage = DWaveSampler(solver={'chip_id': 'Advantage_system6.4'})
    sampler = EmbeddingComposite(qpu_advantage)   
    response = sampler.sample(bqm, num_reads=1000, chain_strength=uniform_torque_compensation(bqm), chain_break_method=majority_vote, auto_scale=True, reduce_intersample_correlation=True)
    
    # Print results
    print("All energies:", response.record['energy'])

    # print("Sample:", response.info, "Energy:", response.energy)

    # Find the best sample (modify this as per your criteria)
    # For simplicity, we're taking the first sample as an example
    best_sample = list(response.first.sample.items())

    # Identify selected features
    selected_features = [int(key) for key, value in best_sample if value == 1]

    # Filter the DataFrame to keep only the selected columns
    filtered_df = df.iloc[:, selected_features]
    
    # 將本次迭代選中的特徵添加到集合中
    all_selected_features.update(selected_features)
    
    # 將本次迭代選中的特徵儲存到字典中
    class_selected_features[i] = selected_features

    # Add the index of the last column (class) to the selected features
    last_column = df[columns[-1]]
    filtered_df = pd.concat([filtered_df, last_column], axis=1)

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(f'../data_p/QA_data.address_class{i}.csv', index=False)

# Print non-duplicate selected features from all iterations
print("Combined Selected Features (No Duplicates):", sorted(all_selected_features))

# 也可以打印出每個 class 的 selected_features 來確認
for class_num, features in class_selected_features.items():
    print(f"Class {class_num} Selected Features:", features)