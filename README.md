# Experiment Steps

1. Collect Data from Wallet Expolrer as mixer and others catagory bitcoin addresses, we collect:
Interactive Addresses: 694,676
Class 0: Exchange 163,537
Class 1: Faucet 14,720
Class 2: Gambling 78,698
Class 3: Market 102,129
Class 4: Mixer 289,006
Class 5: Mining Pool 46,586

2. Collect Data from Bitcoin fullnode from genesis block to latest block
3. Preprocess data with preprocessing.py. It takes at last 4 days for 200 millsion transactions
5. Summary Bitcoin addrees transactions data with summarization.py. It takes 4~8 hours for 200 millsion transactions
6. Quantum proprocess data with quantum data preprocess.ipynb
7. Quantum annealing features with batch_featureSelection_SA_v2.ipynb
8. Supervised Machine Learning for Classification_rm.ipynb and more you want to training.
