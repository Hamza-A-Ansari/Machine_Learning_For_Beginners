'''/// Very Basic Example for Fp_Growth
firsrt install ...
pip install pyfpgrowth
then, execute the below code!'''

import numpy as np
import pandas as pd
import pyfpgrowth


dataset = pd.read_csv('Market.csv', header = None)
transactions = []
for sublist in dataset.values.tolist():
    clean_sublist = [item for item in sublist if item is not np.nan]
    transactions.append(clean_sublist) 



patterns = pyfpgrowth.find_frequent_patterns(transactions, 2)

rules = pyfpgrowth.generate_association_rules(patterns, 0.7)