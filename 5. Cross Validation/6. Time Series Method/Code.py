#TimeSeriesSplit
   
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])

tscv = TimeSeriesSplit(n_splits=3,max_train_size= None)
print(tscv)  

for train, test in tscv.split(X):
    print("%s %s" % (train, test))  
