# Leave One Out

import numpy as np
from sklearn.model_selection import LeaveOneOut
X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
Y = np.array([0, 1, 0, 1])
loo = LeaveOneOut()
loo.get_n_splits(X)
for i, (train_index, test_index) in enumerate(loo.split(X)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

# Leave P Out

import numpy as np
from sklearn.model_selection import LeavePOut
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])
lpo = LeavePOut(2)
lpo.get_n_splits(X)
for i, (train_index, test_index) in enumerate(lpo.split(X)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")