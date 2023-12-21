
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Poly_dataSet.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values


# Fitting RF Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
lin_reg = RandomForestRegressor(n_estimators=10,random_state =0)
lin_reg.fit(X, y)

# Visualising the RF Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the RF Regression results

x_grid=np.arange(min(X),max(X), 0.1)
x_grid =x_grid.reshape(len(x_grid),1)
plt.scatter(X, y, color = 'red')
plt.plot(x_grid, lin_reg.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (DT Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Prdict by RF Regression
lin_reg.predict(np.reshape(6.5,(1,1)))

