import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


samples = 1000

mu = [1.0, 1.0]
covm = [[2.0, 0.0], [0.0, 0.5]]
X = np.random.multivariate_normal(mean=mu, cov=covm, size=samples)
plt.scatter(X[:,0],X[:,1])
plt.title('Original')
plt.show()

std = StandardScaler()
X_std = std.fit_transform(X)
plt.scatter(X_std[:,0],X_std[:,1])
plt.title('Standard')
plt.show()

rob = RobustScaler()
X_rob = rob.fit_transform(X)
plt.scatter(X_rob[:,0],X_rob[:,1])
plt.title('Robust')
plt.show()
print("Robust is better when you have OUTLIERS")

mm = MinMaxScaler()
X_mm = mm.fit_transform(X)
plt.scatter(X_mm[:,0],X_mm[:,1])
plt.title('MinMax')
plt.show()
