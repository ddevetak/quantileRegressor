

import numpy as np
import matplotlib.pyplot as plt
from quantileRegressor import quantileRegressor

# train data
x = np.linspace(-np.pi, np.pi, 50)
y = x**2 + 2*np.sin(2*x) + 1

# test data
x_test = np.linspace(-np.pi, np.pi, 30)

# fit and predict
model = quantileRegressor(20)
model.fit(x, y)
pred_y = model.predict(x_test)

plt.scatter(x, y)
plt.plot(x_test, pred_y, 'r', lw = 1)
plt.show()





