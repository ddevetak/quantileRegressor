import numpy as np
import pandas as pd

class quantileRegressor:

    def __init__(self, n_quantiles):
        self.n_quantiles = n_quantiles
        self.subsets = []
        self.Data = None
        self.meanY = []
        self.predictY = []

    def fit(self, X, y):
             
       tempData = dict(zip(X, y))
       xq = pd.qcut(X, self.n_quantiles)
       
       for interval in xq.categories:
            xx = [i for i in X if i in interval]
            self.subsets.append(xx)
       
       for cc in self.subsets:
            self.meanY.append(np.mean([tempData[i] for i in cc]))

       self.Data = dict(list(zip(self.meanY, self.subsets)))

    def predict(self, X): # x is array

       for i in X:
           for xy in self.Data:
               if self.Data[xy][0] <= i <= self.Data[xy][-1]: 
                   self.predictY.append(xy)  # xy - key
                   break

       return np.array(self.predictY)
