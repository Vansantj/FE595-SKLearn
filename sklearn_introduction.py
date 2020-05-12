# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:56:26 2020

@author: jvan1
"""


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

def bostonCoefficient():
    data = datasets.load_boston()
    x = data.data
    y = data.target
    lr = LinearRegression().fit(x,y)
    max_coefficient  = max([abs(x) for x in lr.coef_])
    name = data.feature_names[[abs(x) for x in lr.coef_].index(max_coefficient)]
    return '%s is the feature with the largest coefficient which is %f'%(name,max_coefficient)

def irisCluster():
    x_values = []
    y_values = []
    data = datasets.load_iris()
    for i in range(1,20):
        curr_k = KMeans(i).fit(data.data,data.target)
        x_values.append(i)
        y_values.append(curr_k.inertia_)
    plt.plot(x_values,y_values)
    plt.xticks(x_values)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Heuristics For Iris Dataset')
    plt.show()

if __name__=='__main__':
    print(bostonCoefficient())
    irisCluster()
    print('The Iris data set looks to have 3 clusters as the slope of inertia massively changes in magnitude between 3 and 4 clusters showing that predictive power is not helped much by adding additional clusters after the third')