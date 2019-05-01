'''
Replica of Jupyter notebook - useful for debugging SA code.
'''

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sa import SACluster, ExponentialCoolingSchedule, CoruCoolingSchedule

if __name__ == '__main__':
    df = pd.read_csv('data/Mall_Customers.csv')
    unlabelled_x = df.iloc[:, [3, 4]].values
    n_clusters = 5
    max_iter = max(150 * unlabelled_x.shape[0], 10000)
    cooling_schedule = CoruCoolingSchedule(1000, max_iter=max_iter)
    #cooling_schedule = ExponentialCoolingSchedule(1000)
    sa = SACluster(n_clusters=n_clusters, cooling_schedule=cooling_schedule,
                dist_metric='euclidean')

    state, energy, search_history = sa.fit(unlabelled_x)

    print(pd.DataFrame(search_history))