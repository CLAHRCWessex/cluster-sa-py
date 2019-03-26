#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:31:34 2019

@author: tom
"""

import numpy as np

class SACluster(object):
    
    def __init__(self, n_clusters, dist_metric='corr'):
        '''
        Keyword arguments:
        
        n_clusters --   The number of clusters to generate.  
                        Each observation is assigned to one cluster
        
        dist_metric --  Distance metric. The cost function is the sum of the 
                        group-average distances between group
                        members. The clustering aims to minimise this 
                        cost function. 
                        Options:
                            'corr' -- correlation (default)
                            'euclidean' -- euclidean distance
                            
        TM To - do:
            Not done anything about plot progress...
            
        '''
        self.n_clusters = n_clusters
        self.dist_metruc = dist_metric
    
    def fit(self, data):
        pass

def acceptance_probability(old_energy, new_energy, temp, 
                           delta_energy_scaling=1):
    '''
    
    Calculates the acceptance probability for the SA.
    
    Returns:
        A float representing the probability that the new state is accepted.
    
    Keyword arguments:
        old_energy -- the energy of the current state
        new_energy -- the energy of a proposed state for transition
        temp -- the temperature of the optimisation
        delta_energy_scaling -- to normalise the delta (default = 1)
    '''
    prob = 0.0
    
    if new_energy < old_energy:
        prob = 1.0
    else:
        delta = (new_energy - old_energy) / delta_energy_scaling
        prob = np.exp(-delta / temp)
        
    return prob
    


