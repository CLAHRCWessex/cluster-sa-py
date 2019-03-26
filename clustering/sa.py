#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:31:34 2019

@author: tom
"""

import numpy as np
from abc import ABC, abstractmethod

class AbstractCoolingSchedule(ABC):
    '''
    Encapsulates a cooling schedule for 
    a simulated annealing algorithm.
    
    Abstract class.  
    '''
    def __init__(self, starting_temp):
        self.starting_temp = starting_temp
    
    @abstractmethod
    def cool_temperature(self, k):
        pass
    

class ExponentialCoolingSchedule(AbstractCoolingSchedule):
    '''
    Expenontial Cooling Scheme.
    
    Source:     
    https://uk.mathworks.com/help/gads/how-simulated-annealing-works.html
    '''
    def cool_temperature(self, k):
        '''
        Cool the temperature using the scheme
        
        T = T0 * 0.95^k.
        
        Where
        T = temperature after cooling
        T0 = starting temperature
        k = iteration number (within temperature?)
        
        Keyword arguments:
        k -- int, iteration number (within temp?)
        '''
        return self.starting_temp * (0.95**k)
    

class CoruCoolingSchedule(AbstractCoolingSchedule):
    
    def __init__(self, starting_temp, max_iter):
        AbstractCoolingSchedule.__init__(self, starting_temp)
        self.max_iter = max_iter
    
    def cool_temperature(self, k):
        '''
        Returns a temperature from 1 tending to 0
        
        Keyword arguments:
            iteration --int. the iteration number
    et about the 5% of the time he wants to use Windows to access legacy model
        '''
        
        #where did this cooling scheme come from?
        #some standard methods: 
        # https://uk.mathworks.com/help/gads/how-simulated-annealing-works.html
        return np.exp( -(k - 1) * 10 / self.iter_max)
    


class SACluster(object):
    
    def __init__(self, n_clusters, dist_metric='corr', 
                 max_iter=np.float32(1e5)):
        '''
        Keyword arguments:
        
        n_clusters --   int: The number of clusters to generate.  
                        Each observation is assigned to one cluster
        
        dist_metric --  Distance metric. The cost function is the sum of the 
                        group-average distances between group
                        members. The clustering aims to minimise this 
                        cost function. 
                        Options:
                            'corr' -- correlation (default)
                            'euclidean' -- euclidean distance
                            
        max_iter --    Int, Optional, default = le5
                       The maximum number of iterations for the SA
                       
                            
        TM To - do:
            Not done anything about plot progress...
            
        '''
        self.n_clusters = n_clusters
        self.dist_metric = dist_metric
        self.max_iter = max_iter
        
    def fit(self, data):
        '''
        Keyword arguments:
            data -- numpy.ndarray of unlabelled x data.  Each row is a 
                    observation and each column an x variable
        '''
        
        n_observations = data.shape[0]
        
        #If we go this long without a change then stop.
        #TM notes - why 3000?
        stopping_distance = max(2*n_observations, 3000) 
        
    
        



'''
function [group_E, grp_count] = cost_func(state, dat, ngroups, distance_metric)
% cs = categories(state);
for i=1:ngroups
    tf = state==i; %cs(i);
    grp_count(i) = nnz(tf);
    all_dist =  pdist( dat(tf,:), distance_metric );
    group_E(i) = sum( all_dist );
end
% diffs = all_dist-all_dist(randperm(numel(all_dist)));
% diff_stdev = stdev(diffs(diffs>0));
end
'''

def cost(state, data, n_clusters, dist_metric):
    #for i in range(1, n_clusters):
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



    


