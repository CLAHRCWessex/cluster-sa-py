#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:31:34 2019

@author: tom
"""

import numpy as np
from scipy.spatial.distance import pdist
from abc import ABC, abstractmethod

class AbstractCoolingSchedule(ABC):
    '''
    Encapsulates a cooling schedule for 
    a simulated annealing algorithm.
    
    Abstract class.
    Concrete implementations can be 
    customised to extend the range of
    cooling schedules available to the SA.
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
    
        '''
        
        #where did this cooling scheme come from?
        #some standard methods: 
        # https://uk.mathworks.com/help/gads/how-simulated-annealing-works.html
        return np.exp( -(k - 1) * 10 / self.max_iter)
    


class SACluster(object):
    
    def __init__(self, n_clusters, cooling_schedule, dist_metric='correlation', 
                 max_iter=np.float32(1e5)):
        '''
        
        Constructor method.
        
        Keyword arguments:
        
        n_clusters --   int: The number of clusters to generate.  
                        Each observation is assigned to one cluster
                        
        cooling_schedule -- AbstractCoolingSchedule: The cooling schedule
                            that is the annealing employs.  Implemented
                            as a concrete instance of AbstractCoolingSchedule
                            Options include:
                                ExponentialCoolingSchedule
                                CoruCoolingSchedule
        
        dist_metric --  Distance metric. The cost function is the sum of the 
                        group-average distances between group
                        members. The clustering aims to minimise this 
                        cost function. 
                        Options:
                            'correlation' -- correlation (default)
                            'euclidean' -- euclidean distance
                            ...others...
                            
        max_iter --    Int, Optional, default = le5
                       The maximum number of iterations for the SA
                       
        
        
        TM TO-DO:
            Not done anything about plot progress...
            
        '''
        self.n_clusters = n_clusters
        self.dist_metric = dist_metric
        self.max_iter = max_iter
        self.cooling_schedule = cooling_schedule
        
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
        
        state_init = np.random.randint(low=0, 
                                       high=self.n_clusters, 
                                       size=n_observations)
        
        state = state_init.copy()
        #call cost function        

    def cost(self, state, data):
        '''
        Calculate the energy of the solution
        
        Keyword arguments:
            state -- numpy.ndarray (vector), each index corresponds to an 
                     observation each value is a int between 0 and 
                     self.n_clusters (exclusive) that indicates to which 
                     cluster the observation has been assigned 
            
            data -- numpy.ndarray (matrix), unlabelled x data
        '''
    
        cluster_count = np.zeros(self.n_clusters, dtype=np.int64)
        cluster_energy = np.zeros(self.n_clusters, dtype=np.float64)
        
        for cluster_index in range(1, self.n_clusters):
            assigned_to = (state == cluster_index)
            cluster_count[cluster_index] = np.count_nonzero(assigned_to)
            cluster_energy[cluster_index] = pdist(data[assigned_to, :], 
                                            self.dist_metric).sum()
        
        
        return cluster_energy, cluster_count
        
    
   
        






        
    


   

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



    


