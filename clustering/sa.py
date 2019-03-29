#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:31:34 2019

@author: tom
"""

from abc import ABC, abstractmethod
from scipy.spatial.distance import pdist, cdist
import numpy as np

class AbstractCoolingSchedule(ABC):
    '''
    Encapsulates a cooling schedule for
    a simulated annealing algorithm.

    Abstract base class.
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
        return np.exp(-(k - 1) * 10 / self.max_iter)


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

        #to-do defensive validation of parameters

        n_observations = data.shape[0]

        #If we go this long without a change then stop.
        #TM notes - why 3000?
        stopping_distance = max(2*n_observations, 3000)

        state_init = np.random.randint(low=0,
                                       high=self.n_clusters,
                                       size=n_observations)

        state = state_init.copy()

        cluster_energy, cluster_count = self.cost(state, data)



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

        for cluster_index in range(self.n_clusters):
            assigned_to = (state == cluster_index)

            cluster_count[cluster_index] = np.count_nonzero(assigned_to)
            cluster_energy[cluster_index] = pdist(data[assigned_to, :],
                                                  self.dist_metric).sum()

        return cluster_energy, cluster_count


    def neighbour(self, state, data, cluster_energies, cluster_counts):
        '''

        Assigns a observation to a random cluster.
        Calculates the change in energy (cost) and number of obs in
        changed clusters

        Returns:
        -----

        Tuple with 3 items

        0. state_new -- updated cluster assignments (the 'neighbour')
        1. new_energies -- updated cluster energies
        2. new_count -- updated counts of energies assigned to clusters

        Dev Notes:
        -----
        Seems like it should be refactored at some point so that
        'state' is a class with meta data energy and count

        Keyword arguments:
        ------
        state -- current cluster assignments by observation
        data -- unlabelled x data
        cluster_energies -- energies (cost) by cluster (ordered by cluster)
        cluster_counts -- count of observations ordered by cluster
        '''
        #sample an observation for state change
        i_to_change, original_cluster_index = self.sample_observation(state)

        #shift the cluster of the sample obs by a random amount
        new_cluster_index = self.random_cluster_shift(original_cluster_index)

        #create neighbour of state with shifted cluster
        state_new = self.generate_neighbour_state(state, i_to_change, 
                                                  new_cluster_index)

        
# =============================================================================
#        Not convinced this bit is a correct port yet.
#        It is tested, but need to think of more tests!
# =============================================================================
        # Find the change in E (cost) for the old cluster
        delta_group_E0_sum = self.delta_cluster_energy(state, data, 
                                                       original_cluster_index, 
                                                       i_to_change)

        
        # Find the change in E (cost) for the new cluster
        delta_group_E1_sum = self.delta_cluster_energy(state, data, 
                                                       new_cluster_index, 
                                                       i_to_change)

        new_energies, new_counts = self.copy_cluster_metadata(cluster_energies,
                                                              cluster_counts)

        #update cluster energies
        new_energies[original_cluster_index] -= delta_group_E0_sum
        new_energies[new_cluster_index] += delta_group_E1_sum;

        #update cluster counts
        new_counts[original_cluster_index] -= 1;
        new_counts[new_cluster_index] += 1;

        return state_new, new_energies, new_counts

    

    def sample_observation(self, state):
        '''
        Sample an individual observation from the state

        Returns
        -----
        Tuple with 2 items
        0. sample_index -- int, the index of the observations within state
        1. state[sample_index] -- int, the cluster number

        Keyword arguments:
        ------
        state -- np.ndarray (vector),
                 current cluster assignments by observation

        '''
        sample_index = np.random.randint(state.shape[0])
        cluster_index = state[sample_index]
        return sample_index, cluster_index


    def random_cluster_shift(self, original_cluster):
        '''

        Shifts a cluster number by a random amount using modulo.

        Assumes that original_cluster is between 0 and self.n_clusters
        (no validation)

        Returns
        ------
        new_cluster -- int, between 0 and self.n_clusters (exclusive)
                       A new cluster number

        Keyword arguments:
        ------
        original_cluster -- int, between 0 and self.n_clusters
                            represents a cluster number

        '''
        n_shift = np.random.randint(self.n_clusters)
        new_cluster = (original_cluster + n_shift - 1) % self.n_clusters
        return new_cluster


    def generate_neighbour_state(self, state, i_to_change, new_cluster):
        '''
        Clones 'state' and updates index 'i_to_change' to 'new_cluster'
        This 'new_state' is a neighbour to state

        Returns
        -----
        state_new -- np.ndarray, a vector of clusters.  The order represents
                     the observations in the unlabelled x's.  Each array
                     element represents the cluster that observation i
                     has been assigned

        Keyword arguments:
        -----
        state -- np.ndarray (vector),
                 current cluster assignments by observation

        i_to_change -- int, index within state to update
        new_cluster -- int, between 0 and self.n_clusters (exclusive)
                       the updated cluster number
        '''
        state_new = state.copy()
        state_new[i_to_change] = new_cluster
        return state_new
    
    
    def delta_cluster_energy(self, state, data, cluster_index, 
                             observation_index):
        '''
        Calculates the amount of energy that an individual 
        observation adds to a cluster (refered to as delta)
        
        Returns:
        -------
        Float, representing the amount of energy an observation
        adds to a specifed cluster
        
        Keyword arguments:
        -------
        state -- np.ndarray (vector),
                 current cluster assignments by observation
                 
        data -- unlabelled x values
        
        cluster_index -- int, unique identifier for the cluster 
    
        observation_index -- int, index within data to analyse
        
        
        '''
        #boolean index of all observations assigned to cluster
        assigned_to_cluster = (state == cluster_index) 
        
        #note: cdist is equivalent of matlab pdist2
        delta_energy = cdist(np.array([data[observation_index, :]]),
                                   data[assigned_to_cluster, : ],
                                   self.dist_metric).sum()
        

        return delta_energy


    def copy_cluster_metadata(self, cluster_energies, cluster_counts):
        '''
        Used to duplicate numpy arrays containing cluster energies
        and counts

        Returns
        ------
        A copy of the cluster energies (costs) np.ndarray (vector) and
        cluster counts (number of obs assigned to each cluster) np.ndarray (vector)

        Keyword arguments:
        -------

        cluster_energies -- np.ndarray (vector), ordered cluster energies
        cluster_counts -- np.ndarray (vector), ordered cluster counts

        '''
        return cluster_energies.copy(), cluster_counts.copy()



def acceptance_probability(old_energy, new_energy, temp,
                           delta_energy_scaling=1):
    '''
    Calculates the acceptance probability for the SA.

    Returns:
    ------

        A float representing the probability that the new state is accepted.

    Keyword arguments:
    -----
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



    
    




