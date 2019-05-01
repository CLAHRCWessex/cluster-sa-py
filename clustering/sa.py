#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes and functions that support unsupervised clustering
of data using Simulated Annealing (SA) based algorithms
"""

from abc import ABC, abstractmethod
import time
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
        ------
        k -- int, iteration number (within temp?)

        Returns:
        ------
        float, new temperature
        '''
        return self.starting_temp * (0.95**k)


class CoruCoolingSchedule(AbstractCoolingSchedule):

    def __init__(self, starting_temp, max_iter):
        AbstractCoolingSchedule.__init__(self, starting_temp)
        self._max_iter = max_iter

    def cool_temperature(self, k):
        '''
        Returns a temperature from 1 tending to 0

        Keyword arguments:
        ------
        iteration --int. the iteration number

        Returns:
        ------
        float, new temperature
        '''

        #where did this cooling scheme come from?
        #some standard methods:
        # https://uk.mathworks.com/help/gads/how-simulated-annealing-works.html
        t =  np.exp(-(k - 1) * 10 / self._max_iter)
        return t


class SACluster(object):
    '''
    Encapsulates a simulated annealing clustering algorithm
    
    Public methods:
        
    fit() -- runs the SA and fits data to clusters
    '''

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
        self._n_clusters = n_clusters
        self._dist_metric = dist_metric
        self._max_iter = max_iter
        self._cooling_schedule = cooling_schedule
        self._search_history = np.empty(self._max_iter, dtype=np.float64)
        
    @property
    def search_history(self):
        '''
        Get the search history (energy levels)
        of the SA.

        Returns
        ------
        np.ndarray containing the energy level 
        of the solution at each iteration of the algorithm
        '''
        return self._search_history


    def fit(self, data):
        '''
        Keyword arguments:
        -----
        data -- numpy.ndarray of unlabelled x data.  Each row is a
                observation and each column an x variable

        Returns:
        ------
        Tuple
        0: state
        1: energy of clusters
        2: search history (to remove after debugging)
        '''

        #to-do defensive validation of parameters

        n_observations = data.shape[0]

        #TM notes: this isn't quite right as already setting this in class
        #constructor
        # MaxIter - CP changed to 150 because it seems to be enough looking at the output
        #TM note - if this varies then we should encapsulate its calculation
        #self._max_iter = max(150 * n_observations, 10000)
        

        #If we go this long without a change then stop.
        #TM notes - why 3000?
        stopping_distance = max(2 * n_observations, 3000)

        state_init = np.random.randint(low=0,
                                       high=self._n_clusters,
                                       size=n_observations)

        state = state_init.copy()

        cluster_energy, cluster_count = self._cost(state, data)

        t0 = time.time() - 3 #why -3?
        n_changes = 0
        delta_sum = 0
        delta_sum_n = 0
        Einit = np.divide(cluster_energy, cluster_count).sum()
        Emin = Einit
        state_min = state_init.copy()
        delta_E_scaling = 1
        #Es = np.zeros(self._max_iter)
        last_change_i = 1


        #why only a single loop for each temp?
        #standard SA has multiple iterations at each temperature...
        for i in range(self._max_iter):
            T = self._cooling_schedule.cool_temperature(i)
            state_new, new_energy, new_count = self._neighbour(state,
                                                               data,
                                                               cluster_energy,
                                                               cluster_count) 
            
            E = np.divide(cluster_energy, cluster_count).sum()
            
            
            Enew = np.divide(new_energy, new_count).sum()

            if Enew > E:
                # Get a running total of delta E so we can calculate 
                # an average and use that to scale the acceptance
                # probability
                delta_sum += (Enew - E)
                delta_sum_n += 1
                
                #mean positive value
                delta_E_scaling = delta_sum / delta_sum_n
                
            P = acceptance_probability(E, Enew, T, delta_E_scaling)
                            
            if P == 1 or P > np.random.rand():
                    
                last_change_i = i
                #TM Note: should be okay but just
                #check if need to use state_new.copy()
                state = state_new.copy()
                E = Enew 
                cluster_energy = new_energy.copy()
                cluster_count = new_count.copy()
                n_changes += 1

                
                if Enew < Emin:
                    Emin = Enew
                    state_min = state_new.copy()
            
            self._search_history[i] = E
                
                #Justin's plot code goes here...
            
            if  i == self._max_iter or i > last_change_i + stopping_distance:
                msg = 'Iter {0}/{1} {2}. Group changes={3}. E={4} E/Einit={5}.'
                msg += ' E/Emin={6}\n'
                
                print(msg.format(i, self._max_iter, 100*i/self._max_iter, 
                                 n_changes, E, E/Einit, E/Emin))
            
            
            # Automatically stop if nothing has changed for 2*nobvs iterations
            if  i > last_change_i + stopping_distance:
                msg = 'STOPPING @ iteration {0}. No change for {1} iterations...\n'
                print(msg.format(i, stopping_distance))
                break
            
            #end of iter loop
            
        if not np.array_equal(state_min, state):
            print('INFO: Returning state_min not current state...')
            state = state_min
            E = Emin
            
            msg = 'Iter {0}/{1} {2}. Group changes={3}. E={4} E/Einit={5}.'
            msg += ' E/Emin={6}\n'
                
            print(msg.format(i, self._max_iter, 100*i/self._max_iter, 
                             n_changes, E, E/Einit, E/Emin))


        return state, E


    def _cost(self, state, data):
        '''
        Calculate the energy of the whole solution.
        

        Keyword arguments:
        ------
        state -- numpy.ndarray (vector), each index corresponds to an
                    observation each value is a int between 0 and
                    self._n_clusters (exclusive) that indicates to which
                    cluster the observation has been assigned

        data -- numpy.ndarray (matrix), unlabelled x data

        Returns:
        -----
        Tuple:
        0: nd.array (vector), cluster energy (cost) 
        1: nd.array (vector), count of points in each cluster
        '''

        cluster_count = np.zeros(self._n_clusters, dtype=np.int64)
        cluster_energy = np.zeros(self._n_clusters, dtype=np.float64)

        for cluster_index in range(self._n_clusters):
            assigned_to = (state == cluster_index)

            cluster_count[cluster_index] = np.count_nonzero(assigned_to)
            cluster_energy[cluster_index] = pdist(data[assigned_to, :],
                                                  self._dist_metric).sum()

        return cluster_energy, cluster_count


    def _neighbour(self, state, data, cluster_energies, cluster_counts):
        '''

        Assigns a observation to a random cluster.
        Calculates the change in energy (cost) and number of obs in
        changed clusters

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

        Returns:
        -----

        Tuple with 3 items

        0. state_new -- updated cluster assignments (the 'neighbour')
        1. new_energies -- updated cluster energies
        2. new_count -- updated counts of energies assigned to clusters
        '''
        #sample an observation for state change
        i_to_change, original_cluster_index = self._sample_observation(state)

        #shift the cluster of the sample obs by a random amount
        new_cluster_index = self._random_cluster_shift(original_cluster_index)

        #create neighbour of state with shifted cluster
        state_new = self._generate_neighbour_state(state, i_to_change,
                                                   new_cluster_index)


# =============================================================================
#        Not convinced this bit is a correct port yet.
#        It is tested, but need to think of more tests!
# =============================================================================
        # Find the change in E (cost) for the old cluster
        delta_group_E0_sum = self._delta_cluster_energy(state, data,
                                                        original_cluster_index,
                                                        i_to_change)


        # Find the change in E (cost) for the new cluster
        delta_group_E1_sum = self._delta_cluster_energy(state, data,
                                                        new_cluster_index,
                                                        i_to_change)

        new_energies, new_counts = self._copy_cluster_metadata(cluster_energies,
                                                               cluster_counts)

        #update cluster energies
        new_energies[original_cluster_index] -= delta_group_E0_sum
        new_energies[new_cluster_index] += delta_group_E1_sum

        #update cluster counts
        new_counts[original_cluster_index] -= 1
        new_counts[new_cluster_index] += 1

        return state_new, new_energies, new_counts



    def _sample_observation(self, state):
        '''
        Sample an individual observation from the state

        Keyword arguments:
        ------
        state -- np.ndarray (vector),
                 current cluster assignments by observation

        Returns
        -----
        Tuple with 2 items
        0. sample_index -- int, the index of the observations within state
        1. state[sample_index] -- int, the cluster number

        '''
        sample_index = np.random.randint(state.shape[0])
        cluster_index = state[sample_index]
        return sample_index, cluster_index


    def _random_cluster_shift(self, original_cluster):
        '''

        Shifts a cluster number by a random amount using modulo.

        Assumes that original_cluster is between 0 and self._n_clusters
        (no validation)

        Keyword arguments:
        ------
        original_cluster -- int, between 0 and self._n_clusters
                            represents a cluster number

        Returns
        ------
        new_cluster -- int, between 0 and self._n_clusters (exclusive)
                       A new cluster number
        '''
        n_shift = np.random.randint(self._n_clusters)
        new_cluster = (original_cluster + n_shift - 1) % self._n_clusters
        return new_cluster


    def _generate_neighbour_state(self, state, i_to_change, new_cluster):
        '''
        Clones 'state' and updates index 'i_to_change' to 'new_cluster'
        This 'new_state' is a neighbour to state

        Keyword arguments:
        -----
        state -- np.ndarray (vector),
                 current cluster assignments by observation

        i_to_change -- int, index within state to update
        new_cluster -- int, between 0 and self._n_clusters (exclusive)
                       the updated cluster number

        Returns
        -----
        state_new -- np.ndarray, a vector of clusters.  The order represents
                     the observations in the unlabelled x's.  Each array
                     element represents the cluster that observation i
                     has been assigned
        '''
        state_new = state.copy()
        state_new[i_to_change] = new_cluster
        return state_new

    
    def _delta_cluster_energy(self, state, data, cluster_index,
                             observation_index):
        '''
        Calculates the amount of energy that an individual
        observation adds to a cluster (refered to as delta)

        Keyword arguments:
        -------
        state -- np.ndarray (vector),
                 current cluster assignments by observation

        data -- unlabelled x values

        cluster_index -- int, unique identifier for the cluster

        observation_index -- int, index within data to analyse

        Returns:
        -------
        Float, representing the amount of energy an observation
        adds to a specifed cluster


        '''
        #boolean index of all observations assigned to cluster
        assigned_to_cluster = (state == cluster_index)

        #note: cdist is equivalent of matlab pdist2
        delta_energy = cdist(np.array([data[observation_index, :]]),
                             data[assigned_to_cluster, : ],
                             self._dist_metric).sum()


        return delta_energy


    def _copy_cluster_metadata(self, cluster_energies, cluster_counts):
        '''
        Used to duplicate numpy arrays containing cluster energies
        and counts

        Keyword arguments:
        -------

        cluster_energies -- np.ndarray (vector), ordered cluster energies
        cluster_counts -- np.ndarray (vector), ordered cluster counts       
        
        Returns
        ------
        A copy of the cluster energies (costs) np.ndarray (vector) and
        cluster counts (number of obs assigned to each cluster) np.ndarray
        (vector)

        '''
        return cluster_energies.copy(), cluster_counts.copy()



def acceptance_probability(old_energy, new_energy, temp,
                           delta_energy_scaling=1):
    '''
    Calculates the acceptance probability for the SA.

    Keyword arguments:
    -----
    old_energy -- the energy of the current state
    new_energy -- the energy of a proposed state for transition
    temp -- the temperature of the optimisation
    delta_energy_scaling -- to normalise the delta (default = 1)   
        
    Returns:
    ------

    A float representing the probability that the new state is accepted.
    '''
    prob = 0.0

    if new_energy < old_energy:
        prob = 1.0
    else:
        delta = (new_energy - old_energy) / delta_energy_scaling
        prob = np.exp(-delta / temp)

    return prob


