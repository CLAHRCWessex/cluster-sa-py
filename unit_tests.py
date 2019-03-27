#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the SA clustering algorithm developed by 
Justin Ashmall and Christina Pagel.  

These tests run with the package pytest 
https://docs.pytest.org/en/latest/

'conda install pytest' or 'pip install pytest'

"""

import pytest

import clustering.sa as csa
from clustering.sa import SACluster, ExponentialCoolingSchedule
import numpy as np

def test_acceptance_probability_100_1():
    '''
    When new_energy > old_energy then it 
    the expected probability = 1.0 regardless of temp
    '''
    old_energy = 10
    new_energy = 9.9
    expected = 1.0
    temp = 100.0
    
    actual = csa.acceptance_probability(old_energy, new_energy, temp)
    
    assert expected == actual
    
    

def test_acceptance_probability_100_2():
    '''
    When new_energy > old_energy then it 
    the expected probability = 1.0 regardless of temp
    
    '''
    old_energy = 10
    new_energy = 9.9
    expected = 1.0
    temp = 0.0
    
    actual = csa.acceptance_probability(old_energy, new_energy, temp)
    
    assert expected == actual
    

def test_acceptance_probability_low_delta():
    '''
    Tests when new_energy >= old_energy
    No normalisation of delta
    '''
    old_energy = 10
    new_energy = 10.1
    temp = 50
    expected = np.exp(-0.1 / 50)
        
    actual = csa.acceptance_probability(old_energy, new_energy, temp)
    
    assert expected == actual
    
    
def test_acceptance_probability_high_delta():
    '''
                   
    Tests when new_energy >= old_energy
    No normalisation of delta
    '''
    old_energy = 10
    new_energy = 1000
    temp = 50
    expected = np.exp(-(new_energy - old_energy) / temp)
        
    actual = csa.acceptance_probability(old_energy, new_energy, temp)
    
    assert expected == actual
    
def test_acceptance_probability_high_delta_2():
    '''
    Tests when new_energy >= old_energy
    No normalisation of delta
    '''
    old_energy = 10
    new_energy = 1000
    temp = 8
    expected = np.exp(-(new_energy - old_energy) / temp)
        
    actual = csa.acceptance_probability(old_energy, new_energy, temp)
    
    assert expected == actual
    
def test_acceptance_probability_high_delta_3():
    '''
    Tests when new_energy >= old_energy
    No normalisation of delta
    '''
    old_energy = 10
    new_energy = 1000
    temp = 100
    expected = np.exp(-(new_energy - old_energy) / temp)
        
    actual = csa.acceptance_probability(old_energy, new_energy, temp)
    
    assert expected == actual
    

def test_acceptance_probability_standardised_delta():
    '''
    Tests when new_energy >= old_energy
    No normalisation of delta
    '''
    old_energy = 10
    new_energy = 1000
    temp = 100
    normaliser = 100
    delta = (new_energy - old_energy) / normaliser
    expected = np.exp(-delta / temp)
        
    actual = csa.acceptance_probability(old_energy, new_energy, temp,
                                        delta_energy_scaling=normaliser)
    
    assert expected == actual

# =============================================================================
# Unit test for CoolingSchedule
# =============================================================================

def test_exp_cooling_schedule():
    '''
    Exponential cool schedule
    ExponentialCoolingSchedule()
    Test different iteration numbers etc.
    '''
    starting_temps = np.arange(100)
    expected = np.full(100, -1)
    actuals = np.full(100, -1)
    ks = np.arange(1, 101)
    
    for i in range(starting_temps.shape[0]):
        schedule = csa.ExponentialCoolingSchedule(starting_temps[i])
        expected[i] = starting_temps[i] * (0.95**ks[i])
        actuals[i] = schedule.cool_temperature(ks[i])
        
    assert np.array_equal(expected, actuals)
    
    
    
def test_coru_cooling_schedule():
    '''
    CORU's custom cooling schedule
    CoruCoolingSchedule
    Test different iteration numbers etc.
    
    Note: what about tests the cause an error?
    '''
    starting_temps = np.arange(100)
    expected = np.full(100, -1)
    actuals = np.full(100, -1)
    ks = np.arange(1, 101)
    max_iter = np.arange(100, 3100, 30)
        
    for i in range(starting_temps.shape[0]):
        schedule = csa.CoruCoolingSchedule(starting_temps[i], max_iter[i])
        expected[i] = np.exp( -(ks[i] - 1) * 10 / max_iter[i])
        actuals[i] = schedule.cool_temperature(ks[i])
        
    assert np.array_equal(expected, actuals)
    

# =============================================================================
# Unit test cost function
# =============================================================================
    
def test_cost_count():
    schedule = csa.ExponentialCoolingSchedule(100)
    sa = SACluster(n_clusters=2, cooling_schedule=schedule, 
                   dist_metric='euclidean')
    #6 observations on 2 variables
    data = np.arange(12).reshape((6, 2))
    state = np.zeros(6)
    state[3:] = 1
    actual_energy, actual_count = sa.cost(state, data)
    expected_count = np.array([3, 3])
    assert np.array_equal(expected_count, actual_count)
    

def test_cost_euclidean():
    '''
    Tests that the cost function calculates
    the weighted cluster euclidean distance correctly
    '''
    
    #cooling schedule selected does not matter for the test
    schedule = ExponentialCoolingSchedule(100)
    sa = SACluster(n_clusters=2, cooling_schedule=schedule, 
                   dist_metric='euclidean')
    
    # 5 observations on 2 variables
    data = np.arange(10).reshape((5, 2))  
    
    #state = [0, 0, 0, 1, 1]
    state = np.zeros(5)
    state[3:] = 1
    
    #calculate energy for state and data
    actual_energy, actual_count = sa.cost(state, data) 
    
    #calculate expected energy based on pairwise euclidean distances
    expected_energy = np.zeros(2)
    expected = 0
    
    for i in range(3):
        for j in range(i, 3):
            expected += euclidean_distance(data[i,:], data[j,:]) 
            
    expected_energy[0] = expected
    expected_energy[1] = euclidean_distance(data[3,:], data[4,:]) 
    
    print('expected {}'.format(expected_energy))

    assert np.array_equal(actual_energy, expected_energy)


def euclidean_distance(city1, city2):
    """
    Calculate euc distance between 2 cities
    5.5 ms to execute
    cities are np.ndarray (vector with 2 points)
    """
    return np.linalg.norm((city1-city2))

# =============================================================================
# Unit test neighbour() and sub functions
# =============================================================================

def test_sample_observation():
    '''
    Test that an observation is sampled correctly 
    from an ordered list of cluster observations
    '''
    #cooling schedule selected does not matter for the test
    schedule = ExponentialCoolingSchedule(100)
    n_clusters = 3
    sa = SACluster(n_clusters=n_clusters, cooling_schedule=schedule)

    state = np.zeros(10)
    state[2:6] = 1
    state[6:] = 2
    
    #for reproducibility
    np.random.seed(seed=101)
    actual_index, actual_value = sa.sample_observation(state)
   
    expected_value = state[actual_index]
    
    assert expected_value == actual_value

def test_generate_neighbour_state():
    '''
    Test that a state is cloned and 
    correct array element is updated 
    '''
    state = np.zeros(10)
    state[2:6] = 1
    state[6:] = 2
    
    exp_state = state.copy()
    
    i_to_change = 3
    new_cluster = 0
        
    exp_state[i_to_change] = new_cluster
    
    schedule = ExponentialCoolingSchedule(100)
    n_clusters = 3
    sa = SACluster(n_clusters=n_clusters, cooling_schedule=schedule)
    actual = sa.generate_neighbour_state(state, i_to_change, new_cluster)
    
    assert np.array_equal(exp_state, actual)

def test_random_cluster_shift_1():
    #cooling schedule selected does not matter for the test
    schedule = ExponentialCoolingSchedule(100)
    n_clusters = 3
    sa = SACluster(n_clusters=n_clusters, cooling_schedule=schedule)
    original_cluster = 0
    
    #control pseudo-random sampling
    np.random.seed(101)
    actual_cluster = sa.random_cluster_shift(original_cluster)
    
    #reset sampling
    np.random.seed(101)
    n_shift = np.random.randint(n_clusters)
    expected = (original_cluster + n_shift - 1) % n_clusters
    
    assert expected == actual_cluster
    

def test_random_cluster_shift_2():
    #cooling schedule selected does not matter for the test
    schedule = ExponentialCoolingSchedule(100)
    n_clusters = 10
    sa = SACluster(n_clusters=n_clusters, cooling_schedule=schedule)
    original_cluster = 3
    
    #control pseudo-random sampling
    np.random.seed(101)
    actual_cluster = sa.random_cluster_shift(original_cluster)
    
    #reset sampling
    np.random.seed(101)
    n_shift = np.random.randint(n_clusters)
    expected = (original_cluster + n_shift - 1) % n_clusters
    
    assert expected == actual_cluster

def test_copy_cluster_metadata_energy():
    #cooling schedule selected does not matter for the test
    schedule = ExponentialCoolingSchedule(100)
    n_clusters = 6
    sa = SACluster(n_clusters=n_clusters, cooling_schedule=schedule)
    
    energy = np.arange(n_clusters)
    count = np.arange(10, 10+n_clusters)
    
    actual_e, actual_c = sa.copy_cluster_metadata(energy, count)
    
    assert np.array_equal(actual_e, energy)
    
    
def test_copy_cluster_metadata_count():
    #cooling schedule selected does not matter for the test
    schedule = ExponentialCoolingSchedule(100)
    n_clusters = 6
    sa = SACluster(n_clusters=n_clusters, cooling_schedule=schedule)
    
    energy = np.arange(n_clusters)
    count = np.arange(10, 10+n_clusters)
    
    actual_e, actual_c = sa.copy_cluster_metadata(energy, count)
    
    assert np.array_equal(actual_c, count)
    


    

