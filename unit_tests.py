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
    

