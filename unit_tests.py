#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:48:01 2019

@author: tom
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


