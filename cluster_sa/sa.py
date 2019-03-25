#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:31:34 2019

@author: tom
"""

import numpy as np


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
    


