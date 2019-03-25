#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:31:34 2019

@author: tom
"""

import numpy as np

'''


function P = acceptance_probability(E, Enew, T, delta_E_scaling)
if Enew < E
    P = 1;
else
    delta = (Enew-E)/delta_E_scaling;
    P = exp(-delta/T);
    % fprintf('Delta: %g. T=%g. Delta/T=%g. P=%g\n', delta, T, delta/T, P);
end
end % acceptance_probability

'''

def acceptance_probability(energy, energy_new, temp, delta_energy_scaling):
    '''
    
    Calculates the acceptance probability for the SA.
    
    Returns:
        A float representing the probability that the new state is accepted.
    
    Keyword arguments:
        energy -- the energy of the current state
        energy_new -- the energy of a proposed state for transition
        temp -- the temperature of the optimisation
        delta_energy_scaling -- ? 
    '''
    prob = 0.0
    
    if energy_new < energy:
        prob = 1.0
    else:
        delta = (energy_new - energy) / delta_energy_scaling
        prob = np.exp(-delta / temp)
        
    return prob
    


