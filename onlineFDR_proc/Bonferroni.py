# Bonferroni

import numpy as np
from numpy import sqrt, log, exp, mean, cumsum, sum, zeros, ones, argsort, argmin, argmax, array, maximum
        
class BONF_proc: 

    tmp = range(1, 10000)

    def __init__(self, alpha0):
        self.alpha0 = alpha0
        self.wealth_vec = [self.alpha0]
        self.alpha = [0, self.alpha0/2]  # vector of alpha_js
        self.last_rej = 0 # save last time of rejection
        self.t_ctr = 1
    
    def next_alpha(self, dummy):
              
        this_alpha = self.alpha[-1] # make sure first one doesn't do bullshit
        
        # Calc wealth
        wealth = self.wealth_vec[-1] - this_alpha
        self.wealth_vec.append(wealth)  

        self.t_ctr += 1
 
        # Calc new alpha
        next_alpha = self.alpha0/(2*self.t_ctr**2)
        self.alpha.append(next_alpha)

        return next_alpha
