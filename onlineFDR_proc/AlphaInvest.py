# Alpha Investing as first proposed in FS '07

import numpy as np
import pdb
from numpy import sqrt, log, exp, mean, cumsum, sum, zeros, ones, argsort, argmin, argmax, array, maximum
        

class ALPHA_proc:
    
    def __init__(self, alpha0):
        self.alpha0 = alpha0
        self.b0 = 0.045
        self.wealth_vec = [0.005]
        self.alpha = [0, self.wealth_vec[-1]/2] # preset alpha_1 for usage. note alpha vec always one longer than wealth vec
        self.t_ctr = 1
        self.last_rej = 0

    def next_alpha(self, rej):
        

        this_alpha = self.alpha[-1] # make sure first one doesn't do bullshit
        
        # Calc wealth 
        wealth = self.wealth_vec[-1] - (1-rej)*this_alpha/(1-this_alpha)  + rej*self.b0
        self.wealth_vec.append(wealth)

        # Set last rejection time
        if (rej == 1) :
            self.last_rej = self.t_ctr
        
        self.t_ctr += 1
       
        # Calc new alpha
        next_alpha = wealth/(1 + self.t_ctr - self.last_rej)
        self.alpha.append(next_alpha)
        
        if wealth < 0:
            pdb.set_trace()

        return next_alpha
