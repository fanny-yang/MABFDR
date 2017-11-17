# wrong FDR

import numpy as np
from numpy import sqrt, log, exp, mean, cumsum, sum, zeros, ones, argsort, argmin, argmax, array, maximum
        
class wrongFDR_proc: 

    tmp = range(1, 10000)
    # discount factor \gamma_m
    gamma_vec =  np.true_divide(log(maximum(tmp,ones(len(tmp))*2)), np.multiply(tmp, exp(sqrt(log(maximum(ones(len(tmp)), tmp))))))
    gamma_vec = gamma_vec / np.float(sum(gamma_vec))    

    def __init__(self, alpha0):
        self.alpha0 = alpha0
        self.w0 = 0.005
        self.wealth_vec = [self.w0]
        self.alpha = [0, self.gamma_vec[0]*self.w0]  # vector of alpha_js
        self.last_rej = 0 # save last time of rejection
        self.t_ctr = 1
        self.first = 0
        self.b0 = 0.045
    
    def next_alpha(self, rej):

        

        # this_alpha = self.alpha[-1] # make sure first one doesn't do bullshit
        
        # # Calc wealth
        # wealth = self.wealth_vec[-1] - this_alpha + rej*self.b0
        # self.wealth_vec.append(wealth)  

        # # Set last rejection time
        # if (rej == 1):
        #     self.last_rej = self.t_ctr    
        
        # self.t_ctr += 1
 
        # Calc new alpha
        next_alpha = self.alpha0
        self.alpha.append(next_alpha)

        return next_alpha
