# This is  LORD GAI++ including memory and weight

import numpy as np
from numpy import sqrt, log, exp, mean, cumsum, sum, zeros, ones, argsort, argmin, argmax, array, maximum

class GAI_MW_proc: 

    tmp = range(1, 10000)

    # discount factor \gamma_m
    gamma_vec =  np.true_divide(log(np.maximum(tmp, ones(len(tmp))*2)), np.multiply(tmp, exp(sqrt(log(np.maximum(ones(len(tmp)), tmp))))))
    gamma_vec = gamma_vec / np.float(sum(gamma_vec))    

    def __init__(self, alpha0, mempar):
        self.alpha0 = alpha0
        self.w0 = 0.05*self.alpha0
        self.wealth_vec = [self.w0]
        self.alpha = [0, self.gamma_vec[0]*self.w0]  # vector of alpha_js
        self.last_rej = 0 # save last time of rejection
        self.t_ctr = 0
        self.first = 0
        self.mempar = mempar
    
    def next_alpha(self, rej, pr_w, pen_w):
        
        self.t_ctr += 1

        this_alpha = self.alpha[-1] # make sure first one doesn't do bullshit
        flag = (self.first == 0)
        
        # Calc b, psi, r
        b_t = self.alpha0
        if (rej == 1):
            if flag:
                self.first = 1
                b_t = b_t - np.true_divide(self.w0, pen_w)
            self.last_rej = self.t_ctr
        r_t = np.true_divide(thresh_func(pen_w), pr_w)
        psi_t = min(pen_w*b_t, r_t - pen_w + pen_w*b_t)

        # Calc wealth
        wealth = self.mempar*self.wealth_vec[-1] + (1-mempar)*flag*self.w0 - this_alpha + rej*psi_t
        self.wealth_vec.append(wealth) 
 
            
        # Calc new alpha
        next_alpha = self.gamma_vec[self.t_ctr - self.last_rej]*self.wealth_vec[self.last_rej]
        self.alpha.append(next_alpha)      

        # Return value that you want to test wrt 
        return (np.true_divide(next_alpha/r_t))
    
    # Can add different thresh function
    def thresh_func(x):
        return x
