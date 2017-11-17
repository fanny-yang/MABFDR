
import numpy as np
from numpy import sqrt, log, exp, mean, cumsum, sum, zeros, ones, argsort, argmin, argmax, array, maximum, concatenate, ceil
from numpy.random import randn, rand, choice
np.set_printoptions(precision = 4)
from scipy.stats import norm, bernoulli, cauchy
# from scipy.stats import bernoulli
#import ipdb
import time

# Import Best arm procedures
import confidence_bounds
import LUCB
import UniformSampling


nounif = 2 # Make sure it's even or it'll break!


def bin_search(a, b, f, MAX, precision = 1e-8):
    # If for max the bound b is already unsatisfied, don't try
    # MAX indicates whether want to find max or min that satisfies f
    if MAX:
        if f(b):
            return 1
        else:
            while b-a > precision:
                pivot = (a+b)/2.
                if f(pivot):
                    a = pivot
                else:
                    b = pivot
    else:
        while b-a > precision:
            pivot = (a+b)/2.
            if f(pivot):
                b = pivot
            else:
                a = pivot
    return pivot

    
'''
Model for row experiment
'''

class rowexp:

    def __init__(self,  ALT, no_arms, k, mu_list, epsilon = 0,  sigma = 1): 
        # n number of arms, ALT: whether alternativ is true
        # epsilon: sigma: of Gaussian distribution,
        self.no_arms = no_arms
        self.ALT = ALT
        self.k = k # might not need it?
        self.mu_list = mu_list
        
        '''
        P-val giving funct: multiple testing/MAB case
        '''

    def get_results(self, i, alg, trunctime):

        # ------------ Calculate p-value -------- #
        # Get control arm
        controlarm = alg.arms[0]
        bound_type = self.bound_type
        # Save best arm
        all = alg.arms
        self.bestarm = sorted(all, key = lambda y: -y['mu_hat'])[0]
        if (alg.total_queries < trunctime) & (alg.controlbest == True):
            self.bestarm = controlarm

        # Get pval, the min p is the one calculated from best arm vs. control
        # By binary search
        # If the best arm is actually the control arm, the p-value is expected to be 1
        
        if alg.improved: 
            unionb_l = 2.*(alg.n-alg.k)
        else:
            unionb_l = alg.n
        if alg.improved: 
            unionb_u = 2.*alg.k
        else:
            unionb_u = alg.n
        cb = confidence_bounds.ConfidenceBound(bound_type)
        f = lambda gamma: cb.lower(self.bestarm['mu_hat'], gamma/unionb_l, self.bestarm['T']) < cb.upper(controlarm['mu_hat'], gamma/unionb_u, controlarm['T'])

        # Get max gamma which satisfies the constraint
        P = bin_search(10**(-10), 1., f, 1)
        self.rightarm = (self.bestarm['index'] == argmax(self.mu_list))
        self.pval[i] = P
        self.total_queries[i] = alg.total_queries
        

    def multi_ab(self, confalpha, trunctimerange, epsilon = 0, bound_type = 'SubGaussian_LIL', alg_name = 'LUCB', sigma = 1, cauchyn = 0, punif = 0, verbose = 0, improved = 1, precision = 1e-8, control_threshold = float('inf')): # trunctime = max no of pulls
        self.bound_type = bound_type
        
        trunctime = max(trunctimerange)
        self.pval = np.zeros(len(trunctimerange))
        self.total_queries = np.zeros(len(trunctimerange))
        tt_counter = 0
        
        if verbose:
            print (self.mu_list)
        # --------- Initialize best arm algorithm to find it---------- #
        if alg_name == 'LUCB_eps':
            alg = LUCB.LUCB(self.no_arms, 1, confalpha, epsilon, bound_type, improved = True, extra_rules = 1)
        elif alg_name == 'MAB-LORD':
            alg = LUCB.LUCB(self.no_arms, 1, confalpha, epsilon, bound_type, improved = False, extra_rules = 0)
        elif alg_name == 'AB-LORD':
            alg = UniformSampling.Uniform(self.no_arms, 1, confalpha, epsilon, bound_type)
        
        alg_start_ts = time.time()
        

        tic = time.time()
        
        # -----  Run algorithm till stopping rule or truncation time --------#
        # Draw all arms nounif times uniformly
        for i in range(nounif):
            for j in range(self.no_arms):
                idx = j
                alg.mu_hat_decreasing.remove(alg.arms[idx])
                alg.ucb_decreasing.remove(alg.arms[idx])

                # Pull arm, i.e. get a {Gaussian, Bernoulli} sample
                if 'Gaussian' in bound_type:
                        X = self.mu_list[idx] + sigma*randn()
                elif 'Bernoulli' in bound_type:
                        X = float(rand()<self.mu_list[idx])
                # Update arm info (empirical mu, LCB, UCB) for pulled arm
                alg.report_answer(idx, X, 0) #verbose) 
        alg.total_queries = nounif*self.no_arms

        timegetquery = 0
        timereportans = 0

        # Draw adaptively              
        while not (alg.should_stop(stop_threshold=control_threshold)) | (alg.total_queries > trunctime-1):
            # Get index to pull in this round
            tic0 = time.time()
            idx = alg.get_query()
            timegetquery = timegetquery + time.time() - tic0

            
            # To prevent another pull when stopping criterion is already met
            if alg._should_stop == 0:

                # Pull arm, i.e. get a {Gaussian, Bernoulli} sample
                if 'Gaussian' in bound_type:
                    if (self.ALT == 1) & (cauchyn == 1):
                        X = self.mu_list[idx] + cauchy.rvs(loc=0, scale=sigma) #sigma*randn()
                    else:
                        X = self.mu_list[idx] + sigma*randn()
                elif 'Bernoulli' in bound_type:
                        X = float(rand()<self.mu_list[idx])
                tic1 = time.time()
                # Update arm info (empirical mu, LCB, UCB) for pulled arm
                alg.report_answer(idx, X, 0) #verbose) # Verbose if you want to ese LCB and UCB at every step precisely
                timereportans = timereportans + time.time() - tic1
            if (alg.total_queries > trunctimerange[tt_counter]):
                self.get_results(tt_counter, alg, trunctimerange[tt_counter])
                tt_counter = tt_counter + 1
        if verbose:    
            print("Time on bandit only was %f " % (time.time() -tic))
 
        self.get_results(tt_counter, alg, trunctimerange[tt_counter])
