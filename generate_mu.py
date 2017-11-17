# FDR framework
import numpy as np
from numpy import sqrt, log, exp, mean, cumsum, sum, zeros, ones, argsort, argmin, argmax, array, maximum, concatenate
from numpy.random import randn, rand
np.set_printoptions(precision = 4)

import os
import scipy.optimize as optim
from scipy.stats import norm
from scipy.stats import bernoulli
import time
from datetime import datetime
import StringIO


def saveres(direc, filename, mat, ext = 'dat', verbose = True):
    filename = "%s.%s" % (filename, ext)
    if not os.path.exists(direc):
        os.makedirs(direc)
    savepath = os.path.join(direc, filename)
    np.savetxt(savepath, mat, fmt='%.3e', delimiter ='\t')
    if verbose:
        print("Saving results to %s" % savepath)

# Generate mu files depending on Gaussian Bernoulli

def generate_mu(dist_type, gap, mu_style, mu_max, max_arms, top_arms, max_hyp, epsilon, sigma = 0):
    # Generate file name
    dirname = './expsettings'
    filename = 'D%d_S%d_G%.1f_E%.1f_Si%.1f_TA%d_MM%.1f_NH%d_NA%d' % (dist_type, mu_style, gap, epsilon, sigma, top_arms, mu_max, max_hyp, max_arms)
        
        
    mu_mat = np.zeros([max_hyp, max_arms])
        
    #1: one peak, rest same low (with sigma noise)
    #2: one high, uniform down
    #3: some same around highest (epsilon), rest same low.
    #4: some same around highest, rest uniform down

    for i in range(max_hyp):
            if mu_style == 1:
                top_mu = [mu_max]
                rest_mu = randn(max_arms - 1)*gap*sigma + mu_max - gap
            elif mu_style == 2:
                top_mu = [mu_max]
                rest_mu = (mu_max - gap)*rand(max_arms - 1)
            elif mu_style == 3:
                # was *epsilon*gap but I don't remember why makes sense
                top_mu = randn(top_arms)*epsilon + mu_max*np.ones(top_arms)
                rest_mu = (mu_max - gap) + randn(max_arms - top_arms)*sigma*gap
            elif mu_style == 4:
                top_mu = randn(top_arms)*epsilon + mu_max*np.ones(top_arms)
                rest_mu = (mu_max - gap)*rand(max_arms - top_arms)

            if dist_type == 0:
                rest_mu = [max(min(x,0.99),0.01) for x in rest_mu]
            mu_list = np.sort(concatenate((top_mu, rest_mu)))
            mu_mat[i] = mu_list[::-1]
            #print mu_mat[i]
    

    
    # Save in file
    saveres(dirname, filename, mu_mat)
    return mu_mat

def generate_hyp(hyp_style, pi1, max_hyp, samples):
    
    hyp_mat = np.zeros([samples, max_hyp])
    num_alt = np.int(np.ceil(max_hyp*pi1))
    prob_lin = np.true_divide(range(max_hyp),np.sum(range(max_hyp)))
    prob = concatenate(( 200*np.ones(int(np.ceil(max_hyp/2.))), 2*np.ones(int(np.floor(max_hyp/2.))) ))
    prob_step = np.true_divide(prob,np.sum(prob))
    # 0: uniform across num_hyp
    # 1: many alt at beginning - lin prob
     #2: many alt at end - lin prob
    # 3: many alt at beginning - step down
    # 4: many alt at beginning - step up
   
    for i in range(samples):
        hyp_row = np.zeros(max_hyp)
        if hyp_style == 0:
            #hyp_row = bernoulli.rvs(pi1, size = max_hyp)
            # Or with fixed length, random assignment of indices 
            indices_alt = np.random.choice(max_hyp, num_alt)
        elif hyp_style == 1:
            #ipdb.set_trace()
            indices_alt = np.random.choice(max_hyp, num_alt, replace = False, p= prob_lin[::-1])
        elif hyp_style == 2:
            indices_alt = np.random.choice(max_hyp, num_alt, replace = False, p= prob_lin)
        elif hyp_style == 3:
            indices_alt = np.random.choice(max_hyp, num_alt, replace = False, p= prob_step)
        elif hyp_style == 4:
            indices_alt = np.random.choice(max_hyp, num_alt, replace = False, p= prob_step[::-1])
        hyp_row[indices_alt] = np.ones(num_alt)
        hyp_mat[i] = hyp_row
        
        #print indices_alt
        #print prob_step
        #print hyp_row
        #ipdb.set_trace()
    
    # Save mat
    dirname = './expsettings'
    filename = "S%d_P%.1f_NH%d_NS%d" % (hyp_style, pi1, max_hyp, samples)
    saveres(dirname, filename, hyp_mat)
    return hyp_mat
        
        
def create_mu(gaprange, toparmrange,  dist_type, ms_range, mu_max, max_arms, max_hyp, sigmarange = [0], eps_range = [0]):
    #idiotensicherung
    toparmrange = [min(x, max_arms) for x in toparmrange]
    gap_mult = 1
    
    # Create data        
    for mu_style in ms_range:
        for gap in gaprange:
            for epsilon in eps_range:
                for top_arms in toparmrange:
                    for sigma in sigmarange:
                        if dist_type == 0:
                            for i_mu_max in np.arange(0.1,0.9,0.3):
                                generate_mu(dist_type, gap, mu_style, i_mu_max, max_arms, top_arms, max_hyp, epsilon, sigma)
                    # For Gaussians, mu_max doesn't matter
                        else:
                            generate_mu(dist_type, gap, mu_style, mu_max, max_arms, top_arms, max_hyp, epsilon, sigma)

def create_hyp(hyprange = [10,50,100,500,1000]):
    for max_hyp in hyprange:
        for hyp_style in range(1,5):
            for pi1 in np.arange(0.1,0.9,0.1): 
                generate_hyp(hyp_style, pi1, max_hyp, 100)
                               
        
        
        
