# FDR framework
import numpy as np
from numpy import sqrt, log, exp, mean, cumsum, sum, zeros, ones, argsort, argmin, argmax, array, maximum, concatenate
from numpy.random import randn, rand
np.set_printoptions(precision = 4)

import os
import time
#import ipdb
from datetime import datetime
#import StringIO

from generate_mu import*
from parse_new_yorker import*

def get_mu(dist_type, gap, mu_style, hyp_style, pi1, no_arms, num_hyp, sigma, epsilon, top_arms, mu_max):
    
    mu_mat = []
    Hypo = []
    #1: one peak, rest same low
    #2: one high, uniform down
    #3: some same around highest, rest same low.
    #4: some same around highest, rest uniform down

    # Read mu from file, fix NUMHYP
    #if dist_type == 1:
    filename_pre = "D%d_S%d_G%.1f_E%.1f_Si%.1f_TA%d_MM%.1f_" % (dist_type, mu_style, gap, epsilon, sigma, top_arms, mu_max)
    #elif dist_type == 0:
    #filename_pre = "D%d_S%d_G%.1f_E%.1f_Si%.1f_TA%d_MM%.1f" % (dist_type, mu_style, gap*0.1, epsilon, sigma, top_arms, mu_max)
    all_filenames = [filename for filename in os.listdir('./expsettings') if filename.startswith(filename_pre)]
    # CHeck if numhyp of filename is bigger, similarly num_arms
    pos_NH_start = [all_filenames[i].index('NH') for i in range(len(all_filenames))]
    pos_NH_end = [all_filenames[i].index('_NA') for i in range(len(all_filenames))]
    NH_vec = [int(all_filenames[i][pos_NH_start[i] + 2:pos_NH_end[i]]) for i in range(len(all_filenames))]
    nh_indices = np.where(np.array(NH_vec) >= num_hyp)[0]

    pos_NA_start = [all_filenames[i].index('NA') for i in range(len(all_filenames))]
    pos_NA_end = [all_filenames[i].index('.dat') for i in range(len(all_filenames))]
    NA_vec = [int(all_filenames[i][pos_NA_start[i] + 2:pos_NA_end[i]]) for i in range(len(all_filenames))]
    na_indices = np.where(np.array(NA_vec) >= no_arms)[0]
    
    #ipdb.set_trace()
    fs_indices = list(set(nh_indices).intersection(na_indices))
    #ipdb.set_trace()
    if (len(fs_indices) > 0):
        mu_filename = all_filenames[fs_indices[0]]
    #ipdb.set_trace()
        mu_mat = np.loadtxt('./expsettings/%s' % mu_filename)    
        #ipdb.set_trace()
        mu_mat = mu_mat[0:num_hyp, 0:no_arms] 
    else:
        # Create if it doesn't exist
        if (mu_style == 0):
            mu_mat = generate_ny_mu()
            mu_mat = mu_mat[0:num_hyp, 0:no_arms]
        else:
            mu_mat = generate_mu(dist_type, gap, mu_style, mu_max, no_arms, top_arms, num_hyp, epsilon, sigma = 0)
        #ipdb.set_trace()
        
    # Read hyp from file
    filename_pre = "S%d_P%.1f_NH%d_" % (hyp_style, pi1, num_hyp)
    hypo_filename = [filename for filename in os.listdir('./expsettings') if filename.startswith(filename_pre)]
    if len(hypo_filename) > 0:
        hyp_mat = np.loadtxt('./expsettings/%s' % hypo_filename[0])    
    else:
        hyp_mat = generate_hyp(hyp_style, pi1, num_hyp, 10)
        print "Hyp file doesn't exist"

    # Choose some Hypvector 
    Hypo = hyp_mat[0]
    for i in range(num_hyp):
        if Hypo[i] == 1:
            # Randomly sample a control depending on mu_style
            ctrlindx = np.random.choice(range(1,min(top_arms+1, no_arms-1)), 1)      
            # Swapping the two
            mu_mat[i][0], mu_mat[i][ctrlindx] = mu_mat[i][ctrlindx], mu_mat[i][0]
        
    return (mu_mat, Hypo)

