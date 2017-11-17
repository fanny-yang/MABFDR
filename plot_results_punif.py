# FDR framework
import numpy as np
from numpy import sqrt, log, exp, mean, cumsum, sum, zeros, ones, argsort, argmin, argmax, array, maximum, concatenate
from numpy.random import randn, rand
np.set_printoptions(precision = 4)

import os
#import matplotlib
# To run on remote machine
#matplotlib.use('agg')
#import matplotlib.pyplot as plt

import scipy.optimize as optim
from scipy.stats import norm
from scipy.stats import bernoulli
import time
from datetime import datetime
import StringIO

# import 
import rowexp_new
# To read mus
import parse_mu
from plotting import*


# ATTENTION USER Only choose num_hyp, top_arms, sigma, epsilon pi1 gap,  that have been created!
def plot_results_punif(truncrange, noarmsrange, algnumrange, dist_type, gap, mu_style, hyp_style, pi1range,  num_hyp, sigma, epsilon, top_arms, FDRrange, mu_max, alpha0, plot_numrun = 5, punif = 0, cauchyn = 0, halt = 0, plot_start=0):
    
    if len(FDRrange) == 1:
        FDR= FDRrange[0]
    if len(noarmsrange) == 1:
        no_arms = noarmsrange[0]
    # Sample complexity vs. no arms for fixed trunc time, different algo
    NUMPLOT = len(algnumrange)
    plot_dirname = './plots'
    
    # Power plots
    # For given MS, NA, HS ... different trunctime different algo

    
        ################## Plot pseudo FDR over time #########################
    if (punif == 1) & (len(pi1range) == 1):
        
        
        filename_pre = 'TD_D%d_MS%d_AG%d_G%.1f_MM%.1f_E%.1f_Si%.1f_TA%d_HS%d_P%.1f_AL%.1f_FDR%d_NH%d_NA%d_TT%d_PU%d_' %  (dist_type, mu_style, algnumrange[0], gap, mu_max, epsilon, sigma, top_arms, hyp_style, pi1range[0], alpha0, FDRrange[0], num_hyp, noarmsrange[0], truncrange[0], punif)
        all_filenames = [filename for filename in os.listdir('./dat') if filename.startswith(filename_pre)]

        # Take the first, take FDR mat
        xs = range(num_hyp)[plot_start:num_hyp]
        FDR_mat = np.loadtxt('./dat/%s' % all_filenames[0])
        plot_numrun = min(len(FDR_mat[0]),plot_numrun)

        ys_mat = np.zeros([plot_numrun, num_hyp-plot_start])

        for run in range(plot_numrun):
            ys = np.array(FDR_mat[plot_start:num_hyp,run])
            ys_mat[run] = ys

        filename= 'FDPvsHYP_D%d_MS%d_AG%d_G%.1f_MM%.1f_E%.1f_Si%.1f_TA%d_HS%d_P%.1f_AL%.1f_FDR%d_NH%d_NA%d_TT%d_PU%d' %  (dist_type, mu_style, algnumrange[0], gap, mu_max, epsilon, sigma, top_arms, hyp_style, pi1range[0], alpha0, FDRrange[0], num_hyp, noarmsrange[0], truncrange[0], punif)
        plotsingle_shaded_mat(xs, ys_mat, plot_dirname, filename,  'Hypothesis index', 'FDP($J$)') 
    
        ################## Plot FDR over pi1 ################################
    elif (punif == 1) & (len(pi1range) > 1):
        for j, FDR in enumerate(FDRrange):
            min_nr = 10000
            # Find all files with the right settings
            filename_pre = 'AD_D%d_MS%d_AG%d_G%.1f_MM%.1f_E%.1f_Si%.1f_TA%d_HS%d_' % (dist_type, mu_style, algnumrange[0], gap, mu_max, epsilon, sigma, top_arms, hyp_style)
            all_filenames0 = [filename for filename in os.listdir('./dat') if filename.startswith(filename_pre)]
            filename_post = 'AL%.1f_FDR%d_NH%d_NA%d_TT%d_PU%d_' %  (alpha0, FDR, num_hyp, noarmsrange[0], truncrange[0], punif)
               
            all_filenames = [s for s in all_filenames0 if filename_post in s]
            
            # Read out all possible  #arms
            pos_P_start = [all_filenames[i].index('_P') for i in range(len(all_filenames))]
            pos_P_end = [all_filenames[i].index('_AL') for i in range(len(all_filenames))]
            P_vec = [float(all_filenames[i][pos_P_start[i] + 2:pos_P_end[i]]) for i in range(len(all_filenames))]            
            order = np.argsort(P_vec)
            # Get distinct NAs, then merge
            P_list = sorted(set(np.array(P_vec)[order]))
            #all_filenames = np.array(all_filenames)[order]     
            if j == 0:
                BDR_av = np.zeros([len(FDRrange), len(P_list)])
                BDR_std = np.zeros([len(FDRrange), len(P_list)])
                samples_av = np.zeros([len(FDRrange), len(P_list)])
                samples_std = np.zeros([len(FDRrange), len(P_list)])
                FDR_av = np.zeros([len(FDRrange), len(P_list)])
                FDR_std = np.zeros([len(FDRrange), len(P_list)])             
                mFDR_av = np.zeros([len(FDRrange), len(P_list)])
                mFDR_std = np.zeros([len(FDRrange), len(P_list)])


            # Merge everything with the same NA and NH
            for k, P in enumerate(P_list):
                indices = np.where(np.array(P_vec) == P)[0]
                result_mat = []
                
                # Load resultmats and append 
                for m, idx in enumerate(indices):
                    result_mat_cache = np.loadtxt('./dat/%s' % all_filenames[idx])
                    if (m == 0):
                        result_mat = result_mat_cache
                    else:
                        result_mat = np.c_[result_mat, result_mat_cache]

                # print len(result_mat[0])
                numrun = len(result_mat[0])
                min_nr = min(min_nr, numrun)
                # Get first vector for BDR
                BDR_vec = result_mat[0]
                BDR_av[j][k] = np.average(BDR_vec)
                BDR_std[j][k] = np.true_divide(np.std(BDR_vec),np.sqrt(numrun))
                # Get last vector for samples
                samples_vec = result_mat[3]
                samples_av[j][k] = np.average(samples_vec)
                samples_std[j][k] = np.true_divide(np.std(samples_vec), np.sqrt(numrun))
                # FDR
                FDR_vec = result_mat[2]
                FDR_av[j][k] = np.average(FDR_vec)
                FDR_std[j][k] = np.true_divide(np.std(FDR_vec), np.sqrt(numrun))

                # for mFDR
                mFDR_num_vec = result_mat[4]
                mFDR_denom_vec = result_mat[5]
                mFDR_av[j][k] = np.true_divide(np.average(mFDR_num_vec), np.average(mFDR_denom_vec) +1)
                mFDR_std[j][k] = 0


        xs = P_list
        # ##### FDR vs. pi1 ####

        filename = 'FDRvsPi_D%d_MS%d_G%.1f_MM%.1f_E%.1f_Si%.1f_TA%d_HS%d_AL%.1f_NH%d_TT%d_NA%d_NR%d' % (dist_type, mu_style, gap, mu_max, epsilon, sigma, top_arms, hyp_style, alpha0, num_hyp, truncrange[0], no_arms, min_nr)
        plot_errors_mat(xs, FDR_av, FDR_std, [proc_list[i] for i in FDRrange], plot_dirname, filename,'Proportion of alternatives $\pi_1$', 'FDR', plots_ind = 2)

 
        ########### mFDR vs. pi1 ##########
        filename = 'mFDRvsPi_D%d_MS%d_G%.1f_MM%.1f_E%.1f_Si%.1f_TA%d_HS%d_AL%.1f_NH%d_TT%d_NA%d_NR%d' % (dist_type, mu_style, gap, mu_max, epsilon, sigma, top_arms, hyp_style, alpha0, num_hyp, truncrange[0],no_arms, min_nr)
        plot_errors_mat(xs, mFDR_av, mFDR_std, [proc_list[i] for i in FDRrange], plot_dirname, filename,'Proportion of alternatives $\pi_1$', 'mFDR', plots_ind = 2)

     
    
    
