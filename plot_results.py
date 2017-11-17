# FDR framework
import numpy as np
from numpy import sqrt, log, exp, mean, cumsum, sum, zeros, ones, argsort, argmin, argmax, array, maximum, concatenate
from numpy.random import randn, rand
np.set_printoptions(precision = 4)

import os

import matplotlib as mpl

import matplotlib.pyplot as plt
#plt.rcParams['font.family'] = 'Times New Roman'

import scipy.optimize as optim
from scipy.stats import norm
from scipy.stats import bernoulli
import pdb
import ipdb
import time
from datetime import datetime
import StringIO

# Import FDR procedures
import onlineFDR_proc.Lord as Lord
import onlineFDR_proc.GAIPlus as GAIPlus
import onlineFDR_proc.AlphaInvest as AlphaInvest
import onlineFDR_proc.Bonferroni as Bonferroni


# import 
import rowexp_new
import parse_mu
from plotting import*
from importme import *


# ATTENTION USER Only choose num_hyp, top_arms, sigma, epsilon pi1 gap,  that have been created!
def plot_results(truncrange, noarmsrange, algnumrange, dist_type, gap, mu_style, hyp_style, pi1,  num_hyp, sigma, epsilon, top_arms, FDR, mu_max, punif = 0, cauchyn = 0, halt = 0, NA_range=[0]):
    
    # Sample complexity vs. no arms for fixed trunc time, different algo
    NUMPLOT = len(algnumrange)
    plot_dirname = './plots'
    # Power plots
    # For given MS, NA, HS ... different trunctime different algo
    # Find all possible trunctimes available for this setting

    #%%%%%%%%%%%%%%%%%%%%  PLOTS vs. truncation time %%%%%%%%%%%%%%%%%%%%%%
    if len(truncrange) > 1:
        # ----------- LOAD DATA --------
        for algnum in algnumrange:
            filename_pre = 'AD_D%d_MS%d_AG%d_G%.1f_MM%.1f_E%.1f_Si%.1f_TA%d_HS%d_P%.1f_FDR%d_NH%d_NA%d_' % (dist_type, mu_style, algnum, gap, mu_max, epsilon, sigma, top_arms, hyp_style, pi1, FDR, num_hyp, noarmsrange[0])
            all_filenames = [filename for filename in os.listdir('./dat') if filename.startswith(filename_pre)]
            
            
            # Read out all possible truncation times
            pos_TT_start = [all_filenames[i].index('TT') for i in range(len(all_filenames))]
            pos_TT_end = [all_filenames[i].index('_PU') for i in range(len(all_filenames))]
            TT_vec = [int(all_filenames[i][pos_TT_start[i] + 2:pos_TT_end[i]]) for i in range(len(all_filenames))]

            order = np.argsort(TT_vec)
            # Get distinct NAs, then merge
            TT_list = sorted(set(np.array(TT_vec)[order]))
        
            if algnum == 0:
                BDR_av = np.zeros([len(algnumrange), len(TT_list)])
                BDR_std = np.zeros([len(algnumrange), len(TT_list)])
                samples_av = np.zeros([len(algnumrange), len(TT_list)])
                samples_std = np.zeros([len(algnumrange), len(TT_list)])
                FDR_av = np.zeros([len(algnumrange), len(TT_list)])
                FDR_std = np.zeros([len(algnumrange), len(TT_list)])             
                mFDR_av = np.zeros([len(algnumrange), len(TT_list)])
                mFDR_std = np.zeros([len(algnumrange), len(TT_list)])
                
            # Only plot the ones in truncrange (TO BE IMPLEMENTED)
            
            # Merge everything with the same NA and NH
            for k, TT in enumerate(TT_list):
                indices = np.where(np.array(TT_vec) == TT)[0]
                result_mat = []
                # Load resultmats and append 
                for j, idx in enumerate(indices):
                    result_mat_cache = np.loadtxt('./dat/%s' % all_filenames[idx])
                    if (j == 0):
                        result_mat = result_mat_cache
                    else:
                        result_mat = np.c_[result_mat, result_mat_cache]
                
                numrun = len(result_mat[0])
                # Get first vector for BDR
                BDR_vec = result_mat[0]
                BDR_av[algnum][k] = np.average(BDR_vec)
                BDR_std[algnum][k] = np.true_divide(np.std(BDR_vec),np.sqrt(numrun))
                # Get last vector for samples
                samples_vec = result_mat[3]
                samples_av[algnum][k] = np.average(samples_vec)
                samples_std[algnum][k] = np.true_divide(np.std(samples_vec), np.sqrt(numrun))
                # FDR
                FDR_vec = result_mat[2]
                FDR_av[algnum][k] = np.average(FDR_vec)
                FDR_std[algnum][k] = np.true_divide(np.std(FDR_vec), np.sqrt(numrun))

                # mFDR
                mFDR_num_vec = result_mat[4]
                mFDR_denom_vec = result_mat[5]
                mFDR_av[algnum][k] = np.true_divide(np.average(mFDR_num_vec), np.average(mFDR_denom_vec) +1)
                mFDR_std[algnum][k] = np.true_divide(max(np.std(mFDR_num_vec), np.std(mFDR_denom_vec)), np.sqrt(numrun))
   
        if (halt == 1):
            ipdb.set_trace()

        # -------- PLOT ---------------
        
        #xs = np.array(TT_vec)[order]
        if dist_type == 0:
            xs = [x/1000 for x in TT_list]
            xtrunc_lbl = 'Truncation time $T_S /1000$'
        else:
            xs = TT_list
            xtrunc_lbl = 'Truncation time $T_S$'
       

        ##### BDR vs. trunc #####
        filename = 'BDRvsTT_D%d_MS%d_G%.1f_E%.1f_Si%.1f_TA%d_HS%d_P%.1f_FDR%d_NH%d_NA%d' % (dist_type, mu_style, gap, epsilon, sigma, top_arms, hyp_style, pi1, FDR, num_hyp, noarmsrange[0])
        plot_errors_mat(xs, BDR_av, BDR_std, alg_list, plot_dirname, filename, xtrunc_lbl, 'BDR')

  
        ##### Samples vs. trunc ####
        filename = 'SPSvsTT_D%d_MS%d_G%.1f_E%.1f_Si%.1f_TA%d_HS%d_P%.1f_FDR%d_NH%d_NA%d' % (dist_type, mu_style, gap, epsilon, sigma, top_arms, hyp_style, pi1, FDR, num_hyp, noarmsrange[0])
        plot_errors_mat(xs, samples_av/1000., samples_std/1000., alg_list, plot_dirname, filename, xtrunc_lbl, 'Total number of samples $/1000$')
 

        ##### FDR vs. trunc ####
        filename = 'FDRvsTT_D%d_MS%d_G%.1f_MM%.1f_E%.1f_Si%.1f_TA%d_HS%d_P%.1f_FDR%d_NH%d_NA%d' % (dist_type, mu_style, gap, mu_max, epsilon, sigma, top_arms, hyp_style, pi1, FDR, num_hyp, noarmsrange[0])
        plot_errors_mat(xs, FDR_av, FDR_std, alg_list, plot_dirname, filename, xtrunc_lbl, 'FDR')

        #### mFDR vs. trunc ####
        filename = 'mFDRvsTT_D%d_MS%d_G%.1f_MM%.1f_E%.1f_Si%.1f_TA%d_HS%d_P%.1f_FDR%d_NH%d_NA%d' % (dist_type, mu_style, gap, mu_max, epsilon, sigma, top_arms, hyp_style, pi1, FDR, num_hyp, noarmsrange[0])
        plot_errors_mat(xs, mFDR_av, mFDR_std, alg_list, plot_dirname, filename, xtrunc_lbl, 'mFDR')



    #%%%%%%%%%%%%%%%%%%%  PLOTS vs. no.arms %%%%%%%%%%%%%%%%%%%%%%%%%%
    elif len(noarmsrange) > 1: 
        # ---------- LOAD DATA --------------
        for algnum in algnumrange:
            filename_pre = 'AD_D%d_MS%d_AG%d_G%.1f_MM%.1f_E%.1f_Si%.1f_TA%d_HS%d_P%.1f_FDR%d_NH%d_' %  (dist_type, mu_style, algnum, gap, mu_max, epsilon, sigma, top_arms, hyp_style, pi1, FDR, num_hyp) 
            all_filenames = [filename for filename in os.listdir('./dat') if filename.startswith(filename_pre)]
            
            if (len(all_filenames) == 0):
                ipdb.set_trace()
                print "well didn't find files"
                return True
            # Get ones with a particular trunctime
            # Read out all possible truncation times
            pos_TT_start = [all_filenames[i].index('TT') for i in range(len(all_filenames))]
            pos_TT_end = [all_filenames[i].index('_PU') for i in range(len(all_filenames))]
            TT_vec = [int(all_filenames[i][pos_TT_start[i] + 2:pos_TT_end[i]]) for i in range(len(all_filenames))]
            # Pick only the ones with right trunctime
            TT_ind = np.where(np.array(TT_vec) == truncrange[0])[0]
            all_filenames = np.array(all_filenames)[TT_ind]
            # Get number of arms

            # Read out all possible  #arms
            pos_NA_start = [all_filenames[i].index('NA') for i in range(len(all_filenames))]
            pos_NA_end = [all_filenames[i].index('_TT') for i in range(len(all_filenames))]
            NA_vec = [int(all_filenames[i][pos_NA_start[i] + 2:pos_NA_end[i]]) for i in range(len(all_filenames))]            

            order = np.argsort(NA_vec)
            if (len(NA_range)>1):
                NA_list = sorted(set(np.array(NA_vec)[order]).intersection(NA_range))
            else:
            # Get distinct NAs, then merge
                NA_list = sorted(set(np.array(NA_vec)[order]))

            # if halt == 0 could be different sizes!
            if (algnum == 0) & (halt == 0):
                BDR_av = np.zeros([len(algnumrange), len(NA_list)])
                BDR_std = np.zeros([len(algnumrange), len(NA_list)])
                samples_av = np.zeros([len(algnumrange), len(NA_list)])
                samples_std = np.zeros([len(algnumrange), len(NA_list)])
                FDR_av = np.zeros([len(algnumrange), len(NA_list)])
                FDR_std = np.zeros([len(algnumrange), len(NA_list)])
            elif (algnum==0) & (halt == 1):
                BDR_av = [None]*len(algnumrange)
                BDR_std = [None]*len(algnumrange)
                FDR_av = [None]*len(algnumrange)
                FDR_std = [None]*len(algnumrange)
                samples_av = [None]*len(algnumrange)
                samples_std = [None]*len(algnumrange)
        

            if (halt == 1):   
                BDR_av[algnum] = np.zeros(len(NA_list))
                BDR_std[algnum] = np.zeros(len(NA_list))
                FDR_av[algnum] = np.zeros(len(NA_list))
                FDR_std[algnum] = np.zeros(len(NA_list))
                samples_av[algnum] = np.zeros(len(NA_list))
                samples_std[algnum] = np.zeros(len(NA_list))
         
            
            # Merge everything with the same NA and NH
            for k, NA in enumerate(NA_list):
                indices = np.where(np.array(NA_vec) == NA)[0]
                result_mat = []
                # Load resultmats and append 
                for j, idx in enumerate(indices):
                    result_mat_cache = np.loadtxt('./dat/%s' % all_filenames[idx])
                    if (j == 0):
                        result_mat = result_mat_cache
                    else:
                        result_mat = np.c_[result_mat, result_mat_cache]
        
                numrun = len(result_mat[0])
                # Get first vector for BDR
                BDR_vec = result_mat[0]
                BDR_av[algnum][k] = np.average(BDR_vec)
                BDR_std[algnum][k] = np.true_divide(np.std(BDR_vec), np.sqrt(numrun))
                # Get last vector for samples
                samples_vec = result_mat[3]
                samples_av[algnum][k] = np.average(samples_vec)
                samples_std[algnum][k] = np.true_divide(np.std(samples_vec), np.sqrt(numrun))
                # FDR
                FDR_vec = result_mat[2]
                FDR_av[algnum][k] = np.average(FDR_vec)
                FDR_std[algnum][k] = np.true_divide(np.std(FDR_vec), np.sqrt(numrun))

            
        if (halt == 1):
            ipdb.set_trace()
            # stop program here
            return True
        # -------- PLOT ---------------
        xs = NA_list

        ##### BDR vs. NA #####
        filename = 'BDRvsNA_D%d_MS%d_G%.1f_MM%.1f_E%.1f_Si%.1f_TA%d_HS%d_P%.1f_FDR%d_NH%d_TT%d' % (dist_type, mu_style, gap, mu_max, epsilon, sigma, top_arms, hyp_style, pi1, FDR, num_hyp, truncrange[0])
        plot_errors_mat(xs, BDR_av, BDR_std, alg_list, plot_dirname, filename, 'Number of arms', 'BDR')


        ##### Samples vs. NA ####
        filename = 'SPSvsNA_D%d_MS%d_G%.1f_MM%.1f_E%.1f_Si%.1f_TA%d_HS%d_P%.1f_FDR%d_NH%d_TT%d' % (dist_type, mu_style, gap, mu_max, epsilon, sigma, top_arms, hyp_style, pi1, FDR, num_hyp, truncrange[0])
        plot_errors_mat(xs, samples_av/1000., samples_std/1000., alg_list, plot_dirname, filename, 'Number of arms', 'Total number of samples $/1000$')

        ##### FDR vs. NAx ####
        filename = 'FDRvsNA_D%d_MS%d_G%.1f_MM%.1f_E%.1f_Si%.1f_TA%d_HS%d_P%.1f_FDR%d_NH%d_TT%d' % (dist_type, mu_style, gap, mu_max, epsilon, sigma, top_arms, hyp_style, pi1, FDR, num_hyp, truncrange[0])
        plot_errors_mat(xs, FDR_av, FDR_std, alg_list, plot_dirname, filename, 'Number of arms', 'FDR')

