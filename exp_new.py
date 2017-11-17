# FDR framework
import numpy as np
from numpy import sqrt, log, exp, mean, cumsum, sum, zeros, ones, argsort, argmin, argmax, array, maximum, concatenate
from numpy.random import randn, rand
np.set_printoptions(precision = 4)

import os
#import matplotlib.pyplot as plt
import scipy.optimize as optim
from scipy.stats import norm
from scipy.stats import bernoulli
#import ipdb
import time
from datetime import datetime
#import StringIO

# Import FDR procedures
import onlineFDR_proc.Lord as Lord
import onlineFDR_proc.GAIPlus as GAIPlus
import onlineFDR_proc.AlphaInvest as AlphaInvest
import onlineFDR_proc.Bonferroni as Bonferroni

# import 
import rowexp_new
from generate_mu import*
from importme import *
# To read mus
import parse_mu


################ Saving and plotting framework ###############

def saveres(direc, filename, mat, ext = 'dat', verbose = True):
    filename = "%s.%s" % (filename, ext)
    if not os.path.exists(direc):
        os.makedirs(direc)
    savepath = os.path.join(direc, filename)
    np.savetxt(savepath, mat, fmt='%.7e', delimiter ='\t')
    if verbose:
        print("Saving results to %s" % savepath)
 
################ Running entire framework  ####################

def run_single(dist_type, gap, mu_style, hyp_style, pi1, no_arms, num_hyp, sigma, epsilon, top_arms, alpha0, trunctimerange, FDR, NUMRUN, mu_max, alg_num = 0, punif = 0, cauchyn = 0, verbose = 1, precision = 1e-8):
    # --------- Explanation for arguments: ----
    # Sigma, epsilon are used for generating mu - sigma is the wiggle and top arms are eps away from each other
    # Top arms: # of top arms that are around epsilon from each other 
    # ------------------------------------------


    #%%%%%%%%%%%%%%%% Initializations %%%%%%%%%%%%%%%%%
    # Initialize results file (for verbose output, useful for debugging per run)
    time_str = datetime.today().strftime("%m%d%y_%H%M")
    if not os.path.exists('./results'):
        os.makedirs('./results')
    res_filename = './results/output_%s.dat' % time_str
    result_file = open(res_filename, 'w')

    # Initialize variables
    alg_name = alg_list[alg_num]
    numtrunc = len(trunctimerange)
    # Load mu_mat with ready to go mu
    (mu_mat, Hypo) = parse_mu.get_mu(dist_type, gap, mu_style, hyp_style, pi1, no_arms, num_hyp, sigma, epsilon, top_arms, mu_max)
    num_alt = sum(Hypo)
    if dist_type == 1:
        bound_type = 'SubGaussian_LIL'
    elif dist_type == 0:
        bound_type = 'Bernoulli_LIL'

    # Initializing all arrays that save results per experiment (run, hypothesis and within hyp time)  
    pval_mat = np.zeros(shape=(numtrunc,num_hyp, NUMRUN))
    rej_mat = np.zeros(shape=(numtrunc,num_hyp, NUMRUN))
    samples_mat = np.zeros(shape=(numtrunc,num_hyp, NUMRUN))
    alpha_mat = np.zeros(shape=(numtrunc,num_hyp, NUMRUN))
    wealth_mat = np.zeros(shape=(numtrunc, num_hyp, NUMRUN))
    FDR_tsr = np.zeros(shape=(numtrunc, num_hyp, NUMRUN))
    falrej_vec = np.zeros([numtrunc,NUMRUN])
    correj_vec = np.zeros([numtrunc,NUMRUN])
    samples_vec = np.zeros([numtrunc,NUMRUN])
    totrej_vec = np.zeros([numtrunc,NUMRUN])
    rightarm_vec = np.zeros([numtrunc,NUMRUN])
    pval_vec = np.zeros(num_hyp)

    #%%%%%%%%%%%%% Differnt runs of experiments %%%%%%%%%%%%%%%%%%%
    
    for l in range(NUMRUN):

        # ---- Initialize FDR procedures, all first values are to be tossed (non-used, including alpha)
        if FDR == 0:
            proc = GAIPlus.GAI_proc(alpha0)
        elif FDR == 1:
            proc = Lord.LORD_proc(alpha0)
        elif FDR == 2:
            proc = GAI_MW.GAI_MW_proc(alpha0)
        elif FDR == 3:
            proc = wrongFDR.wrongFDR_proc(alpha0)
        elif FDR == 4:
            proc = AlphaInvest.ALPHA_proc(alpha0)
        # dummy wrong FDR, always giving same alpha0 or some other constant
        elif FDR == 5:
            proc = Bonferroni.BONF_proc(alpha0)

        tic = time.time()

        #%%%%%%%%%%%%%%% Run MAB for each hypothesis %%%%%%%%%%%%%%%%
        for i in range(num_hyp):
            
            # Get means for the corresponding hypothesis/MAB
            mu_list = mu_mat[i]
            this_exp = rowexp_new.rowexp(Hypo[i], no_arms, 1,  mu_list, 0, 1)

            if verbose:
                result_file.write("Run: %d\n" % l)
                #result_file.write(mu_list)

            # --------- Draw exp if possibly alg-dependent exp                
            this_alpha = proc.alpha[-1]
            this_exp.multi_ab(this_alpha, trunctimerange, epsilon, bound_type, alg_name, 1, cauchyn, punif, verbose = 0,  precision = precision)
            rightarm_b = this_exp.rightarm  # boolean: Whether MAB found best arm
            bestarm_idx = this_exp.bestarm['index'] #
                
            # -------  Compute values and all for different truncation times 
            for q, trunctime in enumerate(trunctimerange):
                ## Get P value at current time
                # pval_vec[i] = this_exp.pval[q]
                ## Get the min over all times till time q
                # pval_mat[q][i][l] = min(this_exp.pval[0:q+1])
                pval_mat[q][i][l] = this_exp.pval[q]
                total_samples = this_exp.total_queries[q]
                
                samples_mat[q][i][l] = total_samples

                # If wealth still positive for that procedure
                if (proc.wealth_vec[-1] >= 0):
                    # Reject
                    rej_mat[q][i][l] = (pval_mat[q][i][l] <= this_alpha + precision)

                    # Total measures
                    falrej_vec[q][l] = falrej_vec[q][l] + rej_mat[q][i][l]*(1-Hypo[i])
                    correj_vec[q][l] = correj_vec[q][l] + rej_mat[q][i][l]*Hypo[i]
                    rightarm_vec[q][l] = rightarm_vec[q][l] + (Hypo[i])*rightarm_b*rej_mat[q][i][l]
                    samples_vec[q][l] = samples_vec[q][l] + total_samples
                    totrej_vec[q][l] = falrej_vec[q][l] + correj_vec[q][l]
                    
                    FDR_tsr[q][i][l] = np.true_divide(falrej_vec[q][l], max(totrej_vec[q][l],1))
                    #ipdb.set_trace()
                    if verbose:
                        result_file.write("true best: %d, found best: %d, queries: %d \n" % (argmax(mu_list), bestarm_idx, total_samples))
                        result_file.write("alpha_j: %f, p_j: %f, rej: %d \n" % (this_alpha, pval_mat[q][i][l], rej_mat[q][i][l]))
            if (proc.wealth_vec[-1] >= 0):
                    # Get next alpha (and wealth) from FDR if theres a next hypothesis to test
                    if i < num_hyp - 1:
                        wealth_mat[q][i][l] = proc.wealth_vec[-1]
                        alpha_mat[q][i][l] = proc.next_alpha(rej_mat[q][i][l]) # use last rejection
        
    if verbose: 
        result_file.write("Time for one complete experiment with %d hypotheses was %f" % (num_hyp, time.time() - tic))


    # Save data
    dir_name = './dat'

    for q, trunctime in enumerate(trunctimerange):    

        #ipdb.set_trace()
        FDR_vec = np.true_divide(falrej_vec[q], [max(totrej_vec[q][l],1) for l in range(len(totrej_vec[q]))])
        TDR_vec = np.true_divide(correj_vec[q], num_alt)
        BDR_vec = np.true_divide(rightarm_vec[q], num_alt)
        FDR_mat = FDR_tsr[q]

        pr_filename = 'PR_D%d_MS%d_AG%d_G%.1f_MM%.1f_E%.1f_Si%.1f_TA%d_HS%d_P%.1f_FDR%d_NH%d_NA%d_TT%d_PU%d_CN%d_NR%d_%s' % (dist_type, mu_style, alg_num, gap, mu_max, epsilon, sigma, top_arms, hyp_style, pi1, FDR, num_hyp, no_arms, trunctime, punif, cauchyn, NUMRUN, time_str)
        ad_filename = 'AD_D%d_MS%d_AG%d_G%.1f_MM%.1f_E%.1f_Si%.1f_TA%d_HS%d_P%.1f_FDR%d_NH%d_NA%d_TT%d_PU%d_CD%d_NR%d_%s' % (dist_type, mu_style, alg_num, gap, mu_max, epsilon, sigma, top_arms, hyp_style, pi1, FDR, num_hyp, no_arms, trunctime, punif, cauchyn, NUMRUN, time_str)
        td_filename = 'TD_D%d_MS%d_AG%d_G%.1f_MM%.1f_E%.1f_Si%.1f_TA%d_HS%d_P%.1f_FDR%d_NH%d_NA%d_TT%d_PU%d_CN%d_NR%d_%s' % (dist_type, mu_style, alg_num, gap, mu_max, epsilon, sigma, top_arms, hyp_style, pi1, FDR, num_hyp, no_arms, trunctime, punif, cauchyn, NUMRUN, time_str)

        # Save data
        saveres(dir_name, td_filename, FDR_mat)
        saveres(dir_name, ad_filename, [BDR_vec, TDR_vec, FDR_vec, samples_vec[q], falrej_vec[q], totrej_vec[q]])
        saveres(dir_name, pr_filename, np.r_[rej_mat[q], pval_mat[q], alpha_mat[q], wealth_mat[q], samples_mat[q]])
    
    result_file.close()
