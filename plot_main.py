# FDR framework
import numpy as np
np.set_printoptions(precision = 4)

import os
import logging, argparse
import time
import itertools
from datetime import datetime
import StringIO
import ipdb

# import 
import exp_new
from importme import *
from ipyparallel import Client
from plot_results import *
from plot_results_punif import *

def main():

    # Get arguments
    ny_data = args.ny_data
    mu_style = args.mu_style
    dist_style = args.dist_style

    numrunrange = [10]

    #### Set experimental settings in paper
    pi1 = 0.4
    hyp_style = 1
    top_arms = 10
    alpha0 = 0.1 # generally 0.1
    FDRrange = [0]
    algrange = [0,1] # both 0: best-arm MAB and 1: Uniform sampling
    FDR = 0
    # Power or FDR plots
    if args.power_plot == 1:
        punif = 0
    else:
        punif = 1

    if (dist_style == 1):
        mu_gap = 3
        mu_best = 8
        num_hyp = 500
        # fix_na = [50]
        # fix_tt = [300]
        # armrange = np.arange(10, 121, 10) 
        # truncrange = np.arange(100, 801, 100)
        armrange = np.arange(10, 41, 10) 
        truncrange = np.arange(100, 301, 100)
        fix_na = [30]
        fix_tt = [300]
        if punif == 1:
            armrange = [30]
            algrange = [0]
            truncrange = [200]
            pi1range = np.arange(0.1,1,0.1)
            FDRrange = [0, 3, 5]
            num_hyp = 1000
            fix_pi1 = [0.4]
            plot_numrun = 80
            plot_start = 1
            
    elif (dist_style == 0):
        mu_gap = 0.3
        mu_best = 0.7
        num_hyp = 50
        # fix_na = [50]
        # fix_tt = [5000]
        # armrange = np.arange(5, 36, 5) 
        # truncrange = np.arange(5000, 25001, 1000)
        truncrange = np.arange(5000, 10001, 5000)
        armrange = np.arange(5, 11, 5)
        fix_na = [10]
        fix_tt = [10000]
    # Overwrite all settings if New Yorker data
    if (ny_data == 1):
        dist_style = 0
        mu_gap = 0
        mu_style = 0
        mu_best = 0
        num_hyp = 30
        top_arms = 5
        fix_tt = [13000]
        # armrange = np.arange(10, 81, 10)
        truncrange = np.arange(13000, 13001, 1000)
        armrange = np.arange(10, 21, 10)

    ##### Plot results
    # Plot vs. FDR
    if args.power_plot == 0:
        # Plot over pi1
        plot_results_punif(truncrange, armrange, [0], dist_style, mu_gap, mu_style, hyp_style, pi1range, num_hyp, 0, 0, top_arms, FDRrange, mu_best, alpha0 , plot_numrun = plot_numrun, punif=1, plot_start = plot_start)
        # Plot over time
        plot_results_punif(truncrange, armrange, algrange, dist_style, mu_gap, mu_style, hyp_style, fix_pi1, num_hyp, 0, 0, top_arms, [0], mu_best, alpha0, plot_numrun = plot_numrun, punif=1, plot_start = plot_start) 

    # Plot vs. TT
    if (ny_data == 0):
        plot_results(truncrange, fix_na, algrange, dist_style, mu_gap, mu_style, hyp_style, pi1, num_hyp, 0, 0, top_arms, FDR, mu_best, 0,0,0) 

    # Plot vs. number of arms
    plot_results(fix_tt, armrange, algrange, dist_style, mu_gap, mu_style, hyp_style, pi1, num_hyp, 0, 0, top_arms, FDR, mu_best, 0,0,0) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mu-style',type=int, default = 2)
    parser.add_argument('--dist-style', type=int, default = 1)
    parser.add_argument('--ny-data', type=int, default = 0)
    parser.add_argument('--power-plot', type=int, default = 1)
    args = parser.parse_args()
    logging.info(args)
    main()
