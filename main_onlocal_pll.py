# FDR framework
import numpy as np
np.set_printoptions(precision = 4)

import os
import logging, argparse
import time
import itertools
from datetime import datetime
import StringIO


# import 
import exp_new
import exp_new_punif
from importme import *
from ipyparallel import Client


def main():

    # Get arguments
    ny_data = args.ny_data
    mu_style = args.mu_style
    dist_style = args.dist_style

    numrunrange = [25]

    #### Set experimental settings in paper
    pi1range = [0.4]
    hyp_style = 1
    top_arms = 10
    alpha0 = 0.1 # generally 0.1
    FDRrange = [0]
    algrange = [0,1] # both 0: best-arm MAB and 1: Uniform sampling

    # Power or FDR plots
    if args.power_plot == 1:
        punif = 0
    else:
        punif = 1

    if (dist_style == 1):
        mu_gap = 3
        mu_best = 8
        num_hyp = 500
        # armrange = np.arange(10, 121, 10) 
        # truncrange = np.arange(100, 801, 100)
        armrange = np.arange(10, 41, 10) 
        truncrange = np.arange(100, 301, 100)
        if punif == 1:
            armrange = [30]
            algrange = [0]
            truncrange = [200]
            pi1range = np.arange(0.1,1,0.1)
            FDRrange = [0, 3, 5]
            num_hyp = 1000
            
    elif (dist_style == 0):
        mu_gap = 0.3
        mu_best = 0.7
        num_hyp = 50
        # armrange = np.arange(5, 36, 5) 
        # truncrange = np.arange(5000, 25001, 1000)
        truncrange = np.arange(5000, 10001, 5000)
        armrange = np.arange(5, 11, 5)

    # Overwrite all settings if New Yorker data
    if (ny_data == 1):

        dist_style = 0
        mu_gap = 0
        mu_style = 0
        mu_best = 0
        num_hyp = 30
        top_arms = 5
        # armrange = np.arange(10, 81, 10)
        truncrange = np.arange(13000, 13001, 1000)
        armrange = np.arange(10, 21, 10)


        
    ##### Run for loop
    forrange = list(itertools.product(armrange, algrange, truncrange,numrunrange, pi1range, FDRrange))

    def wrapper(i):
        (no_arms, algnum, trunctime, numrun, pi1, FDR) = forrange[i]
        if punif == 0:
            exp_new.run_single(dist_style, mu_gap, mu_style, hyp_style, pi1, no_arms, num_hyp, 0,0, top_arms, alpha0,[trunctime], FDR, numrun, mu_best, algnum, verbose = 0)
        else:
            exp_new_punif.run_single(dist_style, mu_gap, mu_style, hyp_style, pi1, no_arms, num_hyp, 0,0, top_arms, alpha0,[trunctime], FDR, numrun, mu_best, algnum, 1, verbose = 0)
        print("time for %d, %d, %d: %d" % (no_arms, algnum, trunctime, time.time() - time1))
        return True

    # Parallelization
    c = Client()
    c.ids
    dview = c[:]
    dview.block = True
    lview = c.load_balanced_view()
    lview.block = True
    dview.execute('import numpy as np')
    dview.execute('import exp_new')
    mydict = dict(forrange = forrange, hyp_style = hyp_style, num_hyp = num_hyp, alpha0 = alpha0, dist_style = dist_style)
    dview.push(mydict)

    lview.map(wrapper, range(len(forrange)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mu-style',type=int, default = 2)
    parser.add_argument('--dist-style', type=int, default = 1)
    parser.add_argument('--ny-data', type=int, default = 0)
    parser.add_argument('--power-plot', type=int, default = 1)
    args = parser.parse_args()
    logging.info(args)
    main()
