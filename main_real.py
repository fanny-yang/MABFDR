# FDR framework
import numpy as np
np.set_printoptions(precision = 4)

import os
import time
import itertools
#from datetime import datetime
# import StringIO

# import 
import exp_new

from ipyparallel import Client


# They are just here as dummies
no_res = 4
no_proc = 1
plot_mode = ['r--', 'g-.', 'b-', 'k:']
plot_col = ['r', 'g', 'b', 'k']
plot_mark = ['o', '^', 'v', 'D', 'x', '+']
proc_list = ['LORD GAI++', 'LORD', 'AlphaInvest', 'Bonferroni']
alg_list = ['LUCB', 'Uniform']
mu_filename = './mu_lists/spikedexamples/mus.dat'

# Set ranges that you want to for loop over
armrange = np.arange(10, 90, 10)
algrange = range(len(alg_list))
truncrange = np.arange(13000, 130001, 1)
FDRrange = [0]
numrun = 100
pi1 = 0.4
num_hyp = 30
alpha0 = 0.1
hyp_style = 1

# Flatten nested for loops
forrange = list(itertools.product(armrange, algrange, truncrange, FDRrange))
forrange_short = list(itertools.product(algrange, FDRrange))

# Run on simulated mu
def wrapper(i):
    #time1 = time.time()
    (no_arms, algnum, trunctime, FDR) = forrange[i]
    exp_new.run_single(0,0,0,hyp_style,pi1,no_arms,num_hyp,0,0,5,alpha0,[trunctime], FDR, numrun, 0, algnum, verbose = 0)
    # print "time for %d, %d, %d: %d" % (no_arms, algnum, trunctime, time.time() - time1)
    return True
 

# Parallelization
c = Client()
c.ids
dview = c[:]
dview.block = True
lview = c.load_balanced_view()
lview.block = True
dview.execute('import exp_new')
dview.execute('import numpy as np')
mydict = dict(forrange = forrange, hyp_style = hyp_style, pi1 = pi1, num_hyp = num_hyp, numrun = numrun, alpha0 = alpha0, truncrange = truncrange)
dview.push(mydict)

lview.map(wrapper, range(len(forrange)))
#lview.map(wrapper2, range(len(forrange_short)))

# def runabunch():
#     for i in range(len(forrange)):
#         (no_arms, algnum, trunctime) = forrange[i]
#         exp_new.run_single(0,3,2,1,0.4,no_arms,100,0.2,0,10,0.1,trunctime,0,50, 0.4, algnum)
