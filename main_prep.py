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

from generate_mu import*

# Setting you want
gaprange = [0.2, 0.4, 0.6]
toparmrange = [1, 5, 10]
dist_type = 0
ms_range = [1, 2, 3, 4]
max_arms = 1000
mu_max = 8 # That's for Gaussian only, for Bernoulli sweeps through
max_hyp = 1000
sigmarange = [0, 0.2]
eps_range = [0, 0.2]

create_mu(gaprange, toparmrange,  dist_type, ms_range, mu_max, max_arms, max_hyp, sigmarange, eps_range)
