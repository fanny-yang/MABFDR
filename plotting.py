import numpy as np
from numpy import sqrt, log, exp, mean, cumsum, sum, zeros, ones, argsort, argmin, argmax, array, maximum, concatenate
from numpy.random import randn, rand
np.set_printoptions(precision = 4)

import os
import matplotlib #as mpl
matplotlib.use('agg')
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 24
matplotlib.rcParams['axes.labelsize'] = 36
matplotlib.rcParams['xtick.labelsize']= 28
matplotlib.rcParams['ytick.labelsize']= 28

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import scipy.optimize as optim
from scipy.stats import norm
from scipy.stats import bernoulli
import StringIO

# import own stuff
from importme import *


def saveplot(direc, filename, lgd, ext = 'pdf',  close = True, verbose = True):
    filename = "%s.%s" % (filename, ext)
    if not os.path.exists(direc):
        os.makedirs(direc)
    savepath = os.path.join(direc, filename)
    if lgd != []:
        plt.savefig(savepath, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        plt.savefig(savepath, bbox_inches='tight')
    if verbose:
        print("Saving figure to %s" % savepath)
    if close:
        plt.close()


def plot_errors_mat(xs, matrix_av, matrix_err, labels, dirname, filename, xlabel, ylabel, plots_ind = 1):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    no_lines = len(matrix_av)
    for i in range(no_lines):
            ys = np.array(matrix_av[i])
            zs = np.array(matrix_err[i])
            ax.errorbar(xs, ys, yerr = zs, color = plot_col[i % len(plot_col)], marker = plot_mark[i % len(plot_mark)], lw=3, markersize =10, label=labels[i])
    if plots_ind == 1:
        lgd = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, handletextpad=0.3,
                       ncol=no_lines, mode="expand", borderaxespad=0.)
    else:
        lgd = ax.legend(loc='upper right')
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_xlim((min(xs), max(xs)))
    ax.grid(True)

    saveplot(dirname, filename, lgd)
    
def plotsingle_shaded_mat(xs, matrix, dirname, filename, xlabel, ylabel):

    plots_ind = 2
    no_lines = len(matrix)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Compute max of all rows and min
    max_vec = np.array(matrix).max(axis=0)
    min_vec = np.array(matrix).min(axis=0)
    mean_vec = np.array(matrix).mean(axis=0)
    # Compute mean
    for i in range(no_lines):
        ys = np.array(matrix[i])
        plt.plot(xs, ys, color = plot_col[plots_ind], linestyle='-', lw=1)
    plt.fill_between(xs, max_vec, min_vec, facecolor=plot_col[plots_ind], alpha=0.2)
    plt.plot(xs, mean_vec, 'r--', lw=3)
    plt.xlabel(xlabel, labelpad=10)
    plt.ylabel(ylabel, labelpad=10)
    plt.ylim((0,0.22))
    plt.grid(True)
    plt.tight_layout()
    saveplot(dirname, filename,[])
