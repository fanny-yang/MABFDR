import pickle
import ipdb
import numpy as np
import os

# caption = []
# funny_count = []
# total_count = []

def saveres(direc, filename, mat, ext = 'dat', verbose = True):
    filename = "%s.%s" % (filename, ext)
    if not os.path.exists(direc):
        os.makedirs(direc)
    savepath = os.path.join(direc, filename)
    np.savetxt(savepath, mat, fmt='%.3e', delimiter ='\t')
    if verbose:
        print("Saving results to %s" % savepath)

#####  Read the pickle

# Read from file
#data is a list of rows, where each row is of the format 
#[contest_id, caption, unfunny counts, somewhat funny counts, funny counts]
def generate_ny_mu():
    data = pickle.load(open('./csvs/agg.pkl','rb'))
    all_contests = list(set([x[0] for x in data]))

    # Initialize
    num_hyp = len(all_contests)
    # Will be list of arrays
    mu_list0 = []
    no_arms = np.zeros(num_hyp, dtype = np.int)
    cont_vec = [x[0] for x in data]

    # Per contest = hypothesis, read only captions for that contest
    for i in range(num_hyp):
            # Get number of total captions/arms
            cur_indices = np.where(np.array(cont_vec) == all_contests[i])[0]
            no_arms[i] = len(cur_indices)
            funny_vec = np.zeros(no_arms[i])
            tot_vec = np.zeros(no_arms[i])
            caption_vec = ['' for k in range(no_arms[i])]
            mu_vec = np.zeros(no_arms[i])

            # Get data for all captions
            for l, j in enumerate(cur_indices):
                    num_funny = float(data[j][3]) + float(data[j][4])
                    num_tot = float(data[j][2]) + num_funny
                    funny_vec[l] = num_funny
                    tot_vec[l] = num_tot
                    mu_vec[l] = num_funny/num_tot
                    caption_vec[l] = data[j][1]

            # Sort and save the list to a list of lists
            mu_vec = np.sort(mu_vec)[::-1]
            mu_list0.append(mu_vec)

    # Trim to a matrix with same length rows #Write to matrix with max number of 
    min_arms = min(no_arms)
    mu_mat = np.zeros([num_hyp, min_arms])
    for i in range(num_hyp):
            mu_mat[i] = mu_list0[i][0:min_arms]


    dirname = './expsettings'
    filename = 'D%d_S%d_G%.1f_E%.1f_Si%.1f_TA%d_MM%.1f_NH%d_NA%d' % (0, 0, 0, 0, 0, 5, 0, num_hyp, min_arms)

    # Save in file
    saveres(dirname, filename, mu_mat)
    return mu_mat
