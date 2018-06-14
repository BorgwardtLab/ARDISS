# -----------------------------------------------------------------------------
# These functions compute the ARD values for the chromosome of interest
# They rely on three external libraries: tensorflow, gpflow and sklearn. The
# Experiment class calls these methods if the ARD option is enabled.
#
# January 2018, M. Togninalli
# -----------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import gpflow
from sklearn import preprocessing
# import matplotlib.pyplot as plt

class GPflowARD(object):
    # The class regroups the ARD optimization steps
    def __init__(self, X, Y, window_size, optimizer=gpflow.train.RMSPropOptimizer(0.1, momentum=0.01), maxiter=100,
                 scale_X=False, verbose=False):
        # Initialize the class and raise warnings depending on options chosen
        self.X = np.copy(X) # The haplotype values, this must be normalized ahead for optimal results
        if scale_X: # If X was not scaled before, we scale it here
            self.X = preprocessing.scale(self.X, axis=1, copy=False)
        self.Y = np.copy(Y) # The typed scores
        self.window_size = window_size # The window size used during optimization, this affects performance
        self.optimizer = optimizer # The chosen optimizer, RMSProp is set as default
        self.maxiter = maxiter # The maximum number of iteration of the optimizer at each window
        self.verbose = verbose
        self.ards = None

    def optimize_weights(self):
        ws = self.window_size
        n_windows = np.int(np.floor(self.X.shape[0] / ws))
        r = np.arange(n_windows)
        i = 0
        ards = []
        for k in r:
            if self.verbose:
                print("Optimization {}/{}".format(i + 1, n_windows))
            i += 1
            # In order to avoid excessive growth of graph, we save the ARD vectors at every iterations and replace them
            with tf.Session(graph=tf.Graph()):
                X_batch = np.array(self.X[ws * k:ws * (k + 1)], dtype=float)  # Optimize on non-overlapping windows
                Y_batch = np.array(self.Y[ws * k:ws * (k + 1)], dtype=float)
                k = gpflow.kernels.Linear(X_batch.shape[1], ARD=True)
                m_ard = gpflow.models.GPR(X_batch, Y_batch, k)
                m_ard.likelihood.variance = 0.1
                m_ard.likelihood.trainable = False
                self.optimizer.minimize(m_ard, maxiter=self.maxiter)
                ards.append(m_ard.kern.variance.read_value())
        self.ards = np.asarray(ards)
        return self.ards

    def compute_avg_ard_weights(self):
        # if weights were not optimized yet:
        if self.ards is None:
            self.optimize_weights()
        return np.mean(self.ards,axis=0)

    def save_weights_to_file(self, output="ARD_weigts.txt", pop_file=None):
        # Writes out the weights to a txt file. If a population_file is provided, saves popnames in columns
        if pop_file is not None:
            # CAREFUL: this pop_file must not be the same as the one provided to load the data
            with open(pop_file, "r") as pop:
                written_lines = []
                firstline=pop.readline()[:-1].split()
                w_l = firstline[0]
                pops_reported = len(firstline)>1
                if pops_reported:
                    w_l += " " + firstline[2] + " " + firstline[1]
                # Need to account for haplotypes, hence duplicate everytime
                written_lines.append(w_l)
                written_lines.append(w_l)
                for line in pop:
                    cols = line[:-1].split()
                    w_l = cols[0]
                    if pops_reported:
                        w_l += " " + cols[2] + " " + cols[1]
                    written_lines.append(w_l)
                    written_lines.append(w_l)
            with open(output, "w") as w:
                for idx,ard in enumerate(self.compute_avg_ard_weights()):
                    w.write(written_lines[idx] + " " + str(ard) + "\n")
        else:
            with open(output, "w") as w:
                for ard in self.compute_avg_ard_weights():
                    w.write(str(ard) + "\n")


    ######## Plotting functions ##########
    def plot_mean_val(self, haps_origin, outfile="ARD_weights"):
        pass

def plot_mean_val(pop):
    # Plot the mean and std of ard values for set of populations
    data = []
    for i in pop.keys():
        data.append(pop[i])
    # plt.boxplot(data, labels=pop.keys())
def collect_pop_ard(haps_origin, ard):
    # Quickly compute the average and std of ARD weights for different samples
    subpops = dict()
    pops = dict()
    # initialize
    for sub in np.unique(np.asarray(haps_origin)[:,0]):
        subpops[sub]=np.array([])
    for pop in np.unique(np.asarray(haps_origin)[:,1]):
        pops[pop] = np.array([])
    for i,ard_i in enumerate(ard):
        pops[haps_origin[i,1]] = np.append(pops[haps_origin[i,1]],ard_i)
        subpops[haps_origin[i,0]] = np.append(subpops[haps_origin[i,0]],ard_i)
    return pops, subpops
def get_labels_pop(pop_file, all_pop_file):
    with open(all_pop_file, "r") as f:
        sample_dict = dict()
        for line in f:
            cols = line[:-1].split()
            sample_dict[cols[0]] = [cols[1], cols[2]] # Get both sub and superpopulation
    with open(pop_file,"r") as p:
        sample_origin = []
        for line in p:
            # Need to account for haplotypes, hence duplicate everytime
            sample_origin.append(sample_dict[line[:-1]])
            sample_origin.append(sample_dict[line[:-1]])
    return sample_origin

def get_idx_top_N(ards, N):
    idx = []
    for ard_weights in ards:
        idxi = np.argsort(ard_weights)[-N:][::-1]
        idx.append(idxi)
    return np.asarray(idx)

def count_top_N(ards, N, haps_origin):
    # Get the top indeces
    top_idx = get_idx_top_N(ards, N)
    # set up counters:
    subpops_dict = {s:0 for s in np.unique(haps_origin[:,0])}
    pops_dict = {s:0 for s in np.unique(haps_origin[:,1])}
    for idx in top_idx:
        for i in idx:
            subpops_dict[haps_origin[i,0]]+=1
            pops_dict[haps_origin[i,1]]+=1
    return subpops_dict, pops_dict