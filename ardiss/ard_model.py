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
import gc

class GPflowARD(object):
    # The class regroups the ARD optimization steps
    def __init__(self,
                 X,
                 Y,
                 window_size,
                 optimizer=gpflow.train.RMSPropOptimizer(0.1, momentum=0.01),
                 maxiter=100,
                 scale_X=False,
                 verbose=False):
        # Initialize the class and raise warnings depending on options chosen
        self.X = np.copy(X) # The haplotype values, this must be normalized ahead for optimal results
        if scale_X: # If X was not scaled before, we scale it here
            if self.X.dtype not in [np.float16, np.float32, np.float64]:
                self.X = self.X.astype(dtype=np.float16, copy=False) # Need to transform it to float to ensure scaling
                gc.collect()
            self.X = preprocessing.scale(self.X, axis=1, copy=False)
            gc.collect()
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
