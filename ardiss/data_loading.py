import numpy as np
from sklearn import preprocessing
import gc

def load_ard_weights(weight_file, pop_file=None):
    # Read the weights from the weight file and check against pop_file
    with open(weight_file, "r") as f:
        weights = []
        ids = []
        for line in f:
            cols = line[:-1].split()
            weights.append(float(cols[-1]))
            if len(cols) > 1:
                ids.append(cols[0])
    if pop_file is not None:
        with open(pop_file, "r") as f:
            ref_ids = []
            for line in f:
                ref_ids.append(line[:-1].split()[0])
        if 2*len(ref_ids) != len(weights):
            print("Warning: the number of weights is different than twice the number of reference samples in the population file")
        if len(ids) != 0:
            for id in ids:
                if id not in ref_ids:
                    print("Warning: sample {}is not referenced in the population file".format(id))
    return np.asarray(weights)

def scale_with_weights(all_haps, typed_index, ard_weights):
    # Scales the haps so as to directly incorporate the ARD weights and speed up later computations
    all_haps = np.sqrt(ard_weights)*all_haps
    gc.collect()
    all_haps = preprocessing.scale(all_haps, axis=1, copy=False)
    gc.collect()
    typed_haps = np.take(all_haps, typed_index, axis=0)
    return all_haps, typed_haps
