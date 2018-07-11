#########
# GARBAGE
#########

import numpy as np
import imputationUtilities as iu
import time, math, datetime
from utilities import elapsed_time

class Experiment(object):
    # A class that regroups data, preprocessing steps and imputation methods

    def __init__(self, haps_file, markers_file, typed_file, array_haps=False, window_size = 100, maf=0, normalization=False,
                 parallelization=False, recomputed_rate=1000, compare=False, masked_file="", debugging=False,
                 timing=True, verbose=True, population_file=None, target_file=None):
        # Initialize the class and raise warnings depending on options chosen
        self.haps_filename = haps_file
        self.markers_filename = markers_file
        self.typed_filename = typed_file
        self.array = array_haps
        if array_haps and haps_file[-4:]!=".npy":
            raise Warning("You selected the nupmy array approach for faster imputation but provided a filename that "
                          "doesn't have the numpy extension (.npy), be careful!")
        if window_size % 2 != 0:
            raise Warning("Careful! You suggested an odd window size...")
        self.window_size = window_size
        self.maf = maf
        self.normalization = normalization
        self.recomputed_rate = recomputed_rate
        self.compare = compare
        self.masked_filename = masked_file
        self.debugging = debugging
        self.timing = timing
        self.verbose = verbose
        if population_file is not None and not array_haps:
            raise ValueError("Please provide a valid population file to select the haplotype columns.")
        self.population_filename = population_file
        if compare and masked_file == "":
            self.compare = False
            raise ValueError("You want comparison but did not provide a masked file to compare with.")

# TODO: add method to modify options even when data is loaded
    def load_all_necessary_files(self):
        """
        Function that loads all the necessary ressources from the provided files
        If the haps are in a numpy array, the loading will follow a different path
        """
        if self.timing:
            startout = time.time()
        # First generate dictionary and list of typed SNPs. All haplotype will be loaded in memory in numpy array format
        if not self.array:
            # If the user selected the regular loading from beagle file
            if self.verbose:
                print("Loading arrays from the beagle file.")
            if self.population_filename is None:
                raise ValueError("You need to provide a valid population file for beagle-file import")
            if self.parallelization:
                haps_dict = get_whole_chr_haps_dict_parallel(self.haps_filename, self.population_filename, self.markers_filename)
            else:
                haps_dict = get_whole_chr_haps_dict(self.haps_filename, self.population_filename, self.markers_filename)
            if self.timing:
                print("Haps loaded. Elapsed time : {}".format(elapsed_time(time.time() - startout)))
            if self.verbose:
                print("Loading frequency dictionary...")
            freq_dict = get_all_freq(haps_dict)
            if self.timing:
                print("Freq dict loaded. Elapsed time : {}".format(elapsed_time(time.time() - startout)))

            all_snps = load_all_snps_from_markers_file(markers_file, freq_dict=freq_dict,
                                                       maf=maf)  # Generates list of snp_id and positions and ref/alt
        else:
            # Direct numpy loading of precomputed/filtered array
            if self.verbose:
                print("Loading haps from numpy file")
            all_haps = np.load(self.haps_filename)  # This loads unfiltered haps, so even maf=0 haps will be here
            if self.timing:
                print("Haps loaded. Elapsed time : {}".format(elapsed_time(time.time() - startout)))
            if self.verbose:
                print("Computing frequencies")
            all_freqs = compute_freqs(all_haps)  # Same as above
            if self.timing:
                print("Elapsed time : {}".format(elapsed_time(time.time() - startout)))
            all_snps, freq_dict = load_all_snps_from_markers_file_freq_array(self.markers_filename, freq_array=all_freqs,
                                                                             maf=self.maf)  # Generates list of snp_id and positions and ref/alt
            if self.verbose:
                print("Filtering haps.")
            all_haps, all_freqs = filter_haps_array(all_haps, all_freqs, maf=self.maf)
            # quick check
            if len(all_haps) != len(all_snps):
                print(
                "WARNING: your haps array and snp array don't have the same length (all_haps: {}, all_snps: {}".format(
                    all_haps.shape, len(all_snps)))

        if self.timing:
            print("Elapsed time : {}".format(elapsed_time(time.time() - startout)))

        # Problem: typed snps in global file are not ordered by position! Hence load them and sort them
        typed_snps = load_typed_snps(typed_file, sorting=True, freq_dict=freq_dict,
                                     maf=maf)  # generates list of snp_id, positions and scores
        # Get idx of typed snps in the all_snps list and build all_dict
        # all_dict contains positions, ref and alt information
        typed_dict = dict()
        typed_index = []
        for i in typed_snps:
            typed_dict[i[0]] = i[1:]
        all_dict = dict()
        for idx, snp in enumerate(all_snps):
            all_dict[snp[0]] = snp[1:]
            if snp[0] in typed_dict:
                typed_index.append(idx)
        del typed_dict
        z_scores_typed = get_zscore_array(typed_snps, all_dict)
        del all_dict
        if not isnumpy:
            all_haps, typed_haps = get_haps_arrays(haps_dict, all_snps, typed_index)
            all_freqs = compute_freqs(all_haps)
            # freq_typed = np.compress(typed_index, all_freqs, axis=0)
            freq_typed = compute_freqs(typed_haps)
        else:
            typed_haps = get_typed_haps_arrays(all_haps, typed_index)
            freq_typed = np.take(all_freqs, typed_index, axis=0)

        print("Loading sigma_tt...")
        sigma_tt = compute_sigma_tt_condensed_storage_array(typed_haps, freq_typed,
                                                            sliding_window_size=window_size,
                                                            maf=maf)
        print("Sigma_tt loaded. Dimensions: {}x{}".format(len(sigma_tt), len(sigma_tt[0])))
        return all_haps, typed_haps, all_freqs, freq_typed, z_scores_typed, typed_snps, typed_index, all_snps, sigma_tt
