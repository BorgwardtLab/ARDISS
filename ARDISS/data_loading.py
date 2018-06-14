import numpy as np
import time
from sklearn import preprocessing
from .utilities import verboseprint, elapsed_time
import gc

def get_whole_chr_haps_dict(haps_file, population_file, markers_file, verbose=False):
    """
    Generates a haps dict for the entire chromosome
    :param haps_file:
    :return:
    """
    # Load population of interest:
    verboseprint("Loading population of interest IDs.", verbose)
    indvs = set()
    popfile = open(population_file, "r")
    for line in popfile:
        line = line[:-1]
        indvs.add(line)
    popfile.close()
    # Load references
    ref_dict = get_snps_reference_catalog(markers_file)
    with open(haps_file, "r") as f:
        verboseprint("Loading haps file, this can take a while...", verbose)
        # Get IDs of interest on the first line of the haps_file
        indv_line = "SNP_id "
        indv_idx = list()
        fline = f.readline()
        cols = fline[:-1].split()
        for i in range(2, len(cols)):
            if (cols[i] in indvs):
                indv_line = indv_line + cols[i] + " "
                indv_idx.append(i)
        haps_dict = dict()
        m = 0
        for line in f:
            cols = line[:-1].split()
            snp_id = cols[1]
            if (snp_id[0:2] != "rs" and snp_id[0:3] != "Chr"): # ONLY FOR HUMAN SAMPLES!! Adapted for A thaliana
                continue
            if not snp_id in ref_dict:
                m += 1
                continue
            str_hap = []
            ref = ref_dict[snp_id]
            for i in indv_idx:
                if cols[i] == ref:
                    str_hap.append(1)
                else:
                    str_hap.append(0)
            haps_dict[snp_id] = np.asarray(str_hap)
        if m != 0:
            verboseprint("{} SNPs had to be skipped because they were not found in the .markers file.", verbose)
        verboseprint("Haps loaded. The dictionary has {} entries.".format(len(haps_dict)), verbose)
    return haps_dict

def get_haps_arrays(haps_dict, all_snps, typed_index, scale=False, verbose=False):
    """
    Return the arrays for the haps of interest
    :param haps_dict:
    :param all_snps:
    :param typed_snps:
    :return:
    """
    all_haps = np.array([haps_dict[x[0]] for x in all_snps])
    if scale:
        verboseprint("Scaling SNP values...", verbose)
        all_haps = preprocessing.scale(all_haps, axis=1, copy=False)
        verboseprint("Done", verbose)
    haps_typed = np.array([np.copy(all_haps[idx]) for idx in typed_index])
    return all_haps, haps_typed

def get_typed_haps_arrays(all_haps, typed_index):
    """
    Return the arrays for the haps of interest
    :param haps_dict:
    :param all_snps:
    :param typed_snps:
    :return:
    """
    # Not copying here since we don't change them anymore
    haps_typed = np.take(all_haps, typed_index, axis=0)
    return haps_typed

def filter_haps_array(all_haps, all_freqs, maf=0):
    # Get mask for haps_indeces:
    mask = (all_freqs > maf) & (all_freqs < 1 - maf)
    mask = mask.reshape(mask.shape[0],)
    filtered_haps = np.compress(mask,all_haps,axis=0)
    filtered_freqs = np.compress(mask,all_freqs,axis=0)
    return filtered_haps, filtered_freqs

def get_snps_reference_catalog(markers_file):
    f = open(markers_file, 'r')
    ref_dict = dict()
    positions = []
    for line in f:
        line = line[:-1]
        cols = line.split()
        # ignore lines that have more than 4 fields
        if (len(cols) != 4):
            continue
        # parse everything out and check for format
        snp_id = cols[0]
        prefix = snp_id[:2]  # should be "rs" for snps
        suffix = snp_id[2:]  # should be a number for snps
        # snp_id should be "rs" followed by a number, otherwise, ignore it
        if ((prefix != "rs" or (not suffix.isdigit())) and snp_id[:3] != "Chr"):
            continue
        # snp_pos should be a number
        snp_pos = cols[1]
        if (not snp_pos.isdigit()):
            continue
        # should be A, T, C, G
        ref = cols[2]
        if (ref != "A" and ref != "T" and ref != "C" and ref != "G"):
            continue
        alt = cols[3]
        # should be A, T, C, G
        if (alt != "A" and alt != "T" and alt != "C" and alt != "G"):
            continue
        ref_dict[snp_id] = ref
    f.close()
    return ref_dict

def load_typed_snps(typed_file, sorting = False, freq_dict=None, maf = 0.0, human_check=False):
    # Utility function that loads the files from the typed_file. Returns list with [id, pos, score] for each typed snp ordered with ascending position
    # Load typed_file
    f = open(typed_file, "r")
    # Skip header
    f.readline()
    if not sorting:
        typed_snps = []
        for line in f:
            cols = line[:-1].split()
            id = cols[0]
            # Check nomenclature for human SNPs
            if human_check and (id[0:2] != "rs" and id[0:3] != "Chr"):
                continue
            if freq_dict != None:
                freq = freq_dict[id]
                if freq <= maf or freq >= 1-maf:
                    continue
            pos = cols[1]
            ref = cols[2]
            alt = cols[3]
            score = cols[4]
            typed_snps.append([id, pos, ref, alt, score])
        f.close()
        return typed_snps
    else:
        typed_dict = dict()
        supp_dict = dict()
        id_list = []
        for line in f:
            cols = line[:-1].split()
            id = cols[0]
            if human_check and (id[0:2] != "rs" and id[0:3] != "Chr"):
                continue
            if freq_dict != None:
                if id in freq_dict:
                    freq = freq_dict[id]
                    if freq <= maf or freq >= 1-maf:
                        continue
            typed_dict[id] = int(cols[1])
            supp_dict[id] = cols[2:]
            id_list.append(id)
        s_keys = sorted(id_list, key=typed_dict.__getitem__)
        typed_snps = []
        for snp_id in s_keys:
            typed_snps.append([snp_id, str(typed_dict[snp_id])] + supp_dict[snp_id])
        f.close()
        return typed_snps

def get_zscore_array(typed_snps, all_dict, filter=False):
    if filter:
        z_scores_typed = []
    else:
        z_scores_typed = np.zeros((len(typed_snps), 1))
    for i, snp in enumerate(typed_snps):
        # for j in range(range_low, range_high):
        id = snp[0]
        ref = snp[2]
        alt = snp[3]
        # Check if conversion is needed
        if id not in all_dict:
            if filter:
                continue
            else:
                conversion = 0
        else:
            all_ref = all_dict[id][1]
            all_alt = all_dict[id][2]
            if ref == all_ref and alt == all_alt:
                conversion = 1
            elif ref == all_alt and alt == all_ref:
                conversion = -1
            else:
                # print("Warning! The ref and alt alleles don't match! Ignoring Z-Score.")
                conversion = 0
        if filter:
            z_scores_typed.append(conversion*float(snp[4]))
        else:
            z_scores_typed[i] = conversion * float(snp[4])
    if filter:
        z_scores_typed = np.asarray(z_scores_typed).reshape(-1,1)
    return z_scores_typed

def load_all_snps_from_markers_file(markers_file, freq_dict=None, maf = 0.0):
    # Utility function that loads the files from the typed_file. Returns list with [id, pos, score] for each typed snp
    # Load typed_file
    f = open(markers_file, "r")
    all_snps = []
    for line in f:
        cols = line[:-1].split()
        id = cols[0]
        # Filter snps format
        prefix = id[:2]  # should be "rs" for snps
        suffix = id[2:]  # should be a number for snps
        # snp_id should be "rs" followed by a number, otherwise, ignore it
        if ((prefix != "rs" or (not suffix.isdigit())) and id[:3] != "Chr"): # Adapted for A thaliana as well
            continue
        ref = cols[2]
        if (ref != "A" and ref != "T" and ref != "C" and ref != "G"):
            continue
        alt = cols[3]
        # should be A, T, C, G
        if (alt != "A" and alt != "T" and alt != "C" and alt != "G"):
            continue
        if freq_dict != None:
            freq = freq_dict[id]
            if freq <= maf or freq >= 1 - maf:
                continue
        pos = cols[1]
        all_snps.append([id, pos, ref, alt])
    f.close()
    return all_snps

def load_all_snps_from_markers_file_freq_array(markers_file, freq_array=None, maf = 0.0):
    # Utility function that loads the files from the typed_file. Returns list with [id, pos, score] for each typed snp
    # Load typed_file
    # Count markers_file length
    if freq_array is not None:
        with open(markers_file, "r") as f:
            markers_len = 0
            for line in f:
                markers_len += 1
        freq_special_indexing = markers_len != len(freq_array)
        print("Freq_special_indexing: {}".format(freq_special_indexing))
    f = open(markers_file, "r")
    all_snps = []
    freq_dict = dict()
    i = 0
    skipped_idx = []
    for idx, line in enumerate(f):
        cols = line[:-1].split()
        id = cols[0]
        # Filter snps format
        prefix = id[:2]  # should be "rs" for snps
        suffix = id[2:]  # should be a number for snps
        # snp_id should be "rs" followed by a number, otherwise, ignore it
        if ((prefix != "rs" or (not suffix.isdigit())) and id[:3] != "Chr"): # Adapted for A thaliana as well
            skipped_idx.append(idx)
            continue
        ref = cols[2]
        if (ref != "A" and ref != "T" and ref != "C" and ref != "G"):
            skipped_idx.append(idx)
            continue
        alt = cols[3]
        # should be A, T, C, G
        if (alt != "A" and alt != "T" and alt != "C" and alt != "G"):
            skipped_idx.append(idx)
            continue
        if freq_array is not None:
            #????
            if freq_special_indexing:
                freq = freq_array[i,0]
                i += 1
            else:
                freq = freq_array[idx,0] 
            freq_dict[id] = freq
            if freq <= maf or freq >= 1 - maf:
                skipped_idx.append(idx)
                continue # No need to add these indices as the haps array goes through the filtering step afterwards
        pos = cols[1]
        all_snps.append([id, pos, ref, alt])
    f.close()
    return all_snps, freq_dict, skipped_idx, idx+1 # Return ids of the SNPs to skip and the total number of SNPs

def get_all_freq(haps_dict):
    # Return a dictionary of all freqs
    freq_dict = dict()
    for idx in haps_dict:
        haps = haps_dict[idx]
        nhaps = len(haps)
        freq = float(haps.sum())
        freq = freq/nhaps
        freq_dict[idx] = freq
    return freq_dict

def compute_freqs(haps):
    n = haps.shape[1]
    freqs = np.sum(haps, axis=1) / float(n)
    return freqs.reshape((freqs.shape[0],1))

def load_all_necessary_files_array(haps_file, markers_file, typed_file, maf, population_file=None, verbose=False, human_check=False):
    """
    Function that loads all the necessary ressources from the provided files
    If the haps are in a numpy array, the loading will follow a different path
    """
    isnumpy = haps_file[-4:]==".npy"
    startout = time.time()
    # First generate dictionary and list of typed SNPs. All haplotype will be loaded in memory in numpy array format
    if not isnumpy:
        verboseprint("Loading arrays from the beagle file.", verbose)
        if population_file is None:
            raise ValueError("You need to provide a valid population file for beagle-file import")
        else:
            haps_dict = get_whole_chr_haps_dict(haps_file, population_file, markers_file, verbose=verbose)
        verboseprint("Elapsed time : {}".format(elapsed_time(time.time() - startout)), verbose)

        verboseprint("Loading frequency dictionary...", verbose)
        freq_dict = get_all_freq(haps_dict)
        verboseprint("Freq dict loaded.", verbose)
        verboseprint("Elapsed time : {}".format(elapsed_time(time.time() - startout)), verbose)

        all_snps = load_all_snps_from_markers_file(markers_file, freq_dict=freq_dict, maf=maf)  # Generates list of snp_id and positions and ref/alt
    else:
        verboseprint("Loading haps from numpy file", verbose)
        all_haps = np.load(haps_file) # This loads unfiltered haps, so even maf=0 haps will be here
        verboseprint("Elapsed time : {}".format(elapsed_time(time.time() - startout)), verbose)
        verboseprint("Computing frequencies", verbose)
        all_freqs = compute_freqs(all_haps) # Same as above
        verboseprint("Elapsed time : {}".format(elapsed_time(time.time() - startout)), verbose)
        all_snps, freq_dict, skipped_idx, n_all_snps = load_all_snps_from_markers_file_freq_array(markers_file, freq_array=all_freqs,
                                                   maf=maf)  # Generates list of snp_id and positions and ref/alt
        verboseprint("Filtering SNPs.", verbose)
        # Check if we need to remove the SNPs by using the IDs (e.g. for A. thaliana, where the haps file still contain the extra SNPs
        if n_all_snps == len(all_haps):
            verboseprint("Removing {} extra SNPs using ids".format(len(skipped_idx)), verbose)
            all_haps = np.delete(all_haps, skipped_idx, axis=0)
            all_freqs = np.delete(all_freqs, skipped_idx, axis=0)
            verboseprint("Final size of haps: {}".format(all_haps.shape), verbose)
        else:
            verboseprint("Removing extra SNPs only with MAF", verbose)
            all_haps, all_freqs = filter_haps_array(all_haps, all_freqs, maf=maf)
        gc.collect()
        # quick check
        if len(all_haps) != len(all_snps):
            print("WARNING: your haps array and snp array don't have the same length (all_haps: {}, all_snps: {}".format(all_haps.shape, len(all_snps)))

    verboseprint("Elapsed time : {}".format(elapsed_time(time.time() - startout)), verbose)

    # Problem: typed snps in global file are not ordered by position! Hence load them and sort them
    typed_snps = load_typed_snps(typed_file, sorting=True, freq_dict=freq_dict,
                                 maf=maf, human_check=human_check)  # generates list of snp_id, positions and scores
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
    filter_zscores = len(typed_snps) != len(typed_index)
    z_scores_typed = get_zscore_array(typed_snps, all_dict, filter=filter_zscores)
    del all_dict
    if not isnumpy:
        all_haps, typed_haps = get_haps_arrays(haps_dict, all_snps, typed_index, scale=True, verbose=verbose)
        verboseprint("Elapsed time : {}".format(elapsed_time(time.time() - startout)), verbose)
        all_freqs = compute_freqs(all_haps)
        typed_freqs = compute_freqs(typed_haps)
    else:
        verboseprint("Scaling the haps matrix...", verbose)
        # Need to change the type of the matrix beforehand to avoid forced copy of the array., np.float16 can be used, maybe add if function to take into account memory available?
        # all_haps = all_haps.astype(dtype=np.float32, copy=False)
        gc.collect()
        all_haps = preprocessing.scale(all_haps, axis=1, copy=False)
        gc.collect()
        typed_haps = get_typed_haps_arrays(all_haps, typed_index)
        typed_freqs = np.take(all_freqs,typed_index, axis=0)
    return all_haps, typed_haps, all_freqs, typed_freqs, z_scores_typed, typed_snps, typed_index, all_snps

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
    typed_haps = get_typed_haps_arrays(all_haps, typed_index)
    return all_haps, typed_haps

#### USE STUDY GENOTYPES ####
def get_haps_from_study_samples(genotypes_file, markers_file, verbose=False, human_check=True):
    """
    Generates a haps dict for the entire chromosome using the original study files (.traw)
    :param haps_file:
    :return:
    """
    # Get the snps that will be of interest
    all_snps_markers = load_all_snps_from_markers_file(markers_file)
    all_dict = dict()
    for snp in all_snps_markers:
        all_dict[snp[0]] = snp[1:]
    del all_snps_markers
    all_snps = []
    with open(genotypes_file, "r") as f:
        verboseprint("Loading haps file, this can take a while...", verbose)
        # Get IDs of interest on the first line of the haps_file
        indv_line = "SNP_id "
        fline = f.readline()
        cols = fline[:-1].split()
        for i in range(2, len(cols)):
            indv_line = indv_line + cols[i] + " "
        haps_dict = dict()
        m = 0
        for line in f:
            cols = line[:-1].split()
            snp_id = cols[1]
            if human_check and (snp_id[0:2] != "rs" and snp_id[0:3] != "Chr"): # ONLY FOR HUMAN SAMPLES!! Adapted for A thaliana
                continue
            if not snp_id in all_dict:
                m += 1
                continue
            all_snps.append([snp_id, all_dict[snp_id][0], all_dict[snp_id][1], all_dict[snp_id][2]])
            str_hap = []
            ref = all_dict[snp_id][1]
            for c in cols[1:]: # Still need to define from where to look at ACGT, depends on format
                if c == ref:
                    str_hap.append(1)
                else:
                    str_hap.append(0)
            haps_dict[snp_id] = np.asarray(str_hap)
        if m != 0:
            verboseprint("{} SNPs had to be skipped because they were not found in the .markers file.", verbose)
        verboseprint("Haps loaded. The dictionary has {} entries.".format(len(haps_dict)), verbose)
    return haps_dict, all_snps
