# -----------------------------------------------------------------------------
# These functions allow for summary statistics imputation using the ARD model
# The ARD weights can be either associated with a specific population or
# optimized via GPflow
#
# January 2018, M. Togninalli
# -----------------------------------------------------------------------------
from .utilities import elapsed_time, compareZsc, correlation, verboseprint
import time, math
import numpy as np
import gpflow
from .data_loading import load_all_necessary_files_array, load_ard_weights, scale_with_weights
from .ard_computation import GPflowARD
from .GPModel import GPModelARD
import gc

def impute_ard(typed_file, haps_file, output_file, population_file, markers_file, masked_file="", window_size = 100, maf = 0, weight_optimization=False, weights_file=None, weight_scaling=False, verbose=False, human_check=False):#, recomputed_rate = 1000, maf = 0, normalization = False, parallelization=False, output_log=False):
    """Usage:
        typed_file (required) specify input file name for available partitioned typed files
        haps_file (required) specify input file name for haps list
        output_file (required) name of the file whereq imputed z-scores will be written
        population_file (required) name of the file containing the samples of the 1000genomes used for the haps dictionary
        markers_file (required) name of the file with information about all the SNPs on the chromosome of interest
        masked_file (optional) name of the original masked file for final comparison (default: "")
        window_size (optional) smaller window size (default: 100, must be pair)
        maf (optional) minimum minor allele frequency for considering SNPs (default: 0)
        weight_optimization (optional) determines whether the ARD weights are optimized prior to the imputation
        human_check (optional) turn on if SNPs ids shall be checked to match human nomenclature (rsxxxx or Chrxxxx)
        Returns: the elapsed time for the computation
        """
    # Generate warning if window size is not pair:
    if window_size % 2 != 0:
        raise Warning("Careful! You suggested an unpair window size...")

    startout = time.time()

    all_haps, typed_haps, all_freqs, freq_typed, z_scores_typed, typed_snps, typed_index, all_snps = \
        load_all_necessary_files_array(haps_file=haps_file, markers_file=markers_file, typed_file=typed_file,
                                       maf=maf, population_file=population_file, verbose=verbose, human_check=human_check)

    ard_weights = np.ones(typed_haps.shape[1])
    if weights_file is not None:
        ard_weights = load_ard_weights(weights_file, population_file)
        weight_optimization = False

    # Choose whether to optimize ARD weights:
    if weight_optimization:
        gpflow_model = GPflowARD(typed_haps, z_scores_typed, window_size, optimizer=gpflow.train.RMSPropOptimizer(0.1, momentum=0.01), maxiter=200,
                 scale_X=False, verbose=verbose)
        ard_weights = gpflow_model.compute_avg_ard_weights()
        gpflow_model.save_weights_to_file(output=output_file+".weights.txt", pop_file=population_file)

    if weight_scaling:
        all_haps, typed_haps = scale_with_weights(all_haps, typed_index, ard_weights)
        gc.collect()
        # Reset weights for the model, as we directly incorporate ARD in the scaling
        ard_weights = np.ones(typed_haps.shape[1])

    # Create the gaussian process model (kernel, ard params, noise variance)
    ard_gp_model = GPModelARD(sigma_noise=0.1, sigma_ard=ard_weights)

    verboseprint("Imputing missing Summary Statistics...", verbose=verbose)
    impute_sumstats_with_ard(typed_haps, all_haps, z_scores_typed, typed_index, all_snps, window_size, output_file,
                             ard_gp_model)

    endout = time.time()
    tottime = endout - startout
    verboseprint("Total elapsed time : {} ({} s)".format(elapsed_time(tottime), tottime), verbose=verbose)

    if masked_file != "":
        # Compare with masked values
        compareZsc(output_file, masked_file, suffix=output_file)
        # Compute the correlation...
        long_output, corr = correlation(output_file + ".comparison")
        print("Final correlation for overall chromosome computation: {}".format(corr))

    return tottime

def impute_sumstats_with_ard(haps_typed, all_haps, z_scores_typed, typed_index, all_snps, window_size, output_file, gp_model):
    w = open(output_file, "w")
    w.write("SNP_id SNP_pos Ref_allele Alt_allele Z-score Var\n")
    n_typed = haps_typed.shape[0]

    # all_positions = np.array([snp[1] for snp in all_snps]).astype(int)
    # typed_positions = np.array([all_positions[a] for a in typed_index])

    skipped=0
    skipped_last = True
    infin = True

    for i in range(n_typed - 1):
        idx1 = typed_index[i]
        idx2 = typed_index[i + 1]
        # Get typed snp and write it on file
        typed_s = all_snps[idx1]
        n_untyped = idx2 - idx1 - 1

        w.write(typed_s[0] + " " + typed_s[1] + " " + typed_s[2] + " " + typed_s[3] + " " +
                str(z_scores_typed[i,0]) + " 1.0\n") # Used to print the number of untyped but will report r2pred from now on

        if n_untyped == 0 and i != 0:
            skipped +=1
            skipped_last = True
            continue
        # Build neighboring map
        # Needs to get dynamic range:
        if i - (window_size / 2 - 1) <= 0:
            range_low = 0
            range_high = min(window_size, n_typed)
        else:
            if i + (window_size / 2 + 1) >= n_typed:
                range_low = n_typed - window_size
                range_high = n_typed
                infin = True
            else:
                range_low = int(i - (window_size / 2 - 1))
                range_high = int(i + (window_size / 2 + 1))
                infin = False
        # TODO: add skipping of MAF
        X_typed = haps_typed[range_low:range_high]
        z_typed = z_scores_typed[range_low:range_high]
        # X_positions = typed_positions[range_low:range_high]
        if skipped_last:
            gp_model.set_X(X_typed) #, X_positions
            gp_model.set_Y(z_typed) # Need to initialize Y after X because it also computes alpha
        elif not infin:
            gp_model.update_model(haps_typed[range_high],z_scores_typed[range_high])

        X_untyped = all_haps[idx1+1:idx2,:]
        # X_untyped_positions = all_positions[idx1+1:idx2]

        z_scores, vars_opt = gp_model.impute_Y(X_untyped,compute_var=True)

        # write on file
        skipped_last = False
        for j,snp in enumerate(all_snps[idx1 + 1: idx2]):
            if math.isnan(z_scores[j]):
                z_scores[j] = 0
            w.write(snp[0] + " " + snp[1] + " " + snp[2] + " " + snp[3] + " " + str(z_scores[j,0]) + " " + str(vars_opt[j]) + "\n")
    w.close()
