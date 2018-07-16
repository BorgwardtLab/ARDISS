# -----------------------------------------------------------------------------
# These functions allow for summary statistics imputation using the ARD model
# The ARD weights can be either associated with a specific population or
# optimized via GPflow
#
# July 2018, M. Togninalli
# -----------------------------------------------------------------------------
from .utilities import elapsed_time, compareZsc, correlation, verboseprint
import time
import math
import numpy as np
import gpflow
from .ard_model import GPflowARD, load_ard_weights, scale_with_weights
from .imputation_model import GPModelARD
from .data_io import ReferenceData, TypedData
import gc
import os

def impute_ard(typed_file, genotype_file, output_file, population_file, markers_file, masked_file=None, window_size=100,
               maf = 0, weight_optimization=True, weights_file=None, verbose=False, snpid_check=False, gpu=False):
    """Usage:
        typed_file (required) specify input file name for available partitioned typed files
        genotype_file (required) specify input file name for genotype list
        output_file (required) name of the file whereq imputed z-scores will be written
        population_file (required) name of the file containing the samples of the 1000genomes used for the genotype
        dictionary
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
        raise Warning("Careful! You suggested an odd number as a window size...")

    startout = time.time()

    # 1. Start from the Reference Panel, this instance is then passed to
    #    the TypedData object
    RefData = ReferenceData(genotype_file, markers_file, population_file, snpid_check, maf=maf, verbose=True)
    RefData.load_files()
    RefData.filter_maf_()

    # 2. Then load the typed files, this way, one can impute multiple
    #    studies using the same reference dataset
    TypData = TypedData(typed_file, snpid_check)
    TypData.load_typed_snps()

    # 3. Extract the indeces of the typed files
    typed_indeces = TypData.get_typed_indeces(RefData.all_snps)

    # 4. Extract the zscores
    z_scores_typed = TypData.get_scores_array(RefData.all_snps_dict)

    all_genotypes = RefData.genotype_array
    typed_genotypes = np.take(all_genotypes, typed_indeces, axis=0)
    verboseprint('Typed extracted', verbose)
    ard_weights = np.ones(typed_genotypes.shape[1])
    if weights_file is not None:
        ard_weights = load_ard_weights(weights_file, population_file)
        print('INFO: You provided a weight file for precomputed weights, weight optimization will NOT occur.')
        weight_optimization = False

    # Choose whether to optimize ARD weights:
    if weight_optimization:
        # Hack for GPU usage, as gpflow is not device-aware
        # (https://github.com/GPflow/GPflow/issues/294)
        if not gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        else:
            cuda_vis = os.environ.get('CUDA_VISIBLE_DEVICES')
            if cuda_vis is None:
                print('WARNING: You chose to use GPUs for your computations, but you have no available GPU : '
                      'CUDA_VISIBLE_DEVICES: {}'.format(cuda_vis))
            verboseprint('INFO: You chose to use GPUs for your computations, make sure that the following environment '
                         'variable is non-null: CUDA_VISIBLE_DEVICES: {}'.format(cuda_vis),verbose)
        gpflow_model = GPflowARD(typed_genotypes, z_scores_typed, window_size,
                                 optimizer=gpflow.train.RMSPropOptimizer(0.1, momentum=0.01), maxiter=200,
                                 verbose=verbose)
        ard_weights = gpflow_model.compute_avg_ard_weights()
        gpflow_model.save_weights_to_file(output=output_file+".weights.txt", pop_file=population_file)


        all_genotypes, typed_genotypes = scale_with_weights(all_genotypes, typed_indeces, ard_weights)
        gc.collect()
        # Reset weights for the model, as we directly incorporate ARD in the scaling
        ard_weights = np.ones(typed_genotypes.shape[1])

    # Create the gaussian process model (kernel, ard params, noise variance)
    ard_gp_model = GPModelARD(sigma_noise=0.1, sigma_ard=ard_weights)

    # Impute the missing values
    verboseprint("Imputing missing Summary Statistics...", verbose=verbose)
    impute_sumstats_with_ard(typed_genotypes, all_genotypes, z_scores_typed, typed_indeces, RefData.all_snps,
                             window_size, output_file, ard_gp_model)

    endout = time.time()
    tottime = endout - startout
    verboseprint("Total elapsed time : {} ({} s)".format(elapsed_time(tottime), tottime), verbose=verbose)

    if masked_file is not None:
        # Compare with masked values
        compareZsc(output_file, masked_file, suffix=output_file)
        # Compute the correlation...
        long_output, corr = correlation(output_file + ".comparison")
        print("Final correlation for overall chromosome computation: {}".format(corr))

    return tottime

def impute_sumstats_with_ard(typed_genotypes, all_genotypes, z_scores_typed, typed_index, all_snps, window_size,
                             output_file, gp_model):
    w = open(output_file, "w")
    w.write("SNP_id SNP_pos Ref_allele Alt_allele Z-score Var\n")
    n_typed = typed_genotypes.shape[0]

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
                str(z_scores_typed[i,0]) + " 1.0\n")

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
        X_typed = typed_genotypes[range_low:range_high]
        z_typed = z_scores_typed[range_low:range_high]
        # X_positions = typed_positions[range_low:range_high]
        if skipped_last:
            gp_model.set_X(X_typed) #, X_positions
            gp_model.set_Y(z_typed) # Need to initialize Y after X because it also computes alpha
        elif not infin:
            gp_model.update_model(typed_genotypes[range_high],z_scores_typed[range_high])

        X_untyped = all_genotypes[idx1+1:idx2,:]

        z_scores, vars_opt = gp_model.impute_Y(X_untyped,compute_var=True)

        # write on file
        skipped_last = False
        for j,snp in enumerate(all_snps[idx1 + 1: idx2]):
            if math.isnan(z_scores[j]):
                z_scores[j] = 0
            w.write(snp[0] + " " + snp[1] + " " + snp[2] + " " + snp[3] + " " + str(z_scores[j,0]) + " "
                    + str(vars_opt[j]) + "\n")
    w.close()
