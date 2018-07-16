# -----------------------------------------------------------------------------
# Refactoring of the imputation script into a class: Experiment, to increase
# modulability of the code.
#
# July 2018, M. Togninalli
# -----------------------------------------------------------------------------
import gpflow
from .ard_model import GPflowARD
from .imputation_model import GPModelARD
from .data_io import ReferenceData, TypedData
import gc

class Experiment(object):
    """
    Class that contains the different steps of the algorithms: data loading,
    weight optimization and imputation.
    """

    def __init__(self,
                 genotype_file,
                 markers_file,
                 typed_scores_file=None,
                 output_file=None,
                 population_file=None,
                 masked_scores_file=None,
                 weights_file=None,
                 window_size=100,
                 window_size_ard=None,
                 maf=0,
                 verbose=False,
                 snpid_check=False):
        """
        Initiate the experiment object with filenames and options
        :param genotype_file: (required) specify input file name for
        haps list
        :param output_file: (required) name of the file whereq imputed
        z-scores will be written
        :param markers_file: (required) name of the file with
        information about all the SNPs on the chromosome of interest
        :param population_file: (optional) name of the file containing
        the samples of the 1000genomes used for the genotype dictionary
        :param masked_file: (optional) name of the original masked file
        for final comparison
        :param weights_file: (optional) file containing precomputed
        weights, can be optain from the GPflowARD object
        :param window_size: (optional) smaller window size (default:
        100, must be pair)
        :param window_size_ard: (optional) window size for ARD
        computations (default: equals to window_size, must be pair),
        only necessary if it differs from window_size
        :param maf: (optional) minimum minor allele frequency for
        considering SNPs (default: 0)
        :param weight_optimization:
        :param weight_scaling:
        :param verbose: boolean to print more info during loading and
        imputation
        :param snpid_check: Boolean to indicate whether the SNP_ids
        should undergo a check to see if they match the human format
        (i.e. rsXXXX, if False, all SNPs from the markers file will be
        included)
        """
        # TODO: initialization with all parameters
        # TODO: gpu usage flag...
        # TODO: Implement an internal flag to mask SNPs and keep them as validation
        # Filenames
        self.genotype_filename = genotype_file
        self.markers_filename = markers_file
        self.typed_scores_filename = typed_scores_file
        self.output_filename = output_file

        # Optional filenames
        self.population_filename = population_file
        self.weights_filename = weights_file
        self.masked_scores_filename = masked_scores_file

        # Experiment options
        if window_size % 2 != 0:
            raise Warning("WARNING: You suggested an unpair window size.")
        self.window_size_imputation = window_size
        self.window_size_ard = window_size_ard if window_size_ard is not None else window_size
        self.maf = maf
        self.verbose = verbose
        self.snpid_check = snpid_check

        # Allocate slots for later use
        self.RefData = None
        self.typed_indeces = None
        self.typed_scores_array = None
        self.ard_weights = None

    # 1. Load all the necessary data
    def load_reference_data(self):
        # 1. Start from the Reference Panel, this instance is then passed to
        #    the TypedData object as
        self.RefData = ReferenceData(self.genotype_filename,
                                     self.markers_filename,
                                     population_filename=self.population_filename,
                                     snpid_check=self.snpid_check,
                                     maf=self.maf,
                                     verbose=self.verbose,
                                     extra_checks=False)
        self.RefData.load_files()

    def load_typed_data(self):
        # Split the loading to allow sequential imputation for multiple
        # phenotypes / score sets
        # 2. Then load the typed files, this way, one can impute multiple
        #    studies using the same reference dataset
        TypData = TypedData(self.typed_scores_filename, self.snpid_check)
        TypData.load_typed_snps()
        # 3. Extract the indeces of the typed files
        self.typed_indeces = TypData.get_typed_indeces(
            self.RefData.all_snps)
        self.typed_genotypes = np.take(self.RefData.genotype_array, self.typed_indeces, axis=0)
        # 4. Extract the scores into an array
        self.typed_scores_array = TypData.get_scores_array(
            self.RefData.all_snps_dict)
        # 5. Initialize the ARD weights
        self.ard_weights = np.ones(self.RefData.genotype_array.shape[1])

        # TODO: implement the online masking...

    def optimize_ard_weights(self, gpu=True):
        # Optimize the weights using ard
        # TODO automatic detection of GPU capacity
        gpflow_model = GPflowARD(self.typed_genotypes, self.typed_scores_array, self.window_size,
                                 optimizer=gpflow.train.RMSPropOptimizer(0.1, momentum=0.01), maxiter=200,
                                 scale_X=True, verbose=verbose)
        ard_weights = gpflow_model.compute_avg_ard_weights()
        gpflow_model.save_weights_to_file(output=output_file + ".weights.txt", pop_file=population_file)

    def load_weights_from_file