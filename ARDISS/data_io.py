import numpy as np
import os
from sklearn import preprocessing
from .utilities import verboseprint, elapsed_time
import gc

# ---------------------------------------------
# Include all the object definitions for the different data sources
# ---------------------------------------------

class ReferenceData(object):
    """
    Class that contains all the files associated with the reference panel
    """
    def __init__(self, genotype_filename, markers_filename, population_filename=None, snpid_check=True):
        """
        Initiate the object with the filenames
        :param genotype_filename: Genotype file, can be a .npy array (with its corresponding SNP id list, TODO) or a .bgl file
        :param markers_filename: File with a list of the SNPs on the chromosome of interest,
        should have no header and the following columns: [SNP_ID, SNP_POS, REF, ALT]
        :param population_filename: List of samples if a subset of the reference panel is to be chosen
        (only works with the BGL file)
        :param snpid_check: Boolean to indicate whether the SNP_ids should undergo a check to see if they match
        the human format (i.e. rsXXXX, if False, all SNPs from the markers file will be included)
        """
        self.genotype_filename = genotype_filename
        self.markers_filename = markers_filename
        self.population_filename = population_filename
        self.snpid_check = snpid_check

        # Loaded files
        self.all_snps = None
        self.genotype_array = None
        self.genotype_map = None
        self.genotype_dict = None
        self.population_list = None

        # Auxiliary data structures
        self.all_snps_dict = None

    # ------------------
    # LOAD MARKERS
    # ------------------
    # Pandas loading and processing is twice as slower
    def _load_markers(self, verbose):
        # Default loading, no frequency filtering
        f = open(self.markers_filename, 'r')
        all_snps = []
        for line in f:
            cols = line[:-1].split()
            id = cols[0]
            if self.snpid_check:
                # snp_id should be 'rs' followed by a number, otherwise, ignore it
                if ((id[:2] != 'rs' or (not id[2:].isdigit())) and id[:3] != 'Chr'):  # Adapted for A thaliana as well
                    continue
            ref = cols[2].upper()
            if (ref != 'A' and ref != 'T' and ref != 'C' and ref != 'G'):
                continue
            alt = cols[3].upper()
            # should be A, T, C, G
            if (alt != 'A' and alt != 'T' and alt != 'C' and alt != 'G'):
                continue
            pos = cols[1]
            all_snps.append([id, pos, ref, alt])
        f.close()
        verboseprint('Markers loaded.', verbose)
        self.all_snps = all_snps

    def _load_population(self):
        # Load the population ids (if needed)
        individuals = set()
        popfile = open(self.population_filename, 'r')
        for line in popfile:
            line = line[:-1]
            individuals.add(line)
        popfile.close()
        self.population_list = individuals

    def _get_all_snps_dict(self):
        # returns the dictionary with all SNPs
        if self.all_snps_dict is None:
            all_dict = dict()
            for snp in self.all_snps:
                all_dict[snp[0]]=snp[1:]
            self.all_snps_dict = all_dict
        return self.all_snps_dict

    def _load_genotypes_bgl(self, verbose=False):
        # Generates a genotype dict for the entire chromosome
        # Check if exists and load population of interest:
        usepop_list = self.population_filename is not None
        if usepop_list:
            verboseprint('Loading population of interest IDs.', verbose)
            self._load_population()

        # Load references
        all_dict = self._get_all_snps_dict()
        with open(self.genotype_filename, 'r') as f:
            verboseprint('Loading genotype file, this can take a while...', verbose)
            # Get IDs of interest on the first line of the haps_file, only if a pop file was indicated
            indv_idx = list()
            fline = f.readline()
            cols = fline[:-1].split()
            if usepop_list:
                for i in range(2, len(cols)):
                    if (cols[i] in self.population_list):
                        indv_idx.append(i)
            else:
                # Use all samples from the reference panel
                indv_idx = range(2, len(cols))

            genotype_dict = dict()
            m = 0
            for line in f:
                cols = line[:-1].split()
                snp_id = cols[1]
                # SNP check
                if self.snpid_check and (snp_id[0:2] != "rs" and snp_id[0:3] != "Chr"):  # ONLY FOR HUMAN SAMPLES!! Adapted for A thaliana
                    continue
                # Skip SNPs that are not in the markers; file
                if not snp_id in all_dict:
                    m += 1
                    continue
                str_hap = []
                ref = all_dict[snp_id][1]
                for i in indv_idx:
                    if cols[i] == ref:
                        str_hap.append(1)
                    else:
                        str_hap.append(0)
                genotype_dict[snp_id] = np.asarray(str_hap)
            if m != 0:
                verboseprint('{} SNPs had to be skipped because they were not found in the .markers file.', verbose)
            verboseprint('Haps loaded. The dictionary has {} entries.'.format(len(genotype_dict)), verbose)
        self.genotype_dict = genotype_dict

    def _bgl_to_array(self):
        """
        Return the arrays for the haps of interest in all_snps
        """
        # Check that array is None
        if self.genotype_array is not None:
            raise Warning('You already loaded a genotype array, overwrite prevented')
        self.genotype_array = np.array([self.genotype_dict[x[0]] for x in self.all_snps])
        self.genotype_map = np.array([x[0] for x in self.all_snps])
        # Clear dict content to save memory and gc
        self.genotype_dict.clear()
        gc.collect()

    # ------------------
    # LOAD GENOTYPES
    # ------------------
    def _load_genotypes(self, verbose=False):
        # Check if the file format is .npy, .bgl or .vcf
        filext = os.path.splitext(self.genotype_filename)
        if filext == '.npy':
            self.genotype_array, self.genotype_map = _load_genotypes_npy(self.genotype_filename, verbose)
            self.filetype = 'numpy'
            # If the genotypes are loaded without a map file (containing all the SNP IDs on separated lines),
            # ARDISS assumes that the numpy array matches the markers file (for the SNPs that pass the format conditions,
            # see _load_markers)
            if self.genotype_map is None:
                assert (len(self.all_snps) == self.genotype_array.shape[0]), \
                    'The number of SNPs in the markers file ({}) doesn\'t match the one in the numpy file({})'.format(len(self.all_snps), self.genotype_array.shape[0])

        elif filext == '.bgl':
            # Try to load the dictionary
            self._load_genotypes_bgl(verbose)
            # Transform bgl to array file
            self._bgl_to_array()
            # If save array flag is on, save numpy array, else print an info message
            # TODO: save array to file?



        elif filext == '.vcf':
            # Rely on script to transform it
            raise Warning('VCF file support is not implemented yet.')
            pass  # TODO: add vcf support
        else:
            raise IOError('The provided file is not supported as an input format for the reference panel genotypes. '
                          'Please use .bgl or .npy')
        verboseprint('Genotypes loaded.', verbose)

    def load_files(self, verbose=False):
        # 1. Load the markers file
        self._load_markers(verbose)
        # 2. Load the genotypes
        self._load_genotypes(verbose)
        # 3. Filter the array for MAF requirement

        pass

def _load_genotypes_npy(filename, verbose=False):
    # Return the loaded array
    # Try to open the numpy file
    try:
        verboseprint('Loading the reference genotypes from the numpy file...', verbose)
        all_geno = np.load(filename)
        # TODO: add checks about dtype and only presence of 0 and 1s
    except IOError:
        print('The provided file is not a numpy array, although its extension is .npy ({})'.format(filename))

    # Check if there is a map file accompanying the array file. The map file is expected to only have the SNPs ID, one per line
    snp_maps = None
    if os.path.isfile(filename + '.map'):
        with open(filename + '.map', 'r') as f:
            snp_maps = []
            for line in f:
                snp_maps.append(line[:-1]) # Append the SNP_id
        # Check if the dimensions match
        assert (len(snp_maps) == all_geno.shape[0]),\
            'The number of SNPs in the map file ({}) doesn\'t match the one in the numpy file({})'.format(len(snp_maps), all_geno.shape[0])
    else:
        verboseprint('WARNING: there are no map files associated with the numpy array ({}). '
                     'ARDISS will assume that the provided array contains all the SNPs in their order'.format(filename + '.map'), verbose)

    return all_geno, snp_maps

def get_haps_arrays(haps_dict, all_snps):
    """
    Return the arrays for the haps of interest in all_snps
    """
    all_haps = np.array([haps_dict[x[0]] for x in all_snps])
    return all_haps

class TypedData(object):
    """
    Class that contains all the files associated with the typed values
    """
    def __init__(self, typed_filename, snpid_check=True):
        """
        Initiate the object with filename and options
        :param typed_filename:
        :param snpid_check:
        """
        self.typed_filename = typed_filename
        self.snpid_check = snpid_check

        # Loaded files
        self.typed_snps = None

        # Auxiliary data structures
        self.typed_dict = None


    def load_typed_snps(self, verbose=False):
        # Load the SNPs from the typed_file.
        # Load typed_file
        f = open(self.typed_filename, "r")
        # Skip header
        f.readline()

        self.typed_dict = dict()
        pos_dict = dict()
        id_list = []
        for line in f:
            cols = line[:-1].split()
            id = cols[0]
            if self.snpid_check and (id[0:2] != "rs" and id[0:3] != "Chr"):
                continue
            pos_dict[id] = int(cols[1])
            self.typed_dict[id] = cols[1:]
            id_list.append(id)
        s_keys = sorted(id_list, key=pos_dict.__getitem__)
        self.typed_snps = []
        for snp_id in s_keys:
            self.typed_snps.append([snp_id] + self.typed_dict[snp_id])
        f.close()

    def