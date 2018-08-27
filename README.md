# ARDISS

Automatic Relevance Determination for Imputation of GWAS Summary Statistics (ARDISS) is an adaptive method to impute
missing GWAS summary statistics (e.g. Z-scores) both for mixed-ethnicity and homogeneous cohorts.

The method is described in [Accurate and Adaptive Imputation of Summary Statistics in Mixed-Ethnicity Cohorts](https://test)
and is distributed as a python package. If you use ARDISS, please consider citing the publication.

## Dependencies

ARDISS relies on the following dependencies:
- numpy
- scipy
- scikit-learn
- gpflow
- tensorflow

In particular, ARDISS relies on [GPflow](https://github.com/GPflow/GPflow) to optimize the individual samples' weights
using Gaussian Process Regression. GPflow is a [TensorFlow](www.tensorflow.org) library for Gaussian process models, it has GPU support (for which tensorflow-gpu is required as an optional dependency).

## Installation

We recommend setting up a virtual environment to install ARDISS:
```console
foo@bar:~$ pip install virtualenv
foo@bar:~$ cd /path/to/venvfolder
foo@bar:venvfolder$ virtualenv -p python3 ardissvenv
foo@bar:venvfolder$ source ardissvenv/bin/activate
```

ARDISS can then simply be installed with the following command:

```console
(ardissvenv) foo@bar:~$ pip install ardiss
```

## Imputation
Once the installation is complete, ARDISS is straightforward to run:

```console
(ardissvenv) foo@bar:~$ ardiss --typed_scores /path/to/typed_scores --reference_genotypes /path/to/genotypes --markers /path/to/markers_file
```
See the file format section to find out what structure ARDISS expects. Reference panel datasets are available for the human genome (see Reference panel data section below). Here is a list of all arguments:
```console
  # Required arguments
  --typed_scores        Path to the typed file
  --reference_genotypes Path to the reference file containing the genotypes
  --markers             Path to the reference file containing the details for genotyped SNPs
  
  # Optional arguments
  --output OUTPUT       Path to the output file
  --population POPULATION
                        Path to the population file
  --masked MASKED       Path to the masked file
  --weights WEIGHTS     Path to the pre-computed weights file
  -w WINDOWSIZE, --windowsize WINDOWSIZE
                        Window size for the moving window, must be even (default: 100)
  -m MAF, --maf MAF     Minor Allele Frequency used for filtering (default: 0.0)
  -v, --verbose         Tuns on verbose mode
  -g, --gpu             Turn on GPU usage
  --no_weight_optimization
                        Skip the ARD weight optimization and impute with
                        uniform weights
```
The population file can be used to only select a subset of the samples in the reference panel (only works with `.bgl`), it is a simple file containing the selected sample ids on separate lines.
The user can also provide a file with masked values to evaluate the method on the fly, Pearson's correlation between the imputed and the real values are computed.

## File formats
ARDISS must be provided with three files to run: a file containing the available summary statistics, a file containing information about the SNPs in the reference panel and a file containing the genotypes of the reference panel. These three files are also the ones used by ImpG-Summary and follow the same structure:
- Typed Summary Statistics:
   The file should contain the following columns in a tab separated file: SNP_id, SNP_position, Reference_allele, Alternate_allele, Score. The SNP ids should match the ones in the genotype file / markers files
- Markers:
   This file contains the information about the SNPs in the reference genotypes, it has the following columns in a tab-separated file: SNP_id, SNP_position, Reference_allele, Alternate_allele. We provide reference panel files for the human genome.
- Genotypes:
   ARDISS accepts the `.bgl` format required by ImpG-Summary. The method then automatically transforms the strings into a numpy array. Alternatively, one can directly use numpy arrays with reverse encoding (major alleles -> `1`, minor allele -> `0`, one line per ). A numpy array can be accompanied by a `map` file, a simple text file that has the SNP ids of all the SNP genotyped in the numpy array, if this file is absent, ARDISS assumes that the numpy array contains the same SNPs as the markers file.
   
## Reference panel data
Reference panel datasets for the human genome are provided:
- markers: [markers]()
- genotypes: [beagle format](), [numpy format]()
   
## Tips and tricks
- If you have another favorite reference panel available in `.bgl` format and would like to use it for more than a single imputation, ARDISS also has a command-line tool to transform these into numpy arrays that are loaded **much** faster. Consider running the following command, which will generate a numpy array and a map file containing the ids of the SNPs of interest.
```console
ardiss-transform --reference_genotypes /path/to/bgl_genotypes --markers /path/to/markers_file
```
- GPU support requires `tensorflow-gpu` as a supplementary dependency, this, however only accelerate the weight optimization part but doesn't speed up the imputation step.
