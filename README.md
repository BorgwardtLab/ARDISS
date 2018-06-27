# ARDISS

Automatic Relevance Determination for Imputation of GWAS Summary Statistics (ARDISS) is an adaptive method to impute
missing GWAS summary statistics (e.g. Z-scores) both for mixed-ethnicity and homogeneous cohorts.

The method is described in [Accurate and Adaptive Imputation of Summary Statistics in Mixed-Ethnicity Cohorts](https://test)
and is distributed as a python package

## Dependencies

ARDISS relies on the following dependencies

In particular, ARDISS relies on [GPflow](https://github.com/GPflow/GPflow) to optimize the individual samples' weights
using Gaussian Process Regression.

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