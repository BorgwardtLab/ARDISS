import argparse
import ardiss

def ardiss_transform_console():
    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(description='Transform available bgl files into numpy arrays for faster '
                                                 'loading times.')
    parser.add_argument('-g','--reference_genotypes', required=True,
                        help='Filename for the reference SNPs values in bgl format.')
    parser.add_argument('-m','--markers', required=True,
                        help='Path to the markers file')
    parser.add_argument('--population', required=False, default=None,
                        help='Path to the population file')
    parser.add_argument('-v','--verbose', required=False, action='store_true',
                        help='Tuns on verbose mode')
    parser.add_argument('-c','--no_snpid_check', required=False, action='store_true',
                        help='Tuns off SNP id checking')

    args = parser.parse_args()
    snpid_check = not args.no_snpid_check

    # 1. Use the available reference data class
    RefData = ardiss.ReferenceData(args.reference_genotypes, args.markers, args.population,
                                   snpid_check=snpid_check, verbose=args.verbose)
    RefData.load_files(scale=False)

    # 2. Save to numpy array with map
    RefData.save_to_npy()

def ardiss_console():
    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(description='Impute Summary Statistics using Automatic Relevance Determination')
    parser.add_argument('--typed_scores', required=True,
                        help='Filename for the typed values.')
    parser.add_argument('--reference_genotypes', required=True,
                        help='Filename for the reference SNPs values.')
    parser.add_argument('--markers', required=True,
                        help='Path to the markers file')
    parser.add_argument('--output', required=False, default=None,
                        help='Path to the output file')
    parser.add_argument('--population', required=False, default=None,
                        help='Path to the population file')
    parser.add_argument('--masked', required=False, default=None,
                        help='Path to the masked file')
    parser.add_argument('--weights', required=False, default=None,
                        help='Path to the pre-computed weights file')
    # Imputation parameters
    parser.add_argument('-w', '--windowsize', required=False, default=100, type=int,
                        help='Window size for the moving window, must be even (default: 100)')
    parser.add_argument('-m', '--maf', required=False, default=0.0, type=float,
                        help='Minor Allele Frequency used for filtering (default: 0.0)')
    parser.add_argument('-v','--verbose', required=False, action='store_true',
                        help='Tuns on verbose mode')
    parser.add_argument('-g','--gpu', required=False, action='store_true',
                        help='Turn on GPU usage')
    parser.add_argument('--no_weight_optimization', required=False, action='store_true',
                        help='Skip the ARD weight optimization and impute with uniform weights')

    args = parser.parse_args()

    weight_opt = not args.no_weight_optimization
    if args.output is None:
        args.output = args.typed_scores + '.imputed'

    ardiss.impute_ard(args.typed_scores, args.reference_genotypes, args.output, args.population, args.markers,
                      masked_file=args.masked, weights_file=args.weights, window_size=args.windowsize, maf=args.maf,
                      verbose=args.verbose, snpid_check=True, gpu=args.gpu, weight_optimization=weight_opt)