# -----------------------------------------------------------------------------
# This is the wrapper for the FLASH module
#
# July 2018, M. Togninalli
# -----------------------------------------------------------------------------
import argparse
import ARDISS

def main():
    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(description="Impute Summary Statistics using Automatic Relevance Determination")
    parser.add_argument("--typed", required=True,
                        help="Filename for the typed values.")
    parser.add_argument("--haps", required=True,
                        help="Filename for the reference SNPs values.")
    parser.add_argument("--out", required=True,
                        help="Path to the output file")
    parser.add_argument("--pop", required=True,
                        help="Path to the population file")
    parser.add_argument("--markers", required=True,
                        help="Path to the markers file")
    parser.add_argument("--masked", required=False, default="",
                        help="Path to the masked file")

    args = parser.parse_args()

    ARDISS.impute_ard(args.typed, args.haps, args.out, args.pop, args.markers, masked_file=args.masked, window_size=100,
                      maf=0.01, verbose=True)

if __name__ == "__main__":
    main()
