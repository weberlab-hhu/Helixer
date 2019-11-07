"""check whether the coverage actually looks like RNAseq reads come from a stranded protocol"""
import argparse
import h5py


def match_strands(h5):
    """returns list of (idx +, idx -) tuples for matching positions in h5"""
    pass


def select_chunks(n, coverage_min, idx_pairs):
    """takes random draws from idx_pairs until n pairs of chunks passing the coverage min have been found"""
    pass


def correlation_stats(chunk_pairs):
    """calculates std quantiles for the pearson correlation between all chunk_pairs"""
    pass


def main(h5_data, select_n, coverage_min):
    h5 = h5py.File(h5_data, 'r')
    idx_pairs = match_strands(h5)
    chunk_pairs = select_chunks(select_n, coverage_min, idx_pairs)
    corr_quantiles = correlation_stats(chunk_pairs)
    # print / save results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--h5_data', help="h5 produced by rnaseq.py")
    parser.add_argument('-n', '--select_n', help="base estimate on this many randomly selected chunks",
                        default=1000, type=int)
    parser.add_argument('-c', '--coverage_min', help="skip chunks with less than this mean coverage on either strand",
                        default=0.1, type=float)
    args = parser.parse_args()
    main(args.h5_data, args.select_n, args.coverage_min)
