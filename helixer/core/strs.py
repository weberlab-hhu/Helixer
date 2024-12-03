# model modes
TEST = 'test'
VAL = 'val'
TRAIN = 'train'

# additional overarching goals of running the model scripts
# (for flow control)
EVAL = 'eval'
PREDICT = 'predict'

# zarr file data group datasets
DATA_X = 'data/X'  # (Inf, subsequence_length, 4)
DATA_ERR_SAMPLES = 'data/err_samples'  # (Inf)
DATA_FULLY_INTERGENIC_SAMPLES = 'data/fully_intergenic_samples'  # (Inf)
DATA_IS_ANNOTATED = 'data/is_annotated'  # (Inf)
DATA_GENE_LENGTHS = 'data/gene_lengths'  # (Inf, subsequence_length)
DATA_SAMPLE_WEIGHTS = 'data/sample_weights'  # (Inf, subsequence_length)
DATA_SEQIDS = 'data/seqids'  # (Inf)
DATA_SPECIES = 'data/species'  # (Inf)
DATA_START_ENDS = 'data/start_ends'  # (Inf, 2)
DATA_TRANSITIONS = 'data/transitions'  # (Inf, subsequence_length, 6)
DATA_Y = 'data/y'  # (Inf, subsequence_length, 4)
DATA_PHASES = 'data/phases'  # (Inf, subsequence_length, 4)
