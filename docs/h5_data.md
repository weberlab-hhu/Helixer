# Interpreting h5 input / output formats from Helixer
Helixer uses or produces three types of h5 file: 
input data, trained model, and predictions. 

H5 is a standardized yet flexible format for data
storage with APIs existing to read and modify files
in many languages (we use h5py). Below is some information on
the data we are storing there-in and how.

> Please note, this is a flexible format that
> we have modified or added to repeatedly during
> development. We will almost certainly continue
> to do so in future versions of helixer. 

## Trained Model
This is unmodified from the Keras standard.
See, e.g. https://keras.io/guides/serialization_and_saving/#keras-h5-format
for more information.

## Input data
This is created by the 'exoprt.py' script that
transforms the existing annotations into a variety of
matrices. It can be added to with other scripts in 
helixer/evaluation.

### the 'data' group
This should always be present, an example
file (with 4018 'subsequences' of 21,384bp each) 
would have the following datasets
```
X                        Dataset {4018/Inf, 21384, 4}
err_samples              Dataset {4018/Inf}
fully_intergenic_samples Dataset {4018/Inf}
gene_lengths             Dataset {4018/Inf, 21384}
sample_weights           Dataset {4018/Inf, 21384}
seqids                   Dataset {4018/Inf}
species                  Dataset {4018/Inf}
start_ends               Dataset {4018/Inf, 2}
transitions              Dataset {4018/Inf, 21384, 6}
y                        Dataset {4018/Inf, 21384, 4}
phase                    Dataset {4018/Inf, 21384, 4}
```

#### The key-data
Goes into (almost) every training run of the network.

##### X
This is the sequence data. The last dimension 
corresponds to C, A, T, G in that order. So
`[0, 1, 0, 0]` encodes an 'A'. Ambiguity is
encoded with fractions, so `[0.25, 0.25, 0.25, 0.25]`
encodes the fully ambiguous 'N'. 

The X-data is also critical for indicating where
sequences are padded; `[0, 0, 0, 0]` indicates
padding (of short sequences or sequence ends).

Will be used as input for the network.

##### y
This is the reference annotation data, in matrix
format. The last dimension corresponds to 
[intergenic, UTR, CDS, Intron].

Will be used as labels for the network.

##### phase
This is the phase of the reference annotation, in matrix
format. The last dimension corresponds to 
[None, 0, 1, 2].

Where 'None' is used for non-CDS regions, and within
CDS regions
0, 1, 2 correspond to phase (number of base pairs
until the start of the next codon).

Will be used as labels for the network.

##### sample_weights
One value per base pair. This is used to adjust
the sample_weights and there-by the loss, so 
the network can avoid learning from regions that 
are 

 - 1 if base pair is valid
 - 0 if base pair is invalid (in GeenuFF error mask)

#### Where the subsequence are in the genome(s)
Critical for interpreting anything / quickly knowing
where the data (and later predictions) came from.

#####  species
Species ID, repeated once per subsequence. This is, of
course, important when multiple species are 
in one file (e.g. when training with 6+ species
as was done for vertebrate and plant models).

##### seqids
The ID of the sequence (chromosome, scaffold, contig,
etc...) that a subsequence comes from.

##### start_ends
the start and end
coordinates of a
given subsequence (in that order). When end < start then 
the subsequence is from the minus strand. 
So [0, 21384] is the first 21384bp of a given
sequence and is reverse complented by [21384, 0].
On the plus strand start is inclusive and end
exclusive. 

If |start - end| is less than the subsequence size then
there is padding. You can identify where the padding
is according to data/X or data/y above.

(todo, explain padding better)

#### Additional
Various data matrices that were or are used in a
more experimental or peripheral fashion. To check for biases,
experiment with filtering, or more.
 
##### err_samples
One value per subsequence, derived from sample_weights
 - 1 if the subsequence contains any valid data
 - 0 if the subsequence contains only invalid data

(where valid is the default and invalid means 
any base pairs where GeenuFF masked an error)


##### fully_intergenic_samples 

One value per subsequence, useful for quickly filtering
entirely intergenic regions
 - True if only intergenic base pairs present in reference annotation
 - False otherwise
 
##### gene_lengths
One value per base pair

The length of the _longest_ pre-mRNA overlapping
the base pair.

##### transitions
Last dimension has six possible categories for each base pair, and contains
a binary encoding of the starts and ends of gene features.
Specifically, the six categories are:
```
[transcription start site, 
 start codon,
 donor splice site,
 transscription stop site,
 stop codon,
 acceptor splice site]
```

So, as an example, a stop codon is encoded with 
`[0, 0, 0, 0, 1, 0]`.
Of course, the vast majority of sites are
`[0, 0, 0, 0, 0, 0]` for no transition.
Coordinates should match their usage in GeenuFF, except
that here there are always 2bp marked (this is subject to change).

### The 'evaluation' group (optional)
These two evaluation datasets can be added by the script
helixer/evaluation/training_rnaseq.py, which takes
an indexed bam file and calculates the coverage, and spliced_coverage
for each position on the genome. 

Both data sets have the dimensions 'number subsequences'
by 'subsequence size' in bp, so one value per bp of the genome. 

```
coverage                 Dataset {4018/Inf, 21384}
spliced_coverage         Dataset {4018/Inf, 21384}
```

Coverage is the number of reads mapping at a given position (cigar =, M, or X).

Spliced coverage is the number of reads mapping with a gap, that is a read deletion
or spliced out section at a given position (cigar N or D).

### The 'scores' group
The scores in this group are a sort of RNAseq-based
confidence score for the reference annotation. They are
derived from the data in the evaluation group. 

There are various aggregations and normalizations, but
probably the most useful is 'by_bp', with one score
per genomic bp. A value approaching 0 means poor RNAseq support
for the reference annotation, a value approaching 1 means strong
RNAseq support for the reference annotation. 

For more details see helixer/evaluation/training_rnaseq.py.

Interpret with caution as RNAseq too is far from perfect,
particularly if the RNAseq data isn't of highest quality
and from a variety of tissues & conditions. 

Todo: more complete list

### The 'meta' group
Some meta-data on the different _species_ in any .h5 data file.
Most related to RNAseq and provides values for potentially
normalizing the raw RNAseq some. Yes, this is all very n21384aive
still.

### the Alternative/* groups
The script geenuff2h5.py can be used to add alternative (non-reference, non-helixer)
annotations to the .h5 file, exactly aligned with the other
data. 

Any of these that have been added will be in alternative/<your name here>
and from there have the same format as the original 'data' group.

## predictions output
The predictions file holds, well, the predictions. It does not stand 
alone, but rather requires the sequence location and padding information
from the Input Data above to be interpretable. There is a 1:1 correspondance
between positions in the predictions output and the input data. 
 
### predictions
This is output of the network that has been trained to predict `data/y`
from the input data, thus its categories are [intergenic, UTR, CDS, intron]

### predictions_phase
This is output of the network that has been trained to predict `data/phase`
from the input data, thus its categories are [None, 0, 1, 2]

