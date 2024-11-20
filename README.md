[![Python CI](https://github.com/weberlab-hhu/Helixer/actions/workflows/python-app.yml/badge.svg)](https://github.com/weberlab-hhu/Helixer/actions/workflows/python-app.yml)   
![](img/helixer6.svg)   
Helixer is a tool for structural genome annotation. It utilizes Deep
Neural Networks and a Hidden Markov Model to directly provide primary
gene models in a gff3 file. It’s performant and applicable to a wide
variety of genomes. However, users should be aware that this software
is under ongoing development and improvements.

## Table of contents
1. [Goal](#goal)
2. [Web tool](#web-tool)
3. [Install](#install)
4. [Network architecture](#helixers-architecture)
5. [Example usage](#example-usageinference-gene-calling)
6. [Expert mode](#expert-mode)
7. [Citation](#citation)

## Goal
Perform _ab initio_ prediction of the gene structure for your species.
That is, to perform "gene calling" and identify
which base pairs in a genome belong to the UTR/CDS/Intron regions of genes. 
We have four trained models available for the four lineages: fungi,
land_plant, vertebrate and invertebrate.

## Web tool
Inference on one to a few genomes can be performed
using the Helixer web tool: https://plabipd.de/helixer_main.html.
You can then skip the installation instructions down below.
> **Submission instructions:**
> - submit your genome/sequence in a valid FASTA format
> - minimum sequence length of a single record: 25 kbp
> - maximum file size (including all records): 1 GByte (Hint: if your 
> genome exceeds the file size you could split your genome by chromosome
> or submit a compressed file ('.gz' '.zip' and '.bz2' are supported)

## Install
The installation time depends on the installation method you are using (e.g.
[docker/singularity](#via-docker--singularity-recommended) or
[manual installation](#manual-installation)) and your experience in using GitHub, Python and
CUDA. The time it takes a decently experienced user to install Helixer is 20-30 minutes.

### GPU requirements
For realistically sized datasets, a GPU will be necessary
for acceptable performance.

The example below and all provided models should run on 
an Nvidia GPU with 11GB Memory (e.g. GTX 1080 Ti) and with 8 Gb (e.g. GTX 1080).

The driver for the GPU must also be installed.
The following drivers (top level version) were shown to work with Helixer
(you DON'T need to install one of these versions specifically,
every Nvidia driver should work):

* nvidia-driver-495
* nvidia-driver-510
* nvidia-driver-525
* nvidia-driver-555



### via Docker / Singularity (recommended)
See https://github.com/gglyptodon/helixer-docker

> Additionally, please see notes on usage, which will differ
> slightly from the example below. 

### Manual installation
Please see [full installation instructions](docs/manual_install.md)

## Galaxy ToolShed
There is also a [Galaxy installation](https://usegalaxy.eu/?tool_id=toolshed.g2.bx.psu.edu%2Frepos%2Fgenouest%2Fhelixer%2Fhelixer%2F0.3.3%2Bgalaxy1&version=latest)
of Helixer which you can use for inference.
## Helixer's architecture
![](img/network.png)
## Example usage/inference (gene calling)
If you want to use Helixer to annotate a genome with a provided model, start here.
The best models are:

| Lineage (choose the lineage your species belongs to for prediction) | Model filename              | Available since (year/month/date) |
|:--------------------------------------------------------------------|:----------------------------|:----------------------------------|
| fungi                                                               | fungi_v0.3_a_0100.h5        | 2022/11/21                        |                                            
| land_plant                                                          | land_plant_v0.3_a_0080.h5   | 2022/11/28                        |
| vertebrate                                                          | vertebrate_v0.3_m_0080.h5   | 2022/12/30                        |
| invertebrate                                                        | invertebrate_v0.3_m_0100.h5 | 2022/12/30                        |

### Acquire models
The best models for all lineages are best downloaded by running:

```bash
# the models will be at /home/<user>/.local/share/Helixer/models
scripts/fetch_helixer_models.py
```

If desired, the `--lineage` (`land_plant`, `vertebrate`, `invertebrate`,
and `fungi`) can be specified, or `--all` released models
can be fetched. 

Downloaded models (and any new releases) can also be found at
https://zenodo.org/records/10836346, but we recommend simply using
the autodownload.


>Note: to use a non-default model, set
`--model-filepath <path/to/model.h5>'`,
to override the lineage default for `Helixer.py`. 

### 1-step inference (recommended)
The command below converts the input DNA sequence to numerical
matrices, predicts base-wise class probabilities (is a base pair
part of the **intergenic region**, **UTR**, **CDS** or **intron**)
with a Deep Learning based model and post-processes those probabilities
into primary gene models returning a gff3 output file.
Explanations for the parameters used in this example can be found
[a little further down below](#1-step-inference-parameters).
```bash
# download an example chromosome
wget ftp://ftp.ensemblgenomes.org/pub/plants/release-47/fasta/arabidopsis_lyrata/dna/Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa.gz
# you can also unzip the fasta file (i.e. gunzip Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa.gz),
# but it's not necessary as Helixer can handle zipped fasta files as well

# run all Helixer components from fa to gff3
Helixer.py --lineage land_plant --fasta-path Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa.gz  \
  --species Arabidopsis_lyrata --gff-output-path Arabidopsis_lyrata_chromosome8_helixer.gff3
```
##### 1-step inference parameters
| Parameter         | Default | Explanation                                                                                       |
|:------------------|:--------|:--------------------------------------------------------------------------------------------------|
| --fasta-path      | /       | FASTA input file                                                                                  |
| --gff-output-path | /       | Output GFF3 file path                                                                             |
| --species         | /       | Species name. Will be added to the GFF3 file.                                                     |
| --lineage         | /       | What model to use for the annotation. Options are: vertebrate, land_plant, fungi or invertebrate. |

### 3-step inference
The three main steps the command above executes can also be run separately:
- [fasta2h5.py](fasta2h5.py): conversion of the DNA sequence to numerical matrices
- [HybridModel.py](helixer/prediction/HelixerModel.py): prediction of base-wise
probabilities with the Deep Learning based model defined/programmed in this file
- [helixer_post_bin](https://github.com/TonyBolger/HelixerPost) (part of another
repository): post-processing into primary gene models

Explanations for the parameters used in this example can be found
[a little further down below](#3-step-inference-parameters). You can also check out
the respective help functions or the [Helixer options documentation](docs/helixer_options.md) for
additional usage information, if necessary.
```bash
# example broken into individual steps
# ---------------------------------------
# Consider adding the --subsequence-length parameter:  This number should be large enough to contain typical gene lengths of your species
# while being divisible by at least the timestep width of the used model, which is typically 9. (Lineage dependent defaults)
# Recommendations per lineage: vertebrate: 213840, land_plant: 106920, fungi: 21384, invertebrate: 213840
# Default: 21384
fasta2h5.py --species Arabidopsis_lyrata --h5-output-path Arabidopsis_lyrata.h5 --fasta-path Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa.gz

# the exact location ($HOME/.local/share/) of the model comes from appdirs
# the model was downloaded when fetch_helixer_models.py was called above
# this example code is for _linux_ and will need to be modified for other OSs
# the command runs HybridModel.py in verbose mode with overlap (this will
# improve prediction quality at subsequence ends by creating and overlapping 
# sliding-window predictions.)
HybridModel.py --load-model-path $HOME/.local/share/Helixer/models/land_plant/land_plant_v0.3_a_0080.h5 \
     --test-data Arabidopsis_lyrata.h5 --overlap --val-test-batch-size 32 -v --predict-phase

# order of input parameters:
# helixer_post_bin <genome.h5> <predictions.h5> <window_size> <edge_threshold> <peak_threshold> <min_coding_length> <output.gff3>
helixer_post_bin Arabidopsis_lyrata.h5 predictions.h5 100 0.1 0.8 60 Arabidopsis_lyrata_chromosome8_helixer.gff3
```

**Output:** The main output of the above commands is the gff3 file
(Arabidopsis_lyrata_chromosome8_helixer.gff3) which contains the predicted
genic structure (where the exons, introns, and coding regions are
for every predicted gene in the genome). You can find more about the format 
[here](https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md).
You can readily derive other files, such as a fasta file of the proteome or
the transcriptome, using a standard parser, for instance [gffread](https://github.com/gpertea/gffread).  

##### 3-step inference parameters
###### fasta2h5.py
| Parameter            | Default | Explanation                                                               |
|:---------------------|:--------|:--------------------------------------------------------------------------|
| --fasta-path         | /       | **Required**; FASTA input file                                            |
| --h5-output-path     | /       | **Required**; HDF5 output file for the encoded data. Must end with ".h5". |
| --species            | /       | **Required**; Species name. Will be added to the .h5 file.                |
###### HybridModel.py
| Parameter             | Default | Explanation                                                                                                                                                                                                                  |
|:----------------------|:--------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -l/--load-model-path  | /       | Path to a trained/pretrained model checkpoint. (HDF5 format)                                                                                                                                                                 |
| -t/--test-data        | /       | Path to one test HDF5 file.                                                                                                                                                                                                  |
| --overlap             | False   | Add to improve prediction quality at subsequence ends by creating and overlapping sliding-window predictions (with proportional increase in time usage).                                                                     |
| --val-test-batch-size | 32      | Batch size for validation/test data                                                                                                                                                                                          |
| -v/--verbose          | False   | Add to run HybridModel.py in verbosity mode (additional information will be printed)                                                                                                                                         |
| --predict-phase       | False   | Add this to also predict phases for CDS (recommended);  format: [None, 0, 1, 2]; 'None' is used for non-CDS regions, within CDS regions 0, 1, 2 correspond to phase (number of base pairs until the start of the next codon) |
###### helixer_post_bin
(positional arguments, not specified via name but order)   
   
| Parameter         | Parameter position | Default | Explanation                                                                                                                                                                                           |
|:------------------|:-------------------|:--------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| genome.h5         | 1                  | /       | HDF5 file containing the genome assembly; output of `fasta2h5.py`                                                                                                                                     |
| predictions.h5    | 2                  | /       | HDF5 file containing the predictions from Helixer; output of `HybridModel.py`                                                                                                                         |
| window-size       | 3                  | 100     | Width of the sliding window that is assessed for intergenic vs genic (UTR/Coding Sequence/Intron) content                                                                                             |
| edge-threshold    | 4                  | 0.1     | Threshold specifies the genic score which defines the start/end boundaries of each candidate region within the sliding window                                                                         |
| peak-threshold    | 5                  | 0.8     | Threshold specifies the minimum peak genic score required to accept the candidate region; the candidate region is accepted if it contains at least one window with a genic score above this threshold |
| min-coding-length | 6                  | 60      | Output is filtered to remove genes with a total coding length shorter than this value                                                                                                                 |
| output.gff3       | 7                  | /       | Output GFF3 file path                                                                                                                                                                                 |

### Genome dependent parameters
Most parameters from `Helixer.py` have been set to a reasonable default (again you can look
at the [Helixer options documentation](docs/helixer_options.md)); but nevertheless there
are a couple where the best setting is genome dependent.  

1. `--lineage` or `--model-filepath`   
It is of course critical to choose a model appropriate for your phylogenetic
range / trained on species that generalize well to your target species. When
in doubt selection via `--lineage` is recommended, as this will use the best
available model for that lineage (one of `land_plant`, `vertebrate`, `invertebrate`,
and `fungi`.).

2. `--subsequence-length` and overlapping parameters
    > From v0.3.1 onwards these parameters are set to reasonable defaults (see the
     [general recommendations section](#general-recommendations-for-inference))
     when `--lineage` is used, but `--subsequence-length` will still need to be specified
     when using `--model-filepath`, while the overlapping parameters can be derived
     automatically. These parameters are:
    >- `--overlap-offset`: Distance to 'step' between predicting subsequences when overlapping.
    Default: subsequence-length/2  
    >- `--overlap-core-length`:  Predicted sequences will be cut to this length to increase
    prediction quality if overlapping is enabled. Default: subsequence-length*3/4

    Subsequence length controls how much of the genome the Neural Network can see at once,
    and should ideally be comfortably longer than the typical gene. 
\
\
    For genomes with large genes (i.e. there are frequently > 20kbp genomic loci),
    `--subsequence-length` should be increased. This is particularly common for
    vertebrates and invertebrates but can also happen in plants. For efficiency,
    the overlap parameters should increase as well. It might then be necessary to
    decrease `--batch-size` if the GPU runs out of memory.
\
\
    However, the overlap parameters should definitely not be higher than the N50,
    or even the N90 of the genome. Nor so high a reasonable batch size cannot be used. 
    
    ##### General recommendations for inference
    | model         | --subsequence-length        | --overlap-offset           | --overlap-core-length      |
    |:--------------|:----------------------------|:---------------------------|:---------------------------|
    | fungi         | 21384                       | 10692                      | 16038                      | 
    | plants        | 64152 (or try up to 106920) | 32076 (or try up to 53460) | 48114 (or try up to 80190) |
    | invertebrates | 213840                      | 106920                     | 160380                     |
    | vertebrates   | 213840                      | 106920                     | 160380                     |

3. `--peak-threshold` affects the precision <-> recall balance  
In particular, increasing the peak threshold from the default of 0.8 has been reported to increase the precision
of predictions, with very minimal reduction in e.g. [BUSCO](https://busco.ezlab.org/) scores. Values such as 0.9, 0.95 and 0.975 are 
very reasonable to try. 

## Expert mode
### Developer installation
For developers and experts please see [dev installation instructions](docs/dev_install.md).

### Training and evaluation
If the provided models don't work for your needs, 
information on [training and evaluating](docs/training.md) the models can be found in the [docs folder](docs), 
as well as notes on experimental ways to [fine-tune](docs/fine_tuning.md) 
the network for target species including a hack to include RNA-seq data in the input.

## Citation

##### Full Applicable Tool 

Felix Holst, Anthony Bolger, Christopher Günther, Janina Maß, Sebastian Triesch, Felicitas Kindel, Niklas Kiel, Nima Saadat, Oliver Ebenhöh, Björn Usadel, Rainer Schwacke, Marie Bolger, Andreas P.M. Weber, Alisandra K. Denton.
Helixer&mdash;_de novo_ Prediction of Primary Eukaryotic Gene Models Combining Deep Learning and a Hidden Markov Model.
_bioRxiv_ 2023.02.06.527280; doi: https://doi.org/10.1101/2023.02.06.527280 

##### Original Development and Description of Deep Neural Network for base-wise predictions

Felix Stiehler, Marvin Steinborn, Stephan Scholz, Daniela Dey, Andreas P M Weber, Alisandra K Denton.
Helixer: Cross-species gene annotation of large eukaryotic genomes using deep learning. _Bioinformatics_, btaa1044, 
https://doi.org/10.1093/bioinformatics/btaa1044

