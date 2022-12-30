[![Python CI](https://github.com/weberlab-hhu/Helixer/actions/workflows/python-app.yml/badge.svg)](https://github.com/weberlab-hhu/Helixer/actions/workflows/python-app.yml)

# Helixer
Gene calling with Deep Neural Networks.

## Disclaimer
This software is undergoing active testing and development.
Build on it at your own risk.

## Goal
Setup and train models for _de novo_ prediction of gene structure.
That is, to perform "gene calling" and identify
which base pairs in a genome belong to the UTR/CDS/Intron of genes. 
Train one model for a wide variety of genomes.

## Install
### GPU requirements
For realistically sized datasets, a GPU will be necessary
for acceptable performance.

The example below and all provided models should run on 
an nvidia GPU with 11GB Memory (e.g. GTX 1080 Ti) 

The diver for the GPU must also be installed.
During development we have used

* nvidia-driver-495
* nvidia-driver-510

and many in between.

### via Docker / Singularity (recommended)
See https://github.com/gglyptodon/helixer-docker

> Additionally, please see notes on usage, which will differ
> slightly from the example below. 

### Manual installation
Please see [full installation instructions](docs/manual_install.md)

#### contributors & team members
Please additionally see [dev installation instructions](docs/dev_install.md)

## Example
This example focuses only on applying trained models for gene calling, only.
Information on training and evaluating the models can be found in `docs`.

### Using trained models
> NOTE: the extensively evaluated models from the paper are available by
> running `git checkout v0.2.0` and following the instructions
> there in. But they were not yet _applicable_ for generating gff3 files.

We are working towards training another round of models w/ the current
architecture. For now a preliminary land plant model is available and
will be used for the rest of the example. 

#### Acquire models
The best models for each or all lineages can automatically 
downloaded with the `fetch_helixer_models.py` script.

The available lineages are `land_plant`, `vertebrate`, `invertebrate`,
and `fungi`.

Info on the downloaded models (and any new releases) can be found here:
https://uni-duesseldorf.sciebo.de/s/lQTB7HYISW71Wi0


>Note: to use a non-default model, set
`--model-filepath <path/to/model.h5>'`,
to override the lineage default for `Helixer.py`. 

#### Run on target genome
```bash
# download an example chromosome
wget ftp://ftp.ensemblgenomes.org/pub/plants/release-47/fasta/arabidopsis_lyrata/dna/Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa.gz
gunzip Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa.gz
# run all Helixer components from fa to gff3
Helixer.py --lineage land_plant --fasta-path Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa  \
  --species Arabidopsis_lyrata --gff-output-path Arabidopsis_lyrata_chromosome8_helixer.gff3
```

The above runs three main steps: conversion of sequence to numerical matrices in preparation (`fasta2h5.py`),
prediction of base-wise probabilities with the Deep Learning based model (`helixer/prediction/HybridModel.py`),
post-processing into primary gene models (`helixer_post_bin`). See respective help functions for additional
usage information, if necessary.

```bash
# example broken into individual steps
fasta2h5.py --species Arabidopsis_lyrata --h5-output-path Arabidopsis_lyrata.h5 --fasta-path Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa
# the exact location ($HOME/.local/share/) of the model comes from appdirs
# the model was downloaded when Helixer.py was called above
# this example code is for _linux_ and will need to be modified for other OSs
HybridModel.py --load-model-path $HOME/.local/share/Helixer/models/land_plant.h5 --test-data Arabidopsis_lyrata.h5 --overlap --val-test-batch-size 32 -v
helixer_post_bin Arabidopsis_lyrata.h5 predictions.h5 100 0.1 0.8 60 Arabidopsis_lyrata_chromosome8_helixer.gff3
```

**Output:** The main output of the above commands is the gff3 file (Arabidopsis_lyrata_chromosome8_helixer.gff3)
which contains the predicted genic structure (where the exons, introns, and coding regions are
for every predicted gene in the genome). You can find more about the format 
[here](https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md),
and you can readily derive other formats, such as a fasta file of the proteome, using
a standard parser, for instance [gffread](https://github.com/gpertea/gffread).

#### Citation

Felix Stiehler, Marvin Steinborn, Stephan Scholz, Daniela Dey, Andreas P M Weber, Alisandra K Denton, 
Helixer: Cross-species gene annotation of large eukaryotic genomes using deep learning, Bioinformatics, , btaa1044, 
https://doi.org/10.1093/bioinformatics/btaa1044

Please check back to see if the post-processor and additional developments to the core Helixer
functionality have been published.
