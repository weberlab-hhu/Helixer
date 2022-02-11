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

> Coming soon: docker and singularity containers. 
> 
> Until then: see the full install-process below

### Get the code
First, download and checkout the latest release
```shell script
# from a directory of your choice
git clone https://github.com/weberlab-hhu/Helixer.git
cd Helixer
# git checkout dev # v0.2.0
```

### System dependencies

#### Python 3.8 or later

#### Python development libraries
Ubuntu (& co.)
```shell script
sudo apt install python3-dev
```
Fedora (& co.)
```shell script
sudo dnf install python3-devel
```

### Virtualenv (optional)
We recommend installing all the python packages in a
virtual environment: https://docs.python-guide.org/dev/virtualenvs/

For example, create and activate an environment called 'env': 
```shell script
python3 -m venv env
source env/bin/activate
```
The steps below assume you are working in the same environment.

### GPU requirements (optional, but highly recommended for realistically sized datasets)
And to run on a GPU (highly recommended for realistically sized datasets),
everything for tensorflow-gpu is required, 
see: https://www.tensorflow.org/install/gpu


The following has been most recently tested.

python packages:
* tensorflow-gpu==2.7.0

system packages:
* cuda-11-2
* libcudnn8
* libcudnn8-dev
* nvidia-driver-495

A GPU with 11GB Memory (e.g. GTX 1080 Ti) can run the largest 
configurations described below, for smaller GPUs you might
have to reduce the network or batch size.

### Post processor

https://github.com/TonyBolger/HelixerPost

Setup according to included instructions and
further add the compiled `helixer_post_bin` to 
your system PATH. 

### Most python dependencies 
```shell script
pip install -r requirements.txt
```

### Helixer itself

```shell script
# from the Helixer directory
pip install .  # or `pip install -e .`, if you will be changing the code
```

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

```bash
# current code assumes exact directory structure
mkdir models
wget https://uni-duesseldorf.sciebo.de/s/4NqBSieS9Tue3J3/download
mv download models/land_plant.h5
```
Info on the downloaded model (and any new releases) can be found here:
https://uni-duesseldorf.sciebo.de/s/lQTB7HYISW71Wi0

#### Run on target genome
```bash
# download an example chromosome
wget ftp://ftp.ensemblgenomes.org/pub/plants/release-47/fasta/arabidopsis_lyrata/dna/Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa.gz
gunzip Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa.gz
# run all Helixer componets from fa to gff3
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
helixer/prediction/HybridModel.py --load-model-path models/land_plant.h5 --test-data Arabidopsis_lyrata.h5 --overlap --val-test-batch-size 32 -v
helixer_post_bin Arabidopsis_lyrata.h5 predictions.h5 100 0.1 0.8 60 Arabidopsis_lyrata_chromosome8_helixer.gff3
```

#### Citation

Felix Stiehler, Marvin Steinborn, Stephan Scholz, Daniela Dey, Andreas P M Weber, Alisandra K Denton, 
Helixer: Cross-species gene annotation of large eukaryotic genomes using deep learning, Bioinformatics, , btaa1044, 
https://doi.org/10.1093/bioinformatics/btaa1044

Please check back to see if the post-processor and additional developments to the core Helixer
functionality have been published.
