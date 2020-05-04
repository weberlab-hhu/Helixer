# HelixerPrep
Gene calling with Deep Neural Networks.

## Disclaimer
This is beta, or maybe alpha... there is nothing stable here.

## Goal
Setup and train models for _de novo_ prediction of gene structure.
That is, to perform "gene calling" and identify
which base pairs in a genome belong to the UTR/CDS/Intron of genes. 
Train one model for a wide variety of genomes.

## Install 
Preferably in a virtual environment:

### Non PyPi Requirements
Both can best be installed according to their own installation instructions
* dustdas, https://github.com/janinamass/dustdas
* geenuff, https://github.com/weberlab-hhu/GeenuFF

### The Rest
```
pip install -r requirements.txt
```

### HelixerPrep itself

```
# from the HelixerPrep directory
python setup.py develop  # or `install`, if someone who isn't working on this actually installs it
```

## Example
We set up a working example with a limited amount of data. For this, we will utilize the GeenuFF database in located at `example/three_algae.sqlite3` to generate a training, validation and testing dataset and then use those to first train a model and then to make gene predictions with Helixer.

Our example database contains the genetic information (that is the information of the FASTA and GFF3 files from the Phytosome 13 database) of three very small algae: *Ostreococcus_sp._lucimarinus*, *Micromonas_sp._RCC299* and *Micromonas_pusilla*. We will train our model with the first two and then predict on the third.

```
# generating the training and validation dataset
# the genome names are a bit different due to the naming used in Phytosome
python3 export.py --db-path-in example/three_algae.sqlite3 --genomes Olucimarinus,MspRCC299 --out-dir example/train 

```

```
# generating the test dataset
python3 export.py --db-path-in example/three_algae.sqlite3 --genomes MpusillaCCMP1545 --out-dir example/test --only-test-set 

```
