# Helixer
Gene calling with Deep Neural Networks.

## Disclaimer
This is beta, there is nothing stable here.

## Goal
Setup and train models for _de novo_ prediction of gene structure.
That is, to perform "gene calling" and identify
which base pairs in a genome belong to the UTR/CDS/intron of genes.

## Install 
Preferably in a virtual environment

### non PyPi requirement
Can best be installed according it's own installation instructions
* geenuff, use the dev branch, https://github.com/weberlab-hhu/GeenuFF/tree/dev

### the rest
```
pip install -r requirements.txt
```

### Helixer itself

```
# from the Helixer directory
python setup.py develop  # or `install`, if someone who isn't working on this actually installs it
```
