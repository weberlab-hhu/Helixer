# HelixerPrep
Gene calling with Deep Neural Networks.

## Disclaimer
This is beta, or maybe alpha... there is nothing stable here.

## Goal
Setup and train models for _de novo_ prediction of gene structure.
That is, to perform "gene calling" and identify
which base pairs in a genome belong to the UTR/CDS/intron of genes.

## Install 
Preferably in a virtual environment

### non PyPi requirements
Both can best be installed according to their own installation instructions
* dustdas, https://github.com/janinamass/dustdas
* geenuff, https://github.com/weberlab-hhu/GeenuFF

### the rest
```
pip install -r requirements.txt
```

### HelixerPrep itself

```
# from the HelixerPrep directory
python setup.py develop  # or `install`, if someone who isn't working on this actually installs it
```
