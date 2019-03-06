# HelixerPrep
https://xkcd.com/927/ sigh, but bioformats _are_ really frustrating, 
and I need to pre-process them

## Disclaimer
This is beta, or maybe alpha... there is nothing stable here.

## Goal
Pre-processing of sequence, annotation and expression (todo) data with any parsing
troubles being taken care of early, meta-data retained and train/dev/test set assignment.

Further providing api to quickly generate numeric matrices (training examples) 
from the 'cleaned' formats.

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

### Todo, setup.py...

## Example usage
Download your sequence and annotation files 
(e.g. from https://phytozome.jgi.doe.gov/pz/portal.html or
https://www.ensembl.org/index.html) and structure as follows:
```
{somespecies}/
{somespecies}/input/{sequence}.fa
{somespecies}/input/{annotation}.gff3
```
where names contained in brackets `{}` are user-defined.

Then pre-process sequence and annotation data.
```bash
# just pre-process
python prep_example.py --basedir {somespecies}
# divvy up train/dev/test sets
python prep_example.py --basedir {somespecies} --slice
```

Note that this still has some major performance issues, for a small to moderate
sized plant genome, the above command should take something like 5-10min; 
but the bottom command is easily 10x longer.

Also note that the `prep_example.py` script chooses what to re-run for the
first pre-processing based on the presence / absence of files in the `{somespecies}/output/`
directory. So you might have to delete (some of) these files to rerun it... or write
a better example script :-)

