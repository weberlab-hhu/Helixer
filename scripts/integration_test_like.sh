#!/bin/bash
# runs all the major modes such as training/inference += weighting, overlapping, etc; does not (yet) check results
# create directory for all files
wdir=integration_test_like_working_dir
mkdir $wdir
# cache full path for simplicity
wdir=`readlink -f $wdir`
cd $wdir

## work tracking prep
echo_both ()
{
  echo "$@"
  echo "$@" >&2
}
## DATA PREP
echo_both "------ DATA PREP ------"

# download chromosome file to train/validation/test (note this should normally be done _between_ species)
mkdir datain
cd datain
# test
mkdir -p test/input
cd test/input
wget ftp://ftp.ensemblgenomes.org/pub/plants/release-47/gff3/arabidopsis_lyrata/Arabidopsis_lyrata.v.1.0.47.chromosome.8.gff3.gz
wget ftp://ftp.ensemblgenomes.org/pub/plants/release-47/fasta/arabidopsis_lyrata/dna/Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa.gz
gunzip *
# val
cd $wdir/datain
mkdir -p validation/input
cd validation/input
wget ftp://ftp.ensemblgenomes.org/pub/plants/release-47/gff3/arabidopsis_lyrata/Arabidopsis_lyrata.v.1.0.47.chromosome.7.gff3.gz
wget ftp://ftp.ensemblgenomes.org/pub/plants/release-47/fasta/arabidopsis_lyrata/dna/Arabidopsis_lyrata.v.1.0.dna.chromosome.7.fa.gz
gunzip *
# train
cd $wdir/datain
mkdir -p training/input
cd training/input
wget ftp://ftp.ensemblgenomes.org/pub/plants/release-47/gff3/arabidopsis_lyrata/Arabidopsis_lyrata.v.1.0.47.chromosome.1.gff3.gz
wget ftp://ftp.ensemblgenomes.org/pub/plants/release-47/fasta/arabidopsis_lyrata/dna/Arabidopsis_lyrata.v.1.0.dna.chromosome.1.fa.gz
gunzip *

## DATA PRE-PROCESSING
echo_both "------ DATA PRE-PROCESSING ------"

# geenuff
cd $wdir/datain
for bdir in `ls`;
do
  import2geenuff.py --basedir $bdir --species Arabidopsis_lyrata
done

# export to fully labeled h5 files
cd $wdir
mkdir -p h5s/train
mkdir h5s/test
for mode in training validation
do
  geenuff2h5.py --h5-output-path h5s/train/${mode}_data.h5 \
    --input-db-path $wdir/datain/$mode/output/Arabidopsis_lyrata.sqlite3
done
geenuff2h5.py --h5-output-path h5s/test/test_data.h5 \
  --input-db-path $wdir/datain/test/output/Arabidopsis_lyrata.sqlite3

## TRAINING
echo_both "------ TRAINING ------"

cd $wdir
mkdir training
cd training
# very short training with fancier parameters called
$hppath/helixer/prediction/HybridModel.py -d $wdir/h5s/train --class-weights "[0.7, 1.6, 1.2, 1.2]" \
  --transition-weights "[1, 12, 3, 1, 12, 3]" --predict-phase -e 5 --learning-rate 1e-2

## EVAL
cd $wdir
mkdir inference
cd inference
echo_both "------ EVAL -------"
echo_both "- eval vanilla - "
$hppath/helixer/prediction/HybridModel.py --test-data $wdir/h5s/test/test_data.h5 --predict-phase --eval \
  --load-model-path ../training/best_model.h5
echo_both "- eval overlap - "
$hppath/helixer/prediction/HybridModel.py --test-data $wdir/h5s/test/test_data.h5 --predict-phase --eval --overlap \
  --load-model-path ../training/best_model.h5

## PREDICT
echo_both "------ PREDICT ------"
echo_both "- pred vanilla -"
$hppath/helixer/prediction/HybridModel.py --test-data $wdir/h5s/test/test_data.h5 --predict-phase \
  --prediction-output-path predictions_retrained.h5 --load-model-path ../training/best_model.h5
echo_both "- pred overlap -"
$hppath/helixer/prediction/HybridModel.py --test-data $wdir/h5s/test/test_data.h5 --predict-phase --overlap \
 --prediction-output-path predictions_retrained_overlap.h5 --load-model-path ../training/best_model.h5

## HELIXER POST
cd $wdir
mkdir helixer_post
cd helixer_post
helixer_post_bin $wdir/h5s/test/test_data.h5 $wdir/inference/predictions_retrained_overlap.h5 100 0.1 0.8 60 Arabidopsis_lyrata_hp.gff3

## Helixer.py
# not sure this will work w/o being in exactly the Helixer repository, but it will need to
echo_both "------ Helixer.py ------"
cd $wdir
mkdir models
wget https://uni-duesseldorf.sciebo.de/s/4NqBSieS9Tue3J3/download
mv download models/land_plant.h5

mkdir Helixer_py
Helixer.py --lineage land_plant --fasta-path $wdir/datain/test/input/*.fa  \
  --species Arabidopsis_lyrata --gff-output-path Helixer_py/Arabidopsis_lyrata_chromosome8_helixer.gff3

## todo, test other major functionality
# coverage weighting
# (also adding coverage and scoring)
# all models (not just Hybrid)
# uncertainty?
#
## todo, actually CHECK output, right now this still has to be done manually
# i.e. if you run this, for the moment, grep stderr for -i error and make sure at least
# both .gff3 files are produced and are non-empty