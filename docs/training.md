# Training example

We will set up a working example with a limited amount of data. 
For this, we will process species for each of training, validation and testing.
We will pre-process each species with GeenuFF, and then write to h5 files containing
the numerical matrices that will be directly used during training.

## Generating training-ready data

### Pre-processing w/ GeenuFF
> Note: you will be able to skip working with GeenuFF if you
> only wish to predict, and not train. See instead the
> [fasta2h5.py options](helixer_options.md#2-fasta2h5py-options).

First we will need to pre-process the data (Fasta & GFF3 files)
using GeenuFF. This provides a more biologically-realistic
representation of gene structures, and, most importantly
right now, provides identification and masking of invalid
gene structures from the gff file (e.g. those that don't
have _any_ UTR between coding and intergenic, those where
the start codon is not ATG, or those that
have overlapping exons within one transcript). 

See the GeenuFF repository for more information.

For now, we can just run the provided example script to
download and pre-process our example algae data.

```shell script
# this script downloads + organizes data
# and then runs import2geenuff.py for three algae genomes
wget https://raw.githubusercontent.com/weberlab-hhu/GeenuFF/main/example.sh
# execution will take ~ 1-2 minutes depending on your system
bash example.sh
# store full path in a variable for later usage
export data_at=`readlink -f three_algae`
```

This downloads and pre-processes data for the species
(Chlamydomonas_reinhardtii,  Cyanidioschyzon_merolae, and  Ostreococcus_lucimarinus),
as you can see with `ls $data_at`.

To run other genomes, simply provide a fasta and gff3 
file to the `import2geenuff.py` script according to the help function/the [Helixer options documentation](helixer_options.md#5-import2geenuffpy-options),
and supply a species name. The example.sh shows one way to do so.

### numeric encoding of data
To actually train (or predict) we will need to encode the
data numerically (e.g. as 1s and 0s). 

```shell script
mkdir -p example/h5s
for species in `ls $data_at`
do
  mkdir example/h5s/$species
  geenuff2h5.py --input-db-path $data_at/$species/output/$species.sqlite3 \
    --h5-output-path example/h5s/$species/test_data.h5
done
```
To create the simplest working example, we will use Chlamydomonas_reinhardtii 
for training and Cyanidioschyzon_merolae for validation (normally you
would include multiple species for each) and use Ostreococcus_lucimarinus for
testing / predicting / etc.

```shell script
# The training script requires at least two files in the data folder: one matching
# training_data*h5 and one matching validation_data*h5 (* = bash wildcard), respectively.
#
# For this as-simple-as-possible example we will point to one species each
# of training and validation with symlinks
mkdir example/train
cd example/train/
# set training data 
ln -s ../h5s/Chlamydomonas_reinhardtii/test_data.h5 training_data.h5
# and validation
ln -s ../h5s/Cyanidioschyzon_merolae/test_data.h5 validation_data.h5
cd ../..
```

We should now have the following files in `example/`. 

```
example
├── h5s
│   ├── Chlamydomonas_reinhardtii
│   │   └── test_data.h5
│   ├── Cyanidioschyzon_merolae
│   │   └── test_data.h5
│   └── Ostreococcus_lucimarinus
│       └── test_data.h5
└── train
    ├── training_data.h5 -> ../h5s/Chlamydomonas_reinhardtii/test_data.h5
    └── validation_data.h5 -> ../h5s/Cyanidioschyzon_merolae/test_data.h5
```
> **Side note: Including multiple species in datasets.**
> 
> In order for the model to generalize across species, it's of course
> important to train it on _multiple_ training species, and also to
> validate it on _multiple_ validation species.
>
> All files matching `training_data*h5` and `validation_data*h5` (* = bash wildcard),
> that are found in the directory supplied to `--data-dir` below will be used
> for training, and validation, respectively.
>
> So you could, for instance, include multiple species with naming such
> as that shown below:
>
> ```
> train
> ├── training_data.species_01.h5
> ├── training_data.species_02.h5
> ├── training_data.species_03.h5
> ├── training_data.species_04.h5
> ├── validation_data.species_05.h5
> ├── validation_data.species_06.h5
> ├── validation_data.species_07.h5
> ├── validation_data.species_08.h5
> └── validation_data.species_09.h5
> ```


## Model training
Now we use the datasets in `example/train/` to train a model with our 
LSTM architecture for 5 epochs and save the best iteration 
(according to the Genic F1 on the validation dataset) to 
`example/best_helixer_model.h5`. The parameter `--predict-phase`
is necessary so that the resulting models are compatible with post-processing
via HelixerPost. For a detailed explanation of all possible parameters see the
[Helixer options documentation](helixer_options.md#3-hybridmodelpy-options).

```shell script
HybridModel.py --data-dir example/train/ --save-model-path example/best_helixer_model.h5 \
  --epochs 5 --predict-phase
```

The rest of this example will continue with the model example/best_helixer_model.h5 produced above. 

### Interlude: Full size model
The current 'full size' architecture that has been performing well
in hyperparameter optimization runs is:

```shell script
# the indicated batch size and val-test-batch size have been chosen to work on a GTX 2080ti with 11GB RAM
# and should be set as large as the graphics card will allow. For instance, much of our training was
# done on RTX 8000s with 48GB of ram, and there we could set `--batch-size 240 --val-test-batch-size 480`
# for otherwise comparable hyperparameters
HybridModel.py -v --pool-size 9 --batch-size 50 --val-test-batch-size 100 \
  --class-weights "[0.7, 1.6, 1.2, 1.2]" --transition-weights "[1, 12, 3, 1, 12, 3]" --predict-phase \
  --lstm-layers 3 --cnn-layers 4 --units 128 --filter-depth 96 --kernel-size 10 \
  --data-dir example/train/ --save-model-path example/fullsize_helixer_model.h5
```

But make sure you have a full size dataset to go with that model, and up to a couple of days,
if you're going to train it.

## Model evaluation
We can now use the mini model trained above to produce predictions for our test dataset. 

We could even skip right on ahead to generating gff3 files by
calling `Helixer.py` with `--model-filepath example/best_helixer_model.h5`, but that is covered
in the main README.md, and if you're training models you might want some finer
tuned / intermediate predictions and evaluation, so see below:

> NOTE: Generating predictions can produce very large files as 
> we save every individual softmax value in 32 bit floating point format. 
> For this very small genome the predictions require 524MB of disk space. 
```shell script
HybridModel.py --load-model-path example/best_helixer_model.h5 \
  --test-data example/h5s/Ostreococcus_lucimarinus/test_data.h5 \
  --prediction-output-path example/Ostreococcus_lucimarinus_predictions.h5
```

Or we can directly evaluate the predictive performance of our model. 

```shell script
HybridModel.py --load-model-path example/best_helixer_model.h5 \
  --test-data example/h5s/Ostreococcus_lucimarinus/test_data.h5 \
  --predict-phase --eval
```

The last command can be sped up with a higher batch size and should give us the same break down that is performed 
during training at each check (i.e. 1/epoch by default), but on the test data:

```
+confusion_matrix------+----------+-----------+-------------+
|            | ig_pred | utr_pred | exon_pred | intron_pred |
+------------+---------+----------+-----------+-------------+
| ig_ref     | 1297055 | 151436   | 692534    | 188423      |
| utr_ref    | 159457  | 32821    | 99856     | 16345       |
| exon_ref   | 5014284 | 2002047  | 2001614   | 134915      |
| intron_ref | 199752  | 62334    | 89647     | 9550        |
+------------+---------+----------+-----------+-------------+

+normalized_confusion_matrix------+-----------+-------------+
|            | ig_pred | utr_pred | exon_pred | intron_pred |
+------------+---------+----------+-----------+-------------+
| ig_ref     | 0.5568  | 0.065    | 0.2973    | 0.0809      |
| utr_ref    | 0.5169  | 0.1064   | 0.3237    | 0.053       |
| exon_ref   | 0.5478  | 0.2187   | 0.2187    | 0.0147      |
| intron_ref | 0.5529  | 0.1725   | 0.2481    | 0.0264      |
+------------+---------+----------+-----------+-------------+

+F1_summary--+---------+-----------+--------+----------+
|            | norm. H | Precision | Recall | F1-Score |
+------------+---------+-----------+--------+----------+
| ig         |         | 0.1944    | 0.5568 | 0.2882   |
| utr        |         | 0.0146    | 0.1064 | 0.0257   |
| exon       |         | 0.6941    | 0.2187 | 0.3326   |
| intron     |         | 0.0273    | 0.0264 | 0.0269   |
|            |         |           |        |          |
| legacy_cds |         | 0.6916    | 0.2350 | 0.3508   |
| sub_genic  |         | 0.6221    | 0.2114 | 0.3156   |
| genic      |         | 0.3729    | 0.2081 | 0.2671   |
+------------+---------+-----------+--------+----------+
Total acc: 0.2749

metrics calculation took: 0.08 minutes
```

In the demo run above (yours may vary) the model predicted
perhaps better than random, but poorly. It practically
needs more data and more phylogenetic variation in the dataset (different species),
as well as for the network to be larger and trained for a longer period of time.

## Practical considerations
While the above covers technically how to train a model, here are some
tips on how to make it a _good_ model. 

### Scale
Make sure to use the 'full size' model above, or larger if your resources permit.
Scale up (quality) training data as you scale up the model.

### Training species selection
Unfortunately we have no single 'best method' to pick training species at this point,
but nevertheless some patterns are clear. You will probably want to:
- train on a phylogenetically _wider_ selection of species than you need
  the model to predict on 
- train on a phylogenetically balanced selection of species across the range
  you need the model to predict on (e.g. not 50% mice for a vertebrate model)
- include more and _more diverse_ species to boost performance (current performant released models were trained
  on many dozens of species)
- include only higher quality annotations to boost performance
- figure out the best tradeoff between phylogenetic variety and availability of high quality annotations

Trial and error has been a major part of the training process, 
particularly for species selection. Eventually we automated that
trial and error with many random draws of training species and a 2-fold cross validation 
process [here](https://github.com/alisandra/SpeciesSelector).
Given some compute resources and a target set of genomes, this
is a fairly safe way to achieve a baseline and often quite respectable model.

### Validation Species selection
It is critical here that your validation species are
representative of your target prediction range. As with training,
it is better to have a wider selection than your target predictive
range than narrower. While model parameters are not directly optimized
for validation species, these species _are_ used to select the best
model; so it is critical that they are of high enough quality that
metrics improve when the model is improving and get worse when the model is getting
worse. Validation files _can_ be 
[down sampled](https://github.com/weberlab-hhu/helixer_scratch/blob/master/data_scripts/sample-single-genomes.py)
for speed purposes. 

### Hyperparameter optimization
The Helixer codebase is built to work with [nni](https://github.com/microsoft/nni)
for hyperparameter optimization. If you want to optimize the hyperparameters, we recommend
following standard nni instructions on setting up the config.yml
and search_space.json files and additionally adding
`--nni` to the `HybridModel.py` command.

