# Helixer
Gene calling with Deep Neural Networks.

## Disclaimer
This software is very much beta and probably not stable enough to build on (yet).

## Goal
Setup and train models for _de novo_ prediction of gene structure.
That is, to perform "gene calling" and identify
which base pairs in a genome belong to the UTR/CDS/Intron of genes. 
Train one model for a wide variety of genomes.

## Install 

### Get the code
First, download and checkout the latest release
```shell script
# from a directory of your choice
git clone https://github.com/weberlab-hhu/Helixer.git
cd Helixer
# git checkout dev # v0.2.0
```

### Virtualenv (optional)
We recommend installing all the python packages in a
virtual environment:
https://docs.python-guide.org/dev/virtualenvs/

### System dependencies

#### Python 3.6 or later

#### Python development libraries
Ubuntu (& co.)
```shell script
sudo apt install python3-dev
```
Fedora (& co.)
```shell script
sudo dnf install python3-devel
```

### GPU requirements (optional, but highly recommended for realistically sized datasets)
And to run on a GPU (highly recommended for realistically sized datasets),
everything for tensorflow-gpu is required, 
see: https://www.tensorflow.org/install/gpu


python packages:
* tensorflow-gpu>=2.6.2

system packages:
* cuda-11-0
* libcudnn8
* libcudnn8-dev
* nvidia-driver-450

A GPU with 11GB Memory (e.g. GTX 1080 Ti) can run the largest 
configurations described below, for smaller GPUs you might
have to reduce the network or batch size. 
  
### Most python dependencies

```shell script
pip3 install -r requirements.txt
```

### Helixer itself

```shell script
# from the Helixer directory
python3 setup.py install  # or `develop`, if you will be changing the code
```

## Example
We set up a working example with a limited amount of data. 
For this, we will utilize the GeenuFF database in 
located at `example/three_algae.sqlite3` to generate a training, 
validation and testing dataset and then use those to first train a model 
and then to make gene predictions with Helixer.

### Generating training-ready data

#### Pre-processing w/ GeenuFF
>Note: you will be able to skip working with GeenuFF if
> only wish to predict, and not train. See instead the
> --direct-fasta-to-h5-path parameter of Helixer/export.py

First we will need to pre-process the data (Fasta & GFF3 files)
using GeenuFF. This provides a more biologically-realistic
representation of gene structures, and, most importantly
right now, provides identification and masking of invalid
gene structures from the gff file (e.g. those that don't
have _any_ UTR between coding and intergenic or those that
have overlapping exons within one transcript). 

See the GeenuFF repository for more information.

For now we can just run the provided example script to
download and pre-process some algae data.

```shell script
cd <your/path/to>/GeenuFF
bash example.sh
# store full path in a variable for later usage
data_at=`readlink -f three_algae`
cd <your/path/to/Helixer>
```
This downloads and pre-processes data for the species
(Chlamydomonas_reinhardtii,  Cyanidioschyzon_merolae, and  Ostreococcus_lucimarinus)
as you can see with `ls $data_at`
#### numeric encoding of data
To actually train (or predict) we will need to encode the
data numerically (e.g. as 1s and 0s). 

```shell script
mkdir example/h5s
for species in `ls $data_at`
do
  mkdir example/h5s/$species
  python3 export.py --input-db-path $data_at/$species/output/$species.sqlite3 \
    --output-path example/h5s/$species/test_data.h5
done
```
To create the simples working example, we will use Chlamydomonas_reinhardtii 
for training and Cyanidioschyzon_merolae for validation (normally you
would merge multiple species for each) and use Ostreococcus_lucimarinus for
testing / predicting / etc.

```shell script
# The training script requires two files in one folder named
# training_data.h5 and validation_data.h5
#
# while we would need to merge multiple datasets to create our
# training_data.h5 and validation_data.h5 normally, for this as-simple-
# as-possible example we will point to one species each with symlinks
mkdir example/train
cd example/train/
# set training data 
ln -s ../h5s/Chlamydomonas_reinhardtii/test_data.h5 training_data.h5
# and validation
ln -s ../h5s/Cyanidioschyzon_merolae/test_data.h5 validation_data.h5
cd ../..
```

We should now have the following files in `example/`. 
Note that the test data was not split into a training 
and validation set due to the `--only-test-set` option: 
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

### Model training
Now we use the datasets in `example/train/` to train a model with our 
LSTM architeture for 5 epochs and save the best iteration 
(according to the Genic F1 on the validation dataset) to 
`example/best_helixer_model.h5`. 

```shell script
python3 helixer/prediction/DanQModel.py --data-dir example/train/ --save-model-path example/best_helixer_model.h5 --epochs 5 
```

Right before the training starts we may get one or two warnings about 
the data that are not relevant for this example. This trains a very 
small single layer BLSTM model with about 60.000 parameters that 
receives 10 basepairs for each time step. After each epoch, 
an evaluation of the last model iteration is performed and 
the results are printed to the console. If we would like to train a model 
with exactly the architecture and weights that was used to train our 
actual plant models one would use the following command: 

```shell script
python3 helixer/prediction/LSTMModel.py --data-dir example/train/ --save-model-path example/best_helixer_model.h5 --epochs 5 --units 256 --pool-size 10 --batch-size 52 --layers 4 --layer-normalization --class-weights '[0.7, 1.6, 1.2, 1.2]'
```

### Model evaluation
We can now use this model to produce predictions for our test dataset. 
WARNING: Generating predictions can produce very large files as 
we save every individual softmax value in 32 bit floating point format. 
For this very small genome the predictions require 524MB of disk space. 
```shell script
python3 helixer/prediction/LSTMModel.py --load-model-path example/best_helixer_model.h5 --test-data example/test/test_data.h5 --prediction-output-path example/mpusillaCCMP1545_predictions.h5
```

Or we can directly evaluate the predictive performance of our model. It is necessary to generate a test data file per species to get results for just that species.

```shell script
python3 helixer/prediction/LSTMModel.py --load-model-path example/best_helixer_model.h5 --test-data example/test/test_data.h5 --eval
```

The last command can be sped up with a higher batch size and should give us the same break down that is performed during after a training epoch, but on the test data:

```
+confusion_matrix------+----------+-----------+-------------+
|            | ig_pred | utr_pred | exon_pred | intron_pred |
+------------+---------+----------+-----------+-------------+
| ig_ref     | 2563098 | 0        | 3752365   | 0           |
| utr_ref    | 118706  | 0        | 489995    | 0           |
| exon_ref   | 888744  | 0        | 13404195  | 0           |
| intron_ref | 522242  | 0        | 1370375   | 0           |
+------------+---------+----------+-----------+-------------+

+normalized_confusion_matrix------+-----------+-------------+
|            | ig_pred | utr_pred | exon_pred | intron_pred |
+------------+---------+----------+-----------+-------------+
| ig_ref     | 0.4058  | 0.0      | 0.5942    | 0.0         |
| utr_ref    | 0.195   | 0.0      | 0.805     | 0.0         |
| exon_ref   | 0.0622  | 0.0      | 0.9378    | 0.0         |
| intron_ref | 0.2759  | 0.0      | 0.7241    | 0.0         |
+------------+---------+----------+-----------+-------------+

+F1_summary--+-----------+--------+----------+
|            | Precision | Recall | F1-Score |
+------------+-----------+--------+----------+
| ig         | 0.6262    | 0.4058 | 0.4925   |
| utr        | 0.0000    | 0.0000 | 0.0000   |
| exon       | 0.7049    | 0.9378 | 0.8048   |
| intron     | 0.0000    | 0.0000 | 0.0000   |
|            |           |        |          |
| legacy_cds | 0.7769    | 0.9128 | 0.8394   |
| sub_genic  | 0.7049    | 0.8282 | 0.7615   |
| genic      | 0.7049    | 0.7981 | 0.7486   |
+------------+-----------+--------+----------+
Total acc: 0.6909
```

In the demo run above (yours may vary) the model only predicted 
intergenic and CDS bases. The main reason for this is likely that 
the data contains very few bases of the other two kinds. 
A more varied training dataset or uneven class weights 
could change that (this result is for the small model example).

When we did write out the predictions to disk we can use an experimental 
visualization to display the predictions together with the reference:
```shell script
python3 helixer/visualization/visualize.py --predictions example/mpusillaCCMP1545_predictions.h5 --test-data example/test/test_data.h5
```

### Using trained models
> WARNING: to use the pre-trained models you will need the published
> version of the code! Run `git checkout v0.2.0` and follow the instructions
> there in. As soon as pre-trained phase-containing models are generally 
> available, this section will be updated.

#### Citation

Felix Stiehler, Marvin Steinborn, Stephan Scholz, Daniela Dey, Andreas P M Weber, Alisandra K Denton, Helixer: Cross-species gene annotation of large eukaryotic genomes using deep learning, Bioinformatics, , btaa1044, https://doi.org/10.1093/bioinformatics/btaa1044
