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
git checkout dev # v0.2.0
```

### Virtualenv (optional)
We recommend installing all the python packages in a
virtual environment:
https://docs.python-guide.org/dev/virtualenvs/

### System dependencies
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
see: https://www.tensorflow.org/install/gpu#older_versions_of_tensorflow

Most recently tested with the following (but in theory any valid
tensorflow-gpu setup >2.0 should work).

python packages:
* tensorflow-gpu==2.3.0

system packages:
* cuda-toolkit-10-1
* libcudnn7
* libcudnn7-dev
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
Our example database contains the genetic information 
(that is the information of the FASTA and GFF3 files from the 
Phytosome 13 database) of three very small algae: 
*Ostreococcus_sp._lucimarinus*, *Micromonas_sp._RCC299* and *Micromonas_pusilla*. 
We will train our model with the first two and then predict on the third.

```shell script
# generating the training and validation dataset
# the genome names are a bit different due to the naming used in Phytosome
python3 export.py --db-path-in example/three_algae.sqlite3 \
  --genomes Olucimarinus,MspRCC299 --out-dir example/train
```

```shell script
# generating the test dataset
python3 export.py --db-path-in example/three_algae.sqlite3 --genomes MpusillaCCMP1545 \
  --out-dir example/test --only-test-set 
```

We should now have the following files in `example/`. 
Note that the test data was not split into a training 
and validation set due to the `--only-test-set` option: 
```
example/
├── test
│   └── test_data.h5
├── three_algae.sqlite3
└── train
    ├── training_data.h5
    └── validation_data.h5
```

### Model training
Now we use the datasets in `example/train/` to train a model with our 
LSTM architeture for 5 epochs and save the best iteration 
(according to the Genic F1 on the validation dataset) to 
`example/best_helixer_model.h5`. 

```shell script
python3 helixer/prediction/LSTMModel.py --data-dir example/train/ --save-model-path example/best_helixer_model.h5 --epochs 5 --units 64 --pool-size 10
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
> WARNING: we have only tested it briefly, so while it does appear that
> the LSTM models that were trained with tensorflow 1 work with tensorflow 2,
> we make no promises.

We have uploaded pre-trained models under https://zenodo.org/record/3974409. 

#### animals
The animal models were trained on: 
Anabas_testudineus,
Drosophila_melanogaster,
Gallus_gallus,
Mus_musculus,
Oryzias_latipes, and
Theropithecus_gelada

The 'animal' models are recommended for usage within vertebrates.

#### plants
The plant models were trained on:
Arabidopsis_thaliana,
Brachypodium_distachyon,
Chlamydomonas_reinhardtii,
Glycine_max,
Mimulus_guttatus,
Marchantia_polymorpha,
Populus_trichocarpa,
Setaria_italica, and
Zea_mays

The 'plant' models are recommended for usage within Embryophyta.

#### demo
While not strictly recommended (performance is lower than for
within Embryophyta), for compatibility with the rest of the
example, we will demonstrate how to obtain and use the best plant
model for our algae test data. 

```shell script
# download
wget https://zenodo.org/record/3974409/files/plants_a_e10.h5 -P example
# change only the --load-model-path and evaluate as above 
# you don't need to respecify any training parameters unless --pool-size was non-default
python3 helixer/prediction/LSTMModel.py --load-model-path example/plants_a_e10.h5 -test-data example/test/test_data.h5 --eval
```

The same idea can be used for predictions and visualization. To achieve top performance
for *predictions* you should also consider adding the parameter `--overlap` 
and setting `--batch-size` higher (depending on GPU memory).
