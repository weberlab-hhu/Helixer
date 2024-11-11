# Helixer options
The most important scripts and their options are listed.
1. [Helixer.py](#1-helixerpy-options)
2. [fasta2h5.py](#2-fasta2h5py-options)
3. [HybridModel.py](#3-hybridmodelpy-options)
4. [HelixerPost](#4-helixerpost-options) (the same as the [post-processing parameters](#post-processing-parameters) for Helixer.py)
5. [import2geenuff.py](#5-import2geenuffpy-options)
6. [geenuff2h5.py](#6-geenuff2h5py-options)

## 1. Helixer.py options
Helixer.py always searches for the configuration file ``config/helixer_config.yaml`` in the current
working directory. If that file isn't provided, the parameters are expected to be given via the
command line.

### General parameters
| Parameter            | Default                                                                   | Explanation                                                                                                                                                                                                                                                                                   |
|:---------------------|:--------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --fasta-path         | /                                                                         | FASTA input file                                                                                                                                                                                                                                                                              |
| --gff-output-path    | /                                                                         | Output GFF3 file path                                                                                                                                                                                                                                                                         |
| --species            | /                                                                         | Species name. Will be added to the GFF3 file.                                                                                                                                                                                                                                                 |
| --temporary-dir      | system default                                                            | Use supplied (instead of system default) for temporary directory (place where temporary h5 files from fasta to h5 conversion and Helixer's raw base-wise predictions get saved)                                                                                                               |
| --subsequence-length | vertebrate: 213840, land_plant: 64152, fungi: 21384, invertebrate: 213840 | How to slice the genomic sequence. Set moderately longer than length of typical genic loci. Tested up to 213840. Must be evenly divisible by the timestep width of the used model, which is typically 9. (Lineage dependent defaults)                                                         |
| --write-by           | 20_000_000                                                                | Convert genomic sequence in super-chunks to numerical matrices with this many base pairs, which will be rounded to be divisible by subsequence-length; needs to be equal to or larger than subsequence length; for lower memory consumption, consider setting a lower number                  |
| --lineage            | /                                                                         | What model to use for the annotation. Options are: vertebrate, land_plant, fungi or invertebrate.                                                                                                                                                                                             |
| --model-filepath     | /                                                                         | Set this to override the default model for any given lineage and instead take a specific model                                                                                                                                                                                                |

### Prediction parameters
| Parameter             | Default                                                                                                     | Explanation                                                                                                                                                                                                                                                                                                                                                                                 |
|:----------------------|:------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --batch-size          | 32                                                                                                          | The batch size for the raw predictions in TensorFlow. Should be as large as possible on your GPU to save prediction time.                                                                                                                                                                                                                                                                   |
| --no-overlap          | False                                                                                                       | Switches off the overlapping after predictions are made. Overlap will improve prediction quality at subsequence ends by creating and overlapping sliding-window predictions. Predictions without overlapping will be faster, but will have lower quality towards the start and end of each subsequence. With this parameter --overlap-offset and --overlap-core-length will have no effect. |
| --overlap-offset      | vertebrate: 106920, land_plant: 32076, fungi: 10692, invertebrate: 106920 (i.e. subsequence_length / 2)     | Distance to 'step' between predicting subsequences when overlapping. Smaller values may lead to better predictions but will take longer. The subsequence_length should be evenly divisible by this value.                                                                                                                                                                                   |
| --overlap-core-length | vertebrate: 160380, land_plant: 48114, fungi: 16038, invertebrate: 160380 (i.e. subsequence_length * 3 / 4) | Predicted sequences will be cut to this length to increase prediction quality if overlapping is enabled. Smaller values may lead to better predictions but will take longer. Has to be smaller than subsequence_length.                                                                                                                                                                     |
| --debug               | False                                                                                                       | Add this to quickly run the code through without loading/predicting on the full file                                                                                                                                                                                                                                                                                                        |

### Post-processing parameters
| Parameter           | Default | Explanation                                                                                                                                                                                           |
|:--------------------|:--------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --window-size       | 100     | Width of the sliding window that is assessed for intergenic vs genic (UTR/Coding Sequence/Intron) content                                                                                             |
| --edge-threshold    | 0.1     | Threshold specifies the genic score which defines the start/end boundaries of each candidate region within the sliding window                                                                         |
| --peak-threshold    | 0.8     | Threshold specifies the minimum peak genic score required to accept the candidate region; the candidate region is accepted if it contains at least one window with a genic score above this threshold |
| --min-coding-length | 60      | Output is filtered to remove genes with a total coding length shorter than this value                                                                                                                 |

## 2. fasta2h5.py options
fasta2h5.py always searches for the configuration file ``config/fasta2h5_config.yaml`` in the current
working directory. If that file isn't provided, the parameters are expected to be given via the
command line.

| Parameter            | Default    | Explanation                                                                                                                                                                                                                         |
|:---------------------|:-----------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --fasta-path         | /          | **Required**; FASTA input file                                                                                                                                                                                                      |
| --h5-output-path     | /          | **Required**; HDF5 output file for the encoded data. Must end with ".h5".                                                                                                                                                           |
| --species            | /          | **Required**; Species name. Will be added to the .h5 file.                                                                                                                                                                          |
| --subsequence-length | 21384      | Size of the chunks each genomic sequence gets cut into.                                                                                                                                                                             |
| --write-by           | 20_000_000 | Write in super-chunks with this many base pairs, which will be rounded to be divisible by subsequence-length; needs to be equal to or larger than subsequence length; for lower memory consumption, consider setting a lower number |
## 3. HybridModel.py options
(for training and evaluation)
### General parameters
| Parameter            | Default         | Explanation                                                                                                                                                                                       |
|:---------------------|:----------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -d/--data-dir        | /               | Directory containing training and validation data (.h5 files). The naming convention for the training and validation files is "training_data[...].h5" and "validation_data[...].h5" respectively. |
| -s/--save-model-path | ./best_model.h5 | Path to save the best model (model with the best validation genic F1 (the F1 for the classes CDS, UTR and Intron)) to.                                                                            |

### Model parameters
| Parameter      | Default | Explanation                                                                                           |
|:---------------|:--------|:------------------------------------------------------------------------------------------------------|
| --cnn-layers   | 1       | Number of convolutional layers                                                                        |
| --lstm-layers  | 1       | Number of bidirectional LSTM layers                                                                   |
| --units        | 32      | Number of LSTM units per bLSTM layer                                                                  |
| --filter-depth | 32      | Filter depth for convolutional layers                                                                 |
| --kernel-size  | 26      | Kernel size for convolutional layers                                                                  |
| --pool-size    | 9       | Best set to a multiple of 3 (codon/nucleotide triplet size)                                           |
| --dropout1     | 0.0     | If > 0, will add dropout layer with given dropout probability after the CNN. (range: 0.0-1.0)         |
| --dropout2     | 0.0     | If > 0, will add dropout layer with given dropout probability after the bLSTM block. (range: 0.0-1.0) |


### Training parameters
| Parameter               | Default   | Explanation                                                                                                                                                                                                                  |
|:------------------------|:----------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -e/--epochs             | 10,000    | Number of training runs                                                                                                                                                                                                      |
| -b/--batch-size         | 8         | Batch size for training data                                                                                                                                                                                                 |
| --val-test-batch-size   | 32        | Batch size for validation/test data                                                                                                                                                                                          |
| --loss                  | /         | Loss function specification                                                                                                                                                                                                  |
| --patience              | 3         | Allowed epochs without the validation genic F1 improving before stopping training                                                                                                                                            |
| --check-every-nth-batch | 1,000,000 | Check validation genic F1 every nth batch, on default this check gets executed once every epoch regardless of the number of batches                                                                                          |
| --optimizer             | adamw     | Optimizer algorithm; options: adam or adamw                                                                                                                                                                                  |
| --clip-norm             | 3.0       | The gradient of each weight is individually clipped so that its norm is no higher than this value                                                                                                                            |
| --learning-rate         | 3e-4      | Learning rate for training                                                                                                                                                                                                   |
| --weight-decay          | 3.5e-5    | Weight decay for training; penalizes complexity and prevents overfitting                                                                                                                                                     |
| --class-weights         | /         | Weighting of the 4 classes [intergenic, UTR, CDS, Intron] (Helixer predictions)                                                                                                                                              |
| --transition-weights    | /         | Weighting of the 6 transition categories [transcription start site, start codon, donor splice site, transcription stop site, stop codon, acceptor splice site]                                                               |
| --predict-phase         | False     | Add this to also predict phases for CDS (recommended);  format: [None, 0, 1, 2]; 'None' is used for non-CDS regions, within CDS regions 0, 1, 2 correspond to phase (number of base pairs until the start of the next codon) |
| --resume-training       | False     | Add this to resume training (pretrained model checkpoint necessary)                                                                                                                                                          |

### Testing/Predicting parameters
| Parameter                   | Default                    | Explanation                                                                                                                                                                                                             |
|:----------------------------|:---------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -l/--load-model-path        | /                          | Path to a trained/pretrained model checkpoint. (HDF5 format)                                                                                                                                                            |
| -t/--test-data              | /                          | Path to one test HDF5 file.                                                                                                                                                                                             |
| -p/--prediction-output-path | predictions.h5             | Output path of the HDF5 prediction file. (Helixer base-wise predictions)                                                                                                                                                |
| --compression               | gzip                       | compression used for datasets in predictions h5 file ("lzf" or "gzip").                                                                                                                                                 |
| --eval                      | False                      | Add to run test/validation run instead of predicting.                                                                                                                                                                   |
| --overlap                   | False                      | Add to improve prediction quality at subsequence ends by creating and overlapping sliding-window predictions (with proportional increase in time usage).                                                                |
| --overlap-offset            | subsequence_length / 2     | Distance to 'step' between predicting subsequences when overlapping. Smaller values may lead to better predictions but will take longer. The subsequence_length should be evenly divisible by this value.               |
| --core-length               | subsequence_length * 3 / 4 | Predicted sequences will be cut to this length to increase prediction quality if overlapping is enabled. Smaller values may lead to better predictions but will take longer. Has to be smaller than subsequence_length. |

### Resources parameters
| Parameter         | Default | Explanation                                                                                               |
|:------------------|:--------|:----------------------------------------------------------------------------------------------------------|
| --float-precision | float32 | Precision of model weights and biases                                                                     |
| --gpu-id          | 1       | Sets GPU index, use if you want to train on one GPU on a multi-GPU machine without a job scheduler system |
| --workers         | 1       | Number of threads used to fetch input data for training. Consider setting to match the number of GPUs     |

### Miscellaneous parameters
| Parameter          | Default | Explanation                                                                                                                                                                                                                                                                                                                |
|:-------------------|:--------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --save-every-check | False   | Add to save a model checkpoint every validation genic F1 check (see --check-every-nth-batch in [training parameters](#training-parameters)                                                                                                                                                                                 |
| --nni              | False   | [nni](https://github.com/microsoft/nni) = Neural Network Intelligence,  automates feature engineering, neural architecture search, hyperparameter tuning, and model compression for deep learning; add this in addition to following the standard nni instructions on setting up the config.yml and search_space.json file |
| -v/--verbose       | False   | Add to run HybridModel.py in verbosity mode (additional information will be printed)                                                                                                                                                                                                                                       |
| --debug            | False   | Add to run in debug mode; truncates input data to small example (for training: just runs a few epochs)                                                                                                                                                                                                                     |

### Fine tuning parameters
| Parameter                    | Default | Explanation                                                                                                                             |
|:-----------------------------|:--------|:----------------------------------------------------------------------------------------------------------------------------------------|
| --fine-tune                  | False   | Add/Use with --resume-training to replace and fine tune just the very last layer                                                        |
| --pretrained-model-path      | /       | Required when predicting with a model fine tuned with coverage                                                                          |
| --input-coverage             | False   | Add to use "evaluation/rnaseq_(spliced_)coverage" from HDF5 training/validation files as additional input for a late layer of the model |
| --coverage-norm              | None    | None, linear or log (recommended); how coverage will be normalized before inputting                                                     |
| --post-coverage-hidden-layer | False   | Adds extra dense layer between concatenating coverage and final output layer                                                            |

## 4. HelixerPost options
The options for HelixerPost are either chosen when directly using Helixer.py (see 
[post-processing parameters](#post-processing-parameters)) or by using HelixerPost directly after
HybridModel.py. In that case the parameters are not defined by name but position.
```bash
helixer_post_bin <genome.h5> <predictions.h5> <window_size> <edge_threshold> <peak_threshold> \
<min_coding_length> <output.gff3>
```

## 5. import2geenuff.py options
(see [GeenuFF repository](https://github.com/weberlab-hhu/GeenuFF/tree/main))
### Configuration parameters
| Parameter     | Default           | Explanation                                                                                             |
|:--------------|:------------------|:--------------------------------------------------------------------------------------------------------|
| --base-dir    | /                 | Organized output (& input) directory. If this is not set, all four custom input parameters must be set. |
| --config-file | config/import.yml | .yml file containing configuration parameters                                                           |

### Override default with custom parameters
These parameters are **required** if ``--base-dir`` is not set.

| Parameter  | Default                                                | Explanation                                                                    |
|:-----------|:-------------------------------------------------------|:-------------------------------------------------------------------------------|
| --gff3     | /                                                      | GFF3 formatted file to parse / standardize                                     |
| --fasta    | /                                                      | Fasta file to parse standardize (has to be the same assembly as the GFF3 file) |
| --db-path  | /                                                      | Output path of the GeenuFF database                                            |
| --log-file | basedir/output/import.log (when ``--base-dir`` is set) | Output path for import2geenuff log file                                        |

### Possible genome attribute parameters
| Parameter       | Default | Explanation                                    |
|:----------------|:--------|:-----------------------------------------------|
| --species       | /       | **Required**; (Scientific) name of the species |
| --accession     | /       | Genome assembly accession                      |
| --version       | /       | Genome assembly version                        |
| --acquired-from | /       | Genome source (example: NCBI_RefSeq)           |

### Miscellaneous parameter
| Parameter    | Default | Explanation                                                                                            |
|:-------------|:--------|:-------------------------------------------------------------------------------------------------------|
| --replace-db | /       | Whether to override a GeenuFF database file found at the default location/at the location of --db_path |

## 6. geenuff2h5.py options
geenuff2h5.py always searches for the configuration file ``config/fasta2h5_config.yaml`` in the current
working directory. If that file isn't provided, the parameters are expected to be given via the
command line.

| Parameter            | Default        | Explanation                                                                                                                                                                                                                                                                                                                                                             |
|:---------------------|:---------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --input-db-path      | /              | **Required**; Path to the GeenuFF SQLite input file/database (has to contain only one genome)                                                                                                                                                                                                                                                                           |
| --h5-output-path     | /              | **Required**; HDF5 output file for the encoded data. Must end with ".h5"                                                                                                                                                                                                                                                                                                |
| --add-additional     | /              | Outputs the datasets under alternatives/{add-additional}/ (and checks sort order against existing "data" datasets). Use to add e.g. additional annotations from Augustus                                                                                                                                                                                                |
| --subsequence-length | 21384          | Length of the subsequences that the model will use at once.                                                                                                                                                                                                                                                                                                             |
| --modes              | all            | Either "all" (default), or a comma separated list with desired members of the following {X, y, anno_meta, transitions} that should be exported. This can be useful, for instance when skipping transitions (to reduce size/mem) or skipping X because you are adding an additional annotation set to an existing file (i.e. y,anno_meta,transitions <- no whitespaces!) |
| --write-by           | 21,384,000,000 | Write in super-chunks with this many base pairs, which will be rounded to be divisible by subsequence-length; needs to be equal to or larger than subsequence length                                                                                                                                                                                                    |
