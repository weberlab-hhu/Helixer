# Fine-tuning
Current Helixer models are a work in progress and performance
in practice can be lower for some targets, 
particularly where no proximal high quality 
genomes were sufficiently represented in the training data.

One option here that is potentially easier than fully retraining
is fine-tuning. This has not been properly evaluated, although
anecdotal evidence has been promising.

There are three main options for tine tuning
- fine tune the whole network
- freeze most of the network and train (a) new final layer(s)
- new final layer(s), with extrinsic information added before

> NOTE: these are currently experimental, meaning
> - they should be compared to alternatives and optimized as necessary
> - they are in some cases a process and have not yet been
>   stream-lined for user-friendliness

## Whole Network
The training process here is identical to training from scratch,
except that the weights are initialized based on the trained model,
instead of randomly.

This is very comparable to training from scratch, you just add
`--load-model-path <trained_model.h5> --resume-training`
to the training command with `HybridModel.py` otherwise described 
in [training.md](training.md#model-training).
Note also that the architecture is taken from the pre-trained network,
so parameters affecting the architecture (e.g. `--lstm-layers` and `--filter-depth`)
are not necessary and will have no effect. 

Note that this is potentially just as subject to over fitting
as training from scratch, and similar amounts of data should be
considered. A validation set that is representative of the prediction
target is very critical. 

##### additional useful parameters
Possible parameters that may help catch the sweet spot where
tuning is helping before over fitting starts hurting are
reducing the `--learning-rate` and the `--check-every-nth-batch`.
The idea behind both of these changes is to reduce how much the
model updates between checkpoints, so that "finding the sweet spot"
is less luck dependent.

The default learning rate is `3e-4`, so values such as `1e-4` or `3e-5` might
be helpful. This causes the network to make smaller updates to the weights at
each step.

The default checkpointing occurs once per epoch, you can add additional checks
based on the number of batches by adding `--check-every-nth-batch`; this then
should be set to something smaller than the batches per epoch. In the example
below, there are 1302 batches per epoch. 

```
266/1302 [=====>........................] - ETA: 7:13 - loss: 0.1457 - genic_loss: 0.1703 - phase_loss: 0.0476   
```

While simple, and requiring slightly less training time;
the advantages over retraining from scratch may be limited. 

## Freeze + new final layers.
Here, most of the network weights are frozen and won't change. 
Either the final weights connecting the last embedding to the 
output are replaced and trained, or a final hidden layer is included as well.
This results in much fewer trainable parameters and lower data
requirements than training from scratch or tuning the whole
network. Thus, this is a potential option, even training _within_
one species. 

To run this, add `--fine-tune` as well as the parameters from above:
`--load-model-path <trained_model.h5> --resume-training`. 

As before, data quality and that the data represents the prediction
target is critical. A training + validation set from one species
may predict extra well within that species, but fail to generalize
to even a close relative. Similarly a bias in the training + validation
set towards highly expressed genes, or highly conserved genes will 
likely be mirrored by the network.

Species specific tuning raises the same conundrum for Helixer that it does with 
any gene calling tool; in that it requires existing gene models
to train. Established (e.g. data-centric, or self-supervised + data supported
as used in genemark-ETP) methods can be used here, although care
should be taken that not-just-genes, but also a representative
proportion of intergenic sequences are included. It's also critical
to keep in mind that the Neural Network is unconstrained in using all available 
information that helps it
to fit the training set. So while it's common and reasonable to train
an HMM on centered genes with their flanking intergenic regions;
if these examples were less than the subsequence length (21834bp),
the Neural Network would most likely learn to always predict centered genes,
a trait that would be extremely detrimental at test time. Contiguous
regions of at least the subsequence length with random centering,
and a genic vs intergenic distribution representative of the whole
genome should be used. 

Insofar as the existing pre-trained Helixer model has respectable (if improvable)
performance; there is an additional new option using a first round 
of Helixer's predictions as pseudolabels for fine-tuning.

### Create pseudolabels with Helixer for fine-tuning
Regions within genomes differ in their difficulty level
for gene callers. The following is an idea to leverage these differences,
by fine-tuning Helixer on regions predicted confidently, so-as to make
predictions overall, but especially in harder regions better. 

Before starting, download a couple of helper scripts:
[filter-to-most-certain.py](https://raw.githubusercontent.com/weberlab-hhu/helixer_scratch/master/data_scripts/filter-to-most-certain.py) and 
[n90_train_val_split.py](https://raw.githubusercontent.com/weberlab-hhu/helixer_scratch/master/data_scripts/n90_train_val_split.py)]; and put them in the same folder.

- first, setup numeric data (`fasta2zarr.py`), 
  raw predictions (`HybridModely.py`), and post-processed predictions 
  (`helixer_post_bin`) according to the [three-step process described in 
  the main readme](../README.md#run-on-target-genomes-3-step-method)
- second, convert the gff3 output by HelixerPost to Helixer's training
  data format

```commandline
# read into the GeenuFF database
import2geenuff.py --fasta <your_genome.fa> --gff3 <helixer_post_output.gff3> \
  --db-path <your_species>.sqlite3 --log-file <your_species_import>.log \
  --species <speces_name_or_prefix>
# export to numeric matrices
geenuff2zarr.py --h5-output-path <your_species_helixer_post>.h5 \
  --input-db-path <your_species>.sqlite3 
```

- third, select the 'most confident' predictions (subsequences, default of 21384bp each). 
  Here, 'most confident' is defined
  as the predictions with the smallest absolute discrepancy between the
  raw predictions and the post-processed predictions; stratified by fraction
  of the intergenic class. Stratification is necessary to avoid selecting 
  all intergenic (which tends to be predicted very confidently), and 
  teaching the fine-tuned model to predict only intergenic. Assuring that 
  selected tuning examples are representative for the genome could, and should,
  be improved further.

```
python3 filter-to-most-certain.py --write-by 6415200 \
    --h5-to-filter <your_species_helixer_post.h5> --predictions <predictions.h5> \
    --keep-fraction 0.2 --output-file <filtered.h5>
```

- fourth, we'll split the resulting confident predictions into
  train and validation files. Each sequence from the fasta file will be
  fully assigned to either train _OR_ validation, which helps avoid
  having highly conserved tandem duplicates in both sets; but might not be 
  sufficient to reduce overfitting in e.g. recent polyploids or highly 
  duplicated genomes, so take care if that applies.

```commandline
mkdir <fine_tuning_data_dir>
python3 n90_train_val_split.py --write-by 6415200 \
    --h5-to-split <filtered.h5> --output-pfx <fine_tuning_data_dir>/
# note that any directories in the output-pfx should exist
# it need not be an empty directory, but it is the simplest here

# check that training and validation files were created
ls -sSh <fine_tuning_data_dir>/
```
> Note, if you've been taking the example from the readme or any other
> example so small that it has just one chromosome; we're about to hit a tiny-example specific problem.
> Specifically, the full sequence split will mean that the one chromosome available was assigned
> entirely to training and that the validation file is empty here. This should not occur
> with real data, and you should also _definitely not_ do the following hack with real data,
> as it just about guarantees overfitting. But just for the sake of *making an example run*,
> and if and only if the above applies to you, copy the training file to be a mock non-empty validation file, 
> (e.g. `cp <fine_tuning_data_dir>/training_data.h5 <fine_tuning_data_dir>/validation_data.h5`

- fifth, tune the model! 
```commandline
# model architecture parameters are taken from the loaded model
# but training, weighting and loss parameters do need to be specified
# appropriate batch sizes depend on the size of your GPU
HybridModel.py -v --batch-size 50 --val-test-batch-size 100 -e100 \
  --class-weights "[0.7, 1.6, 1.2, 1.2]" --transition-weights "[1, 12, 3, 1, 12, 3]" \
  --predict-phase --learning-rate 0.0001 --resume-training --fine-tune \
  --load-model-path <$HOME/.local/share/Helixer/models/land_plant/land_plant_v0.3_a_0080.h5> \
  --data-dir <fine_tuning_data_dir> --save-model-path <tuned_for_your_species_best_model.h5>
```

## Tuning with extrinsic information
Extrinsic information that could be used to help gene calling
is extremely varied in both technology and execution. 
Moreover, much is sequencing based and has
a tendency to be large and require extensive processing. New
developments and improvements are coming out continuously. 
Training a network to generalize for e.g. RNAseq input would
require training time input of data from high-quality, degraded,
contaminated, normalized and unnormalized,
low and high tissue-coverage RNA. 
It would require this from Illumina and IsoSeq, 
with random and poly-T priming, with and without 5' tagging, with
minimally and over PCR amplified input; and that for many or
most training species. Thus, we have not trained broadly 
generalizable models with extrinsic input.
**However, adding extrinsic input at fine-tune
time and tuning on exactly the same extrinsic input you will 
test with opens new possibilities.**

This process starts as **Freeze + new final layers** above,
but before the third step of selecting the 
most confident predictions, coverage track(s) are added 
to the h5 file of pseudolabels from Helixer post.

This assumes you already have aligned reads in .bam format
with information relating to genic regions (e.g. RNAseq,
CAGE). 

#### Add aligned reads to the h5 file as coverage tracks
*RECOMMENDED:* make a back up of your helixer-post-output-converted-back-to-h5 from above
as this process will change it in place

```commandline
cp <your_species_helixer_post.h5> <your_species_helixer_post_backup.h5>
```
If anything goes wrong, you can copy this back to start over from here.

Now on to adding the extrinsic data

```commandline
# in the provided containers, replace <path_to> below with /home/helixer_user
# otherwise with the path to where you've cloned the repository
python3 <path_to>/Helixer/helixer/evaluation/add_ngs_coverage.py \
  -s <species_name_or_prefix> --second-read-is-sense-strand 
  --bam <your_sorted_indexed_bam_file(s)> --h5-data <your_species_helixer_post.h5> \
   --dataset-prefix rnaseq --threads 1
```

Where one of `--second-read-is-sense-strand`, `--first-read-is-sense-strand`,
or `--unstranded` is chosen to match the protocol. For the common dUTP
stranded protocol (Illumina stranded libraries) you will want `--second-read-is-sense-strand`
as in the example. 

You can add multiple bam files `--bam A.bam B.bam C.bam` or `--bam a/path/*.bam`,
as long as their srandedness matches. If you want to add reads with different
protocols, run the above script once per strandedness.

This script changes the file given to `--h5-data` in place, adding
the datasets `rnaseq_coverage` and `rnaseq_spliced_coverage` which
will be used as coverage input

#### Select confident and split train / val
Once you've done this, continue with selecting the most confident
subsequences, and splitting them into training and validation sets 
as before.

#### Tuning with RNAseq
This is very similar to the fine-tuning above, but requires
a few extra parameters.

```
HybridModel.py -v --batch-size 140 --val-test-batch-size 280 \
   --class-weights "[0.7, 1.6, 1.2, 1.2]" --transition-weights "[1, 12, 3, 1, 12, 3]" \
   --predict-phase --learning-rate 0.0001 --resume-training --fine-tune \
   --load-model-path <$HOME/.local/share/Helixer/models/land_plant/land_plant_v0.3_a_0080.h5> \
   --input-coverage --coverage-norm log --data-dir --save-model-path <best_tuned_rnaseq_model.h5>
```

The new parameters are `--input-coverage`, which causes any data
in the h5 datasets `rnaseq_coverage` and `rnaseq_spliced_coverage` 
to be provided to the network after the frozen weights, but before
the new final layer(s); and `--coverage-norm log` (recommended for RNAseq)
which causes this
data to be log transformed before being input to the network.
Additionally, you can add `--post-coverage-hidden-layer` to add and tune not
1, but 2 final layers.

In this way, the network will learn the typical relation between high confidence
gene models and the supplied RNAseq data, and can use this to help predict
_all_ gene models. Thus, _in theory_ if the data has 3' bias the network will learn to use
it for the 3' end of the gene only, and if it has DNA contamination and resulting
background reads, the network will learn to ignore the appropriate amount of
background, and if the data is high quality and has very consistent correspondence
to gene regions, the network will learn to trust it heavily. _In theory._

> Note that this could be extended for any extrinsic data from which base
level data can be created; but only input of data from .bam files is implemented
here. 

## Inference with tuned models
For both tuning options without coverage, there are no special
requirements at inference time. Just set `--model-filepath`
to the fine-tuned model, and `--subsequence-length` to a 
value substantially above the typical gene length and divisible
by the pool-size used during training (generally 9) for `Helixer.py`. 
If using the three-step process, just point `--load-model-path`
to the fine-tuned model when running `Helixer.py`.

### Coverage models
Inference with coverage is a bit more complicated.

First, and unsurprisingly, you must provide the model
coverage at inference time. This means that
- you will have to take the three-step inference process,
  and make sure the h5 file has coverage
  - yes, you could take the file from above, if and only if
    the subsequence length (default 21384)
    is substantially longer than the typical genetic loci length; i.e. this 
    probably works for plants and fungi, not for animals.
  - if you need a longer subsequence-length at inference time,
    the only currently implemented option is to make an h5 each for training
    and inference and then add coverage to each. **Make sure the coverage is 
    added (i.e. the bam files are specified) in exactly the same order as at training time!**
- You will have to specify parameters at inference time, as done at 
  train time. These are `--input-coverage`, `--coverage-norm <log>`,
  `--predict-phase`,
  and `--post-coverage-hidden-layer` (if used).
- Finally, you will have to provide `Helixer.py` the path not just to
  the fine-tuned model with `--load-model-path`; but also provide the 
  pretrained model on which the tuning was performed under 
  `--pretrained-model-path`.

## Feedback very welcome
As this remains very experimental, we would highly encourage 
you to share your experience either with these methods or alternatives
you develop yourself; be it simply as a github issue, as a tutorial, a 
manuscript or anything in between.
