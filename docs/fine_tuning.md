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
to the command `HybridModel.py` otherwise described in [training.md].

Note that this is potentially just as subject to over fitting
as training from scratch, and similar amounts of data should be
considered. A validation set that is representative of the prediction
target is very critical. 

Possible parameters that may help catch the sweet spot where
tuning is helping before over fitting starts hurting are
reducing the `--learning-rate` and the `--check-every-nth-batch`.

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
`--load-model-path <trained_model.h5> --resume-training`. Further,
you can include `--post-coverage-hidden-layer` to allow for a potentially
more complex fit. 

As before, data quality and that the data represents the prediction
target is critical. A training + validation set from one species
may predict extra well within that species, but fail to generalize
to even a close relative. 

Species specific tuning raises the same conundrum for Helixer that it does with 
any gene calling tool; in that it requires existing gene models
to train. Established (e.g. data-centric) methods can be used here, although care
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

- first, setup numeric data (`fasta2h5.py`), 
  raw predictions (`HybridModely.py`), and post-processed predictions 
  (`helixer_post_bin`) according to the three-step process described in 
  the main readme
- second, convert the gff3 output by HelixerPost to Helixer's training
  data format

```commandline
# read into the GeenuFF database
import2geenuff.py --fasta <your_genome.fa> --gff3 <helixer_post_output.gff3> \
  --db-path <your_species>.sqlite3 --log-file <your_species_import>.log \
  --species <speces_name_or_prefix>
# export to numeric matrices
geenuff2h5.py --h5-output-path <your_species_helixer_post>.h5 \
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
python filter-to-most-certain.py --write-by 6415200 \
    --h5-to-filter <your_species_helixer_post.h5> --predictions <predictions.h5> \
    --keep-fraction 0.2 --output-file <filtered.h5>
```

where filter-to-most-certain.py is the script 
[here](https://raw.githubusercontent.com/weberlab-hhu/helixer_scratch/master/data_scripts/filter-to-most-certain.py)

- fourth, we'll split the resulting confident predictions into
  train and validation files. Each sequence from the fasta file will be
  fully assigned to either train _OR_ validation, which helps avoid
  having highly conserved tandem duplicates in both sets; but might not be 
  sufficient to reduce overfitting in e.g. recent polyploids or highly 
  duplicated genomes, so take care if that applies.

```commandline
mkdir <fine_tuning_data_dir>
python n90_train_val_split.py --write-by 6415200 \
    --h5-to-split <filtered.h5> --output-pfx <fine_tuning_data_dir>/
# note that any directories in the output-pfx should exist
# it need not be an empty directory, but it is the simplest here
```

- fifth, tune the model! 
```commandline
# model architecture parameters are taken from the loaded model
# but training, weighting and loss parameters do need to be specified
# appropriate batch sizes depend on the size of your GPU
HybridModel.py -v --batch-size 140 --val-test-batch-size 280 \
  --class-weights "[0.7, 1.6, 1.2, 1.2]" --transition-weights "[1, 12, 3, 1, 12, 3]" \
  --predict-phase --learning-rate 0.0001 --resume-training --fine-tune 
  --load-model-path <$HOME/.local/share/Helixer/models/land_plant/land_plant_v0.3_a_0080.h5> \
  --data-dir <fine_tuning_data_dir> --save-model-path <tuned_for_your_species_best_model.h5>
```

## Tuning with extrinsic information
Extrinsic information that could be used to help gene calling
is extremely varied. Moreover, much is sequencing based and has
a tendency to be large, require extensive processing, and then 
be replaced by a better method a year later.