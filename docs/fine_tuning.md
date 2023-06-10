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
proportion of intergenic sequences are included. Moreover, insofar
as the existing pre-trained Helixer model has respectable (if improvable)
performance; there is an additional new option. 

### Create pseudolabels with Helixer for fine-tuning
Areas and genes within genomes differ in their difficulty level
for gene callers. The following is an idea to leverage these differences,
by fine-tuning Helixer on regions predicted confidently, so-as to make
predictions in other regions better. 

- first, setup numeric data (`fasta2h5.py`), 
  raw predictions (`HybridModely.py`), and post-processed predictions 
  (`helixer_post_bin`) according to the three-step process described in 
  the main readme
- second, convert the gff3 output by HelixerPost to Helixer's training
  data format

```commandline
TODO
```
- 
- third, select the 'most confident' predictions (subsequences, normally 21384bp each). 
  Here, 'most confident' is defined
  as the predictions with the smallest absolute discrepancy between the
  raw predictions and the post-processed predictions; stratified by fraction
  of the intergenic class. Stratification is necessary to avoid selecting 
  all intergenic (which tends to be predicted confidently) examples, and 
  teaching the fine-tuned model to predict only intergenic. Assuring that 
  selected tuning examples are representative for the genome could and should
  be improved further.

```
python filter-to-most-certain.py --write-by 6415200 \
    --h5-to-filter <your_species_helixer_post.h5> --predictions <predictions.h5> \
    --keep-fraction 0.2 --output-file <filtered.h5>
```

where filter-to-most-certain.py is the script here TODO

- fourth,
  