# Comparing annotations
Genome annotations from differing gene callers, i.e. from Helixer and others
can be compared via various methods. We used:
- BUSCO (v5.2.2) (Simão et al., 2015)
- Gffcompare (v0.12.8) (Pertea and Pertea, 2020)
- Comparing with the reference genome (precision, recall, F1)
   
The code, bash and python scripts as well as jupyter notebooks, aren't
optimized, but will work on the output from the version of the tools
we used.

### Important Definitions:
- **UTR**: untranslated region; present in the mature RNA, but doesn't get
translated; two UTRs per gene: 5' UTR at the start of the gene and 3' UTR
at the end
- **CDS**: coding sequence; present in the mature RNA, gets translated
- **Exon**: contains UTR and CDS regions   


## Running other gene callers (short)
We compared Helixer to two different _ab initio_ gene callers:
- GeneMark-ES (v4.71_lic) (Lomsadze et al., 2005, Ter-Hovhannisyan et al., 2008)
- AUGUSTUS (v3.3.2) (Stanke et al., 2006)

The commands we used to run them were (please look up the tools
own documentation for more details):
```bash
# genemark
gmes_petap.pl --sequence <genome_assembly.fa>  --ES --cores 24  # add: --fungus for fungal genomes

# augustus
augustus --species=<augustus_species> <genome_assembly.fa> --softmasking=1 \
    --noInFrameStop=true --stopCodonExcludedFromCDS=false \
    --gff3=on --UTR=on > <output>.gff3
    # utr = on if available
```
We converted GeneMark-ES's gtf output file into a gff3 file. Then,
we removed the non-ID parts of the fasta headers with the script
[fix_gm_names.py](https://github.com/weberlab-hhu/helixer_scratch/blob/master/data_scripts/fix_gm_names.py).
```bash
# convert to gff3
gffread genemark.gtf -o genemark.gff3

# fix ID names
python3 fix_gm_names.py genemark.gff3 > <species>_genemark.gff3
```
Gff3 files produced from AUGUSTUS called with `--gff3=on --UTR=off` were corrected
by inserting exons for every CDS and changing transcript features to mRNA. We used
the script [clean_UTRoff_gff3.py](https://github.com/weberlab-hhu/helixer_scratch/blob/master/method_comp/clean_UTRoff_gff3.py).
```bash
python3 clean_UTRoff_gff3.py --gff-in <augustus_annotation_utr_off>.gff3 --out <cleaned_annotation>.gff3
```
## BUSCO
First, we ran BUSCO for each lineage (plant, fungi, vertebrate, invertebrate),
species and gene caller prediction (AUGUSTUS, GeneMark, Helixer), as well as for
each reference genome. The BUSCO lineage datasets used were:

| Lineage                     | BUSCO lineage       | File version (date) | domain    |
|:----------------------------|:--------------------|:--------------------|:----------|
| fungi                       | fungi_odb10         | 28.06.2021          | Eukaryota |
| plant                       | viridiplantae_odb10 | 10.09.2020          | Eukaryota |
| vertebrate and invertebrate | metazoa_odb10       | 24.02.2021          | Eukaryota |
   
```bash
bash busco_local.sh <genome_assembly.fa> <genome_annotation.gff> <busco_lineage>
```
Contents of `busco_local.sh`:
```bash
genome_fa=$1
gff=$2
lineage=$3

# activate python environment where BUSCO is installed
source <busco_virtual_environment>/bin/activate
# path to the busco_downloads folder where all BUSCO datasets are located
# when BUSCO is installed locally
bp=<path_to_BUSCO>/busco_downloads

# find source files
annodir=`echo $gff |sed 's@/[^/]*gff3@@g'`
proteins=$annodir/protein.fa

# extract protein sequences from the GFF file
gffread $gff -g $genome_fa -y $proteins

# create output directory and remove a potential old one
busco_out=$annodir/busco
rm -r $busco_out
mkdir -p $busco_out

# run BUSCO
busco --in $fa_in --out $busco_out --offline --mode "prot" -l $lineage -f --download_path $bp
```
We imported the BUSCO results into the jupyter notebook
[busco_performance_comparison.ipynb](busco_performance_comparison.ipynb)
to create plots and tables.


## Gffcompare
### Filtering: longest gene isoform
Since Helixer only predicts the splice variants producing the longest protein,
we first filter out all other splice variants from annotations of other gene
callers and the reference with the script [lala_longest.py](https://github.com/weberlab-hhu/helixer_scratch/blob/master/misc_scripts/lala_longest.py).
```bash
lala_longest.py --gff-file <gff_to_filter> > <filtered_gff>
```
### Compare without UTRs and Exons
Next we filter out the 5' and 3' UTR regions. AUGUSTUS and GeneMark don't
generally predict UTRs and the combination of biological variability
of transcript start/ends and a lot of technical error, leads to all tools
approaching 0% exactly right UTRs (gffcompare doesn't use base-wise
statistics, but only scores something as true positive if it's exactly right.).
Since 99.9 % accuracy for CDS and Intron could point to a frame shift
and therefore result in a non-functional protein, non-base-wise statistics
make sense. For UTRs even just 90 % accuracy wouldn't have that effect.
Next, since we filtered out the UTRs, the exons are redundant (see
definition above). Therefore, we compare CDS, intron, intron chain (any
predicted transcript for which all of its introns can be found, with the
same exact intron coordinates in a reference transcript (with the same
number of introns)) and transcript annotations.

```bash
# command
./gffcompare.sh <refrence.gff3> <alternate.gff3> <output_directory>
```
Contents of `gffcompare.sh`:
```bash
#! /bin/bash
ref=$1
gff=$2
outdir=$3

mkdir -p $outdir
mkdir -p tmp_gffcompare_cleaner
cat $ref|awk '$3 != "exon"' |awk '$3 != "five_prime_UTR"'|awk '$3 != "three_prime_UTR"' > tmp_gffcompare_cleaner/reference.gff3
cat $gff|awk '$3 != "exon"' |awk '$3 != "five_prime_UTR"'|awk '$3 != "three_prime_UTR"' > tmp_gffcompare_cleaner/alternate.gff3
gffcompare -r tmp_gffcompare_cleaner/reference.gff3 \
  -o $outdir/gffcompare \
  tmp_gffcompare_cleaner/alternate.gff3

rm tmp_gffcompare_cleaner/reference.gff3*
rm tmp_gffcompare_cleaner/alternate.gff3*
```

### Precision, recall and F1
The percentages for precision and recall (here: sensitivity) can be
found in the output file `gffcompare.stats` written to the designated
output directory. The F1 can be calculated using the precision and
recall values. This calculation and post-processing of the gffcompare
output are documented in
[vs_reference_performance_gffcompare.ipynb](vs_reference_performance_gffcompare.ipynb).

## Vs. reference
We compared three different annotation categories:
- **genic class**: UTR, CDS, Intron (present in the gene region, not present
in the mature RNA)
- **coding phase**: None, phase 0, phase 1, phase 2 – indicating the number
of base-pairs until the start of the next codon (three consecutive
nucleotides in the CDS coding for a specific amino acid)
- **subgenic class**: CDS and Intron

We used precision, recall and F1  to evaluate the accuracy of the gene
predictions of AUGUSTUS, GeneMark and Helixer versus the reference.
To calculate these metrics base-wise, we followed a three-step workflow:
   
1. Converting the genome annotations to sqlite3 with [GeenuFF](https://github.com/weberlab-hhu/GeenuFF)
    
   ```bash
   import2geenuff.py --gff3 <genome_annotation.gff> --fasta <genome_assembly.fa> --db-path <species.sqlite3> --log-file import.log --species <species>
    ```
   
2. Importing the slite3 databases to h5 files with [geenuff2h5.py](../geenuff2h5.py)
    
   ```bash
   geenuff2h5.py --h5-output-path <species>/test_data.h5 --input-db-path <species.sqlite3> --subsequence-length 21384 --write-by 2138400
    ```
3. Computing accuracy metrics files with [accs_genic_intergenic.py](../scripts/accs_genic_intergenic.py)
   ```bash
   python Helixer/scripts/accs_genic_intergenic.py \
    --data <species>.h5 --predictions <species>/test_data.h5 \
    --h5_prediction_dataset 'data/y' --stats_dir summary/<lineage>/<species>/vs_ref/<gene_caller>
    ```
   - `<species>.h5` contains the reference annotation (stored in `data/y`)
   - `h5_prediction_dataset` defines where in the predictions h5 file the
   annotation to be compared to the reference is stored
   - `stats_dir`: export several csv files of the calculated statistics in
   this (nested) directory
   
\
After the creation of the metric files: `F1_summary.csv`, `confusion_matrix.csv`
and `normalized_confusion_matrix.csv` for genic class and coding phase for
every species and gene caller we can analyze and plot the results. An example
to generate plots, tables and statistics can be found in the jupyter notebook
[vs_reference_performance_comparison.ipynb](vs_reference_performance_comparison.ipynb).

### Citations
Geo Pertea and Mihaela Pertea. Gff utilities: Gffread and gffcompare.
_F1000Research_, 9:304, 2020. URL
https://doi.org/10.12688/f1000research.23297.2   
   
Felipe A Simão, Robert M Waterhouse, Panagiotis Ioannidis, Evgenia V
Kriventseva, and Evgeny M Zdobnov. Busco:assessing genome assembly and
annotation completeness with single-copy orthologs. _Bioinformatics_,
31(19):3210–3212, 2015
   
Mario Stanke, Oliver Keller, Irfan Gunduz, Alec Hayes, Stephan Waack,
and Burkhard Morgenstern. Augustus: ab initio prediction of alternative
transcripts. _Nucleic acids research_, 34(suppl_2):W435–W439, 2006
   
Alexandre Lomsadze, Vardges Ter-Hovhannisyan, Yury O. Chernoff, and Mark
Borodovsky. Gene identification in novel eukaryotic genomes by self-training
algorithm. _Nucleic Acids Research_, 33(20):6494–6506, 01 2005. ISSN
0305-1048. doi: 10.1093/nar/gki937. URL https://doi.org/10.1093/nar/gki937
   
Vardges Ter-Hovhannisyan, Alexandre Lomsadze, Yury O Chernoff, and Mark
Borodovsky. Gene prediction in novel fungal genomes using an ab initio
algorithm with unsupervised training. _Genome research_, 18(12):1979–1990, 2008
