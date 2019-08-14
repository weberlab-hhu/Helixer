#! /usr/bin/env python3
import os

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

base_path = '/mnt/data/ali/share/phytozome_organized/ready/train/'
columns = {
    # gff_features
    'n_CDS': [],
    'n_exon': [],
    'five_prime_UTR': [],
    'three_prime_UTR': [],
    'gene': [],
    'mRNA': [],
    # quast
    'total_len': [],
    'total_len_gt_1k': [],
    'total_len_gt_5k': [],
    'total_len_gt_10k': [],
    'total_len_gt_25k': [],
    'total_len_gt_50k': [],
    'n_contigs': [],
    'n_contigs_gt_1k': [],
    'n_contigs_gt_5k': [],
    'n_contigs_gt_10k': [],
    'n_contigs_gt_25k': [],
    'n_contigs_gt_50k': [],
    'largest_contig': [],
    'gc_content': [],
    'N50': [],
    'N75': [],
    'L50': [],
    'L75': [],
    # busco
    'busco_C_geno': [],
    'busco_S_geno': [],
    'busco_D_geno': [],
    'busco_F_geno': [],
    'busco_M_geno': [],
    'busco_C_prot': [],
    'busco_S_prot': [],
    'busco_D_prot': [],
    'busco_F_prot': [],
    'busco_M_prot': [],
    'busco_C_tran': [],
    'busco_S_tran': [],
    'busco_D_tran': [],
    'busco_F_tran': [],
    'busco_M_tran': [],
    # jellyfish
    'n_kmers_A': [],
    'n_kmers_C': [],
    'n_kmers_N': [],
    'n_kmers_AA': [],
    'n_kmers_AC': [],
    'n_kmers_AG': [],
    'n_kmers_AT': [],
    'n_kmers_AT': [],
    'n_kmers_CA': [],
    'n_kmers_CC': [],
    'n_kmers_CG': [],
    'n_kmers_GA': [],
    'n_kmers_GC': [],
    'n_kmers_TA': [],
}


for genome_path in listdir_fullpath(base_path):
    # gff_features




