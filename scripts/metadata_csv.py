#! /usr/bin/env python3
import os
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--scale', action='store_true', help='Whether to scale the kmer and busco columns')
parser.add_argument('--basepath', default='/mnt/data/ali/share/phytozome_organized/ready/train/',
                    help="path to the folder containing organized species folders")
args = parser.parse_args()

base_path = args.basepath
delimiter = ','
columns = {
    'species': [],
    # gff_features
    'CDS': [],
    'exon': [],
    'five_prime_UTR': [],
    'three_prime_UTR': [],
    'gene': [],
    'mRNA': [],
    # quast
    'contigs': [],
    'contigs_gt_1k': [],
    'contigs_gt_5k': [],
    'contigs_gt_10k': [],
    'contigs_gt_25k': [],
    'contigs_gt_50k': [],
    'total_len': [],
    'total_len_gt_1k': [],
    'total_len_gt_5k': [],
    'total_len_gt_10k': [],
    'total_len_gt_25k': [],
    'total_len_gt_50k': [],
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
    'A': [],
    'C': [],
    'N': [],
    'AA': [],
    'AC': [],
    'AG': [],
    'AT': [],
    'AT': [],
    'CA': [],
    'CC': [],
    'CG': [],
    'GA': [],
    'GC': [],
    'TA': [],
}

quast_key_matches = {
    '# contigs (>= 0 bp)': 'contigs',
    '# contigs (>= 1000 bp)': 'contigs_gt_1k',
    '# contigs (>= 5000 bp)': 'contigs_gt_5k',
    '# contigs (>= 10000 bp)': 'contigs_gt_10k',
    '# contigs (>= 25000 bp)': 'contigs_gt_25k',
    '# contigs (>= 50000 bp)': 'contigs_gt_50k',
    'Total length (>= 0 bp)': 'total_len',
    'Total length (>= 1000 bp)': 'total_len_gt_1k',
    'Total length (>= 5000 bp)': 'total_len_gt_5k',
    'Total length (>= 10000 bp)': 'total_len_gt_10k',
    'Total length (>= 25000 bp)': 'total_len_gt_25k',
    'Total length (>= 50000 bp)': 'total_len_gt_50k',
    'Largest contig': 'largest_contig',
    'GC (%)': 'gc_content',
    'N50': 'N50',
    'N75': 'N75',
    'L50': 'L50',
    'L75': 'L75',
}

busco_key_matches = {
    'Complete BUSCOs (C)': 'busco_C',
    'Complete and single-copy BUSCOs (S)': 'busco_S',
    'Complete and duplicated BUSCOs (D)': 'busco_D',
    'Fragmented BUSCOs (F)': 'busco_F',
    'Missing BUSCOs (M)': 'busco_M',
}

for genome in sorted(os.listdir(base_path)):
    columns['species'].append(genome)
    genome_path = os.path.join(base_path, genome)
    # gff_features
    # some values may be missing so setup dict with default values
    gff_values = {
        'CDS': '0',
        'exon': '0',
        'five_prime_UTR': '0',
        'three_prime_UTR': '0',
        'gene': '0',
        'mRNA': '0',
    }
    for line in open(os.path.join(genome_path, 'meta_collection', 'gff_features', 'counts.txt')):
        parts = line.strip().split(' ')
        gff_values[parts[1]] = parts[0]
    for gff_type, count in gff_values.items():
        columns[gff_type].append(count)
    # quast
    for line in open(os.path.join(genome_path, 'meta_collection', 'quast', 'geno', 'report.tsv')):
        parts = line.strip().split('\t')
        if parts[0] in quast_key_matches:
            columns[quast_key_matches[parts[0]]].append(parts[1])
    # busco
    for busco_type in ['geno', 'prot', 'tran']:
        path = os.path.join(genome_path, 'meta_collection', 'busco', busco_type)
        for line in open(glob(os.path.join(path, 'short_summary*.txt'))[0]):
            parts = line.strip().split('\t')
            if len(parts) > 1 and parts[1] in busco_key_matches:
                if args.scale:
                    value = '{:.4f}'.format(int(parts[0]) / 430.0)
                else:
                    value = parts[0]
                columns[busco_key_matches[parts[1]] + '_' + busco_type].append(value)
    # jellyfish
    for kmer_len in ['1', '2']:
        file_name = 'k' + kmer_len + 'mer_counts.tsv'
        for line in open(os.path.join(genome_path, 'meta_collection', 'jellyfish', file_name)):
            parts = line.strip().split('\t')
            if args.scale:
                value = '{:.4f}'.format(int(parts[0]) / float(columns['total_len'][-1]))
            else:
                value = parts[0]
            columns[parts[1]].append(value)

# checks
species_len = len(columns['species'])
for key, value_list in columns.items():
    assert len(value_list) == species_len

# output csv
print(delimiter.join(list(columns.keys())))
for i in range(len(columns['species'])):
    print(delimiter.join([columns[key][i] for key in columns.keys()]))
