#! /usr/bin/env python3
import os
from glob import glob

base_path = '/mnt/data/ali/share/phytozome_organized/ready/train/'
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

for genome in os.listdir(base_path):
    columns['species'].append(genome)
    genome_path = os.path.join(base_path, genome)
    # gff_features
    for line in open(os.path.join(genome_path, 'meta_collection', 'gff_features', 'counts.txt')):
        parts = line.strip().split(' ')
        columns[parts[1]].append(parts[0])
    # quast
    for line in open(os.path.join(genome_path, 'meta_collection', 'quast', 'geno', 'report.tsv')):
        parts = line.strip().split('\t')
        if parts[0] in quast_key_matches:
            columns[quast_key_matches[parts[0]]] = parts[1]
    # busco
    for busco_type in ['geno', 'prot', 'tran']:
        path = os.path.join(genome_path, 'meta_collection', 'busco', busco_type)
        import pudb; pudb.set_trace()
        for line in open(glob(os.path.join(path, 'short_summary*.txt'))[0]):
            parts = line.strip().split('\t')
            if len(parts) > 1 and parts[1] in busco_key_matches:
                columns[busco_key_matches[parts[1]] + '_' + busco_type] = parts[0]
