#! /usr/bin/env python3
import pandas as pd

df = pd.read_csv('alternative_splicing_results.csv', sep=',',
                 names=['seqid', 'strand', 'start', 'end', 'gene_name', 'n_transcripts',
                        'ig_f1', 'utr_f1', 'intron_f1', 'exon_f1', 'genic_f1'])
df['length_bin'] = pd.cut((df.start - df.end).abs(),
                           bins=[0, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000],
                           labels=False)

dfg = df.groupby(['length_bin']).mean()
dfg = dfg.loc[:, 'ig_f1': 'genic_f1']
# df = df['ig_f1', 'utr_f1', 'intron_f1', 'exon_f1', 'genic_f1']

import pudb; pudb.set_trace()
pass



