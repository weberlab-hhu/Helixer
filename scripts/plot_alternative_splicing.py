#! /usr/bin/env python3
import pandas as pd

df = pd.read_csv('alternative_splicing_results.csv', sep=',',
                 names=['seqid', 'n_transcripts', 'genic_f1'])

import pudb; pudb.set_trace()
pass



