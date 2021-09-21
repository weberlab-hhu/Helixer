#! /usr/bin/env python3
import sys
import h5py
import argparse
import numcodecs
import numpy as np
from pathlib import Path
import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast, Trainer, TrainingArguments
from helixer.pretrain.base import HelixerDatasetBase, HelixerModelBase

class HelixerDatasetFinetune(HelixerDatasetBase):
    def __init__(self, args, split):
        super().__init__(args)

        h5_file = h5py.File(f'{args.data_dir}/{split}_data.h5', 'r')

        # turn one hot encoding into strings again ...
        X = np.full(h5_file['data/X'].shape[:2], 'N', dtype='|S1')
        # get indices of all ATCG bases, the rest gets encoded as 'N'
        batch_size = 20000
        bases = ['C', 'A', 'T', 'G']
        for offset in range(0, len(X), batch_size):
            idx_all = np.where(h5_file['data/X'][offset:offset+batch_size] == 1.)
            for i in range(4):
                idx_base = idx_all[2] == i
                X[idx_all[0][idx_base], idx_all[1][idx_base]] = bases[i]


        self._tokenize(kmer_seqs, '')

    def __getitem__(self, idx):
        item = {key: torch.tensor(np.frombuffer(self.compressor.decode(val[idx]), dtype=np.int8))
                for key, val in self.encodings.items()}
        return item

class HelixerModelFinetune(HelixerModelBase):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('-d', '--data-dir', type=str, required=True)
        self.parser.add_argument('-l', '--load-model-path', type=str, default='')
        self.parser.add_argument('--n-lstm-layers', type=int, default=1)
        self.parser.add_argument('--n-lstm-units', type=int, default=128)
        self.parse_args()
        self.run()

    def run(self):
        args = self.args

        configuration = BertConfig()
        model = BertForMaskedLM(configuration)

        trainer = Trainer(
            model=model,
            args=self.training_args(),
            train_dataset=HelixerDatasetFinetune(args, 'training'),
            eval_dataset=HelixerDatasetFinetune(args, 'validation')
        )
        trainer.train()

if __name__ == '__main__':
    model = HelixerModelFinetune()
