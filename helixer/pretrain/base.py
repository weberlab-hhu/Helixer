from abc import ABC, abstractmethod
import sys
import torch
from pprint import pprint
import argparse
import numcodecs
import numpy as np
from collections import defaultdict
from transformers import BertTokenizerFast, TrainingArguments

class HelixerDatasetBase(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizerFast('vocab', do_lower_case=False)
        self.compressor = numcodecs.blosc.Blosc(cname='blosclz', clevel=4, shuffle=2)
        self.encodings = defaultdict(list)

    def _tokenize(self, kmer_seqs):
        """Tokenizes a sequence into 3-mers and adds the compressed arrays to self.encodings.
        Do in batches to not run into mem limits."""
        batch_size = 50000
        for offset in range(0, len(kmer_seqs), batch_size):
            tokenized_seqs = self.tokenizer(kmer_seqs[offset:offset+batch_size], padding=True, return_special_tokens_mask=True)
            # convert int lists to int8 np arrays and append to tokenized_seqs
            for key, vals in tokenized_seqs.items():
                key_seqs_int8 = [self.compressor.encode(np.array(arr, dtype=np.int8)) for arr in vals]
                self.encodings[key].extend(key_seqs_int8)
            print(f'processed {min(offset+batch_size, len(kmer_seqs))}/{len(kmer_seqs)}')
        mem_footprints = {key:sum([sys.getsizeof(e) for e in vals]) / 2 ** 20 for key, vals in self.encodings.items()}
        mem_footprints_str = {key:f'{val:.2f} MB' for key, val in mem_footprints.items()}
        print(f'memory footprints: {mem_footprints_str}')

    def __len__(self):
        return len(self.encodings['input_ids'])


class HelixerModelBase(ABC):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--n-epochs', type=int, default=3)
        self.parser.add_argument('--batch-size-train', type=int, default=16)
        self.parser.add_argument('--batch-size-valid', type=int, default=64)
        self.parser.add_argument('--warmup-steps', type=int, default=500)
        self.parser.add_argument('--weight-decay', type=int, default=0.01)

    def parse_args(self):
        args = self.parser.parse_args()
        self.args = args

        print('Config:')
        pprint(vars(self.args))
        print()

    def training_args(self):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=self.args.n_epochs,
            per_device_train_batch_size=self.args.batch_size_train,
            per_device_eval_batch_size=self.args.batch_size_valid,
            warmup_steps=self.args.warmup_steps,
            weight_decay=self.args.weight_decay,
            logging_dir='./logs',
            logging_steps=10,
        )
        return training_args

    @abstractmethod
    def run(self):
        pass

