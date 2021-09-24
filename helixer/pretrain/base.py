from abc import ABC, abstractmethod
import sys
import torch
from pprint import pprint
import argparse
import numcodecs
import numpy as np
from collections import defaultdict
from transformers import BertTokenizerFast, TrainingArguments

def print_model_parameter_counts(model):
    """Taken from https://stackoverflow.com/questions/48393608/pytorch-network-parameter-calculation"""
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    total_param_str = f'{total_param:,}'.replace(',', '.')
    print(f'total parameters: {total_param_str}')


def print_tensors():
    """Prints the currently active GPU tensors for debugging."""
    import gc
    from tabulate import tabulate
    rows = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                size = obj.size()
                if obj.dtype == torch.float16:
                    byte_size = 2
                elif obj.dtype == torch.float32:
                    byte_size = 4
                elif obj.dtype == torch.float64:
                    byte_size = 8
                else:
                    byte_size = 4
                if len(size) > 0:
                    rows.append([str(size),
                                 f'{np.prod(list(size)) * byte_size / 2 ** 30:.4f} GB',
                                 str(type(obj)),
                                 str(obj.dtype),
                                 str(obj.requires_grad)])
        except:
            pass
    print(tabulate(rows, headers=['shape', 'size', 'class', 'dtype', 'requires_grad']))
    print(f'pytorch total mem: {torch.cuda.get_device_properties(0).total_memory / 2 ** 30:.4f} GB')
    print(f'pytorch reserved mem: {torch.cuda.memory_reserved(0) / 2 ** 30:.4f} GB')
    print(f'pytorch allocated mem: {torch.cuda.memory_allocated(0) / 2 ** 30:.4f} GB')


class HelixerDatasetBase(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizerFast('vocab', do_lower_case=False)
        self.compressor = numcodecs.blosc.Blosc(cname='blosclz', clevel=4, shuffle=2)
        self.encodings = defaultdict(list)

    def _tokenize(self, seq, num_max_tokens=502, pretrain=True):
        """Tokenizes a sequence into 3-mers and adds the compressed arrays to self.encodings.
        Done in batches to not run into mem limits."""
        if pretrain:
            seq = seq.upper()
        kmer_seqs = []
        for offset in range(0, len(seq), num_max_tokens - 2):
            # 502 chars would make 500 3-mers, which become 502 tokens with [CLS] and [SEP]
            # as each kmer represents a single base we need sequences that overlap by 2 bases
            # otherwise there would not be predictions for the last 2 bases of each subsequence
            seq_part = seq[offset:offset+num_max_tokens+2]
            kmer_seqs.append(' '.join([seq_part[i:i+3] for i in range(num_max_tokens - 2)]))  # convert to 3-mers
        del seq

        batch_size = 40000
        n_short_samples_per_seq = 40
        for offset in range(0, len(kmer_seqs), batch_size):
            tokenized_seqs = self.tokenizer(kmer_seqs[offset:offset+batch_size], padding=True,
                                            return_special_tokens_mask=pretrain)
            # convert int lists to int8 np arrays and append to tokenized_seqs
            for key, vals in tokenized_seqs.items():
                key_seqs_int8_flat = [self.compressor.encode(np.array(arr, dtype=np.int8)) for arr in vals]
                if pretrain:
                    self.encodings[key].extend(key_seqs_int8_flat)
                else:
                    # append list of grouped sequences for finetuning or inference
                    for i in range(0, len(key_seqs_int8_flat), n_short_samples_per_seq):
                        self.encodings[key].append(key_seqs_int8_flat[i:i+n_short_samples_per_seq])
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
        self.parser.add_argument('--pretrain-input-len', type=int, default=502, help='Including [CLS] and [SEP]')

    def parse_args(self):
        args = self.parser.parse_args()
        self.args = args

        print('Config:')
        pprint(vars(self.args))
        print()

    @staticmethod
    def print_model_info(model, prefix):
        print(f'\n{prefix} config:')
        print(model.config)
        print(f'{prefix} model: ')
        print(model)
        print_model_parameter_counts(model)
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
