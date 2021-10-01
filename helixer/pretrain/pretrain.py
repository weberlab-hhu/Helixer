#! /usr/bin/env python3
from abc import ABC, abstractmethod
import sys
import torch
from pprint import pprint
import argparse
import numcodecs
import numpy as np
from pathlib import Path
from collections import defaultdict
from dustdas import fastahelper
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast, Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling

from helixer.pretrain.base import HelixerDatasetBase, HelixerModelBase

class HelixerDataCollator(DataCollatorForLanguageModeling):
    def torch_mask_tokens(self, inputs, special_tokens_mask):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        Override this function so we mask k kmers in a row.
        """
        def extend_mask(mask):
            return mask | torch.roll(mask, 1, 1) | torch.roll(mask, 2, 1)

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        # Devide probability by ~3 so we end up with ~15% of 3-mer tokens masked
        probability_matrix = torch.full(labels.shape, self.mlm_probability / 2.8)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix[:, -3:-1] = 0.0  # Don't mask the last two token so we always mask full k-mers
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        indices_replaced_extended = extend_mask(indices_replaced)
        inputs[indices_replaced_extended] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        # 0.48 instead of 0.5 to make up for random replacements superseding [MASK] replacements
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.48)).bool() & masked_indices & ~indices_replaced
        indices_random_extended = extend_mask(indices_random)
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random_extended] = random_words[indices_random_extended]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class HelixerDatasetPretrain(HelixerDatasetBase):
    def __init__(self, args):
        super().__init__(args)

        fp = fastahelper.FastaParser()
        for fasta_file in Path(args.fasta_folder).glob('*.fa'):
            print(f'starting with {fasta_file.name}')
            for i, (fasta_header, seq) in enumerate(fp.read_fasta(fasta_file)):
                if len(seq) < 1e6:
                    print(f'skipping {fasta_header} (length: {len(seq)})')
                    continue
                else:
                    print(f'starting with {fasta_header} (length: {len(seq)})')

                self._tokenize(seq, num_max_tokens=self.args.pretrain_input_len, pretrain=True)
                if args.debug:
                    break
            if args.debug:
                break

    def __getitem__(self, idx):
        item = {key: torch.tensor(np.frombuffer(self.compressor.decode(val[idx]), dtype=np.int8))
                for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

class HelixerModelPretrain(HelixerModelBase):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('--fasta-folder', type=str, required=True,
                            help='Path to a folder with fasta files which will be used for pre-training.')
        # the defaults are for the BERT-Mini variant from https://github.com/google-research/bert
        self.parser.add_argument('--n-layers', type=int, default=4)
        self.parser.add_argument('--n-attention-heads', type=int, default=4)
        self.parser.add_argument('--hidden-size', type=int, default=256)
        self.parser.add_argument('--intermediate-size', type=int, default=1024)
        self.parse_args()
        self.run()

    def run(self):
        args = self.args
        train_dataset = HelixerDatasetPretrain(args)

        configuration = BertConfig(num_hidden_layers=args.n_layers,
                                   num_attention_heads=args.n_attention_heads,
                                   hidden_size=args.hidden_size,
                                   intermediate_size=args.intermediate_size,
                                   max_position_embeddings=args.pretrain_input_len,
                                   vocab_size=train_dataset.tokenizer.vocab_size)
        model = BertForMaskedLM(configuration)
        collator = HelixerDataCollator(train_dataset.tokenizer)

        HelixerModelBase.print_model_info(model, 'Pretraining')

        trainer = Trainer(
            model=model,
            args=self.training_args(),
            train_dataset=train_dataset,
            # eval_dataset=val_dataset,
            data_collator=collator,
        )
        trainer.train()

if __name__ == '__main__':
    model = HelixerModelPretrain()
