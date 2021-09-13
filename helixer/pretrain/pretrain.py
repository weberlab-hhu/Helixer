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

class HelixerDataset(torch.utils.data.Dataset):
    def __init__(self, fasta_folder):
        self.tokenizer = BertTokenizerFast('vocab', do_lower_case=False)
        self.compressor = numcodecs.blosc.Blosc(cname='blosclz', clevel=4, shuffle=2)
        self.encodings = defaultdict(list)

        fp = fastahelper.FastaParser()
        for fasta_file in Path(fasta_folder).glob('*.fa'):
            print(f'starting with {fasta_file.name}')
            for i, (fasta_header, seq) in enumerate(fp.read_fasta(fasta_file)):
                # if i == 0:
                    # continue
                if len(seq) < 1e6:
                    print(f'skipping {fasta_header} (length: {len(seq)})')
                    continue
                seq = seq.upper()
                kmer_seqs = []
                for offset in range(0, len(seq), 512):
                    seq_part = seq[offset:offset+512]  # 512 chars make 510 3-mers, which become 512 tokens with [CLS] and [SEP]
                    kmer_seqs.append(' '.join([seq_part[i:i+3] for i in range(510)]))  # convert to 3-mers
                del seq

                # do in batches to not run into mem limits
                batch_size = 50000
                for offset in range(0, len(kmer_seqs), batch_size):
                    tokenized_seqs = self.tokenizer(kmer_seqs[offset:offset+batch_size], padding=True, return_special_tokens_mask=True)
                    # convert int lists to int8 np arrays and append to tokenized_seqs
                    for key, vals in tokenized_seqs.items():
                        key_seqs_int8 = [self.compressor.encode(np.array(arr, dtype=np.int8)) for arr in vals]
                        self.encodings[key].extend(key_seqs_int8)
                    print(f'processed {min(offset+batch_size, len(kmer_seqs))}/{len(kmer_seqs)} of {fasta_header}')
                mem_footprints = {key:sum([sys.getsizeof(e) for e in vals]) / 2 ** 20 for key, vals in self.encodings.items()}
                print(f'memory footprints in MB: {mem_footprints}')
                break
            break

    def __getitem__(self, idx):
        item = {key: torch.tensor(np.frombuffer(self.compressor.decode(val[idx]), dtype=np.int8))
                for key, val in self.encodings.items()}
        return item

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

    @abstractmethod
    def run(self):
        pass

class HelixerModelPreTrain(HelixerModelBase):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('--fasta-folder', type=str, required=True,
                            help='Path to a folder with fasta files which will be used for pre-training.')
        self.parser.add_argument('--n-layers', type=int, default=3)
        self.parse_args()
        self.run()

    def run(self):
        args = self.args
        train_dataset = HelixerDataset(args.fasta_folder)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=args.n_epochs,
            per_device_train_batch_size=args.batch_size_train,
            per_device_eval_batch_size=args.batch_size_valid,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            logging_dir='./logs',
            logging_steps=10,
        )

        configuration = BertConfig(num_hidden_layers=args.n_layers)
        model = BertForMaskedLM(configuration)
        collator = HelixerDataCollator(train_dataset.tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=val_dataset,
            data_collator=collator
        )
        trainer.train()

if __name__ == '__main__':
    model = HelixerModelPreTrain()
