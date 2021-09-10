#! /usr/bin/env python3
import torch
import numpy as np
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
        masked_indices = masked_indices | torch.roll(masked_indices, 1, 1) | torch.roll(masked_indices, 2, 1)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class HelixerDataset(torch.utils.data.Dataset):
    def __init__(self, seq_file):
        self.tokenizer = BertTokenizerFast('vocab', do_lower_case=False)

        fp = fastahelper.FastaParser()
        for i, (fasta_header, seq) in enumerate(fp.read_fasta(seq_file)):
            seq = seq.upper()
            kmer_seqs = []
            for offset in range(0, len(seq), 512):
                seq_part = seq[offset:offset+512]  # 512 chars make 510 3-mers, which become 512 tokens with [CLS] and [SEP]
                kmer_seqs.append(' '.join([seq_part[i:i+3] for i in range(510)]))  # convert to 3-mers

            # do in batches to not run into mem limits
            self.encodings = defaultdict(list)
            batch_size = 50000
            for offset in range(0, len(kmer_seqs), batch_size):
                tokenized_seqs = self.tokenizer(kmer_seqs[offset:offset+batch_size], padding=True, return_special_tokens_mask=True)
                # convert int lists to int8 np arrays and append to tokenized_seqs
                for key, vals in tokenized_seqs.items():
                    key_seqs_int8 = [np.array(arr, dtype=np.int8) for arr in vals]
                    self.encodings[key].extend(key_seqs_int8)
                print(f'processed {min(offset+batch_size, len(kmer_seqs))}/{len(kmer_seqs)} of {fasta_header}')
            break

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = HelixerDataset('/home/felix/Desktop/helixer/GCF_000001405.39_GRCh38.p13_genomic.fa')

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

configuration = BertConfig(num_hidden_layers=3)
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




