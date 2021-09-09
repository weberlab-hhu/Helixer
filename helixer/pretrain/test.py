#! /usr/bin/env python3
import torch
from dustdas import fastahelper
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast, Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling

class HelixerDataset(torch.utils.data.Dataset):
    def __init__(self, seq_file):
        self.tokenizer = BertTokenizerFast('vocab', do_lower_case=False)

        fp = fastahelper.FastaParser()
        for i, (fasta_header, seq) in enumerate(fp.read_fasta(seq_file)):
            seq = seq.upper()
            kmer_seq = [seq[i:i+3] for i in range(len(seq) - 2)]  # convert to 3-mers
            kmer_seq_sentences = [' '.join(kmer_seq[i:i+510]) for i in range(0, len(kmer_seq), 510)]

            # do in batches to not run into mem limits
            tokenized_seqs = []
            batch_size = 50000
            for offset in range(0, len(kmer_seq_sentences), batch_size):
                tokenized_seqs = self.tokenizer(kmer_seq_sentences[offset:offset+batch_size], padding=True, return_special_tokens_mask=True)
                if offset == 0 and i == 0:
                    self.encodings = tokenized_seqs
                else:
                    for key in self.encodings.keys():
                        self.encodings[key] += tokenized_seqs[key]
                print(f'processed {min(offset+batch_size, len(kmer_seq_sentences))}/{len(kmer_seq_sentences)} of {fasta_header}')
            continue

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


train_dataset = HelixerDataset('helixer/GCF_000001405.39_GRCh38.p13_genomic.fa')

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

configuration = BertConfig(num_hidden_layers=1)
model = BertForMaskedLM(configuration)
collator = DataCollatorForLanguageModeling(train_dataset.tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=val_dataset,
    data_collator=collator
)

trainer.train()




