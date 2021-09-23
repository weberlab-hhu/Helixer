#! /usr/bin/env python3
import sys
import h5py
import argparse
import numcodecs
import numpy as np
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizerFast, Trainer, TrainingArguments, BertForTokenClassification
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from helixer.pretrain.base import HelixerDatasetBase, HelixerModelBase

class HelixerDatasetFinetune(HelixerDatasetBase):
    def __init__(self, args, split):
        super().__init__(args)

        h5_file = h5py.File(f'{args.data_dir}/{split}_data.h5', 'r')

        # turn one hot encoding into strings again ...
        # all kinds of bugs here with padding, sample_weights, etc... just a quick test for downstream code
        debug_size = 100
        X = np.full((debug_size, 20000), 'N', dtype='|S1')
        self.labels = np.full((debug_size * 40, 502), 0, dtype=np.int8)  # this sets the labels for CLS and SEP to 0
        self.labels[:, 1:-1] = np.argmax(h5_file['/data/y'][:debug_size], axis=-1).reshape(-1, 500)
        # self.labels = np.argmax(h5_file['/data/y'][:debug_size], axis=-1).reshape(-1, 500)

        # get indices of all ATCG bases, the rest gets encoded as 'N'
        batch_size = debug_size
        bases = ['C', 'A', 'T', 'G']
        for offset in range(0, len(X), batch_size):
            idx_all = np.where(h5_file['data/X'][offset:offset+batch_size] == 1.)
            for i in range(4):
                idx_base = idx_all[2] == i
                X[idx_all[0][idx_base], idx_all[1][idx_base]] = bases[i]
            break
        self._tokenize(X.flatten().tobytes().decode())

    def __getitem__(self, idx):
        item = {key: torch.tensor(np.frombuffer(self.compressor.decode(val[idx]), dtype=np.int8)).int()
                for key, val in self.encodings.items()
                if key != 'special_tokens_mask'}
        item['labels'] = torch.tensor(self.labels[idx]).long()
        return item


""" Adapted code from BertForTokenClassification to work the Helixer way"""
class HelixerBert(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, args):
        self.config = BertConfig(num_labels=4)
        super().__init__(self.config)

        self.num_labels = 4
        self.bert = BertModel.from_pretrained(args.load_model_path)

        self.classifier = torch.nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        special_tokens_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids.int(),
            attention_mask=attention_mask.int(),
            token_type_ids=token_type_ids.int(),
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                attention_mask[:, 0] = 0
                attention_mask[:, -1] = 0
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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

        # config = BertConfig(num_labels=4, max_position_embeddings=502, vocab_size=69, num_hidden_layers=3)
        # finetuning_model = BertForTokenClassification.from_pretrained(args.load_model_path, config=config)
        finetuning_model = HelixerBert(args)
        HelixerModelBase.print_model_info(finetuning_model, 'Finetuning')

        trainer = Trainer(
            model=finetuning_model,
            args=self.training_args(),
            train_dataset=HelixerDatasetFinetune(args, 'training'),
            eval_dataset=HelixerDatasetFinetune(args, 'validation')
        )
        trainer.train()

if __name__ == '__main__':
    model = HelixerModelFinetune()
