#! /usr/bin/env python3
import sys
import h5py
import argparse
import numcodecs
import numpy as np
from pathlib import Path
import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast, Trainer, TrainingArguments
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from helixer.pretrain.base import HelixerDatasetBase, HelixerModelBase

class HelixerDatasetFinetune(HelixerDatasetBase):
    def __init__(self, args, split):
        super().__init__(args)

        h5_file = h5py.File(f'{args.data_dir}/{split}_data.h5', 'r')

        # turn one hot encoding into strings again ...
        # all kinds of bugs here with padding, sample_weights, etc... just a quick test for downstream code
        X = np.full((1000, 20000), 'N', dtype='|S1')
        self.labels = h5_file['/data/y'][:1000]
        # get indices of all ATCG bases, the rest gets encoded as 'N'
        batch_size = 1000
        bases = ['C', 'A', 'T', 'G']
        for offset in range(0, len(X), batch_size):
            idx_all = np.where(h5_file['data/X'][offset:offset+batch_size] == 1.)
            for i in range(4):
                idx_base = idx_all[2] == i
                X[idx_all[0][idx_base], idx_all[1][idx_base]] = bases[i]
            break
        self._tokenize(X.flatten().tobytes().decode())

    def __getitem__(self, idx):
        item = {key: torch.tensor(np.frombuffer(self.compressor.decode(val[idx]), dtype=np.int8))
                for key, val in self.encodings.items()}
        return item


""" Adapted code from BertForTokenClassification to work the Helixer way"""
class HelixerBert(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, args):
        self.config = BertConfig(num_labels=4)
        super().__init__(self.config)
        self.bert = BertForMaskedLM.from_pretrained(args.load_model_path)

        self.classifier = torch.nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward( self,
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

        finetuning_model = HelixerBert(args)
        print('Finetuning config:')
        print(finetuning_model.config)

        trainer = Trainer(
            model=finetuning_model,
            args=self.training_args(),
            train_dataset=HelixerDatasetFinetune(args, 'training'),
            eval_dataset=HelixerDatasetFinetune(args, 'validation')
        )
        trainer.train()

if __name__ == '__main__':
    model = HelixerModelFinetune()
