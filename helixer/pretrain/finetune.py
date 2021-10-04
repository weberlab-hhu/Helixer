#! /usr/bin/env python3
import sys
import h5py
import argparse
import numcodecs
import numpy as np
from pathlib import Path
from functools import partial
from collections import defaultdict
from sklearn.metrics import confusion_matrix

import torch
from torch.nn import CrossEntropyLoss
from transformers import (BertConfig, BertModel, BertForMaskedLM, BertTokenizerFast, Trainer,
                          TrainingArguments, BertForTokenClassification, TrainerCallback)
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from helixer.pretrain.base import HelixerDatasetBase, HelixerModelBase, print_tensors
from helixer.pretrain.Metrics import HelixerConfusionMatrixGenic


class HelixerDatasetFinetune(HelixerDatasetBase):

    def __init__(self, args, split):
        super().__init__(args)

        print(f'loading all {split} data in memory')
        h5_file = h5py.File(f'{args.data_dir}/{split}_data.h5', 'r')
        X_dset = h5_file['/data/X']

        self.labels, self.sample_weights = [], []
        load_batch_size = 100 if args.debug else 10000
        bases = ['C', 'A', 'T', 'G']
        for offset in range(0, len(X_dset), load_batch_size):
            batch_slice = slice(offset, offset + load_batch_size)
            X_one_hot = X_dset[batch_slice]
            X_batch = np.full(X_one_hot.shape[:2], 'N', dtype='|S1')
            # get indices of all ATCG bases, the rest gets encoded as 'N'
            idx_all = np.where(X_one_hot == 1.)
            for i in range(4):
                idx_base = idx_all[2] == i
                X_batch[idx_all[0][idx_base], idx_all[1][idx_base]] = bases[i]
            # load and compress labels and sample weights
            y_batch = np.argmax(h5_file['/data/y'][batch_slice], axis=-1).astype(np.uint8)
            sw_batch = h5_file['/data/sample_weights'][batch_slice]
            self.labels.extend([self.compressor.encode(e) for e in list(y_batch)])
            self.sample_weights.extend([self.compressor.encode(e) for e in list(sw_batch)])

            # the following could be improved by concatenating seqs with the same seqid
            for i in range(len(X_batch)):
                self._tokenize(X_batch[i].tobytes().decode(), pretrain=False)

            print(f'{offset + load_batch_size}/{len(X_dset)}')

            if args.debug:
                break

    def __getitem__(self, idx):
        item = {key: torch.tensor(
                    [np.frombuffer(self.compressor.decode(sub_val), dtype=np.int8) for sub_val in val[idx]]).int()
                for key, val in self.encodings.items()}
        label = np.frombuffer(self.compressor.decode(self.labels[idx]), dtype=np.uint8)
        sample_weight = np.frombuffer(self.compressor.decode(self.sample_weights[idx]), dtype=np.int8)
        item['labels'] = torch.tensor(label).long()
        item['sample_weights'] = torch.tensor(sample_weight).bool()
        return item


""" Adapted code from BertForTokenClassification to work the Helixer way"""
class HelixerBert(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, args):
        self.args = args
        self.config = BertConfig(num_labels=4)  # not sure about this
        super().__init__(self.config)

        self.num_labels = 4
        self.bert = BertModel.from_pretrained(args.load_model_path)

        self.lstm = torch.nn.LSTM(input_size=self.bert.config.hidden_size,
                                  hidden_size=args.n_lstm_units,
                                  num_layers=args.n_lstm_layers,
                                  bidirectional=True)
        self.classifier = torch.nn.Linear(args.n_lstm_units * 2, self.config.num_labels)
        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        sample_weights=None,
        special_tokens_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # call the BERT model for each of the 40 subsequences individually, but for the whole batch
        input_ids = input_ids.int()
        attention_mask = attention_mask.int()
        token_type_ids = token_type_ids.int()
        n_sub_seqs = input_ids.shape[1]
        bert_outputs = []
        # only backprop a small number of bert forward passes
        backprop_idxs = np.random.choice(range(n_sub_seqs), self.n_backprob_samples, replace=False)
        for i in range(n_sub_seqs):
            bert_call = partial(self.bert,
                torch.squeeze(input_ids[:, i]),
                attention_mask=torch.squeeze(attention_mask[:, i]),
                token_type_ids=torch.squeeze(token_type_ids[:, i]),
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if i in backprop_idxs:
                out = bert_call()
            else:
                with torch.no_grad():
                    out = bert_call()
            bert_outputs.append(out[0])

        lstm_input = torch.stack(bert_outputs, axis=1)
        lstm_input = lstm_input[:, :, 1:-1].reshape(lstm_input.shape[0], -1, lstm_input.shape[-1])
        lstm_outputs = self.lstm(lstm_input)
        logits = self.classifier(lstm_outputs[0])

        loss = None
        if labels is not None:
            # hardcoded weights for now
            loss_fct = CrossEntropyLoss(weight=torch.Tensor([0.7, 1.6, 1.2, 1.2]).cuda())
            # only keep active parts of the loss
            attention_mask = attention_mask[:, :, 1:-1]
            active_loss = (attention_mask.reshape(-1) == 1) & sample_weights.reshape(-1)
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(loss=loss, logits=logits)


class HelixerEvalCallback(TrainerCallback):
    def __init__(self, cli_args, model, eval_dataset):
        self.batch_size = cli_args.batch_size_valid
        self.model = model
        self.eval_dataset = eval_dataset

    def on_epoch_end(self, args, state, control, **kwargs):
        cm = HelixerConfusionMatrixGenic()
        input_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'sample_weights']
        # construct batches by hand, there probably is code for that in transformers.Trainer
        for offset in range(0, len(self.eval_dataset), self.batch_size):
            print(f'{offset}/{len(self.eval_dataset)}')
            inputs = defaultdict(list)
            for i in range(min(self.batch_size, (len(self.eval_dataset) - offset))):
                data = self.eval_dataset[offset+i]
                for key in input_keys:
                    inputs[key].append(data[key])

            batched_inputs = {key:torch.stack(vals, axis=0).cuda() for key, vals in inputs.items()}
            with torch.no_grad():
                out = self.model.forward(**batched_inputs)
                # accumulate cm on the GPU
                y_pred = out[1].view(-1, 4)
                y_true = batched_inputs['labels'].view(-1)
                # sw = batched_inputs['sample_weights'].view(-1)
                # y_pred = y_pred[sw]
                # y_true = y_true[sw]
                cm.add_to_cm(y_true, y_pred)
        cm.print_cm()


class HelixerModelFinetune(HelixerModelBase):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('-d', '--data-dir', type=str, required=True)
        self.parser.add_argument('-l', '--load-model-path', type=str, default='')
        self.parser.add_argument('--n-lstm-layers', type=int, default=1)
        self.parser.add_argument('--n-lstm-units', type=int, default=128)
        self.parser.add_argument('--n-backprop-samples', type=int, default=2)
        self.parse_args()
        self.run()

    def run(self):
        train_dataset = HelixerDatasetFinetune(self.args, 'training')
        eval_dataset = HelixerDatasetFinetune(self.args, 'validation')

        finetuning_model = HelixerBert(self.args)
        HelixerModelBase.print_model_info(finetuning_model, 'Finetuning')

        trainer = Trainer(
            model=finetuning_model,
            args=self.training_args(),
            train_dataset=train_dataset,
            callbacks=[HelixerEvalCallback(self.args, finetuning_model, eval_dataset)],
        )
        trainer.train()

if __name__ == '__main__':
    model = HelixerModelFinetune()
