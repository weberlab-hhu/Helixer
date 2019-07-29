#! /usr/bin/env python3

import os

nni_base = '/home/felix-stiehler/nni/experiments'
nni_id = 'jaeWb7Ff'
trials_folder = '{}/{}/trials'.format(nni_base, nni_id)

print(','.join(['genome', 'loss', 'acc_overall', 'acc_transcript', 'acc_coding', 'acc_intron', 'nni id']))
for folder in os.listdir(trials_folder):
    # get genome name
    parameters = eval(open('{}/{}/parameter.cfg'.format(trials_folder, folder)).read())
    path = parameters['parameters']['test_data']
    genome = path.split('/')[6]

    # get overall performance value and print as csv
    log_file = open('{}/{}/trial.log'.format(trials_folder, folder))
    for line in log_file:
        pass
    results = eval(''.join(line.strip().split(' ')[4:]))
    print(','.join([genome] + ['{:.4f}'.format(r) for r in results.values()] + [folder]))
