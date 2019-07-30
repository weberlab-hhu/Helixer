#! /usr/bin/env python3

import os

nni_base = '/home/felix-stiehler/nni/experiments'
nni_id = 'jaeWb7Ff'
trials_folder = '{}/{}/trials'.format(nni_base, nni_id)

print(','.join(['genome', 'loss', 'acc_overall', 'f1_total_0', 'f1_total_1', 'nni_id']))
for folder in os.listdir(trials_folder):
    # get genome name
    parameters = eval(open('{}/{}/parameter.cfg'.format(trials_folder, folder)).read())
    path = parameters['parameters']['test_data']
    genome = path.split('/')[6]

    # get loss and overall acc
    log_file = open('{}/{}/trial.log'.format(trials_folder, folder))
    total_count = 0
    for line in log_file:
        if line.startswith('| total 0'):
            total_count += 1
            if total_count == 3:
                # overall total
                f1_total_0 = line.split()[8]
                line = next(log_file)
                f1_total_1 = line.split()[8]

    overall_results = eval(''.join(line.strip().split(' ')[4:]))
    numbers = [overall_results['loss'], overall_results['acc']]
    print(','.join([genome] + ['{:.4f}'.format(n) for n in numbers] + [f1_total_0, f1_total_1, folder]))
