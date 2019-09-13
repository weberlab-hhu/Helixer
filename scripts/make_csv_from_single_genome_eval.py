#! /usr/bin/env python3

import os

nni_base = '/home/felix/nni/experiments'
nni_id = 'adPNZ2sW'
trials_folder = '{}/{}/trials'.format(nni_base, nni_id)

print(','.join(['genome', 'loss', 'acc_overall', 'acc_genic', 'acc_intergenic', 'acc_UTR',
                'acc_CDS', 'acc_intron', 'nni_id']))
for folder in os.listdir(trials_folder):
    # get genome name
    parameters = eval(open('{}/{}/parameter.cfg'.format(trials_folder, folder)).read())
    path = parameters['parameters']['test_data']
    genome = path.split('/')[6]

    # parse confusion matrix
    log_file = open('{}/{}/trial.log'.format(trials_folder, folder))
    for line in log_file:
        if 'array([[' in line:  # start of cm
            cm_str = line.split('array(')[1].strip()
            for i in range(3):
                cm_str += next(log_file).strip()
            cm_str = cm_str[:-1]  # remove last round bracket
    cm = eval(cm_str)
    acc_ig, acc_UTR, acc_CDS, acc_intron = cm[0][0], cm[1][1], cm[2][2], cm[3][3]

    # parse metrics
    overall_results = eval(''.join(line.strip().split(' ')[4:]))
    numbers = [
        overall_results['loss'],
        overall_results['main_output_acc'],
        overall_results['main_output_acc_g_oh'],
        acc_ig,
        acc_UTR,
        acc_CDS,
        acc_intron
    ]
    print(','.join([genome] + ['{:.4f}'.format(n) for n in numbers] + [folder]))
