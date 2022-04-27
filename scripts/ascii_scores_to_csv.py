#! /usr/bin/env python3
"""
Extracts individual tables from text file with 1+ ASCII terminal tables and save to individual csv files.
"""

import re
import argparse
import os


def parse_table(splittable):
    out = []
    # splittable = table.split('\n')
    header = splittable[0]
    header = re.sub('\+|-', '', header)
    for line in splittable[1:]:
        if not line.startswith('+'):
            line = line.replace(' ', '')
            line = re.sub('\|', ',', line)
            line = re.sub('^,|,$', '', line)
            out.append(line)

    return header, '\n'.join(out)


def gen_tables(filein):
    out = []
    with open(filein) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('|') or line.startswith('+'):
                out.append(line)
            else:
                # ignore double non-table lines etc...
                if out:
                    yield out
                    out = []
    if out:
        yield out


def main(filein, dirout):
    i = 0
    x = 0
    prefixes = ["genic_CM_", "phase_CM_", "phase_with_intersect_CM_"]
    if not os.path.exists(dirout):
        os.mkdir(dirout)
    for table in gen_tables(filein):
        i += 1
        if i % 3 != 0:
            pass
        else:
            x += 1
            pfx = prefixes[x]
        header, tab = parse_table(table)
        fileout = '{}/{}{}.csv'.format(dirout, pfx, header)
        with open(fileout, 'w') as f:
            f.write(tab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filein', required=True,
                        help='input scores file with ascii tables (e.g. stdout accs_genic_intergenic.py)')
    parser.add_argument('-o', '--outdir', required=True,
                        help='directory for output files (files themselves get table names)')
    args = parser.parse_args()
    main(args.filein, args.outdir)
