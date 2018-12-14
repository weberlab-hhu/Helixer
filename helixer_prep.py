import logging
import argparse
import os

import sequences
import gff_2_annotations


def gff3_to_json(gff3, db_path, sequence_path, prob_path):
    #db_path = 'sqlite:///dummy.db'
    controller = gff_2_annotations.ImportControl(database_path=db_path, err_path=prob_path)
    controller.add_sequences(sequence_path)
    controller.add_gff(gff3)


# todo, maybe this should be a method on the StructuredGenome...? or just in main
def fasta_to_json(fasta, json, smallest_mer=2, largest_mer=2):
    sg = sequences.StructuredGenome()
    sg.add_fasta(fasta, smallest_mer=smallest_mer, largest_mer=largest_mer)
    sg.to_json(json)
    return sg


def load_sequence_json(json):
    sg = sequences.StructuredGenome()
    sg.from_json(json)
    return sg


class PathFinder(object):
    INPUT = 'input'
    OUTPUT = 'output'

    def __init__(self, basedir, fasta=None, gff=None):
        # directories
        self.basedir = basedir
        self.input = '{}/{}/'.format(self.basedir, PathFinder.INPUT)
        self.output = '{}/{}/'.format(self.basedir, PathFinder.OUTPUT)
        for dir in [self.basedir, self.input, self.output]:
            os.makedirs(dir, exist_ok=True)
        # files
        self.fasta_in = self._get_fa(fasta)
        self.gff_in = self._get_gff(gff)
        self.annotations_out = 'sqlite:///{}annotation.sqlitedb'.format(self.output)
        self.sequence_out = '{}sequence.json'.format(self.output)
        self.problems_out = '{}problems.txt'.format(self.output)
        self.sliced_sequence_out = '{}sliced_sequence.json'.format(self.output)
        self.sliced_annotations_out = 'sqlite:///{}sliced_annotation.sqlitedb'.format(self.output)

    def _get_fa(self, provided):
        if provided is not None:
            return provided
        maybe = os.listdir(self.input)
        # todo, actual file type detection
        maybe = [x for x in maybe if (x.endswith('.fa') or x.endswith('.fasta'))]
        self._confirm_exactly_one(maybe, 'fasta')
        return self.input + maybe[0]

    def _get_gff(self, provided):
        if provided is not None:
            return provided
        maybe = os.listdir(self.input)
        maybe = [x for x in maybe if (x.endswith('.gff') or x.endswith('.gff3'))]
        self._confirm_exactly_one(maybe, 'gff')
        return self.input + maybe[0]

    @staticmethod
    def _confirm_exactly_one(possibilities, info):
        assert len(possibilities) == 1, 'no(n) unique {} file found as input. Found: {}'.format(info, possibilities)


def main(gff3, fasta, basedir, smallest_mer=2, largest_mer=2, slice=True, seed='puma'):
    logging.basicConfig(level=logging.WARNING)
    #annotation = gff3_to_json(gff3)
    paths = PathFinder(basedir, fasta=fasta, gff=gff3)
    if not os.path.exists(paths.sequence_out):
        sequences = fasta_to_json(paths.fasta_in, paths.sequence_out, smallest_mer=smallest_mer,
                                  largest_mer=largest_mer)
    else:
        sequences = load_sequence_json(paths.sequence_out)  # todo, choose one spot to import this & clean up
    gff3_to_json(paths.gff_in, paths.annotations_out, paths.sequence_out, paths.problems_out)
    if slice:
        sequences.divvy_each_sequence(seed)
        sequences.to_json(paths.sliced_sequence_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', help='organized output (& input) directory', required=True)
    custominput = parser.add_argument_group("Override default with custom input location:")
    custominput.add_argument('--gff3', help='gff3 formatted file to parse / standardize')
    custominput.add_argument('--fasta', help='fasta file to parse standardize')
    fasta_specific = parser.add_argument_group("Fasta meta_info customizable:")
    fasta_specific.add_argument('--min_k', help='minumum size kmer to calculate from sequence', default=2, type=int)
    fasta_specific.add_argument('--max_k', help='maximum size kmer to calculate from sequence', default=2, type=int)
    slicing = parser.add_argument_group("Split data into train/test/dev sets")
    slicing.add_argument('--slice', action='store_true', help='use this flag to make additional split output files')
    slicing.add_argument('--seed', default='puma',
                         help="random seed is md5sum(sequence) + this parameter; don't change without cause")
    args = parser.parse_args()
    main(args.gff3, args.fasta, args.basedir, smallest_mer=args.min_k, largest_mer=args.max_k, slice=args.slice,
         seed=args.seed)
