from dustdas import gffhelper
import argparse
import os

import sequences


def gff3_to_json(gff3):
    gh = gffhelper.read_gff_file(infile=gff3)
    transcripts = 0
    for entry in gh:
        if entry.type == 'transcript':
            transcripts += 1
    print(transcripts)


# todo, maybe this should be a method on the StructuredGenome...? or just in main
def fasta_to_json(fasta, json, smallest_mer=2, largest_mer=2):
    sg = sequences.StructuredGenome()
    sg.add_fasta(fasta, smallest_mer=smallest_mer, largest_mer=largest_mer)
    sg.to_json(json)


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
        self.annotations_out = '{}annotation.json'.format(self.output)
        self.sequence_out = '{}sequence.json'.format(self.output)

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


def main(gff3, fasta, basedir, smallest_mer=2, largest_mer=2):
    #annotation = gff3_to_json(gff3)
    paths = PathFinder(basedir, fasta=fasta, gff=gff3)
    fasta_to_json(paths.fasta_in, paths.sequence_out, smallest_mer=smallest_mer, largest_mer=largest_mer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', help='organized output (& input) directory', required=True)
    custominput = parser.add_argument_group("Override default with custom input location:")
    custominput.add_argument('--gff3', help='gff3 formatted file to parse / standardize')
    custominput.add_argument('--fasta', help='fasta file to parse standardize')
    fasta_specific = parser.add_argument_group("Fasta meta_info customizable:")
    fasta_specific.add_argument('--min_k', help='minumum size kmer to calculate from sequence', default=2, type=int)
    fasta_specific.add_argument('--max_k', help='maximum size kmer to calculate from sequence', default=2, type=int)
    args = parser.parse_args()
    main(args.gff3, args.fasta, args.basedir, smallest_mer=args.min_k, largest_mer=args.max_k)
