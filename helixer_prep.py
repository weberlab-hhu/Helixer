from sequence_helpers import reverse_complement
from dustdas import gffhelper, fastahelper
import argparse
import itertools
import copy


def gff3_to_json(gff3):
    gh = gffhelper.read_gff_file(infile=gff3)
    transcripts = 0
    for entry in gh:
        if entry.type == 'transcript':
            transcripts += 1
    print(transcripts)


def fasta_to_json(fasta):
    fh = fastahelper.FastaParser()
    meta_genome = MetaInfoGenome(fasta)
    for infos, seq in fh.read_fasta(fasta):
        mis = MetaInfoSequence(fasta_header=infos, sequence=seq)
        meta_genome.add_sequence_meta_info(mis)
        print(mis.to_json())
        #x = reverse_complement(seq)
        #for _ in gen_mers(seq, 2):
        #    pass
        #print(x[:20])
        print(infos)
        print(seq[:20])
    print(meta_genome.__dict__)


class AnnotationSequence(object):
    def __init__(self, seqid, ):
        pass


class MetaInfoGenome(object):
    def __init__(self, fasta_path=None, species=None, accession=None, version=None, acquired_from=None):
        self.species = None
        if fasta_path is not None:
            # todo parse more guesses out of this
            self.species = self.species_from_path(fasta_path)
        # include anything the user has set
        if species is not None:
            self.species = species
        self.accession = accession
        self.version = version
        self.acquired_from = acquired_from
        self.total_bp = 0
        self.gc_content = 0
        self.cannonical_kmer_content = {}
        self.ambiguous_content = 0


    @staticmethod
    def species_from_path(fasta_path):
        if '/' in fasta_path:
            fasta_path = fasta_path.split('/')[-1]
        if fasta_path.endswith('.fasta'):
            species = fasta_path.replace('.fasta', '')
        elif fasta_path.endswith('.fa'):
            species = fasta_path.replace('.fa', '')
        else:
            species = fasta_path
        return species

    def add_sequence_meta_info(self, seq_met_info):
        self.total_bp += seq_met_info.total_bp
        self.gc_content += seq_met_info.gc_content
        # todo, add kmer counts
        #for kmer in seq_met_info
        self.cannonical_kmer_content = add_paired_dictionaries(self.cannonical_kmer_content,
                                                               seq_met_info.cannonical_kmer_content)
        self.ambiguous_content += seq_met_info.ambiguous_content


class MetaInfoSequence(object):
    def __init__(self, fasta_header, sequence, id_delim=' ', smallest_mer=2, largest_mer=2):
        self.deprecated_header = fasta_header
        self.seqid = fasta_header.split(id_delim)[0]
        self.total_bp = len(sequence)
        self.gc_content = None
        self.cannonical_kmer_content = None
        self.ambiguous_content = None
        self.calculate_and_set_meta(sequence.lower(), smallest_mer, largest_mer)

    def calculate_and_set_meta(self, sequence, smallest_mer, largest_mer):
        # go through all the bp and count all the things
        gc = 0
        gc_bps = ('g', 'c')
        ambiguous = 0
        known_bps = ('a', 't', 'c', 'g')
        for bp in sequence:
            if bp.lower() in gc_bps:
                gc += 1
            if bp not in known_bps:
                ambiguous += 1
        # count all kmers
        self.cannonical_kmer_content = {}
        for k in range(smallest_mer, largest_mer + 1):
            mer_counter = MerCounter(k)
            for mer in gen_mers(sequence, k):
                mer_counter.add_mer(mer)
            self.cannonical_kmer_content[k] = mer_counter.export()
        # record
        self.gc_content = gc
        self.ambiguous_content = ambiguous

    def to_json(self):
        return self.__dict__


class MerCounter(object):
    amb = 'ambiguous_mers'

    def __init__(self, k):
        self.k = k
        self.counts = {}
        # calculate all possible mers of this length, and set counter to 0
        for mer in itertools.product('atcg', repeat=k):
            mer = ''.join(mer)
            self.counts[mer] = 0
        self.counts[MerCounter.amb] = 0
        # most recent base pairs of up to length k
        self.sliding_mer = []

    def add_mer(self, mer):
        try:
            self.counts[mer] += 1
        except KeyError:
            self.counts[MerCounter.amb] += 1

    def export(self):
        out = copy.deepcopy(self.counts)
        for key in self.counts:
            if key != MerCounter.amb:
                # collapse to cannonical kmers
                rc_key = reverse_complement(key)
                if key != min(key, rc_key):
                    out[rc_key] += out[key]
                    out.pop(key)
        return out


def gen_mers(sequence, k):
    for i in range(len(sequence) - k):
        yield sequence[i:(i+k)]


def add_paired_dictionaries(add_to, add_from):
    print('add_to', add_to, type(add_to))
    print('add_from', add_from, type(add_from))
    add_to = copy.deepcopy(add_to)
    for key in add_from:
        if key not in add_to:
            add_to[key] = copy.deepcopy(add_from[key])
        elif isinstance(add_to[key], dict):
            add_to[key] = add_paired_dictionaries(add_to[key], add_from[key])
        else:
            add_to[key] += add_from[key]
    return add_to


def main(gff3, fasta, fileout):
    #annotation = gff3_to_json(gff3)
    sequences = fasta_to_json(fasta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gff3', help='gff3 formatted file to parse / standardize')
    parser.add_argument('--fasta', help='fasta file to parse standardize', required=True)
    parser.add_argument('-o', '--out', help='output prefix, defaults to "standardized"', default="standardized")
    args = parser.parse_args()
    main(args.gff3, args.fasta, args.out)
