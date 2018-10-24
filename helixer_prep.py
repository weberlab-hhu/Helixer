from dustdas import gffhelper, fastahelper
import argparse


def gff3_to_json(gff3):
    gh = gffhelper.read_gff_file(infile=gff3)
    transcripts = 0
    for entry in gh:
        if entry.type == 'transcript':
            transcripts += 1
    print(transcripts)


def fasta_to_json(fasta):
    pass  # todo, this would be a good spot to actually start


class AnnotationSequence(object):
    def __init__(self, seqid, ):
        pass


def main(gff3, fileout):
    annotation = gff3_to_json(gff3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gff3', help='gff3 formatted file to parse / standardize')
    parser.add_argument('-o', '--out', help='output prefix, defaults to "standardized"', default="standardized")
    args = parser.parse_args()
    main(args.gff3, args.out)