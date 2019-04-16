from geenuff.applications import gffimporter
from geenuff import orm

from helixerprep.datas.sequences import StructuredGenome

import hashlib


# mod default importer to use json instead of .fa for sequences
# todo, test import from _json_
class ImportControl(gffimporter.ImportControl):
    @property
    def genome_handler_type(self):
        return GenomeHandler


class GenomeHandler(gffimporter.GenomeHandler):
    def add_sequences(self, seq_file):
        self.add_json(seq_file)

    def add_json(self, seq_file):
        genome = StructuredGenome()
        genome.from_json(seq_file)
        for seq in genome.sequences:
            sequence = seq.full_sequence()
            sha1 = hashlib.sha1()
            sha1.update(sequence.encode())
            # todo, parallelize sequence & annotation format, then import directly from sequence_info (~Slice)
            orm.Coordinate(seqid=seq.meta_info.seqid, start=0, sha1=sha1.hexdigest(),
                           end=seq.meta_info.total_bp, genome=self.data)
