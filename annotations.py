from dustdas import gffhelper
import logging
import copy
import intervaltree
import helpers
import enum

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Table, Column, Integer, ForeignKey, String, Enum, CheckConstraint
from sqlalchemy.orm import relationship

class FeatureDecoder(object):
    def __init__(self):
        self.error_buffer = 2000
        # gene like, generally having a collection of transcripts
        self.gene = 'gene'
        self.super_gene = 'super_gene'
        self.ncRNA_gene = 'ncRNA_gene'
        self.pseudogene = 'pseudogene'
        self.gene_level = [self.gene, self.super_gene, self.ncRNA_gene, self.pseudogene]

        # transcript like, generally having a collection of exons, indicating how they are spliced
        # also ultimately, if not explicitly having a transcription start and termination site
        self.mRNA = 'mRNA'
        self.tRNA = 'tRNA'
        self.rRNA = 'rRNA'
        self.miRNA = 'miRNA'
        self.snoRNA = 'snoRNA'
        self.snRNA = 'snRNA'
        self.SRP_RNA = 'SRP_RNA'
        self.lnc_RNA = 'lnc_RNA'
        self.pre_miRNA = 'pre_miRNA'
        self.RNase_MRP_RNA = 'RNase_MRP_RNA'
        self.transcript = 'transcript'
        self.primary_transcript = 'primary_transcript'
        self.pseudogenic_transcript = 'pseudogenic_transcript'  # which may or may not be transcribed, hard to say
        self.transcribed = [self.mRNA, self.transcript, self.tRNA, self.primary_transcript, self.rRNA, self.miRNA,
                            self.snoRNA, self.snRNA, self.SRP_RNA, self.lnc_RNA, self.pre_miRNA, self.RNase_MRP_RNA,
                            self.pseudogenic_transcript]

        # regions of original (both) or processed (exon) transcripts
        self.exon = 'exon'
        self.intron = 'intron'
        self.sub_transcribed = [self.exon, self.intron]

        # sub-exon-level categorization (but should have transcribed as parent)
        self.cds = 'CDS'
        self.five_prime_UTR = 'five_prime_UTR'
        self.three_prime_UTR = 'three_prime_UTR'
        self.coding_info = [self.cds, self.five_prime_UTR, self.three_prime_UTR]

        # point annotations
        self.TSS = 'TSS'  # transcription start site
        self.TTS = 'TTS'  # transcription termination site
        self.start_codon = 'start_codon'
        self.stop_codon = 'stop_codon'
        self.donor_splice_site = 'donor_splice_site'
        self.acceptor_splice_site = 'acceptor_splice_site'
        # use the following when the far side of a splice site is on a different strand and/or sequence
        # does not imply an intron!
        # e.g. for trans-splicing
        self.trans_donor_splice_site = 'trans_donor_splice_site'
        self.trans_acceptor_splice_site = 'trans_acceptor_splice_site'
        self.point_annotations = [self.TSS, self.TTS, self.start_codon, self.stop_codon, self.donor_splice_site,
                                  self.acceptor_splice_site, self.trans_donor_splice_site,
                                  self.trans_acceptor_splice_site]

        # regions (often but not always included so one knows the size of the chromosomes / contigs / whatever
        self.region = 'region'
        self.chromosome = 'chromosome'
        self.supercontig = 'supercontig'
        self.regions = [self.region, self.chromosome, self.supercontig]

        # things that don't appear to really be annotations
        self.match = 'match'
        self.cDNA_match = 'cDNA_match'
        self.ignorable = [self.match, self.cDNA_match]

        # for mistakes or near-mistakes / marking partials
        self.error = 'error'
        self.status_coding = 'status_coding'
        self.status_intron = 'status_intron'
        self.status_five_prime_UTR = 'status_five_prime_UTR'
        self.status_three_prime_UTR = 'status_three_prime_UTR'
        self.status_intergenic = 'status_intergenic'
        self.statuses = [self.status_coding, self.status_intron, self.status_five_prime_UTR,
                         self.status_three_prime_UTR, self.status_intergenic]
        # and putting them together
        self.on_sequence = self.sub_transcribed + self.coding_info + self.point_annotations
        self.known = self.gene_level + self.transcribed + self.sub_transcribed + self.coding_info + \
                     self.point_annotations + self.regions + self.ignorable + [self.error] + self.statuses


# setup classes for data holding
Base = declarative_base()


class AnnotatedGenome(Base):
    # todo, go ahead and remove this WEDNESDAY
    __tablename__ = 'annotated_genomes'

    # data
    id = Column(Integer, primary_key=True)
    species = Column(String)
    accession = Column(String)
    version = Column(String)
    acquired_from = Column(String)
    sequence_infos = relationship("SequenceInfo", back_populates="annotated_genome")

    gffkey = FeatureDecoder()
    # todo, figure out if one wants to keep the id makers like this...
#    transcript_ider = helpers.IDMaker(prefix='trx')
#    protein_ider = helpers.IDMaker(prefix='prt')
#    feature_ider = helpers.IDMaker(prefix='ftr')
#
#    def add_gff(self, gff_file, genome, err_file='trans_splicing.txt'):
#        sls = SuperLociSlice()
#        sls.genome = self
#        sls.add_gff(gff_file, genome, err_file=err_file)
#        self.super_loci_slices.append(sls)
#
#    def divvy_up_super_loci(self, divvied_sequences):
#        # todo: load to interval tree
#        # todo: represent partial super_loci
#        # todo: code split super_loci
#        # todo: put it together
#        pass
#
#    def clean_post_load(self):
#        for sl in self.super_loci_slices:
#            sl.genome = self


#class MetaInfoAnnotation(GenericData):
#    def __init__(self):
#        super().__init__()
#        self.spec += [('number_genes', True, int, None),
#                      ('bp_intergenic', True, int, None),
#                      ('bp_coding', True, int, None),
#                      ('bp_intronic', True, int, None),
#                      ('bp_3pUTR', True, int, None),
#                      ('bp_5pUTR', True, int, None)]
#
#        # todo, does this make sense, considering that any given bp could belong to multiple of the following
#        self.number_genes = 0
#        self.bp_intergenic = 0
#        self.bp_coding = 0
#        self.bp_intronic = 0
#        self.bp_3pUTR = 0
#        self.bp_5pUTR = 0
#
#
#class FeatureLike(GenericData):
#    def __init__(self):
#        super().__init__()
#        self.spec += [('id', True, str, None),
#                      ('type', True, str, None),
#                      ('is_partial', True, bool, None),
#                      ("gff_entry", False, gffhelper.GFFObject, None),
#                      ]
#        self.id = ''
#        self.type = ''
#        self.is_partial = False
#        self.gff_entry = None
#        # self.is_reconstructed = False
#        # self.is_type_in_question = False

class ProcessingSet(enum.Enum):
    train = 1
    dev = 2
    test = 3


class SequenceInfo(Base):
    __tablename__ = "sequence_infos"

    id = Column(Integer, primary_key=True)
    annotated_genome_id = Column(Integer, ForeignKey('annotated_genomes.id'), nullable=False)
    annotated_genome = relationship('AnnotatedGenome', back_populates="sequence_infos")
    processing_set = Column(Enum(ProcessingSet))
    coordinates = relationship('Coordinates', back_populates="sequence_info")
    super_loci = relationship('SuperLocus', back_populates="sequence_info")
#        self.mapper = helpers.Mapper()
#
#    @property
#    def gffkey(self):
#        return self.genome.gffkey
#

    @property
    def seq_info(self):
        try:
            _ = self._seq_info
        except AttributeError:
            seq_info = {}
            for x in self.coordinates:
                seq_info[x.seqid] = x
            self._seq_info = seq_info  # todo, is there a sqlalchemy Base compatible way to add attr in init funciton?
        return self._seq_info
#
#    def _add_sequences(self, genome):
#        for seq in genome.sequences:
#            mi = CoordinateInfo()
#            mi.seqid = seq.meta_info.seqid
#            mi.start = 1
#            mi.end = seq.meta_info.total_bp
#            self.coordinates.append(mi)
#
#    def add_gff(self, gff_file, genome, err_file='trans_splicing.txt'):
#        err_handle = open(err_file, 'w')
#        self._add_sequences(genome)
#
#        gff_seq_ids = helpers.get_seqids_from_gff(gff_file)
#        mapper, is_forward = helpers.two_way_key_match(self.seq_info.keys(), gff_seq_ids)
#        self.mapper = mapper
#
#        if not is_forward:
#            raise NotImplementedError("Still need to implement backward match if fasta IDs are subset of gff IDs")
#
#        for entry_group in self.group_gff_by_gene(gff_file):
#            new_sl = SuperLocus()
#            new_sl.slice = self
#            new_sl.add_gff_entry_group(entry_group, err_handle)
#
#            self.super_loci.append(new_sl)
#            if not new_sl.transcripts and not new_sl.features:
#                print('{} from {} with {} transcripts and {} features'.format(new_sl.id,
#                                                                              entry_group[0].source,
#                                                                              len(new_sl.transcripts),
#                                                                              len(new_sl.features)))
#        err_handle.close()
#
#    def useful_gff_entries(self, gff_file):
#        skipable = self.gffkey.regions + self.gffkey.ignorable
#        reader = gffhelper.read_gff_file(gff_file)
#        for entry in reader:
#            if entry.type not in self.gffkey.known:
#                raise ValueError("unrecognized feature type fr:qom gff: {}".format(entry.type))
#            if entry.type not in skipable:
#                yield entry
#
#    def group_gff_by_gene(self, gff_file):
#        reader = self.useful_gff_entries(gff_file)
#        gene_group = [next(reader)]
#        for entry in reader:
#            if entry.type in self.gffkey.gene_level:
#                yield gene_group
#                gene_group = [entry]
#            else:
#                gene_group.append(entry)
#        yield gene_group
#
#    def load_to_interval_tree(self):
#        trees = {}
#        for seqid in self.seq_info:
#            trees[seqid] = intervaltree.IntervalTree()
#        for sl in self.super_loci:
#            for fkey in sl.features:
#                feature = sl.features[fkey]
#                trees[feature.seqid][feature.py_start:feature.py_end] = feature
#        return trees
#
#    def slice_further(self, seqid, slice_id, start, end, processing_set, trees):
#        # setup new slice
#        new = SuperLociSlice()
#        mi = CoordinateInfo()
#        mi.seqid = seqid
#        mi.start = start
#        mi.end = end
#        new.coordinates = mi
#        new.slice_id = slice_id
#        new.processing_set = processing_set
#        # and get all features
#        tree = trees[seqid]
#        branch = tree[start - 1:end]  # back to python coordinates  # todo, double check this gets overlaps not contains
#        features_by_sl = {}
#        for intvl in branch:
#            sl_id = intvl.data.super_locus.id
#            if sl_id in features_by_sl:
#                features_by_sl[sl_id].append(intvl.data)
#            else:
#                features_by_sl[sl_id] = [intvl.data]
#        for sl_id in features_by_sl:
#            super_locus = features_by_sl[sl_id][0].super_locus
#            for transcript in super_locus.transcripts:
#                trimmed_transcript = transcript.reconcile_with_slice(seqid, start, end)  # todo
#                # todo, add transcript & features to new slice
#            # todo add sl
#
#    def add_slice(self, seqid, slice_id, start, end, processing_set):
#        pass #todo
#
#    def to_example(self):
#        raise NotImplementedError
#
#    def clean_post_load(self):
#        for sl in self.super_loci:
#            sl.slice = self
#
#    def __deepcopy__(self, memodict={}):
#        raise NotImplementedError  # todo


class Coordinates(Base):
    __tablename__ = 'coordinates'

    id = Column(Integer, primary_key=True)
    start = Column(Integer, nullable=False)
    end = Column(Integer, nullable=False)
    seqid = Column(Integer, nullable=False)
    sequence_info_id = Column(Integer, ForeignKey('sequence_infos.id'))
    sequence_info = relationship('SequenceInfo', back_populates='coordinates')

    __table_args__ = (
        CheckConstraint(start >= 1, name='check_start_1plus'),
        CheckConstraint(end >= start, name='check_end_gr_start'),
        {})


class SuperLocusAliases(Base):
    __tablename__ = 'super_locus_aliases'

    id = Column(Integer, primary_key=True)
    alias = Column(String)
    super_locus_id = Column(Integer, ForeignKey('super_loci.id'))
    super_locus = relationship('SuperLocus', back_populates='aliases')


class SuperLocus(Base):
    __tablename__ = 'super_loci'
    # normally a loci, some times a short list of loci for "trans splicing"
    # this will define a group of exons that can possibly be made into transcripts
    # AKA this if you have to go searching through a graph for parents/children, at least said graph will have
    # a max size defined at SuperLoci
#    t_transcripts = 'transcripts'
#    t_proteins = 'proteins'
#    t_feature_holders = 'generic_holders'
#    types_feature_holders = [t_transcripts, t_proteins, t_feature_holders]
#
#        self.transcripts = {}
#        self.proteins = {}
#        self.generic_holders = {}  # this is here for import from gff only
#        self.ids = []
#        self._dummy_transcript = None
    id = Column(Integer, primary_key=True)
    sequence_info_id = Column(Integer, ForeignKey('sequence_infos.id'))
    sequence_info = relationship('SequenceInfo', back_populates='super_loci')
    # things SuperLocus can have a lot of
    aliases = relationship('SuperLocusAliases', back_populates='super_locus')
    features = relationship('Feature', back_populates='super_locus')
    generic_holders = relationship('GenericHolder', back_populates='super_locus')
    transcribeds = relationship('Transcribed', back_populates='super_locus')
    translateds = relationship('Translated', back_populates='super_locus')
#    def short_str(self):
#        string = "{}\ntranscripts: {}\nproteins: {}\ngeneric holders: {}".format(self.id, list(self.transcripts.keys()),
#                                                                                 list(self.proteins.keys()),
#                                                                                 list(self.generic_holders.keys()))
#        return string
#
#    @property
#    def genome(self):
#        return self.slice.genome
#
#    def dummy_transcript(self):
#        if self._dummy_transcript is not None:
#            return self._dummy_transcript
#        else:
#            # setup new blank transcript
#            transcript = FeatureHolder()
#            transcript.id = self.genome.transcript_ider.next_unique_id()  # add an id
#            transcript.super_locus = self
#            self._dummy_transcript = transcript  # save to be returned by next call of dummy_transcript
#            self.generic_holders[transcript.id] = transcript  # save into main dict of transcripts
#            return transcript
#
#    def add_gff_entry(self, entry):
#        exceptions = entry.attrib_filter(tag="exception")
#        for exception in [x.value for x in exceptions]:
#            if 'trans-splicing' in exception:
#                raise TransSplicingError('trans-splice in attribute {} {}'.format(entry.get_ID(), entry.attribute))
#        gffkey = self.genome.gffkey
#        if entry.type in gffkey.gene_level:
#            self.type = entry.type
#            gene_id = entry.get_ID()
#            self.id = gene_id
#            self.ids.append(gene_id)
#            self.gff_entry = entry
#        elif entry.type in gffkey.transcribed:
#            parent = self.one_parent(entry)
#            assert parent == self.id, "not True :( [{} == {}]".format(parent, self.id)
#            transcript = FeatureHolder()
#            transcript.add_data(self, entry)
#            self.generic_holders[transcript.id] = transcript
#        elif entry.type in gffkey.on_sequence:
#            feature = StructuredFeature()
#            feature.add_data(self, entry)
#            self.features[feature.id] = feature
#
#    def _add_gff_entry_group(self, entries):
#        entries = list(entries)
#        for entry in entries:
#            self.add_gff_entry(entry)
#
#    def add_gff_entry_group(self, entries, ts_err_handle):
#        try:
#            self._add_gff_entry_group(entries)
#            self.check_and_fix_structure(entries)
#        except TransSplicingError as e:
#            self._mark_erroneous(entries[0])
#            logging.warning('skipping but noting trans-splicing: {}'.format(str(e)))
#            ts_err_handle.writelines([x.to_json() for x in entries])
#            # todo, log to file
#
#    @staticmethod
#    def one_parent(entry):
#        parents = entry.get_Parent()
#        assert len(parents) == 1
#        return parents[0]
#
#    def _mark_erroneous(self, entry):
#        assert entry.type in self.genome.gffkey.gene_level
#        logging.warning(
#            '{species}:{seqid}, {start}-{end}:{gene_id} by {src}, No valid features found - marking erroneous'.format(
#                src=entry.source, species=self.genome.meta_info.species, seqid=entry.seqid, start=entry.start,
#                end=entry.end, gene_id=self.id
#            ))
#        sf = StructuredFeature()
#        feature = sf.add_erroneous_data(self, entry)
#        self.features[feature.id] = feature
#
#    def check_and_fix_structure(self, entries):
#        # if it's empty (no bottom level features at all) mark as erroneous
#        if not self.features:
#            self._mark_erroneous(entries[0])
#
#        # todo, but with reconstructed flag (also check for and mark pseudogenes)
#        to_remove = []
#        for key in copy.deepcopy(list(self.generic_holders.keys())):
#            transcript = self.generic_holders[key]
#            old_features = copy.deepcopy(transcript.features)
#
#            t_interpreter = TranscriptInterpreter(transcript)
#            transcript = t_interpreter.transcript  # because of non-inplace shift from ofs -> transcripts, todo, fix
#            t_interpreter.decode_raw_features()
#            # no transcript, as they're already linked
#            self.add_features(t_interpreter.clean_features, feature_holders=None)
#            transcript.delink_features(old_features)
#            to_remove += old_features
#        self.remove_features(to_remove)
#
#    def add_features(self, features, feature_holders=None, holder_type=None):
#        if holder_type is None:
#            holder_type = SuperLocus.t_feature_holders
#
#        feature_holders = none_to_list(feature_holders)
#
#        for feature in features:
#            self.features[feature.id] = feature
#            for holder in feature_holders:
#                feature.link_to_feature_holder_and_back(holder.id, holder_type)
#
#    def remove_features(self, to_remove):
#        for f_key in to_remove:
#            self.features.pop(f_key)
#
#    def exons(self):
#        return [self.features[x] for x in self.features if self.features[x].type == self.genome.gffkey.exon]
#
#    def coding_info_features(self):
#        return [self.features[x] for x in self.features if self.features[x].type in self.genome.gffkey.coding_info]
#
#    def check_sequence_assumptions(self):
#        pass
#
#    def clean_post_load(self):
#        for key in self.transcripts:
#            self.transcripts[key].super_locus = self
#
#        for key in self.features:
#            self.features[key].super_locus = self

#    def __deepcopy__(self, memodict={}):
#        new = SuperLocus()
#        copy_over = copy.deepcopy(list(new.__dict__.keys()))
#
#        for to_skip in ['slice']:
#            copy_over.pop(copy_over.index(to_skip))
#
#        # copy everything
#        for item in copy_over:
#            new.__setattr__(item, copy.deepcopy(self.__getattribute__(item)))
#
#        new.slice = self.slice
#
#        # fix point back references to point to new
#        for val in new.transcripts.values():
#            val.super_locus = new
#
#        for val in new.features.values():
#            val.super_locus = new
#
#        return new
#
#
class NoTranscriptError(Exception):
    pass


class TransSplicingError(Exception):
    pass


association_generic_holder_to_features = Table('association_generic_holder_to_features', Base.metadata,
    Column('generic_holder_id', Integer, ForeignKey('generic_holders.id')),
    Column('feature_id', Integer, ForeignKey('features.id'))
)

association_transcribeds_to_features = Table('association_transcribeds_to_features', Base.metadata,
    Column('transcribed_id', Integer, ForeignKey('transcribeds.id')),
    Column('feature_id', Integer, ForeignKey('features.id'))
)

association_translateds_to_features = Table('association_translateds_to_features', Base.metadata,
    Column('translated_id', Integer, ForeignKey('translateds.id')),
    Column('feature_id', Integer, ForeignKey('features.id'))
)

association_translateds_to_transcribeds = Table('association_translateds_to_transcribeds', Base.metadata,
    Column('translated_id', Integer, ForeignKey('translateds.id')),
    Column('transcribed_id', Integer, ForeignKey('transcribeds.id'))
)

class GenericHolder(Base):
    __tablename__ = 'generic_holders'

    id = Column(Integer, primary_key=True)
    super_locus_id = Column(Integer, ForeignKey('super_loci.id'))
    super_locus = relationship('SuperLocus', back_populates='generic_holders')

    features = relationship('Feature', secondary=association_generic_holder_to_features,
                            back_populates='generic_holders')
#    holder_type = SuperLocus.t_feature_holders
#
#    def __init__(self):
#        super().__init__()
#        self.spec += [('super_locus', False, SuperLocus, None),
#                      ('features', True, list, None),
#                      ('next_feature_5p', True, str, None),
#                      ('next_feature_3p', True, str, None)]
#
#        self.features = []
#        self.next_feature_5p = None
#        self.next_feature_3p = None
#
#    def add_data(self, super_locus, gff_entry):
#        self.super_locus = super_locus
#        self.id = gff_entry.get_ID()
#        self.type = gff_entry.type
#        self.gff_entry = gff_entry
#
#    def link_to_feature(self, feature_id, at=None):
#        assert feature_id not in self.features, "{} already in features {} for {} {} in loci {}".format(
#            feature_id, self.features, self.type, self.id, self.super_locus.id)
#        if at is None:
#            self.features.append(feature_id)
#        else:
#            self.features.insert(at, feature_id)
#
#    def remove_feature(self, feature_id):
#        at = self.features.index(feature_id)
#        self.features.pop(at)
#        return at
#
#    def short_str(self):
#        return '{}. --> {}'.format(self.id, self.features)
#
#    def delink_features(self, features):
#        for feature in features:
#            holder_type = type(self).holder_type
#            try:
#                self.super_locus.features[feature].de_link_from_feature_holder(self.id, holder_type)
#            except ValueError:
#                feature_fholder_list = self.super_locus.features[feature].__getattribute__(holder_type)
#                raise ValueError("{} not in feature's {}: {}".format(self.id, holder_type,
#                                                                     feature_fholder_list))
#
#    def reconcile_with_slice(self, seqid, start, end):
#        pass  #todo, WAS HERE, make valid (partial) transcript within slice
#
#    def __deepcopy__(self, memodict={}):
#        new = type(self)()
#        copy_over = copy.deepcopy(list(new.__dict__.keys()))
#
#        for to_skip in ['super_locus']:
#            copy_over.pop(copy_over.index(to_skip))
#
#        # copy everything
#        for item in copy_over:
#            new.__setattr__(item, copy.deepcopy(self.__getattribute__(item)))
#
#        new.super_locus = self.super_locus  # fix super_locus
#        if new.gff_entry is None:
#            print('warning, gff_entry none at {}'.format(new.id))
#        return new
#
#    def clone_but_swap_type(self, new_holder_type):
#        # todo, should this actually go into some sort of generic ordered subclass?
#        old_holder_type = type(self).holder_type
#        assert new_holder_type in SuperLocus.types_feature_holders  # todo, stop retyping this
#        assert new_holder_type != old_holder_type  # Shouldn't call swap_type if one has the right type already
#
#        # setup new type with all transferable attributes
#        to_transfer = self.__deepcopy__()  # can copy from
#        holder_type = [x for x in [Transcribed, Translated, FeatureHolder] if x.holder_type == new_holder_type][0]
#        new = holder_type()
#        transferable = set(to_transfer.__dict__.keys())
#        transferable.remove('spec')
#        transferable = transferable.intersection(set(new.__dict__.keys()))
#        for item in transferable:
#            new.__setattr__(item, to_transfer.__getattribute__(item))
#
#        add_to = self.super_locus.__getattribute__(new_holder_type)
#
#
#        # put new in requested place
#        add_to[new.id] = new
#        # swap feature links from old to new
#        for fkey in copy.deepcopy(self.features):
#            feature = self.super_locus.features[fkey]
#            feature.link_to_feature_holder(new.id, new_holder_type)
#        # remove old from
#        return new
#
#    def swap_type(self, new_holder_type):
#        new = self.clone_but_swap_type(new_holder_type)
#        old_holder_type = type(self).holder_type
#        for fkey in copy.deepcopy(self.features):
#            feature = self.super_locus.features[fkey]
#            feature.de_link_from_feature_holder(self.id, old_holder_type)
#            #feature.link_to_feature_holder(new.id, new_holder_type)
#
#        remove_from = self.super_locus.__getattribute__(old_holder_type)
#        remove_from.pop(self.id)
#        return new
#
#    def feature_obj(self, feature_id):
#        return self.super_locus.features[feature_id]
#
#    def feature_objs(self):
#        return [self.feature_obj(x) for x in self.features]
#
#    def replace_id_everywhere(self, new_id):
#        holder_type = type(self).holder_type
#        # new object with new id
#        new = self.__deepcopy__()
#        new.id = new_id
#        # add to super_locus
#        add_to = self.super_locus.__getattribute__(holder_type)
#        add_to[new_id] = new
#
#        old_id = self.id
#        # replace all references to features
#        for fobj in self.feature_objs():
#            at = fobj.de_link_from_feature_holder(old_id, holder_type)  # todo, keep the position!
#            fobj.link_to_feature_holder(new_id, holder_type, at=at)
#        # replace any other by-id references
#        self.replace_subclass_only_ids(new_id)
#        # remove self from superlocus
#        add_to.pop(self.id)
#        # convenience
#        return new
#
#    def replace_subclass_only_ids(self, new_id):
#        pass
#
class Transcribed(Base):
    __tablename__ = 'transcribeds'

    id = Column(Integer, primary_key=True)
    super_locus_id = Column(Integer, ForeignKey('super_loci.id'))
    super_locus = relationship('SuperLocus', back_populates='transcribeds')

    features = relationship('Feature', secondary=association_transcribeds_to_features,
                            back_populates='transcribeds')

    translateds = relationship('Translated', secondary=association_translateds_to_transcribeds,
                            back_populates='transcribeds')
#    holder_type = 'transcripts'
#
#        self.proteins = []  # list of protein IDs, matching subset of keys in self.super_locus.proteins
#
#    def replace_subclass_only_ids(self, new_id):
#        for prot in self.protein_objs():
#            i = prot.transcripts.index(self.id)
#            prot.transcripts.pop(i)
#            prot.transcripts.insert(i, new_id)
#
#    def protein_obj(self, prot_id):
#        return self.super_locus.proteins[prot_id]
#
#    def protein_objs(self):
#        return [self.protein_obj(x) for x in self.proteins]
#
class Translated(Base):
    __tablename__ = 'translateds'

    id = Column(Integer, primary_key=True)
    super_locus_id = Column(Integer, ForeignKey('super_loci.id'))
    super_locus = relationship('SuperLocus', back_populates='translateds')

    features = relationship('Feature', secondary=association_translateds_to_features,
                            back_populates='translateds')

    transcribeds = relationship('Transcribed', secondary=association_translateds_to_transcribeds,
                               back_populates='translateds')
#    holder_type = 'proteins'
#
#        self.transcripts = []  # list of transcript IDs, matching subset of keys in self.super_locus.transcripts
#
#    def replace_subclass_only_ids(self, new_id):
#        for transcript in self.transcript_objs():
#            i = transcript.transcripts.index(self.id)
#            transcript.transcripts.pop(i)
#            transcript.transcripts.insert(i, new_id)
#
#    def transcript_obj(self, prot_id):
#        return self.super_locus.transcripts[prot_id]
#
#    def transcript_objs(self):
#        return [self.transcript_obj(x) for x in self.transcripts]
#
#
class Feature(Base):
    __tablename__ = 'features'
    # basic attributes
    id = Column(Integer, primary_key=True)
    # relations
    # GenericHolder
    # Translated
    # Transcribed
    # SuperLocus
    super_locus_id = Column(Integer, ForeignKey('super_loci.id'))
    super_locus = relationship('SuperLocus', back_populates='features')

    generic_holders = relationship('GenericHolder', secondary=association_generic_holder_to_features,
                                   back_populates='features')

    transcribeds = relationship('Transcribed', secondary=association_transcribeds_to_features,
                                   back_populates='features')

    translateds = relationship('Translated', secondary=association_translateds_to_features,
                                   back_populates='features')
#        self.spec += [('start', True, int, None),
#                      ('end', True, int, None),
#                      ('seqid', True, str, None),
#                      ('strand', True, str, None),
#                      ('score', True, float, None),
#                      ('source', True, str, None),
#                      ('phase', True, int, None),
#
#        self.start = -1
#        self.end = -1
#        self.seqid = ''
#        self.strand = '.'
#        self.phase = None
#        self.score = None
#        self.source = ''
#        self.transcripts = []
#        self.proteins = []
#        self.generic_holders = []
#        self.super_locus = None
#
#    @property
#    def py_start(self):
#        return self.start - 1
#
#    @property
#    def py_end(self):
#        return self.end
#
#    def short_str(self):
#        return '{} is {}: {}-{} on {}. --> {}|{}|{}'.format(self.id, self.type, self.start, self.end, self.seqid,
#                                                            self.transcripts, self.proteins, self.generic_holders)
#
#    def add_data(self, super_locus, gff_entry):
#        gffkey = super_locus.genome.gffkey
#        try:
#            fid = gff_entry.get_ID()
#        except TypeError:
#            fid = None
#            logging.debug('no ID in attr {} in {}, making new unique ID'.format(gff_entry.attribute, super_locus.id))
#        self.gff_entry = gff_entry
#        self.super_locus = super_locus
#        self.id = super_locus.genome.feature_ider.next_unique_id(fid)
#        self.type = gff_entry.type
#        self.start = int(gff_entry.start)
#        self.end = int(gff_entry.end)
#        self.strand = gff_entry.strand
#        self.seqid = self.super_locus.slice.mapper(gff_entry.seqid)
#        if gff_entry.phase == '.':
#            self.phase = None
#        else:
#            self.phase = int(gff_entry.phase)
#        try:
#            self.score = float(gff_entry.score)
#        except ValueError:
#            pass
#        new_transcripts = gff_entry.get_Parent()
#        if not new_transcripts:
#            self.type = gffkey.error
#            logging.warning('{species}:{seqid}:{fid}:{new_id} - No Parents listed'.format(
#                species=super_locus.genome.meta_info.species, seqid=self.seqid, fid=fid, new_id=self.id
#            ))
#        for transcript_id in new_transcripts:
#            new_t_id = transcript_id
#            if new_t_id not in super_locus.generic_holders:
#                if transcript_id == super_locus.id:
#                    # if we just skipped the transcript, and linked to gene, use dummy transcript in between
#                    transcript = super_locus.dummy_transcript()
#                    logging.info(
#                        '{species}:{seqid}:{fid}:{new_id} - Parent gene instead of transcript, recreating'.format(
#                            species=super_locus.genome.meta_info.species, seqid=self.seqid, fid=fid, new_id=self.id
#                        ))
#                    new_t_id = transcript.id
#                else:
#                    self.type = gffkey.error
#                    new_t_id = None
#                    logging.warning(
#                        '{species}:{seqid}:{fid}:{new_id} - Parent: "{parent}" not found at loci'.format(
#                            species=super_locus.genome.meta_info.species, seqid=self.seqid, fid=fid, new_id=self.id,
#                            parent=transcript_id
#                        ))
#            self.link_to_feature_holder_and_back(new_t_id, SuperLocus.t_feature_holders)
#
#    def add_erroneous_data(self, super_locus, gff_entry):
#        self.super_locus = super_locus
#        feature_e = self.clone()
#        feature_e.start = int(gff_entry.start)
#        feature_e.end = int(gff_entry.end)
#        feature_e.strand = gff_entry.strand
#        feature_e.seqid = gff_entry.seqid
#        feature_e.change_to_error()
#        return feature_e
#
#    def change_to_error(self):
#        self.type = self.super_locus.genome.gffkey.error
#
#    def link_to_feature_holder_and_back(self, holder_id, holder_type=None, at=None):
#        #print('link_to_feature_holder_and_back ({}) {} {}'.format(self.short_str(), holder_id, holder_type))
#        if holder_type is None:
#            holder_type = SuperLocus.t_feature_holders
#
#        sl_holders = self.super_locus.__getattribute__(holder_type)
#        holder = sl_holders[holder_id]  # get feature holder
#        holder.link_to_feature(self.id, at)  # link to and from self
#        # get ordered feature holder (transcripts / proteins / feature_holders)
#        self.link_to_feature_holder(holder_id, holder_type)
#
#    def link_to_feature_holder(self, holder_id, holder_type=None, at=None):
#        #print('link_fo_feature_holder ({}) {} {}'.format(self.short_str(), holder_id, holder_type))
#        if holder_type is None:
#            holder_type = SuperLocus.t_feature_holders
#        holder = self.__getattribute__(holder_type)
#        assert holder_type in SuperLocus.types_feature_holders
#        e = "{} already in {}: {}".format(
#            holder_id, holder_type, holder
#        )
#        assert holder_id not in holder, e
#        if at is None:
#            holder.append(holder_id)
#        else:
#            holder.insert(at, holder_id)
#
#    def de_link_from_feature_holder(self, holder_id, holder_type=None):
#        if holder_type is None:
#            holder_type = SuperLocus.t_feature_holders
#        assert holder_type in SuperLocus.types_feature_holders
#        sl_holders = self.super_locus.__getattribute__(holder_type)
#        holder = sl_holders[holder_id]  # get transcript
#
#        at = holder.remove_feature(self.id)  # drop other
#        # and drop from local ordered feature holder set
#        holders = self.__getattribute__(holder_type)
#        holders.pop(holders.index(holder_id))
#        return at
#
#    def fully_overlaps(self, other):
#        should_match = ['type', 'start', 'end', 'seqid', 'strand', 'phase']
#        does_it_match = [self.__getattribute__(x) == other.__getattribute__(x) for x in should_match]
#        same_gene = self.super_locus is other.super_locus
#        out = False
#        if all(does_it_match + [same_gene]):
#            out = True
#        return out
#
#    def is_contained_in(self, other):
#        should_match = ['seqid', 'strand', 'phase']
#        does_it_match = [self.__getattribute__(x) == other.__getattribute__(x) for x in should_match]
#        same_gene = self.super_locus is other.super_locus
#        coordinates_within = self.start >= other.start and self.end <= other.end
#        return all(does_it_match + [coordinates_within, same_gene])
#
#    def reconstruct_exon(self):
#        """creates an exon exactly containing this feature"""
#        exon = self.clone()
#        exon.type = self.super_locus.genome.gffkey.exon
#        return exon
#
#    def clone(self, copy_feature_holders=True):
#        """makes valid, independent clone/copy of this feature"""
#        new = StructuredFeature()
#        copy_over = copy.deepcopy(list(new.__dict__.keys()))
#
#        for to_skip in ['super_locus', 'id', 'transcripts']:
#            copy_over.pop(copy_over.index(to_skip))
#
#        # handle can't just be copied things
#        new.super_locus = self.super_locus
#        new.id = self.super_locus.genome.feature_ider.next_unique_id()
#        if copy_feature_holders:
#            for transcript in self.transcripts:
#                new.link_to_feature_holder_and_back(transcript, SuperLocus.t_transcripts)
#            for protein in self.proteins:
#                new.link_to_feature_holder_and_back(protein, SuperLocus.t_proteins)
#            for ordf in self.generic_holders:
#                new.link_to_feature_holder_and_back(ordf, SuperLocus.t_feature_holders)
#
#        # copy the rest
#        for item in copy_over:
#            new.__setattr__(item, copy.deepcopy(self.__getattribute__(item)))
#        if new.gff_entry is None:
#            raise ValueError('want real gff entry')
#        return new
#
#    def __deepcopy__(self, memodict={}):
#        new = StructuredFeature()
#        copy_over = copy.deepcopy(list(new.__dict__.keys()))
#
#        for to_skip in ['super_locus']:
#            copy_over.pop(copy_over.index(to_skip))
#
#        # copy everything
#        for item in copy_over:
#            new.__setattr__(item, copy.deepcopy(self.__getattribute__(item)))
#
#        new.super_locus = self.super_locus  # fix super_locus
#
#        return new
#
#    def merge(self, other):
#        assert self is not other
#        # move transcript reference from other to self
#        for fset in SuperLocus.types_feature_holders:
#            for ordf in copy.deepcopy(other.__getattribute__(fset)):
#                self.link_to_feature_holder_and_back(ordf, fset)
#                other.de_link_from_transcript(ordf)
#
#    def is_plus_strand(self):
#        if self.strand == '+':
#            return True
#        elif self.strand == '-':
#            return False
#        else:
#            raise ValueError('strand should be +- {}'.format(self.strand))
#
#    def upstream(self):
#        if self.is_plus_strand():
#            return self.start
#        else:
#            return self.end
#
#    def downstream(self):
#        if self.is_plus_strand():
#            return self.end
#        else:
#            return self.start
#
#    # inclusive and from 1 coordinates
#    def upstream_from_interval(self, interval):
#        if self.is_plus_strand():
#            return interval.begin + 1
#        else:
#            return interval.end
#
#    def downstream_from_interval(self, interval):
#        if self.is_plus_strand():
#            return interval.end
#        else:
#            return interval.begin + 1
#
#    def reconcile_with_slice(self, seqid, start, end, status, last_before_slice):
#        #overlap_status = OverlapStatus()
#        #overlap_status.set_status(self, seqid, start, end)
#        #status = overlap_status.status
#        if status == OverlapStatus.contained:
#            pass  # leave it alone
#        elif status == OverlapStatus.no_overlap:
#            # todo, if it is the last feature before the slice (aka, if the next one is contained)
#            if last_before_slice:
#                self.shift_phase(start, end)
#                pass  # todo, change to 1bp status_at (w/ phase if appropriate)
#            pass  # todo, delete (and from transcripts / super_locus)
#        elif status == OverlapStatus.overlaps_upstream:
#            self.shift_phase(start, end)
#            self.crop(start, end)
#        elif status == OverlapStatus.overlaps_downstream:
#            # just crop
#            self.crop(start, end)

#    def length_outside_slice(self, start, end):
#        if self.is_plus_strand():
#            length_outside_slice = start - self.start
#        else:
#            length_outside_slice = self.end - end
#        return length_outside_slice
#
#    def crop(self, start, end):
#        if self.start < start:
#            self.start = start
#        if self.end > end:
#            self.end = end
#
#    def shift_phase(self, start, end):
#        if self.phase is not None:
#            l_out = self.length_outside_slice(start, end)
#            self.phase = (l_out - self.phase) % 3


class OverlapStatus(object):
    contained = 'contained'
    contains = 'contains'
    no_overlap = 'no_overlap'
    overlaps_upstream = 'overlaps_upstream'
    overlaps_downstream = 'overlaps_downstream'
    accepted_stati = (contained, no_overlap, overlaps_upstream, overlaps_downstream)

    def __init__(self):
        self._status = None

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        assert status in OverlapStatus.accepted_stati
        self._status = status

    def set_status(self, feature, seqid, start, end):
        err_str = 'Non handled overlap feature({}, {}, {}) vs slice({}, {}, {})'.format(
                feature.seqid, feature.start, feature.end,
                seqid, start, end
            )
        overlaps_at_start = False
        overlaps_at_end = False
        if feature.seqid != seqid:
            out = OverlapStatus.no_overlap
        elif feature.start >= start and feature.end <= end:
            out = OverlapStatus.contained
        elif feature.start < start and feature.end > end:
            out = OverlapStatus.contains
        elif feature.end < start or feature.start > end:
            out = OverlapStatus.no_overlap
        elif feature.start < start and feature.end >= start:
            overlaps_at_start = True
        elif feature.end > end and feature.start <= end:
            overlaps_at_end = True
        else:
            raise ValueError(err_str)

        plus_strand = feature.is_plus_strand()
        if overlaps_at_start and overlaps_at_end:
            raise ValueError(err_str + ' Overlaps both ends???')  # todo, test this properly and remove run time check

        if (overlaps_at_start and plus_strand) or (overlaps_at_end and not plus_strand):
            out = OverlapStatus.overlaps_upstream
        if (overlaps_at_end and plus_strand) or (overlaps_at_start and not plus_strand):
            out = OverlapStatus.overlaps_downstream
        self.status = out




#### section TranscriptInterpreter, might end up in a separate file later
class TranscriptStatus(object):
    """can hold and manipulate all the info on current status of a transcript"""
    def __init__(self):
        # initializes to intergenic
        self.genic = False  # todo, have some thoughts about how trans-splicing will fit in
        self.in_intron = False
        self.seen_start = False
        self.seen_stop = False
        self.phase = None  # todo, proper tracking / handling

    def saw_tss(self):
        self.genic = True

    def saw_start(self, phase):
        self.genic = True
        self.seen_start = True
        self.phase = phase

    def saw_stop(self):
        self.seen_stop = True
        self.phase = None

    def saw_tts(self):
        self.genic = False

    def splice_open(self):
        self.in_intron = True

    def splice_close(self):
        self.in_intron = False

    def is_5p_utr(self):
        return self.genic and not any([self.in_intron, self.seen_start, self.seen_stop])

    def is_3p_utr(self):
        return all([self.genic, self.seen_stop, self.seen_start]) and not self.in_intron

    def is_coding(self):
        return self.genic and self.seen_start and not any([self.in_intron, self.seen_stop])

    def is_intronic(self):
        return self.in_intron and self.genic

    def is_intergenic(self):
        return not self.genic


class TranscriptInterpBase(object):
    def __init__(self, transcript):
        self.status = TranscriptStatus()
        self.transcript = transcript

    @property
    def super_locus(self):
        return self.transcript.super_locus

    @property
    def gffkey(self):
        return self.transcript.super_locus.genome.gffkey


class TranscriptTrimmer(TranscriptInterpBase):
    """takes pre-cleaned/explicit transcripts and crops to what fits in a slice"""
    def __init__(self, transcript):
        super().__init__(transcript)

    def crop_to_slice(self, seqid, start, end):
        """crops transcript in place"""
        pass

    def transition_5p_to_3p(self):
        pass


class TranscriptInterpreter(TranscriptInterpBase):
    """takes raw/from-gff transcript, and makes totally explicit"""
    def __init__(self, transcript):
        super().__init__(transcript)
        self.clean_features = []  # will hold all the 'fixed' features
        self.transcript = transcript.swap_type('transcripts')
        self.protein_id_key = self._get_raw_protein_ids()
        self.proteins = self._setup_proteins()

    # todo, divvy features to transcript or proteins
    # todo, get_protein_id function (protein_id, Parent of CDS, None to IDMAker)
    # todo, make new protein when ID changes / if we've hit stop codon?

    def new_feature(self, template, **kwargs):
        try:
            new = template.clone()
        except KeyError as e:
            print(self.transcript.super_locus.short_str())
            print(template.short_str())
            raise e
        for key in kwargs:
            new.__setattr__(key, kwargs[key])
        return new

    @staticmethod
    def pick_one_interval(interval_set, target_type=None):
        if target_type is None:
            return interval_set[0]
        else:
            return [x for x in interval_set if x.data.type == target_type][0]

    def _get_protein_id_from_cds(self, cds_feature):
        assert cds_feature.gff_entry.type == self.gffkey.cds, "{} != {}".format(cds_feature.gff_entry.type,
                                                                                self.gffkey.cds)
        # check if anything is labeled as protein_id
        protein_id = cds_feature.gff_entry.attrib_filter(tag='protein_id')
        # failing that, try and get parent ID (presumably transcript, maybe gene)
        if not protein_id:
            protein_id = cds_feature.gff_entry.get_Parent()
        # hopefully take single hit
        if len(protein_id) == 1:
            protein_id = protein_id[0]
            if isinstance(protein_id, gffhelper.GFFAttribute):
                protein_id = protein_id.value
                assert len(protein_id) == 1
                protein_id = protein_id[0]
        # or handle other cases
        elif len(protein_id) == 0:
            protein_id = None
        else:
            raise ValueError('indeterminate single protein id {}'.format(protein_id))
        print(protein_id, type(protein_id), 'pid, type')
        return protein_id

    def _get_raw_protein_ids(self):
        # only meant for use before feature interpretation
        protein_ids = set()
        for fkey in self.transcript.features:
            feature = self.super_locus.features[fkey]
            if feature.type == self.gffkey.cds:
                protein_id = self._get_protein_id_from_cds(feature)
                protein_ids.add(protein_id)
        # map if necessary to unique / not-None IDs
        prot_id_key = {}
        for pkey in protein_ids:
            prot_id_key[pkey] = self.super_locus.genome.protein_ider.next_unique_id(pkey)
        return prot_id_key

    def _setup_proteins(self):
        proteins = {}
        for key in self.protein_id_key:
            val = self.protein_id_key[key]
            print('making protein {} (ori was {})'.format(val, key))
            protein = self.transcript.clone_but_swap_type(SuperLocus.t_proteins)
            protein = protein.replace_id_everywhere(val)
            proteins[val] = protein
        return proteins

    def _mv_coding_features_to_proteins(self):
        print('proteins to move to', self.proteins)
        print('known proteins in sl', self.super_locus.proteins.keys())
        print('proteins in key', self.protein_id_key)
        for protein in self.proteins:
            print(protein, self.super_locus.proteins[protein].features, 'pid, p features')
        # only meant for use after feature interpretation
        for feature in self.clean_features:
            if feature.type in [self.gffkey.status_coding, self.gffkey.stop_codon, self.gffkey.start_codon]:
                assert len(feature.transcripts) == 1
                feature.de_link_from_feature_holder(
                    holder_id=feature.transcripts[0],
                    holder_type=SuperLocus.t_transcripts
                )
                protein_id = self.protein_id_key[self._get_protein_id_from_cds(feature)]
                feature.link_to_feature_holder(protein_id, SuperLocus.t_proteins)

    def is_plus_strand(self):
        features = [self.super_locus.features[f] for f in self.transcript.features]
        seqids = [x.seqid for x in features]
        if not all([x == seqids[0] for x in seqids]):
            raise TransSplicingError("non matching seqids {}, for {}".format(seqids, self.super_locus.id))
        if all([x.strand == '+' for x in features]):
            return True
        elif all([x.strand == '-' for x in features]):
            return False
        else:
            raise TransSplicingError("Mixed strands at {} with {}".format(self.super_locus.id,
                                                                          [(x.seqid, x.strand) for x in features]))

    def interpret_transition(self, ivals_before, ivals_after, plus_strand=True):
        sign = 1
        if not plus_strand:
            sign = -1
        before_types = self.possible_types(ivals_before)
        after_types = self.possible_types(ivals_after)
        # 5' UTR can hit either start codon or splice site
        if self.status.is_5p_utr():
            # start codon
            self.handle_from_5p_utr(ivals_before, ivals_after, before_types, after_types, sign)
        elif self.status.is_coding():
            self.handle_from_coding(ivals_before, ivals_after, before_types, after_types, sign)
        elif self.status.is_3p_utr():
            self.handle_from_3p_utr(ivals_before, ivals_after, before_types, after_types, sign)
        elif self.status.is_intronic():
            self.handle_from_intron()
        elif self.status.is_intergenic():
            self.handle_from_intergenic()
        else:
            raise ValueError('unknown status {}'.format(self.status.__dict__))

    def handle_from_coding(self, ivals_before, ivals_after, before_types, after_types, sign):
        assert self.gffkey.cds in before_types
        # stop codon
        if self.gffkey.three_prime_UTR in after_types:
            self.handle_control_codon(ivals_before, ivals_after, sign, is_start=False)
        # splice site
        elif self.gffkey.cds in after_types:
            self.handle_splice(ivals_before, ivals_after, sign)

    def handle_from_intron(self):
        raise NotImplementedError  # todo later

    def handle_from_3p_utr(self, ivals_before, ivals_after, before_types, after_types, sign):
        assert self.gffkey.three_prime_UTR in before_types
        # the only thing we should encounter is a splice site
        if self.gffkey.three_prime_UTR in after_types:
            self.handle_splice(ivals_before, ivals_after, sign)
        else:
            raise ValueError('wrong feature types after three prime: b: {}, a: {}'.format(
                [x.data.type for x in ivals_before], [x.data.type for x in ivals_after]))

    def handle_from_5p_utr(self, ivals_before, ivals_after, before_types, after_types, sign):
        assert self.gffkey.five_prime_UTR in before_types
        # start codon
        if self.gffkey.cds in after_types:
            self.handle_control_codon(ivals_before, ivals_after, sign, is_start=True)
        # intron
        elif self.gffkey.five_prime_UTR in after_types:
            self.handle_splice(ivals_before, ivals_after, sign)
        else:
            raise ValueError('wrong feature types after five prime: b: {}, a: {}'.format(
                [x.data.type for x in ivals_before], [x.data.type for x in ivals_after]))

    def handle_from_intergenic(self):
        raise NotImplementedError  # todo later

    def is_gap(self, ivals_before, ivals_after, sign):
        """checks for a gap between intervals, and validates it's a positive one on strand of interest"""
        after0 = self.pick_one_interval(ivals_after)
        before0 = self.pick_one_interval(ivals_before)
        before_downstream = before0.data.downstream_from_interval(before0)
        after_upstream = after0.data.upstream_from_interval(after0)
        is_gap = before_downstream + 1 * sign != after_upstream
        if is_gap:
            # if there's a gap, confirm it's in the right direction
            gap_len = (after_upstream - (before_downstream + 1 * sign)) * sign
            assert gap_len > 0, "inverse gap between {} and {} at putative control codon seq {}, gene {}, " \
                                "features {} {}".format(
                before_downstream, after_upstream, after0.data.seqid, self.super_locus.id, before0.data.id,
                after0.data.id
            )
        return is_gap

    def handle_control_codon(self, ivals_before, ivals_after, sign, is_start=True):
        target_after_type = None
        target_before_type = None
        if is_start:
            target_after_type = self.gffkey.cds
        else:
            target_before_type = self.gffkey.cds

        after0 = self.pick_one_interval(ivals_after, target_after_type)
        before0 = self.pick_one_interval(ivals_before, target_before_type)
        # make sure there is no gap
        is_gap = self.is_gap(ivals_before, ivals_after, sign)

        if is_start:
            if is_gap:
                self.handle_splice(ivals_before, ivals_after, sign)

            template = after0.data
            # it better be std phase if it's a start codon
            at = template.upstream_from_interval(after0)
            if template.phase == 0:  # "non-0 phase @ {} in {}".format(template.id, template.super_locus.id)
                start, end = min_max(at, at + 2 * sign)
                start_codon = self.new_feature(template=template, start=start, end=end, type=self.gffkey.start_codon)
                self.status.saw_start(phase=0)
                self.clean_features.append(start_codon)
            else:
                upstream_buffered = before0.data.upstream_from_interval(before0) - sign * self.gffkey.error_buffer
                err_start, err_end = min_max(at - 1 * sign, upstream_buffered)
                feature_e = self.new_feature(template=template, type=self.gffkey.error,
                                             start=err_start, end=err_end, phase=None)
                coding_status = self.new_feature(template=template, type=self.gffkey.status_coding, start=at, end=at)
                self.status.saw_start(template.phase)
                self.clean_features += [feature_e, coding_status]
        else:
            # todo, confirm phase for stop codon
            template = before0.data
            at = template.downstream_from_interval(before0)
            start, end = min_max(at, at - 2 * sign)
            stop_codon = self.new_feature(template=template, start=start, end=end, type=self.gffkey.stop_codon)
            self.status.saw_stop()
            self.clean_features.append(stop_codon)
            if is_gap:
                self.handle_splice(ivals_before, ivals_after, sign)

    def handle_splice(self, ivals_before, ivals_after, sign):
        target_type = None
        if self.status.is_coding():
            target_type = self.gffkey.cds

        before0 = self.pick_one_interval(ivals_before, target_type)
        after0 = self.pick_one_interval(ivals_after, target_type)
        donor_tmplt = before0.data
        acceptor_tmplt = after0.data
        donor_at = donor_tmplt.downstream_from_interval(before0) + (1 * sign)
        acceptor_at = acceptor_tmplt.upstream_from_interval(after0) - (1 * sign)
        # add splice sites if there's a gap
        between_splice_sites = (acceptor_at - donor_at) * sign
        min_intron_len = 3  # todo, maybe get something small but not entirely impossible?
        if between_splice_sites > min_intron_len - 1:  # -1 because the splice sites are _within_ the intron
            donor = self.new_feature(template=donor_tmplt, start=donor_at, end=donor_at, phase=None,
                                     type=self.gffkey.donor_splice_site)
            # todo, check position of DSS/ASS to be consistent with Augustus, hopefully
            acceptor = self.new_feature(template=acceptor_tmplt, start=acceptor_at, end=acceptor_at,
                                        type=self.gffkey.acceptor_splice_site)
            self.clean_features += [donor, acceptor]
        # do nothing if there is just no gap between exons for a techinical / reporting error
        elif between_splice_sites == -1:
            pass
        # everything else is invalid
        else:
            feature_e = before0.data.clone()
            all_coords = [before0.data.start, before0.data.end, after0.data.start, after0.data.end]
            feature_e.start = sorted(all_coords)[0]
            feature_e.end = sorted(all_coords)[-1]
            feature_e.type = self.gffkey.error
            self.clean_features.append(feature_e)

    def interpret_first_pos(self, intervals, plus_strand=True):
        i0 = self.pick_one_interval(intervals)
        at = i0.data.upstream_from_interval(i0)
        possible_types = self.possible_types(intervals)
        if self.gffkey.five_prime_UTR in possible_types:
            # this should indicate we're good to go and have a transcription start site
            tss = self.new_feature(template=i0.data, type=self.gffkey.TSS, start=at, end=at, phase=None)
            self.clean_features.append(tss)
            self.status.saw_tss()
        elif self.gffkey.cds in possible_types:
            # this could be first exon detected or start codon, ultimately, indeterminate
            cds_feature = self.pick_one_interval(intervals, target_type=self.gffkey.cds).data
            coding = self.new_feature(template=cds_feature, type=self.gffkey.status_coding, start=at, end=at)
            self.clean_features.append(coding)
            self.status.saw_start(phase=coding.phase)
            # mask a dummy region up-stream as it's very unclear whether it should be intergenic/intronic/utr
            if plus_strand:
                # unless we're at the start of the sequence
                start_of_sequence = self.get_seq_start(cds_feature.seqid)
                if at != start_of_sequence:
                    feature_e = self.new_feature(template=cds_feature, type=self.gffkey.error,
                                                 start=max(start_of_sequence, at - self.gffkey.error_buffer - 1),
                                                 end=at - 1, phase=None)
                    self.clean_features.insert(0, feature_e)
            else:
                end_of_sequence = self.get_seq_end(cds_feature.seqid)
                if at != end_of_sequence:
                    feature_e = self.new_feature(template=cds_feature, type=self.gffkey.error, start=at + 1,
                                                 end=min(end_of_sequence, at + self.gffkey.error_buffer + 1),
                                                 phase=None)
                    feature_e.type = self.gffkey.error
                    self.clean_features.insert(0, feature_e)
        else:
            raise ValueError("why's this gene not start with 5' utr nor cds? types: {}, interpretations: {}".format(
                [x.data.type for x in intervals], possible_types))

    def interpret_last_pos(self, intervals, plus_strand=True):
        i0 = self.pick_one_interval(intervals)
        at = i0.data.downstream_from_interval(i0)
        possible_types = self.possible_types(intervals)
        if self.gffkey.three_prime_UTR in possible_types:
            # this should be transcription termination site
            tts = self.new_feature(template=i0.data, type=self.gffkey.TTS, start=at, end=at, phase=None)
            self.clean_features.append(tts)
            self.status.saw_tts()
        elif self.gffkey.cds in possible_types:
            # may or may not be stop codon, but will just mark as error (unless at edge of sequence)
            start_of_sequence = self.get_seq_start(i0.data.seqid)
            end_of_sequence = self.get_seq_end(i0.data.seqid)
            if plus_strand:
                if at != end_of_sequence:
                    feature_e = self.new_feature(template=i0.data, type=self.gffkey.error, start=at + 1, phase=None,
                                                 end=min(at + 1 + self.gffkey.error_buffer, end_of_sequence))
                    self.clean_features.append(feature_e)
            else:
                if at != start_of_sequence:
                    feature_e = self.new_feature(template=i0.data, type=self.gffkey.error, end=at - 1, phase=None,
                                                 start=max(start_of_sequence, at - self.gffkey.error_buffer - 1))
                    self.clean_features.append(feature_e)
        else:
            raise ValueError("why's this gene not end with 3' utr/exon nor cds? types: {}, interpretations: {}".format(
                [x.data.type for x in intervals], possible_types)
            )

    def intervals_5to3(self, plus_strand=False):
        interval_sets = list(self.organize_and_split_features())
        if not plus_strand:
            interval_sets.reverse()
        return interval_sets

    def decode_raw_features(self):
        plus_strand = self.is_plus_strand()
        interval_sets = self.intervals_5to3(plus_strand)
        self.interpret_first_pos(interval_sets[0], plus_strand)
        for i in range(len(interval_sets) - 1):
            ivals_before = interval_sets[i]
            ivals_after = interval_sets[i + 1]
            self.interpret_transition(ivals_before, ivals_after, plus_strand)

        self.interpret_last_pos(intervals=interval_sets[-1])
        self._mv_coding_features_to_proteins()

    def possible_types(self, intervals):
        # shortcuts
        cds = self.gffkey.cds
        five_prime = self.gffkey.five_prime_UTR
        exon = self.gffkey.exon
        three_prime = self.gffkey.three_prime_UTR

        # what we see
        observed_types = [x.data.type for x in intervals]
        set_o_types = set(observed_types)
        # check length
        if len(intervals) not in [1, 2]:
            raise ValueError('check interpretation by hand for transcript start with {}, {}'.format(
                '\n'.join([ival.data.short_str() for ival in intervals]), observed_types
            ))
        # interpret type combination
        if set_o_types == {exon, five_prime} or set_o_types == {five_prime}:
            out = [five_prime]
        elif set_o_types == {exon, three_prime} or set_o_types == {three_prime}:
            out = [three_prime]
        elif set_o_types == {exon}:
            out = [five_prime, three_prime]
        elif set_o_types == {cds, exon} or set_o_types == {cds}:
            out = [cds]
        else:
            raise ValueError('check interpretation of combination for transcript start with {}, {}'.format(
                intervals, observed_types
            ))
        return out

    def organize_and_split_features(self):
        # todo, handle non-single seqid loci
        tree = intervaltree.IntervalTree()
        features = [self.super_locus.features[f] for f in self.transcript.features]
        for f in features:
            tree[f.py_start:f.py_end] = f
        tree.split_overlaps()
        # todo, minus strand
        intervals = iter(sorted(tree))
        out = [next(intervals)]
        for interval in intervals:
            if out[-1].begin == interval.begin:
                out.append(interval)
            else:
                yield out
                out = [interval]
        yield out

    def get_seq_end(self, seqid):
        return self.super_locus.slice.seq_info[seqid].end

    def get_seq_start(self, seqid):
        return self.super_locus.slice.seq_info[seqid].start


def min_max(x, y):
    return min(x, y), max(x, y)


def none_to_list(x):
    if x is None:
        return []
    else:
        assert isinstance(x, list)
        return x


def upstream(x, y, sign):
    if (y - x) * sign >= 0:
        return x
    else:
        return y


def downstream(x, y, sign):
    if (x - y) * sign >= 0:
        return x
    else:
        return y
