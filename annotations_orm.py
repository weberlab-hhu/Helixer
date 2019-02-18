import type_enums

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Table, Column, Integer, ForeignKey, String, Enum, CheckConstraint, Boolean, Float
from sqlalchemy.orm import relationship

# setup classes for data holding
Base = declarative_base()


class AnnotatedGenome(Base):
    __tablename__ = 'annotated_genomes'

    # data
    id = Column(Integer, primary_key=True)
    species = Column(String)
    accession = Column(String)
    version = Column(String)
    acquired_from = Column(String)
    sequence_infos = relationship("SequenceInfo", back_populates="annotated_genome")


class SequenceInfo(Base):
    __tablename__ = "sequence_infos"

    id = Column(Integer, primary_key=True)
    # relations
    annotated_genome_id = Column(Integer, ForeignKey('annotated_genomes.id'), nullable=False)
    annotated_genome = relationship('AnnotatedGenome', back_populates="sequence_infos")
    coordinates = relationship('Coordinates', back_populates="sequence_info")

    def __repr__(self):
        return '<{}: {}, in {}, with {}>'.format(type(self), self.id, self.annotated_genome, self.coordinates)


class Coordinates(Base):
    __tablename__ = 'coordinates'

    id = Column(Integer, primary_key=True)
    start = Column(Integer, nullable=False)
    end = Column(Integer, nullable=False)
    seqid = Column(String, nullable=False)
    sequence_info_id = Column(Integer, ForeignKey('sequence_infos.id'))
    sequence_info = relationship('SequenceInfo', back_populates='coordinates')

    features = relationship('Feature', back_populates='coordinates')

    __table_args__ = (
        CheckConstraint(start >= 0, name='check_start_1plus'),
        CheckConstraint(end > start, name='check_end_gr_start'),
        {})

    def __repr__(self):
        return '<Coordinate {}, {}:{}-{}>'.format(self.id, self.seqid, self.start, self.end)


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

    id = Column(Integer, primary_key=True)
    given_id = Column(String)
    type = Column(Enum(type_enums.SuperLocusAll))
    # things SuperLocus can have a lot of
    aliases = relationship('SuperLocusAliases', back_populates='super_locus')
    features = relationship('Feature', back_populates='super_locus')
    transcribeds = relationship('Transcribed', back_populates='super_locus')
    transcribed_pieces = relationship('TranscribedPiece', back_populates='super_locus')
    translateds = relationship('Translated', back_populates='super_locus')


association_transcribeds_to_features = Table('association_transcribeds_to_features', Base.metadata,  # todo, rename
    Column('transcribed_piece_id', Integer, ForeignKey('transcribed_pieces.id')),
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


class Transcribed(Base):
    __tablename__ = 'transcribeds'

    id = Column(Integer, primary_key=True)
    given_id = Column(String)

    type = Column(Enum(type_enums.TranscriptLevelAll))

    super_locus_id = Column(Integer, ForeignKey('super_loci.id'))
    super_locus = relationship('SuperLocus', back_populates='transcribeds')

    translateds = relationship('Translated', secondary=association_translateds_to_transcribeds,
                               back_populates='transcribeds')

    transcribed_pieces = relationship('TranscribedPiece', back_populates='transcribed')

    pairs = relationship('UpDownPair', back_populates='transcribed')


class TranscribedPiece(Base):
    __tablename__ = 'transcribed_pieces'

    id = Column(Integer, primary_key=True)
    given_id = Column(String)

    super_locus_id = Column(Integer, ForeignKey('super_loci.id'))
    super_locus = relationship('SuperLocus', back_populates='transcribed_pieces')

    transcribed_id = Column(Integer, ForeignKey('transcribeds.id'))
    transcribed = relationship('Transcribed', back_populates='transcribed_pieces')

    features = relationship('Feature', secondary=association_transcribeds_to_features,
                            back_populates='transcribed_pieces')

    def __repr__(self):
        return "<TranscribedPiece, {}: with features {}>".format(self.id, [(x.id, x.start, x.given_id) for x in self.features])


class Translated(Base):
    __tablename__ = 'translateds'

    id = Column(Integer, primary_key=True)
    given_id = Column(String)
    # type can only be 'protein' so far as I know..., so skipping
    super_locus_id = Column(Integer, ForeignKey('super_loci.id'))
    super_locus = relationship('SuperLocus', back_populates='translateds')

    features = relationship('Feature', secondary=association_translateds_to_features,
                            back_populates='translateds')

    transcribeds = relationship('Transcribed', secondary=association_translateds_to_transcribeds,
                                back_populates='translateds')


class Feature(Base):
    __tablename__ = 'features'
    # basic attributes
    id = Column(Integer, primary_key=True)
    given_id = Column(String)

    type = Column(Enum(type_enums.OnSequence))
    bearing = Column(Enum(type_enums.Bearings))
    #seqid = Column(String)
    coordinate_id = Column(Integer, ForeignKey('coordinates.id'))  # any piece of coordinates always has just one seqid
    coordinates = relationship('Coordinates', back_populates='features')
    start = Column(Integer)
    end = Column(Integer)
    is_plus_strand = Column(Boolean)
    score = Column(Float)
    source = Column(String)
    phase = Column(Integer)

    # for differentiating from subclass entries
    subtype = Column(String(20))
    # relations
    super_locus_id = Column(Integer, ForeignKey('super_loci.id'))
    super_locus = relationship('SuperLocus', back_populates='features')

    transcribed_pieces = relationship('TranscribedPiece', secondary=association_transcribeds_to_features,
                                      back_populates='features')

    translateds = relationship('Translated', secondary=association_translateds_to_features,
                               back_populates='features')

    __table_args__ = (
        CheckConstraint(start >= -1, name='check_start_1plus'),
        CheckConstraint(phase >= 0, name='check_phase_not_negative'),
        CheckConstraint(phase < 3, name='check_phase_less_three'),
        {})

    __mapper_args__ = {
        'polymorphic_on': subtype,
        'polymorphic_identity': 'general'
    }

    def __repr__(self):
        s = '<{py_type}, {pk}: {givenid} of type: {type} ({bearing}) from {start}-{end} on {coor}, is_plus: {plus}, ' \
            'phase: {phase}>'.format(
                pk=self.id, bearing=self.bearing,
                type=self.type, start=self.start, end=self.end, coor=self.coordinates, plus=self.is_plus_strand,
                phase=self.phase, givenid=self.given_id, py_type=type(self)
            )
        return s

    def cmp_key(self):  # todo, pos_cmp & full_cmp
        return self.coordinates.seqid, self.is_plus_strand, self.start, self.type

    def pos_cmp_key(self):
        return self.coordinates.seqid, self.is_plus_strand, self.start


class DownstreamFeature(Feature):
    __tablename__ = 'downstream_features'

    id = Column(Integer, ForeignKey('features.id'), primary_key=True)
    pairs = relationship('UpDownPair', back_populates="downstream")

    __mapper_args__ = {
        'polymorphic_identity': 'downstream'
    }


class UpstreamFeature(Feature):
    __tablename__ = 'upstream_features'

    id = Column(Integer, ForeignKey('features.id'), primary_key=True)
    pairs = relationship('UpDownPair', back_populates='upstream')

    __mapper_args__ = {
        'polymorphic_identity': 'upstream'
    }


# todo, association table DownstreamFeature <-> UpstreamFeature + fk -> Transcribed
class UpDownPair(Base):
    __tablename__ = 'up_down_pairs'

    id = Column(Integer, primary_key=True)
    upstream_id = Column(Integer, ForeignKey('upstream_features.id'))
    upstream = relationship('UpstreamFeature', uselist=False, back_populates="pairs")

    downstream_id = Column(Integer, ForeignKey('downstream_features.id'))
    downstream = relationship('DownstreamFeature', uselist=False, back_populates='pairs')

    transcribed_id = Column(Integer, ForeignKey('transcribeds.id'))
    transcribed = relationship('Transcribed', uselist=False, back_populates='pairs')

    def pos_cmp_key(self):
        upstream_key = (None, None, None, None)
        downstream_key = (None, None, None, None)
        if self.upstream is not None:
            upstream_key = self.upstream.pos_cmp_key()
        if self.downstream is not None:
            downstream_key = self.downstream.pos_cmp_key()
        return upstream_key + downstream_key
