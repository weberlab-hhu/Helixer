from geenuff import orm
from sqlalchemy import Column, Enum, Integer, ForeignKey, UniqueConstraint, CheckConstraint, String
from sqlalchemy.orm import relationship
import enum


class ProcessingSet(enum.Enum):
    train = 'train'
    dev = 'dev'
    test = 'test'


# todo, mv to coordinate somehow
class CoordinateSet(orm.Base):
    __tablename__ = "coordinate_set"

    id = Column(Integer, ForeignKey('coordinate.id'), primary_key=True)
    processing_set = Column(Enum(ProcessingSet), nullable=False)

    coordinate = relationship('orm.Coordinate')

    def __repr__(self):
        return '{}: matching {} in set {}'.format(type(self), self.coordinate, self.processing_set)


class Mer(orm.Base):
    __tablename__ = "mer"

    id = Column(Integer, primary_key=True)
    coordinate_id = Column(Integer, ForeignKey('coordinate.id'), nullable=False)

    mer_sequence = Column(String, nullable=False)
    count = Column(Integer)
    length = Column(Integer)

    coordinate = relationship('orm.Coordinate')

    UniqueConstraint('mer_sequence', 'coordinate_id', name='unique_kmer_per_coord')

    __table_args__ = (
        CheckConstraint('length(mer_sequence) > 0', name='check_string_gt_0'),
        CheckConstraint('count >= 0', name='check_count_gt_0'),
        CheckConstraint('length >= 1', name='check_length_gt_1'),
    )
