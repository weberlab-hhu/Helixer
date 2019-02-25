from geenuff import orm
from sqlalchemy import Column, Enum, Integer, ForeignKey
from sqlalchemy.orm import relationship
import enum


class ProcessingSet(enum.Enum):
    train = 'train'
    dev = 'dev'
    test = 'test'


class SequenceInfoSets(orm.Base):
    __tablename__ = "sequence_info_sets"

    id = Column(Integer, ForeignKey('sequence_infos.id'), primary_key=True)
    sequence_info = relationship('orm.SequenceInfo')
    processing_set = Column(Enum(ProcessingSet), nullable=False)

    def __repr__(self):
        return '{}: matching {} in set {}'.format(type(self), self.sequence_info, self.processing_set)


