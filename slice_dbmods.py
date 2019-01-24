import annotations_orm
from sqlalchemy import Column, Enum
import enum


class ProcessingSet(enum.Enum):
    train = 'train'
    dev = 'dev'
    test = 'test'


class SequenceInfoSets(annotations_orm.SequenceInfo):
    __tablename__ = "sequence_info_sets"

    processing_set = Column(Enum(ProcessingSet))
