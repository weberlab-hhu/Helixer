from geenuff import orm
from sqlalchemy import Column, Enum, Integer, ForeignKey
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


