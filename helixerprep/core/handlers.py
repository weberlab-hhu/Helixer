import geenuff
from geenuff.base.orm import Coordinate, Genome
from helixerprep.core.orm import Mer
from helixerprep.core.helpers import MerCounter


class CoordinateHandler(geenuff.handlers.CoordinateHandlerBase):
    def coordinate_set(self, session):
        return session.query(CoordinateSet).filter(CoordinateSet.id == self.data.id).one_or_none()

    def get_processing_set(self, session):
        si_set_obj = self.coordinate_set(session)
        if si_set_obj is None:
            return None
        else:
            return si_set_obj.processing_set.value

    def set_processing_set(self, session, processing_set, create=False):
        current = self.coordinate_set(session)
        if current is None:
            if create:
                current = CoordinateSet(coordinate=self.data, processing_set=processing_set)
                session.add(current)
            else:
                raise CoordinateHandler.CoordinateSetNotExisting()
        else:
            current.processing_set = ProcessingSet[processing_set]
        return current

    def count_mers(self, min_k, max_k):
        mer_counters = []
        # setup all counters
        for k in range(min_k, max_k + 1):
            mer_counters.append(MerCounter(k))

        # count all 'mers
        for bp in self.data.sequence.upper():
            for mer_counter in mer_counters:
                mer_counter.add_bp(bp)

        return mer_counters

    def add_mer_counts_to_db(self, min_k, max_k, session):
        mer_counters = self.count_mers(min_k, max_k)
        # convert to canonical and setup db entries
        for mer_counter in mer_counters:
            for mer_sequence, count in mer_counter.export().items():
                mer = Mer(coordinate=self.data,
                          mer_sequence=mer_sequence,
                          count=count,
                          length=mer_counter.k)
                session.add(mer)
        session.commit()

    class CoordinateSetNotExisting(Exception):
        pass
