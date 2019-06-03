import geenuff

from helixerprep.core.orm import Mer
from helixerprep.core.helpers import MerCounter


class CoordinateHandler(geenuff.handlers.CoordinateHandlerBase):
    def count_mers(self, min_k, max_k):
        mer_counters = []
        # setup all counters
        for k in range(min_k, max_k + 1):
            mer_counters.append(MerCounter(k))

        # count all 'mers
        for i in range(len(self.data.sequence)):
            for k in range(min_k, max_k + 1):
                if i + 1 >= k:
                    substr = self.data.sequence[i-(k-1):i+1]
                    mer_counters[k - 1].add_count(substr)
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
