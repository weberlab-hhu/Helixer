import itertools
import copy

from geenuff.base.helpers import reverse_complement


def min_max(x, y):
    return min(x, y), max(x, y)


class MerCounter(object):

    def __init__(self, k):
        self.k = k
        self.amb = "N" * k
        self.counts = {}
        # calculate all possible mers of this length, and set counter to 0
        for mer in itertools.product('ATCG', repeat=k):
            mer = ''.join(mer)
            self.counts[mer] = 0
        self.counts[self.amb] = 0

    def export(self):
        out = copy.deepcopy(self.counts)
        for key in self.counts:
            if key != self.amb:
                # collapse to cannonical kmers
                rc_key = ''.join(reverse_complement(key))
                if key > rc_key:
                    out[rc_key] += out[key]
                    out.pop(key)
        return out

    def add_sequence(self, sequence):
        # convenience fn for testing, but for efficiency use one loop through
        # sequence for all MerCounters at once
        for i in range(0, len(sequence) - (self.k - 1)):
            self.add_count(sequence[i:i+self.k])

    def add_count(self, mer):
        try:
            self.counts[mer] += 1
        except KeyError:
            self.counts[self.amb] += 1
