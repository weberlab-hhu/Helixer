import itertools
import geenuff
import copy


class IDMaker(object):
    def __init__(self, prefix='', width=6):
        self._counter = 0
        self.prefix = prefix
        self._seen = set()
        self._width = width

    @property
    def seen(self):
        return self._seen

    def next_unique_id(self, suggestion=None):
        if suggestion is not None:
            suggestion = str(suggestion)
            if suggestion not in self._seen:
                self._seen.add(suggestion)
                return suggestion
        # you should only get here if a) there was no suggestion or b) it was not unique
        return self._new_id()

    def _new_id(self):
        new_id = self._fmt_id()
        self._seen.add(new_id)
        self._counter += 1
        return new_id

    def _fmt_id(self):
        to_format = '{}{:0' + str(self._width) + '}'
        return to_format.format(self.prefix, self._counter)


def min_max(x, y):
    return min(x, y), max(x, y)


class MerCounter(object):

    def __init__(self, k):
        self.k = k
        self.counts = {}
        # calculate all possible mers of this length, and set counter to 0
        for mer in itertools.product('ATCG', repeat=k):
            mer = tuple(mer)
            self.counts[mer] = 0
        self.counts[self.amb] = 0
        self.current = []
        self.amb = "N" * k

    def export(self):
        out = copy.deepcopy(self.counts)
        for key in self.counts:
            if key != self.amb:
                # collapse to cannonical kmers
                rc_key = geenuff.helpers.reverse_complement(key)
                if key != min(key, rc_key):
                    out[rc_key] += out[key]
                    out.pop(key)
        return out

    def add_sequence(self, sequence):
        # deprecating in favor of processing all mer lengths at once
        for bp in sequence:
            self.add_bp(bp)

    def add_bp(self, bp):
        self.current.append(bp)
        length = len(self.current)
        # do nothing if we haven't yet reached 'mer length
        if length < self.k:
            pass
        # simply count when we reach 'mer length
        elif length == self.k:
            self.count_current()
        # drop oldest bp if we are over length, then count
        elif length == self.k + 1:
            self.current.pop(0)
            self.count_current()
        else:
            raise ValueError("length of self.current should always be within 0 - (k+ 1), found {}".format(length))

    def count_current(self):
        try:
            self.counts[tuple(self.current)] += 1
        except KeyError:
            self.counts[self.amb] += 1
