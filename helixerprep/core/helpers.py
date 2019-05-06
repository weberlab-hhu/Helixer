import itertools
import copy


def min_max(x, y):
    return min(x, y), max(x, y)


class MerCounter(object):

    def __init__(self, k):
        self.k = k
        self.amb = "N" * k
        self.counts = {}
        # calculate all possible mers of this length, and set counter to 0
        for mer in itertools.product('ATCG', repeat=k):
            mer = tuple(mer)
            self.counts[mer] = 0
        self.counts[self.amb] = 0
        self.current = []

    def export(self):
        pre_out = copy.deepcopy(self.counts)
        for key in self.counts:
            if key != self.amb:
                # collapse to cannonical kmers
                rc_key = tuple(reverse_complement(key))
                if key != min(key, rc_key):
                    pre_out[rc_key] += pre_out[key]
                    pre_out.pop(key)

        # change tuples keys to strings
        out = {}
        for key in pre_out:
            out[''.join(key)] = pre_out[key]

        return out

    def add_sequence(self, sequence):
        # convenience fn for testing, but for efficiency use one loop through sequence for all MerCounters at once
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


def mk_rc_key():
    fw = "ACGTMRWSYKVHDBN"
    rv = "TGCAKYWSRMBDHVN"
    key = {}
    for f, r in zip(fw, rv):
        key[f] = r
    return key


# so one doesn't recalculate it for every call of revers_complement
REV_COMPLEMENT_KEY = mk_rc_key()


def reverse_complement(seq):
    key = REV_COMPLEMENT_KEY
    rc_seq = []
    for base in reversed(seq):
        try:
            rc_seq.append(key[base])
        except KeyError as e:
            raise KeyError('{} caused by non DNA character {}'.format(e, base))
    return rc_seq
