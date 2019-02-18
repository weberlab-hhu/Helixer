import logging


class NonMatchableIDs(Exception):
    pass


class Mapper(object):
    def __call__(self, key, *args, **kwargs):
        return key


class CheckMapper(Mapper):
    def __init__(self, keys):
        self.keys = set(keys)

    def __call__(self, key, *args, **kwargs):
        if key in self.keys:
            return key
        else:
            raise KeyError("{} not in known keys {}".format(key, self.keys))


class DictMapper(Mapper):
    def __init__(self, key_vals):
        if not isinstance(key_vals, dict):
            raise ValueError('key_vals must be a dict instance')
        self.key_vals = key_vals

    def __call__(self, key, *args, **kwargs):
        return self.key_vals[key]


def make_key_mapper(known_keys, other_keys):
    to_match_keys = set(known_keys)
    other_keys = set(other_keys)
    # if we have a perfect match
    if to_match_keys == other_keys or to_match_keys.issuperset(other_keys):
        return CheckMapper(to_match_keys)
    elif to_match_keys.issubset(other_keys):
        raise NonMatchableIDs('known is a subset of other keys')

    logging.info("attempting to match up, non-identical IDs")
    oth2known = {}
    # for each tree key, does it have exactly one match?
    for key in other_keys:
        matches = [x for x in to_match_keys if key in x]
        if len(matches) == 1:
            # setup dict[old_key] = new_key
            oth2known[key] = matches[0]
        elif len(matches) == 0:
            # pretending no match is ok, (a warning will be logged by are_keys_compatible) but seriously NCBI?
            logging.debug('no matches found for {} in known keys, e.g. {}'.format(
                key, list(known_keys)[:min(4, len(known_keys))]
            ))
        else:
            raise NonMatchableIDs('could not identify unique match for {}, but instead got {}'.format(key,
                                                                                                      matches))
    if not oth2known:  # bc we can still get an empty dict for complete unrelated gibberish at this point
        raise NonMatchableIDs('could not uniquely match up known: {} and other: {} keys'.format(known_keys,
                                                                                                other_keys))

    # check we matched trees to _unique_ fasta keys
    if len(oth2known.values()) != len(set(oth2known.values())):
        raise NonMatchableIDs('could not uniquely match up known: {} and other: {} keys'.format(known_keys,
                                                                                                other_keys))
    return DictMapper(oth2known)


def two_way_key_match(known_keys, other_keys):
    forward = True
    try:
        mapper = make_key_mapper(known_keys, other_keys)
    except NonMatchableIDs as e:
        try:
            mapper = make_key_mapper(other_keys, known_keys)
            forward = False
        except NonMatchableIDs:
            raise e
    return mapper, forward


def get_seqids_from_gff(gfffile):
    seqids = set()
    with open(gfffile) as f:
        for line in f:
            if not line.startswith('#'):
                seqids.add(line.split('\t')[0])
    return seqids


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


def as_py_start(start):
    return start - 1


def as_py_end(end):
    return end


def as_bio_start(py_start):
    return py_start + 1


def as_bio_end(py_end):
    return py_end

def min_max(x, y):
    return min(x, y), max(x, y)
