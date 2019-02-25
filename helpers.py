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
