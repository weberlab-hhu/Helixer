import random


class CoordinateGenerator(object):
    def __init__(self, train_size, dev_size, max_len, user_seed) :
        self.train_size = train_size
        self.dev_size = dev_size
        self.max_len = max_len
        self.user_seed = user_seed

    def divvy_coordinates(self, length, seq_hash):
        random.seed(seq_hash + self.user_seed)
        stepper = Stepper(length, self.max_len)
        for begin, end in stepper.step_to_end():
            yield begin, end, choose_set(self.train_size, self.dev_size)


class Stepper(object):
    def __init__(self, end, by):
        self.at = 0
        self.end = end
        self.by = by

    def step(self):
        prev = self.at
        # fits twice or more, just step
        if prev + self.by * 2 <= self.end:
            new = prev + self.by
        # fits less than twice, take half way point (to avoid an end of just a few bp)
        elif prev + self.by < self.end:
            new = prev + (self.end - prev) // 2
        # doesn't fit at all
        else:
            new = self.end
        self.at = new
        return prev, new

    def step_to_end(self):
        while self.at < self.end:
            yield self.step()


def choose_set(train_size, dev_size):
    r = random.random()
    if r < train_size:
        return 'train'
    elif r < train_size + dev_size:
        return 'dev'
    else:
        return 'test'
