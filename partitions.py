from __future__ import division
import random


class CoordinateGenerator(object):
    def __init__(self, user_seed, max_len=2000000):
        self.user_seed = user_seed
        self.max_len = max_len

    def divvy_coordinates(self, seq_seed, length):
        random.seed(seq_seed + self.user_seed)
        stepper = Stepper(length, self.max_len)
        for begin, end in stepper.step_to_end():
            yield begin, end, choose_set()


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


def choose_set(train=0.8, dev=0.1):
    r = random.random()
    if r < train:
        return 'train'
    elif r < train + dev:
        return 'dev'
    else:
        return 'test'
