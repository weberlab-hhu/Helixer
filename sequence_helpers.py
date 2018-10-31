from enum import Enum, unique


def enum_reverse_complement(seq):
    rc_seq = ''
    for base in reversed(seq):
        islower = base.islower()
        new_bp = rc_bp(base.upper())
        new_bp = new_bp.name
        if islower:
            new_bp = new_bp.lower()
        rc_seq += new_bp
    return rc_seq


@unique
class rc_bp(Enum):
    # the base pairs
    A = 'T'
    T = 'A'
    C = 'G'
    G = 'C'
    # ambiguity options
    M = 'K'
    K = 'M'
    R = 'Y'
    Y = 'R'
    W = 'W'
    S = 'S'
    V = 'B'
    B = 'V'
    H = 'D'
    D = 'H'
    N = 'N'


def reverse_complement(seq):
    fw = "ACGTMRWSYKVHDBN"
    rv = "TGCAKYWSRMBDHVN"
    fw += fw.lower()
    rv += rv.lower()
    key = {}
    for f, r in zip(fw, rv):
        key[f] = r
    rc_seq = ''
    for base in reversed(seq):
        try:
            rc_seq += key[base]
        except KeyError as e:
            raise KeyError('{} caused by non DNA character {}'.format(e, base))

    return rc_seq
