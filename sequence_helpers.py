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


def chunk_str(string, length):
    for i in range(0, len(string), length):
        yield string[i:(i+length)]