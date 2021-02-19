# some helpers for handling / sorting / or checking sort of our h5 files
def mk_seqonly_keys(h5):
    return [a + b for a, b in zip(h5['data/species'],
                                  h5['data/seqids'])]


def mk_keys(h5, flip=False):
    first_idx = 0
    second_idx = 1
    if flip:
        first_idx, second_idx = second_idx, first_idx
    return zip(h5['data/species'],
               h5['data/seqids'],
               h5['data/start_ends'][:, first_idx],
               h5['data/start_ends'][:, second_idx])


def get_sp_seq_ranges(h5):
    # dict with {sp: {"start: N,
    #                 "end": N,
    #                 "seqids": {seqid: [start, end], seqid2: [start2, end2], ...}},
    #            sp2: {...},
    #            ...}
    out = {}

    gen = zip(range(h5['data/y'].shape[0]), h5['data/species'], h5['data/seqids'])

    i, prev_sp, prev_seqid = next(gen)
    out[prev_sp] = {"start": i,  # 0
                    "seqids": {prev_seqid: [i]}}
    for i, sp, seqid in gen:
        if sp != prev_sp:
            # end previous sp and seqid
            out[prev_sp]["end"] = i
            out[prev_sp]["seqids"][prev_seqid].append(i)
            # open new
            out[sp] = {"start": i,
                       "seqids": {seqid: [i]}}
        elif seqid != prev_seqid:
            # end previous seqid
            out[sp]["seqids"][prev_seqid].append(i)
            # open new
            out[sp]["seqids"][seqid] = [i]
        prev_sp, prev_seqid = sp, seqid
    # end final sp/seqid
    out[prev_sp]["end"] = i + 1
    out[prev_sp]["seqids"][prev_seqid].append(i + 1)
    return out


def file_stem(path):
    """Returns the file name without extension"""
    import os
    return os.path.basename(path).split('.')[0]

