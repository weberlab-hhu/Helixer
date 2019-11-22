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
