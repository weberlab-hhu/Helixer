# some helpers for handling / sorting / or checking sort of our h5 files
def mk_seqonly_keys(h5):
    return [a + b for a, b in zip(h5['data/species'],
                                  h5['data/seqids'])]


def mk_keys(h5):
    return zip(h5['data/species'],
               h5['data/seqids'],
               h5['data/start_ends'][:, 0],
               h5['data/start_ends'][:, 1])
