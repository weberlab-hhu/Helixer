import sequences
import structure
import annotations


### structure ###
# testing: add_paired_dictionaries
def test_add_to_empty_dictionary():
    d1 = {'a': 1}
    d2 = {}
    d1_2 = structure.add_paired_dictionaries(d1, d2)
    d2_1 = structure.add_paired_dictionaries(d2, d1)
    assert d1 == d1_2
    assert d1 == d2_1


def test_add_nested_dictionaries():
    d1 = {'a': {'b': 1,
                'a': {'b': 10}},
          'b': 100}
    d2 = {'a': {'b': 1,
                'a': {'b': 20}},
          'b': 300}
    dsum = {'a': {'b': 2,
                  'a': {'b': 30}},
            'b': 400}
    d1_2 = structure.add_paired_dictionaries(d1, d2)
    d2_1 = structure.add_paired_dictionaries(d2, d1)
    print('d1_2', d1_2)
    print('d2_1', d2_1)
    print('dsum', dsum)
    assert dsum == d1_2
    assert dsum == d2_1


# testing: class GenericData
class GDataTesting(structure.GenericData):
    def __init__(self):
        super().__init__()
        self.spec += [('expect', False, dict, None)]
        self.expect = {}  # this is what we expect the jsonable to be, assuming we don't change the attributes


class SimpleGData(GDataTesting):
    def __init__(self):
        super().__init__()
        # attribute name, exported_to_json, expected_inner_type, data_structure
        self.spec += [('some_ints', True, int, list),
                      ('a_string', True, str, None)]
        self.some_ints = [1, 2, 3]
        self.a_string = 'abc'
        self.expect = {'some_ints': [1, 2, 3], 'a_string': 'abc'}


class HoldsGdata(GDataTesting):
    def __init__(self):
        super().__init__()
        self.spec += [('a_gdata', True, SimpleGData, None),
                      ('list_gdata', True, SimpleGData, list),
                      ('dict_gdata', True, SimpleGData, dict)]
        self.a_gdata = SimpleGData()
        self.list_gdata = [SimpleGData()]
        self.dict_gdata = {'x': SimpleGData()}
        sgd_expect = self.a_gdata.expect
        self.expect = {'a_gdata': sgd_expect,
                       'list_gdata': [sgd_expect],
                       'dict_gdata': {'x': sgd_expect}}


def test_to_json_4_simplest_of_data():
    x = SimpleGData()
    assert x.to_jsonable() == x.expect


def test_to_json_4_recursive_generic_datas():
    x = HoldsGdata()
    print(x.to_jsonable())
    assert x.to_jsonable() == x.expect


def test_from_json_gdata():
    # make sure we get the same after export, as export->import->export
    x = SimpleGData()
    xjson = x.to_jsonable()
    xjson['a_string'] = 'new_string'
    y = SimpleGData()
    y.load_jsonable(xjson)
    assert y.to_jsonable() == xjson
    # check as above but for more complicated data holder
    holds = HoldsGdata()
    holds.a_gdata = y
    holdsjson = holds.to_jsonable()
    assert holdsjson["a_gdata"]["a_string"] == 'new_string'
    print(holdsjson)
    yholds = HoldsGdata()
    yholds.load_jsonable(holdsjson)
    assert yholds.to_jsonable() == holdsjson


### sequences ###
# testing: counting kmers
def test_gen_mers():
    seq = 'atatat'
    # expect (at x 3) and  (ta x 2)
    mers = list(sequences.gen_mers(seq, 2))
    assert len(mers) == 5
    assert mers[-1] == 'at'
    # expect just 2, w and w/o first/last
    mers = list(sequences.gen_mers(seq, 5))
    assert len(mers) == 2
    assert mers[-1] == 'tatat'


def test_count2mers():
    mc = sequences.MerCounter(2)
    mers = ['aa', 'aa', 'aa']
    for mer in mers:
        mc.add_mer(mer)
    counted = mc.export()
    assert counted['aa'] == 3

    rc_mers = ['tt', 'tt']
    for mer in rc_mers:
        mc.add_mer(mer)
    counted = mc.export()
    assert counted['aa'] == 5

    mc2 = sequences.MerCounter(2)
    seq = 'aaattt'
    mc2.add_sequence(seq)
    counted = mc2.export()
    non0 = [x for x in counted if counted[x] > 0]
    assert len(non0) == 2
    assert counted['aa'] == 4
    assert counted['at'] == 1


# testing parsing matches
def test_fa_matches_sequences_json():
    fa_path = 'testdata/tester.fa'
    json_path = 'testdata/tester.sequence.json'
    sd_fa = sequences.StructuredGenome()
    sd_fa.add_fasta(fa_path)
    sd_json = sequences.StructuredGenome()
    sd_json.from_json(json_path)
    j_fa = sd_fa.to_jsonable()
    j_json = sd_json.to_jsonable()
    for key in j_fa:
        assert j_fa[key] == j_json[key]
    assert sd_fa.to_jsonable() == sd_json.to_jsonable()


### annotations ###
def test_id_maker():
    ider = annotations.IDMaker()
    for i in range(5):
        ider.next_unique_id()
    assert len(ider.seen) == 5
    # add a new id
    suggestion = 'apple'
    new_id = ider.next_unique_id(suggestion)
    assert len(ider.seen) == 6
    assert new_id == suggestion
    # try and add an ID we've now seen before
    new_id = ider.next_unique_id(suggestion)
    assert len(ider.seen) == 7
    assert new_id != suggestion


def setup_testable_super_loci():
    # features [--0--][1][---2---]
    f_coord = [(0, 100), (110, 120), (200, 400)]
    # transcript X: [0, 1], Y: [0, 1, 2], Z: [0]
    t_ids = ['x', 'y', 'z']
    t_features = [(0, 1), (0, 1, 2), (0, )]
    genome = annotations.AnnotatedGenome()
    sl = annotations.SuperLoci(genome)

    # setup transcripts and features
    transcripts = []
    features = []
    for i in range(3):
        t = annotations.Transcribed()
        t.id = t_ids[i]
        t.super_loci = sl
        for j in t_features[i]:
            e = annotations.StructuredFeature()
            e.id = genome.feature_ider.next_unique_id()
            e.transcripts = [t.id]
            c = annotations.StructuredFeature()
            c.transcripts = [t.id]
            c.id = genome.feature_ider.next_unique_id()
            print('transcript {}: [exon: {}, cds: {}, coord: {}]'.format(t.id, e.id, c.id, f_coord[j]))
            e.super_loci = c.super_loci = sl
            e.start, e.end = c.start, c.end = f_coord[j]
            c.type = "CDS"
            e.type = "exon"
            t.features += [e.id, c.id]
            features += [e, c]
        transcripts.append(t)
    for t in transcripts:
        sl.transcripts[t.id] = t
    for f in features:
        sl.features[f.id] = f
    # transcript x: [exon: ftr000000, cds: ftr000001, coord: (0, 100)]
    # transcript x: [exon: ftr000002, cds: ftr000003, coord: (110, 120)]
    # transcript y: [exon: ftr000004, cds: ftr000005, coord: (0, 100)]
    # transcript y: [exon: ftr000006, cds: ftr000007, coord: (110, 120)]
    # transcript y: [exon: ftr000008, cds: ftr000009, coord: (200, 400)]
    # transcript z: [exon: ftr000010, cds: ftr000011, coord: (0, 100)]
    return sl


def test_feature_overlap_detection():
    sl = setup_testable_super_loci()
    assert sl.features['ftr000000'].fully_overlaps(sl.features['ftr000004'])
    assert sl.features['ftr000005'].fully_overlaps(sl.features['ftr000001'])
    # a few that should not overlap
    assert not sl.features['ftr000000'].fully_overlaps(sl.features['ftr000001'])
    assert not sl.features['ftr000000'].fully_overlaps(sl.features['ftr000002'])


def test_collapse_identical_features():
    # features [--0--][1][---2---]
    # transcript X: [0, 1], Y: [0, 1, 2], Z: [0]
    sl = setup_testable_super_loci()
    # setup features
    print(sl.__dict__)
    print('features')
    for key in sorted(sl.features):
        feature_print(sl.features[key])
    print('transcripts')
    for key in sorted(sl.transcripts):
        transcript_print(sl.transcripts[key])

    # starting dimensions
    # total
    assert len(sl.transcripts.keys()) == 3
    assert len(sl.features.keys()) == 12  # 2 x 6
    # by transcript
    assert len(sl.transcripts['y'].features) == 6
    assert len(sl.transcripts['z'].features) == 2

    # collapse
    sl.collapse_identical_features()
    # totals went down
    assert len(sl.transcripts.keys()) == 3
    assert len(sl.features.keys()) == 6  # 2 x 3
    # transcripts kept numbers from before
    assert len(sl.transcripts['y'].features) == 6
    assert len(sl.transcripts['z'].features) == 2
    # transcripts point to exact same features as sl has directly
    f = []
    for t in sl.transcripts:
        f += sl.transcripts[t].features
    feat_ids_frm_transcripts = set(f)
    assert set(sl.features.keys()) == feat_ids_frm_transcripts


def test_add_exon():
    sl = setup_testable_super_loci()

    # add a cds that needs an exon
    new_cds = annotations.StructuredFeature()
    new_cds.super_loci = sl
    new_cds.id = sl.genome.feature_ider.next_unique_id()
    new_cds.start, new_cds.end, new_cds.type = 0, 108, sl.genome.gffkey.cds
    sl.features[new_cds.id] = new_cds
    old_len = len(sl.features.keys())
    sl.maybe_reconstruct_exons()
    assert len(sl.features.keys()) == old_len + 1

    # and add contained features that have an exon
    # cds
    new_cds = annotations.StructuredFeature()
    new_cds.super_loci = sl
    new_cds.id = sl.genome.feature_ider.next_unique_id()
    new_cds.start, new_cds.end, new_cds.type = 21, 100, sl.genome.gffkey.cds
    sl.features[new_cds.id] = new_cds
    # five prime utr
    new_utr = annotations.StructuredFeature()
    new_utr.super_loci = sl
    new_utr.id = sl.genome.feature_ider.next_unique_id()
    new_utr.start, new_utr.end, new_utr.type = 1, 20, sl.genome.gffkey.five_prime_UTR
    sl.features[new_utr.id] = new_utr
    old_len = len(sl.features.keys())
    sl.maybe_reconstruct_exons()
    assert len(sl.features.keys()) == old_len


# todo, make short / long print methods as part of object
def feature_print(feature):
    print('{} is {}: {}-{} on {}. --> {}'.format(
        feature.id, feature.type, feature.start, feature.end, feature.seqid, feature.transcripts))


def transcript_print(transcript):
    print('{}. --> {}'.format(transcript.id, transcript.features))
