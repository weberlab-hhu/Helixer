import sequences
import structure
import annotations
import helpers
import pytest

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
    f_coord = [(1, 100), (111, 120), (201, 400)]
    # transcript X: [0, 1], Y: [0, 1, 2], Z: [0]
    t_ids = ['x', 'y', 'z']
    t_features = [(0, 1), (0, 1, 2), (0, )]
    genome = annotations.AnnotatedGenome()
    # make a dummy sequence
    seq_mi = annotations.MetaInfoAnnoSequence()
    seq_mi.total_bp = 450
    genome.meta_info_sequences[seq_mi.seqid] = seq_mi

    sl = annotations.SuperLoci()
    sl.genome = genome

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
            e.strand = c.strand = '+'
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


def setup_loci_with_utr():
    sl = setup_testable_super_loci()
    for key_1stCDS in ['ftr000001', 'ftr000005', 'ftr000011']:
        sl.features[key_1stCDS].start = 11  # start first CDS later
        sl.features[key_1stCDS].phase = 0  # let's just assume the initial phase is correct

    sl.features['ftr000009'].end = 330  # end first CDS sooner
    return sl


def test_feature_overlap_detection():
    sl = setup_testable_super_loci()
    assert sl.features['ftr000000'].fully_overlaps(sl.features['ftr000004'])
    assert sl.features['ftr000005'].fully_overlaps(sl.features['ftr000001'])
    # a few that should not overlap
    assert not sl.features['ftr000000'].fully_overlaps(sl.features['ftr000001'])
    assert not sl.features['ftr000000'].fully_overlaps(sl.features['ftr000002'])


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


def test_transcript_interpreter():
    sl = setup_loci_with_utr()
    # change so that there are implicit UTRs
    sl.features['ftr000005'].start = 11  # start first CDS later
    sl.features['ftr000009'].end = 330  # end first CDS sooner
    transcript = sl.transcripts['y']
    t_interp = annotations.TranscriptInterpreter(transcript)
    t_interp.decode_raw_features()
    # has all standard features
    assert set([x.type for x in t_interp.clean_features]) == {sl.genome.gffkey.start_codon,
                                                              sl.genome.gffkey.stop_codon,
                                                              sl.genome.gffkey.TTS,
                                                              sl.genome.gffkey.TSS,
                                                              sl.genome.gffkey.donor_splice_site,
                                                              sl.genome.gffkey.acceptor_splice_site}
    assert t_interp.clean_features[-1].end == 400
    assert t_interp.clean_features[0].start == 1


def test_transcript_get_first():
    # plus strand
    sl = setup_loci_with_utr()
    transcript = sl.transcripts['y']
    t_interp = annotations.TranscriptInterpreter(transcript)
    i0 = t_interp.intervals_5to3(plus_strand=True)[0]
    t_interp.interpret_first_pos(i0)
    features = t_interp.clean_features
    status = t_interp.status
    assert len(features) == 1
    f0 = features[0]
    print(f0.short_str())
    print(status.__dict__)
    print(i0[0].data.strand)
    assert f0.start == 1
    assert status.is_5p_utr()
    assert f0.phase is None
    assert f0.strand == '+'

    # minus strand
    sl = setup_loci_with_utr()
    for feature in sl.features.values():  # force minus strand
        feature.strand = '-'

    transcript = sl.transcripts['y']
    t_interp = annotations.TranscriptInterpreter(transcript)
    i0 = t_interp.intervals_5to3(plus_strand=False)[0]
    t_interp.interpret_first_pos(i0, plus_strand=False)
    features = t_interp.clean_features
    status = t_interp.status
    assert len(features) == 1
    f0 = features[0]
    print(f0.short_str())
    print(status)
    print(i0[0].data.strand)
    assert f0.start == 400
    assert status.is_5p_utr()
    assert f0.phase is None
    assert f0.strand == '-'

    # test without UTR (x doesn't have last exon, and therefore will end in CDS)
    transcript = sl.transcripts['x']
    t_interp = annotations.TranscriptInterpreter(transcript)
    i0 = t_interp.intervals_5to3(plus_strand=False)[0]
    t_interp.interpret_first_pos(i0, plus_strand=False)
    features = t_interp.clean_features
    status = t_interp.status
    assert len(features) == 2
    f_err = features[0]
    f_status_coding = features[1]
    print(f_err.short_str())
    print(status)
    print(i0)
    # should get status_coding instead of a start codon
    assert f_status_coding.start == 120
    assert f_status_coding.type == sl.genome.gffkey.status_coding
    assert f_status_coding.strand == '-'
    # region beyond exon should be marked erroneous
    assert f_err.start == 121
    assert f_err.end == 450
    assert f_err.type == sl.genome.gffkey.error
    assert status.is_coding()
    assert status.seen_start
    assert status.genic


def test_transcript_transition_from_5p_to_end():
    sl = setup_loci_with_utr()
    transcript = sl.transcripts['y']
    t_interp = annotations.TranscriptInterpreter(transcript)
    ivals_sets = t_interp.intervals_5to3(plus_strand=True)
    t_interp.interpret_first_pos(ivals_sets[0])
    # hit start codon
    t_interp.interpret_transition(ivals_before=ivals_sets[0], ivals_after=ivals_sets[1], plus_strand=True)
    features = t_interp.clean_features
    assert features[-1].type == sl.genome.gffkey.start_codon
    assert features[-1].start == 11
    assert features[-1].end == 13
    # hit splice site
    t_interp.interpret_transition(ivals_before=ivals_sets[1], ivals_after=ivals_sets[2], plus_strand=True)
    assert features[-1].type == sl.genome.gffkey.acceptor_splice_site
    assert features[-2].type == sl.genome.gffkey.donor_splice_site
    assert features[-2].start == 101  # splice from
    assert features[-1].start == 110  # splice to
    assert t_interp.status.is_coding()
    # hit splice site
    t_interp.interpret_transition(ivals_before=ivals_sets[2], ivals_after=ivals_sets[3], plus_strand=True)
    assert features[-1].type == sl.genome.gffkey.acceptor_splice_site
    assert features[-2].type == sl.genome.gffkey.donor_splice_site
    assert features[-2].start == 121  # splice from
    assert features[-1].start == 200  # splice to
    # hit stop codon
    t_interp.interpret_transition(ivals_before=ivals_sets[3], ivals_after=ivals_sets[4], plus_strand=True)
    assert features[-1].type == sl.genome.gffkey.stop_codon
    assert features[-1].start == 328
    # hit transcription termination site
    t_interp.interpret_last_pos(ivals_sets[4], plus_strand=True)
    assert features[-1].type == sl.genome.gffkey.TTS
    assert features[-1].start == 400


def test_non_coding_transitions():
    sl = setup_testable_super_loci()
    # get single-exon no-CDS transcript
    transcript = sl.transcripts['z']
    transcript.remove_feature('ftr000011')
    print(transcript.short_str())
    t_interp = annotations.TranscriptInterpreter(transcript)
    ivals_sets = t_interp.intervals_5to3(plus_strand=True)
    assert len(ivals_sets) == 1
    t_interp.interpret_first_pos(ivals_sets[0])
    features = t_interp.clean_features
    assert features[-1].type == sl.genome.gffkey.TSS
    t_interp.interpret_last_pos(ivals_sets[0], plus_strand=True)
    assert features[-1].type == sl.genome.gffkey.TTS
    assert features[-1].start == 100
    assert len(features) == 2


def test_errors_not_lost():
    sl = setup_loci_with_utr()
    feature_e = annotations.StructuredFeature()
    feature_e.id = sl.genome.feature_ider.next_unique_id()
    feature_e.super_loci = sl
    sl.features[feature_e.id] = feature_e
    feature_e.start, feature_e.end = 40, 80
    feature_e.change_to_error()
    print('what features did we start with::?')
    for feature in sl.features:
        print(feature)
        print(sl.features[feature].short_str())
    sl.check_and_fix_structure(entries=None)
    print('---and what features did we leave?---')
    for feature in sl.features:
        print(feature)
        print(sl.features[feature].short_str())
    assert feature_e in sl.features.values()


def test_anno2json_and_back():
    # setup the sequence file
    #fa_path = 'testdata/testerSl.fa'
    json_path = 'testdata/testerSl.sequence.json'
    #sd_fa = sequences.StructuredGenome()
    #sd_fa.add_fasta(fa_path)
    #sd_fa.to_json(json_path)
    sd_fa = sequences.StructuredGenome()
    sd_fa.from_json(json_path)

    gfffile = 'testdata/testerSl.gff3'
    json_anno = 'testdata/testerSl.annotation.json'
    ag = annotations.AnnotatedGenome()
    ag.add_gff(gfffile, sd_fa, 'testdata/deletable')
    ag.to_json(json_anno)
    ag_json = annotations.AnnotatedGenome()
    ag_json.from_json(json_anno)
    assert ag.to_jsonable() == ag_json.to_jsonable()
    # check recursive load has worked out
    assert ag_json.super_loci[0].genome is ag_json
    # and same for super_loci
    sl = ag_json.super_loci[0]
    fkey = sorted(sl.features.keys())[0]
    feature = sl.features[fkey]
    assert feature.super_loci is sl
    assert sl.genome.meta_info_sequences[feature.seqid].total_bp == 16000


#### helpers
def test_key_matching():
    # identical
    known = {'a', 'b', 'c'}
    mapper, is_forward = helpers.two_way_key_match(known, known)
    assert is_forward
    assert mapper('a') == 'a'
    assert isinstance(mapper, helpers.CheckMapper)
    with pytest.raises(KeyError):
        mapper('d')

    # subset, should behave as before
    mapper, is_forward = helpers.two_way_key_match(known, {'c'})
    assert is_forward
    assert mapper('a') == 'a'
    assert isinstance(mapper, helpers.CheckMapper)
    with pytest.raises(KeyError):
        mapper('d')

    # superset, should flip ordering
    mapper, is_forward = helpers.two_way_key_match({'c'}, known)
    assert not is_forward
    assert mapper('a') == 'a'
    assert isinstance(mapper, helpers.CheckMapper)
    with pytest.raises(KeyError):
        mapper('d')

    # other is abbreviated from known
    set1 = {'a.seq', 'b.seq', 'c.seq'}
    set2 = {'a', 'b', 'c'}
    mapper, is_forward = helpers.two_way_key_match(set1, set2)
    assert is_forward
    assert mapper('a') == 'a.seq'
    assert isinstance(mapper, helpers.DictMapper)
    with pytest.raises(KeyError):
        mapper('d')

    # cannot be safely differentiated
    set1 = {'ab.seq', 'ba.seq', 'c.seq', 'a.seq', 'b.seq'}
    set2 = {'a', 'b', 'c'}
    with pytest.raises(helpers.NonMatchableIDs):
        mapper, is_forward = helpers.two_way_key_match(set1, set2)
        print(mapper.key_vals)


def test_gff_to_seqids():
    x = helpers.get_seqids_from_gff('testdata/testerSl.gff3')
    assert x == {'NC_015438.2', 'NC_015439.2', 'NC_015440.2'}
