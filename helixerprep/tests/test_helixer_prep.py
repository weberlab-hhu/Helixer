from ..datas import sequences
from ..core import structure
import geenuff
#from geenuff import api as annotations
#from geenuff import orm as annotations_orm
from helixerprep.datas.annotations import slice_dbmods, slicer
from ..core import helpers
#from geenuff import types as type_enums

import pytest
from ..core import partitions
import os
import numpy as np

from sqlalchemy.orm import sessionmaker
import sqlalchemy

from ..numerify import numerify

#from test_geenuff import setup_data_handler, mk_session, TransspliceDemoData
from geenuff.tests.test_geenuff import setup_data_handler, mk_session, TransspliceDemoData

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
    # sd_fa.to_json(json_path)  # can uncomment when one intentionally changed the format, but check
    sd_json = sequences.StructuredGenome()
    sd_json.from_json(json_path)
    j_fa = sd_fa.to_jsonable()
    j_json = sd_json.to_jsonable()
    for key in j_fa:
        assert j_fa[key] == j_json[key]
    assert sd_fa.to_jsonable() == sd_json.to_jsonable()


def test_sequence_slicing():
    json_path = 'testdata/dummyloci.sequence.json'
    sd_fa = sequences.StructuredGenome()
    sd_fa.from_json(json_path)
    sd_fa.divvy_each_sequence(user_seed='', max_len=100)
    print(sd_fa.to_jsonable())
    sd_fa.to_json('testdata/dummyloci.sequence.sliced.json')  # used later, todo, cleanup this sorta of stuff
    for sequence in sd_fa.sequences:
        # all but the last two should be of max_len
        for slice in sequence.slices[:-2]:
            assert len(''.join(slice.sequence)) == 100
            assert slice.end - slice.start == 100
        # the last two should split the remainder in half, therefore have a length difference of 0 or 1
        penultimate = sequence.slices[-2]
        ultimate = sequence.slices[-1]
        delta_len = abs((penultimate.end - penultimate.start) - (ultimate.end - ultimate.start))
        assert delta_len == 1 or delta_len == 0


## slice_dbmods
def test_processing_set_enum():
    # valid numbers can be setup
    ps = slice_dbmods.ProcessingSet(slice_dbmods.ProcessingSet.train)
    ps2 = slice_dbmods.ProcessingSet('train')
    assert ps == ps2
    # other numbers can't
    with pytest.raises(ValueError):
        slice_dbmods.ProcessingSet('training')
    with pytest.raises(ValueError):
        slice_dbmods.ProcessingSet(1.3)
    with pytest.raises(ValueError):
        slice_dbmods.ProcessingSet('Dev')


def test_add_processing_set():
    sess = mk_session()
    ag = geenuff.orm.AnnotatedGenome()
    sequence_info, sequence_infoh = setup_data_handler(slicer.SequenceInfoHandler, geenuff.orm.SequenceInfo,
                                                       annotated_genome=ag)
    sequence_info_s = slice_dbmods.SequenceInfoSets(processing_set='train', sequence_info=sequence_info)
    sequence_info2 = geenuff.orm.SequenceInfo(annotated_genome=ag)
    sequence_info2_s = slice_dbmods.SequenceInfoSets(processing_set='train', sequence_info=sequence_info2)
    sess.add_all([ag, sequence_info_s, sequence_info2_s])
    sess.commit()
    assert sequence_info_s.processing_set.value == 'train'
    assert sequence_info2_s.processing_set.value == 'train'

    sess.add_all([sequence_info, sequence_info2, sequence_info2_s, sequence_info_s])
    sess.commit()
    # make sure we can get the right info together back from the db
    maybe_join = sess.query(geenuff.orm.SequenceInfo, slice_dbmods.SequenceInfoSets).filter(
        geenuff.orm.SequenceInfo.id == slice_dbmods.SequenceInfoSets.id)
    for si, sis in maybe_join.all():
        assert si.id == sis.id
        assert sis.processing_set.value == 'train'

    # and make sure we can get the processing_set from the sequence_info
    sis = sess.query(slice_dbmods.SequenceInfoSets).filter(slice_dbmods.SequenceInfoSets.id == sequence_info.id).all()
    assert len(sis) == 1
    assert sis[0] == sequence_info_s

    # and over api
    sis = sequence_infoh.processing_set(sess)
    assert sis is sequence_info_s
    assert sequence_infoh.processing_set_val(sess) == 'train'
    # set over api
    sequence_infoh.set_processing_set(sess, 'test')
    assert sequence_infoh.processing_set_val(sess) == 'test'

    # confirm we can't have two processing sets per sequence_info
    with pytest.raises(sqlalchemy.orm.exc.FlushError):
        extra_set = slice_dbmods.SequenceInfoSets(processing_set='dev', sequence_info=sequence_info)
        sess.add(extra_set)
        sess.commit()
    sess.rollback()
    assert sequence_infoh.processing_set_val(sess) == 'test'

    # check that absence of entry, is handles with None
    sequence_info3, sequence_info3h = setup_data_handler(slicer.SequenceInfoHandler, geenuff.orm.SequenceInfo,
                                                         annotated_genome=ag)
    assert sequence_info3h.processing_set(sess) is None
    assert sequence_info3h.processing_set_val(sess) is None
    # finally setup complete new set via api
    sequence_info3h.set_processing_set(sess, 'dev')
    assert sequence_info3h.processing_set_val(sess) == 'dev'


#### slicer ####
def test_copy_n_import():
    # bc we don't want to change original db at any point
    destination = 'testdata/tmp.db'
    if os.path.exists(destination):
        os.remove(destination)
    source = 'testdata/dummyloci_annotations.sqlitedb'

    controller = slicer.SliceController(db_path_in=source, db_path_sliced=destination)
    controller.mk_session()
    controller.load_annotations()
    assert len(controller.super_loci) == 1
    sl = controller.super_loci[0].data
    assert len(sl.transcribeds) == 3
    assert len(sl.translateds) == 3
    for transcribed in sl.transcribeds:
        assert len(transcribed.transcribed_pieces) == 1
        piece = transcribed.transcribed_pieces[0]
        print('{}: {}'.format(transcribed.given_id, [x.type.value for x in piece.features]))
    for translated in sl.translateds:
        print('{}: {}'.format(translated.given_id, [x.type.value for x in translated.features]))

    assert len(sl.features) == 24  # if I ever get to collapsing redundant features this will change


def test_intervaltree():
    destination = 'testdata/tmp.db'
    if os.path.exists(destination):
        os.remove(destination)
    source = 'testdata/dummyloci_annotations.sqlitedb'

    controller = slicer.SliceController(db_path_in=source, db_path_sliced=destination)
    controller.mk_session()
    controller.load_annotations()
    controller.fill_intervaltrees()
    print(controller.interval_trees.keys())
    print(controller.interval_trees['1'])
    # check that one known area has two errors, and one transcription termination site as expected
    intervals = controller.interval_trees['1'][400:406]
    assert len(intervals) == 3
    print(intervals, '...intervals')
    print([x.data.data.type.value for x in intervals])
    errors = [x for x in intervals if x.data.data.type.value == geenuff.types.ERROR and
              x.data.data.bearing.value == geenuff.types.END]

    assert len(errors) == 2
    tts = [x for x in intervals if x.data.data.type.value == geenuff.types.TRANSCRIBED and
           x.data.data.bearing.value == geenuff.types.END]

    assert len(tts) == 1
    # check that the major filter functions work
    sls = controller.get_super_loci_frm_slice(seqid='1', start=300, end=405, is_plus_strand=True)
    assert len(sls) == 1
    assert isinstance(list(sls)[0], slicer.SuperLocusHandler)

    features = controller.get_features_from_slice(seqid='1', start=0, end=1, is_plus_strand=True)
    assert len(features) == 3
    starts = [x for x in features if x.data.type.value == geenuff.types.TRANSCRIBED and
              x.data.bearing.value == geenuff.types.START]

    assert len(starts) == 2
    errors = [x for x in features if x.data.type.value == geenuff.types.ERROR and x.data.bearing.value == geenuff.types.START]
    assert len(errors) == 1


def test_order_features():
    destination = 'testdata/tmp.db'
    if os.path.exists(destination):
        os.remove(destination)
    source = 'testdata/dummyloci_annotations.sqlitedb'

    controller = slicer.SliceController(db_path_in=source, db_path_sliced=destination)
    controller.mk_session()
    controller.load_annotations()
    sl = controller.super_loci[0]
    transcript = [x for x in sl.data.transcribeds if x.given_id == 'y'][0]
    transcripth = slicer.TranscribedHandler()
    transcripth.add_data(transcript)
    ti = slicer.TranscriptTrimmer(transcripth, super_locus=None, sess=controller.session)
    assert len(transcript.transcribed_pieces) == 1
    piece = transcript.transcribed_pieces[0]
    # features expected to be ordered by increasing position (note: as they are in db)
    ordered_starts = [0, 10, 100, 110, 120, 200, 300, 400]
    features = ti.sorted_features(piece)
    for f in features:
        print(f)
    assert [x.position for x in features] == ordered_starts
    for feature in piece.features:
        feature.is_plus_strand = False
    features = ti.sorted_features(piece)
    ordered_starts.reverse()
    assert [x.position for x in features] == ordered_starts
    # force erroneous data
    piece.features[0].is_plus_strand = True
    controller.session.add(piece.features[0])
    controller.session.commit()
    with pytest.raises(AssertionError):
        ti.sorted_features(piece)


# todo, we don't actually need slicer for this, mv to test_geenuff
def test_slicer_transition():
    destination = 'testdata/tmp.db'
    if os.path.exists(destination):
        os.remove(destination)
    source = 'testdata/dummyloci_annotations.sqlitedb'

    controller = slicer.SliceController(db_path_in=source, db_path_sliced=destination)
    controller.mk_session()
    controller.load_annotations()
    sl = controller.super_loci[0]
    transcript = [x for x in sl.data.transcribeds if x.given_id == 'y'][0]
    transcripth = slicer.TranscribedHandler()
    transcripth.add_data(transcript)
    ti = slicer.TranscriptTrimmer(transcript=transcripth, super_locus=None, sess=controller.session)
    transition_gen = ti.transition_5p_to_3p()
    transitions = list(transition_gen)
    assert len(transitions) == 8
    statusses = [x[1] for x in transitions]
    features = [x[0][0] for x in transitions]
    ordered_starts = [0, 10, 100, 110, 120, 200, 300, 400]
    assert [x.position for x in features] == ordered_starts
    expected_intronic = [False, False, True, False, True, False, False, False]
    assert [x.in_intron for x in statusses] == expected_intronic
    expected_genic = [True] * 7 + [False]
    print('statuses', [x.genic for x in statusses])
    assert [x.genic for x in statusses] == expected_genic
    expected_seen_startstop = [(False, False)] + [(True, False)] * 5 + [(True, True)] * 2
    assert [(x.seen_start, x.seen_stop) for x in statusses] == expected_seen_startstop


def test_set_updown_features_downstream_border():
    sess = mk_session()
    old_coor = geenuff.orm.Coordinates(seqid='a', start=1, end=1000)
    new_coord = geenuff.orm.Coordinates(seqid='a', start=100, end=200)
    sl, slh = setup_data_handler(slicer.SuperLocusHandler, geenuff.orm.SuperLocus)
    scribed, scribedh = setup_data_handler(slicer.TranscribedHandler, geenuff.orm.Transcribed, super_locus=sl)
    ti = slicer.TranscriptTrimmer(transcript=scribedh, super_locus=slh, sess=sess)
    piece0 = geenuff.orm.TranscribedPiece(super_locus=sl)
    piece1 = geenuff.orm.TranscribedPiece(super_locus=sl)
    scribed.transcribed_pieces = [piece0]
    # setup some paired features
    # new coords, is plus, template, status
    feature = geenuff.orm.Feature(transcribed_pieces=[piece1], coordinates=old_coor, position=110,
                                  is_plus_strand=True, super_locus=sl, type=geenuff.types.CODING,
                                  bearing=geenuff.types.START)

    sess.add_all([scribed, piece0, piece1, old_coor, new_coord, sl])
    sess.commit()
    slh.make_all_handlers()
    # set to genic, non intron area
    status = geenuff.api.TranscriptStatus()
    status.saw_tss()
    status.saw_start(0)

    ti.set_status_downstream_border(new_coords=new_coord, is_plus_strand=True, template_feature=feature, status=status,
                                    old_piece=piece0, new_piece=piece1, old_coords=old_coor, trees={})

    sess.commit()
    assert len(piece1.features) == 3  # feature, 2x upstream
    assert len(piece0.features) == 2  # 2x downstream

    assert set([x.type.value for x in piece1.features]) == {geenuff.types.CODING, geenuff.types.TRANSCRIBED}
    assert set([x.bearing.value for x in piece1.features]) == {geenuff.types.START, geenuff.types.CLOSE_STATUS}

    assert set([x.type.value for x in piece0.features]) == {geenuff.types.CODING, geenuff.types.TRANSCRIBED}
    assert set([x.bearing.value for x in piece0.features]) == {geenuff.types.OPEN_STATUS}

    translated_up_status = [x for x in piece1.features if x.type.value == geenuff.types.CODING and
                            x.bearing.value == geenuff.types.CLOSE_STATUS][0]

    translated_down_status = [x for x in piece0.features if x.type.value == geenuff.types.TRANSCRIBED][0]
    assert translated_up_status.position == 200
    assert translated_down_status.position == 200
    # cleanup to try similar again
    for f in piece0.features:
        sess.delete(f)
    for f in piece1.features:
        sess.delete(f)
    sess.commit()

    # and now try backwards pass
    feature = geenuff.orm.Feature(transcribed_pieces=[piece1], coordinates=old_coor, position=110,
                                  is_plus_strand=False, super_locus=sl, type=geenuff.types.CODING,
                                  bearing=geenuff.types.START)
    sess.add(feature)
    sess.commit()
    slh.make_all_handlers()
    ti.set_status_downstream_border(new_coords=new_coord, is_plus_strand=False, template_feature=feature, status=status,
                                    old_piece=piece0, new_piece=piece1, old_coords=old_coor, trees={})
    sess.commit()

    assert len(piece1.features) == 3  # feature, 2x upstream
    assert len(piece0.features) == 2  # 2x downstream
    assert set([x.type.value for x in piece1.features]) == {geenuff.types.CODING, geenuff.types.TRANSCRIBED}
    assert set([x.bearing.value for x in piece1.features]) == {geenuff.types.START, geenuff.types.CLOSE_STATUS}

    assert set([x.type.value for x in piece0.features]) == {geenuff.types.CODING, geenuff.types.TRANSCRIBED}
    assert set([x.bearing.value for x in piece0.features]) == {geenuff.types.OPEN_STATUS}

    translated_up_status = [x for x in piece1.features if x.type.value == geenuff.types.CODING and
                            x.bearing.value == geenuff.types.CLOSE_STATUS][0]

    translated_down_status = [x for x in piece0.features if x.type.value == geenuff.types.TRANSCRIBED][0]

    assert translated_up_status.position == 99
    assert translated_down_status.position == 99


def test_transition_with_right_new_pieces():
    sess = mk_session()
    old_coor = geenuff.orm.Coordinates(seqid='a', start=1, end=1000)
    # setup two transitions:
    # 1) scribed - [[A,B]] -> AB, -> one expected new piece
    # 2) scribedlong - [[C,D],[A,B]] -> ABCD, -> two expected new pieces
    sl, slh = setup_data_handler(slicer.SuperLocusHandler, geenuff.orm.SuperLocus)
    scribed, scribedh = setup_data_handler(slicer.TranscribedHandler, geenuff.orm.Transcribed, super_locus=sl)
    scribedlong, scribedlongh = setup_data_handler(slicer.TranscribedHandler, geenuff.orm.Transcribed,
                                                   super_locus=sl)

    ti = slicer.TranscriptTrimmer(transcript=scribedh, super_locus=slh, sess=sess)
    tilong = slicer.TranscriptTrimmer(transcript=scribedlongh, super_locus=slh, sess=sess)

    pieceAB = geenuff.orm.TranscribedPiece(super_locus=sl, transcribed=scribed)
    pieceABp = geenuff.orm.TranscribedPiece(super_locus=sl, transcribed=scribedlong)
    pieceCD = geenuff.orm.TranscribedPiece(super_locus=sl, transcribed=scribedlong)

    fA = geenuff.orm.Feature(transcribed_pieces=[pieceAB, pieceABp], coordinates=old_coor, position=190,
                             is_plus_strand=True, super_locus=sl, type=geenuff.types.ERROR,
                             bearing=geenuff.types.START)
    fB = geenuff.orm.UpstreamFeature(transcribed_pieces=[pieceAB, pieceABp], coordinates=old_coor, position=210,
                                     is_plus_strand=True, super_locus=sl, type=geenuff.types.ERROR,
                                     bearing=geenuff.types.CLOSE_STATUS)  # todo, double check for consistency after type/bearing mod to slice...

    fC = geenuff.orm.DownstreamFeature(transcribed_pieces=[pieceCD], coordinates=old_coor, position=90,
                                       is_plus_strand=True, super_locus=sl, type=geenuff.types.ERROR,
                                       bearing=geenuff.types.OPEN_STATUS)
    fD = geenuff.orm.Feature(transcribed_pieces=[pieceCD], coordinates=old_coor, position=110,
                             is_plus_strand=True, super_locus=sl, type=geenuff.types.ERROR, bearing=geenuff.types.END)

    pair = geenuff.orm.UpDownPair(upstream=fB, downstream=fC, transcribed=scribedlong)
    sess.add_all([scribed, scribedlong, pieceAB, pieceABp, pieceCD, fA, fB, fC, fD, pair, old_coor, sl])
    sess.commit()
    short_transition = list(ti.transition_5p_to_3p_with_new_pieces())
    assert len(set([x.replacement_piece for x in short_transition])) == 1
    long_transition = list(tilong.transition_5p_to_3p_with_new_pieces())
    assert len(long_transition) == 4
    assert len(set([x.replacement_piece for x in long_transition])) == 2
    # make sure piece swap is between B(1) & C(2) as expected
    assert long_transition[1].replacement_piece is not long_transition[2].replacement_piece


def test_modify4slice():
    destination = 'testdata/tmp.db'
    if os.path.exists(destination):
        os.remove(destination)
    source = 'testdata/dummyloci_annotations.sqlitedb'

    controller = slicer.SliceController(db_path_in=source, db_path_sliced=destination)
    controller.mk_session()
    controller.load_annotations()
    slh = controller.super_loci[0]
    transcript = [x for x in slh.data.transcribeds if x.given_id == 'y'][0]
    slh.make_all_handlers()
    ti = slicer.TranscriptTrimmer(transcript=transcript.handler, super_locus=slh, sess=controller.session)
    new_coords = geenuff.orm.Coordinates(seqid='1', start=0, end=100)
    newer_coords = geenuff.orm.Coordinates(seqid='1', start=100, end=200)
    ti.modify4new_slice(new_coords=new_coords, is_plus_strand=True)
    assert len(transcript.transcribed_pieces) == 2
    controller.session.add_all([new_coords, newer_coords])
    controller.session.commit()
    print(transcript.transcribed_pieces[0])
    for feature in transcript.transcribed_pieces[1].features:
        print('-- {} --\n'.format(feature))
    print(transcript.transcribed_pieces[1])
    assert {len(transcript.transcribed_pieces[0].features), len(transcript.transcribed_pieces[1].features)} == {8, 4}
    new_piece = [x for x in transcript.transcribed_pieces if len(x.features) == 4][0]
    ori_piece = [x for x in transcript.transcribed_pieces if len(x.features) == 8][0]

    assert set([x.type.value for x in new_piece.features]) == {geenuff.types.TRANSCRIBED,
                                                               geenuff.types.CODING}

    assert set([x.bearing.value for x in new_piece.features]) == {geenuff.types.START, geenuff.types.CLOSE_STATUS}

    print('starting second modify...')
    ti.modify4new_slice(new_coords=newer_coords, is_plus_strand=True)
    for piece in transcript.transcribed_pieces:
        print(piece)
        for f in piece.features:
            print('::::', (f.type.value, f.bearing.value, f.position))
    assert sorted([len(x.features) for x in transcript.transcribed_pieces]) == [4, 4, 8]  # todo, why does this occasionally fail??
    assert set([x.type.value for x in ori_piece.features]) == {geenuff.types.TRANSCRIBED,
                                                               geenuff.types.CODING}
    assert set([x.bearing.value for x in ori_piece.features]) == {geenuff.types.END,
                                                                  geenuff.types.OPEN_STATUS}


def test_modify4slice_directions():
    sess = mk_session()
    old_coor = geenuff.orm.Coordinates(seqid='a', start=1, end=1000)
    # setup two transitions:
    # 2) scribedlong - [[D<-,C<-],[->A,->B]] -> ABCD, -> two pieces forward, one backward
    sl, slh = setup_data_handler(slicer.SuperLocusHandler, geenuff.orm.SuperLocus)
    scribedlong, scribedlongh = setup_data_handler(slicer.TranscribedHandler, geenuff.orm.Transcribed,
                                                   super_locus=sl)

    tilong = slicer.TranscriptTrimmer(transcript=scribedlongh, super_locus=slh, sess=sess)

    pieceAB = geenuff.orm.TranscribedPiece(super_locus=sl)
    pieceCD = geenuff.orm.TranscribedPiece(super_locus=sl)
    scribedlong.transcribed_pieces = [pieceAB, pieceCD]

    fA = geenuff.orm.Feature(transcribed_pieces=[pieceAB], coordinates=old_coor, position=190, given_id='A',
                             is_plus_strand=True, super_locus=sl, type=geenuff.types.TRANSCRIBED,
                             bearing=geenuff.types.START)
    fB = geenuff.orm.UpstreamFeature(transcribed_pieces=[pieceAB], coordinates=old_coor, position=210,
                                     is_plus_strand=True, super_locus=sl, type=geenuff.types.TRANSCRIBED,
                                     bearing=geenuff.types.CLOSE_STATUS, given_id='B')

    fC = geenuff.orm.DownstreamFeature(transcribed_pieces=[pieceCD], coordinates=old_coor, position=110,
                                       is_plus_strand=False, super_locus=sl, type=geenuff.types.TRANSCRIBED,
                                       bearing=geenuff.types.OPEN_STATUS, given_id='C')
    fD = geenuff.orm.Feature(transcribed_pieces=[pieceCD], coordinates=old_coor, position=90,
                             is_plus_strand=False, super_locus=sl, type=geenuff.types.TRANSCRIBED,
                             bearing=geenuff.types.END, given_id='D')

    pair = geenuff.orm.UpDownPair(upstream=fB, downstream=fC, transcribed=scribedlong)

    half1_coords = geenuff.orm.Coordinates(seqid='a', start=1, end=200)
    half2_coords = geenuff.orm.Coordinates(seqid='a', start=200, end=400)
    sess.add_all([scribedlong, pieceAB, pieceCD, fA, fB, fC, fD, pair, old_coor, sl, half1_coords, half2_coords])
    sess.commit()
    slh.make_all_handlers()

    tilong.modify4new_slice(new_coords=half1_coords, is_plus_strand=True)

    sess.commit()
    tilong.modify4new_slice(new_coords=half2_coords, is_plus_strand=True)
    tilong.modify4new_slice(new_coords=half1_coords, is_plus_strand=False)
    for f in sess.query(geenuff.orm.Feature).all():
        assert len(f.transcribed_pieces) == 1
    slice0 = fA.transcribed_pieces[0]
    slice1 = fB.transcribed_pieces[0]
    slice2 = fC.transcribed_pieces[0]
    assert sorted([len(x.features) for x in tilong.transcript.data.transcribed_pieces]) == [2, 2, 2]
    assert set(slice2.features) == {fC, fD}


class TransspliceDemoDataSlice(TransspliceDemoData):
    def __init__(self, sess):
        super().__init__(sess)
        self.old_coor = geenuff.orm.Coordinates(seqid='a', start=1, end=2000)
        # replace handlers with those from slicer
        self.slh = slicer.SuperLocusHandler()
        self.slh.add_data(self.sl)

        self.scribedh = slicer.TranscribedHandler()
        self.scribedh.add_data(self.scribed)

        self.scribedfliph = slicer.TranscribedHandler()
        self.scribedfliph.add_data(self.scribedflip)

        self.ti = slicer.TranscriptTrimmer(transcript=self.scribedh, super_locus=self.slh, sess=sess)
        self.tiflip = slicer.TranscriptTrimmer(transcript=self.scribedfliph, super_locus=self.slh, sess=sess)


def test_piece_swap_handling_during_multipiece_one_coordinate_transition():
    sess = mk_session()
    d = TransspliceDemoDataSlice(sess)  # setup _d_ata
    d.make_all_handlers()
    # forward pass, same sequence, two pieces
    ti_transitions = list(d.ti.transition_5p_to_3p_with_new_pieces())
    pre_slice_swap = ti_transitions[4]
    assert pre_slice_swap.example_feature is not None
    post_slice_swap = ti_transitions[5]
    pre_slice_swap.set_as_previous_of(post_slice_swap)
    assert pre_slice_swap.example_feature is None
    assert post_slice_swap.example_feature is not None
    # two way pass, same sequence, two (one +, one -) piece
    tiflip_transitions = list(d.tiflip.transition_5p_to_3p_with_new_pieces())
    pre_slice_swap = tiflip_transitions[4]
    assert pre_slice_swap.example_feature is not None
    post_slice_swap = tiflip_transitions[5]
    pre_slice_swap.set_as_previous_of(post_slice_swap)
    assert pre_slice_swap.example_feature is None
    assert post_slice_swap.example_feature is not None


class SimplestDemoData(object):
    def __init__(self, sess):
        self.old_coor = geenuff.orm.Coordinates(seqid='a', start=0, end=1000)
        # setup two transitions:
        # 2) scribedlong - [[D<-,C<-],[->A,->B]] -> ABCD, -> two pieces forward, one backward
        self.sl, self.slh = setup_data_handler(slicer.SuperLocusHandler, geenuff.orm.SuperLocus)
        self.scribedlong, self.scribedlongh = setup_data_handler(slicer.TranscribedHandler, geenuff.orm.Transcribed,
                                                                 super_locus=self.sl)

        self.tilong = slicer.TranscriptTrimmer(transcript=self.scribedlongh, super_locus=self.slh, sess=sess)

        self.pieceAB = geenuff.orm.TranscribedPiece(super_locus=self.sl)
        self.pieceCD = geenuff.orm.TranscribedPiece(super_locus=self.sl)
        self.scribedlong.transcribed_pieces = [self.pieceAB, self.pieceCD]

        self.fA = geenuff.orm.Feature(transcribed_pieces=[self.pieceAB], coordinates=self.old_coor, position=190,
                                      given_id='A', is_plus_strand=True, super_locus=self.sl,
                                      type=geenuff.types.TRANSCRIBED, bearing=geenuff.types.START)
        self.fB = geenuff.orm.UpstreamFeature(transcribed_pieces=[self.pieceAB], coordinates=self.old_coor,
                                              is_plus_strand=True, super_locus=self.sl, position=210,
                                              type=geenuff.types.TRANSCRIBED, bearing=geenuff.types.CLOSE_STATUS,
                                              given_id='B')

        self.fC = geenuff.orm.DownstreamFeature(transcribed_pieces=[self.pieceCD], coordinates=self.old_coor,
                                                is_plus_strand=False, super_locus=self.sl, position=110,
                                                type=geenuff.types.TRANSCRIBED, bearing=geenuff.types.OPEN_STATUS,
                                                given_id='C')
        self.fD = geenuff.orm.Feature(transcribed_pieces=[self.pieceCD], coordinates=self.old_coor, position=90,
                                      is_plus_strand=False, super_locus=self.sl,
                                      type=geenuff.types.TRANSCRIBED, bearing=geenuff.types.END, given_id='D')

        self.pair = geenuff.orm.UpDownPair(upstream=self.fB, downstream=self.fC, transcribed=self.scribedlong)

        sess.add_all([self.scribedlong, self.pieceAB, self.pieceCD, self.fA, self.fB, self.fC, self.fD, self.pair,
                      self.old_coor, self.sl])
        sess.commit()
        self.slh.make_all_handlers()


def test_transition_unused_coordinates_detection():
    sess = mk_session()
    d = SimplestDemoData(sess)
    # modify to coordinates with complete contain, should work fine
    new_coords = geenuff.orm.Coordinates(seqid='a', start=0, end=300)
    d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=True)
    d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=False)
    assert d.pieceCD not in d.sl.transcribed_pieces  # confirm full transition
    assert d.pieceAB not in d.scribedlong.transcribed_pieces
    # modify to coordinates across tiny slice, include those w/o original features, should work fine
    d = SimplestDemoData(sess)
    new_coords_list = [geenuff.orm.Coordinates(seqid='a', start=185, end=195),
                       geenuff.orm.Coordinates(seqid='a', start=195, end=205),
                       geenuff.orm.Coordinates(seqid='a', start=205, end=215)]
    print([x.id for x in d.tilong.transcript.data.transcribed_pieces])
    for new_coords in new_coords_list:
        print('fw {}, {}'.format(new_coords.id, new_coords))
        d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=True)
        print([x.id for x in d.tilong.transcript.data.transcribed_pieces])

    new_coords_list = [geenuff.orm.Coordinates(seqid='a', start=105, end=115),
                       geenuff.orm.Coordinates(seqid='a', start=95, end=105),
                       geenuff.orm.Coordinates(seqid='a', start=85, end=95)]
    for new_coords in new_coords_list:
        print('\nstart mod for coords, - strand', new_coords)
        d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=False)
        for piece in d.tilong.transcript.data.transcribed_pieces:
            print(piece, [(f.position, f.type, f.bearing) for f in piece.features])
    assert d.pieceCD not in d.scribedlong.transcribed_pieces  # confirm full transition
    assert d.pieceAB not in d.sl.transcribed_pieces

    # try and slice before coordinates, should raise error
    d = SimplestDemoData(sess)
    new_coords = geenuff.orm.Coordinates(seqid='a', start=0, end=10)
    with pytest.raises(slicer.NoFeaturesInSliceError):
        d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=True)
    # try and slice after coordinates, should raise error
    d = SimplestDemoData(sess)
    new_coords = geenuff.orm.Coordinates(seqid='a', start=399, end=410)
    with pytest.raises(slicer.NoFeaturesInSliceError):
        d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=True)
    # try and slice between slices where there are no coordinates, should raise error
    d = SimplestDemoData(sess)
    new_coords = geenuff.orm.Coordinates(seqid='a', start=149, end=160)
    with pytest.raises(slicer.NoFeaturesInSliceError):
        d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=True)
    d = SimplestDemoData(sess)
    with pytest.raises(slicer.NoFeaturesInSliceError):
        d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=False)


def test_features_on_opposite_strand_are_not_modified():
    sess = mk_session()
    d = SimplestDemoData(sess)
    # forward pass only, back pass should be untouched
    new_coords = geenuff.orm.Coordinates(seqid='a', start=1, end=300)
    d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=True)
    assert d.pieceAB not in d.sl.transcribed_pieces
    assert d.pieceCD in d.sl.transcribed_pieces  # minus piece should not change on 'plus' pass

    d = SimplestDemoData(sess)
    # backward pass only, plus pass should be untouched
    new_coords = geenuff.orm.Coordinates(seqid='a', start=1, end=300)
    d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=False)
    assert d.pieceAB in d.sl.transcribed_pieces  # plus piece should not change on 'minus' pass
    assert d.pieceCD not in d.sl.transcribed_pieces


def test_modify4slice_transsplice():
    sess = mk_session()
    d = TransspliceDemoDataSlice(sess)  # setup _d_ata
    d.make_all_handlers()
    new_coords_0 = geenuff.orm.Coordinates(seqid='a', start=0, end=915)
    new_coords_1 = geenuff.orm.Coordinates(seqid='a', start=915, end=2000)
    d.ti.modify4new_slice(new_coords=new_coords_0, is_plus_strand=True)
    d.ti.modify4new_slice(new_coords=new_coords_1, is_plus_strand=True)
    # we expect 3 new pieces,
    #    1: TSS-start-DonorTranssplice-TTS via <2x status> to (6 features)
    #    2: <2x status>-TSS- via <3x status> to (6 features)
    #    3: <3x status>-AcceptorTranssplice-stop-TTS (6 features)
    pieces = d.ti.transcript.data.transcribed_pieces
    assert len(pieces) == 3
    assert d.pieceA2D not in pieces  # pieces themselves should have been replaced
    sorted_pieces = d.ti.sort_pieces()

    assert [len(x.features) for x in sorted_pieces] == [6, 6, 6]
    ftypes_0 = set([(x.type.value, x.bearing.value) for x in sorted_pieces[0].features])
    assert ftypes_0 == {(geenuff.types.TRANSCRIBED, geenuff.types.START),
                        (geenuff.types.CODING, geenuff.types.START),
                        (geenuff.types.TRANS_INTRON, geenuff.types.START),
                        (geenuff.types.TRANSCRIBED, geenuff.types.END),
                        (geenuff.types.CODING, geenuff.types.CLOSE_STATUS),
                        (geenuff.types.TRANS_INTRON, geenuff.types.CLOSE_STATUS)}

    ftypes_2 = set([(x.type.value, x.bearing.value) for x in sorted_pieces[2].features])
    assert ftypes_2 == {(geenuff.types.CODING, geenuff.types.OPEN_STATUS),
                        (geenuff.types.TRANSCRIBED, geenuff.types.OPEN_STATUS),
                        (geenuff.types.TRANS_INTRON, geenuff.types.OPEN_STATUS),
                        (geenuff.types.CODING, geenuff.types.END),
                        (geenuff.types.TRANSCRIBED, geenuff.types.END),
                        (geenuff.types.TRANS_INTRON, geenuff.types.END)}
    # and now where second original piece is flipped and slice is thus between STOP and TTS
    print('moving on to flipped...')
    d.tiflip.modify4new_slice(new_coords=new_coords_0, is_plus_strand=True)
    assert len(d.tiflip.transcript.data.transcribed_pieces) == 2
    d.tiflip.modify4new_slice(new_coords=new_coords_1, is_plus_strand=False)
    d.tiflip.modify4new_slice(new_coords=new_coords_0, is_plus_strand=False)
    # we expect 3 new pieces,
    #    1: TSS-start-DonorTranssplice-TTS via <2x status> to (6 features)
    #    2: <2x status>-TSS-AcceptorTranssplice-stop via <1x status> to (6 features)
    #    3: <1x status>-TTS (2 features)
    pieces = d.tiflip.transcript.data.transcribed_pieces
    assert len(pieces) == 3
    assert d.pieceA2D not in pieces  # pieces themselves should have been replaced
    sorted_pieces = d.tiflip.sort_pieces()

    assert [len(x.features) for x in sorted_pieces] == [6, 6, 2]
    ftypes_0 = set([(x.type.value, x.bearing.value) for x in sorted_pieces[0].features])
    assert ftypes_0 == {(geenuff.types.TRANSCRIBED, geenuff.types.START),
                        (geenuff.types.CODING, geenuff.types.START),
                        (geenuff.types.TRANS_INTRON, geenuff.types.START),
                        (geenuff.types.TRANSCRIBED, geenuff.types.END),
                        (geenuff.types.TRANS_INTRON, geenuff.types.CLOSE_STATUS),
                        (geenuff.types.CODING, geenuff.types.CLOSE_STATUS)}
    ftypes_2 = set([(x.type.value, x.bearing.value) for x in sorted_pieces[2].features])
    assert ftypes_2 == {(geenuff.types.TRANSCRIBED, geenuff.types.OPEN_STATUS),
                        (geenuff.types.TRANSCRIBED, geenuff.types.END)}
    for piece in sorted_pieces:
        for f in piece.features:
            assert f.coordinates in {new_coords_1, new_coords_0}


def test_modify4slice_2nd_half_first():
    # because trans-splice occasions can theoretically hit transitions in the 'wrong' order where the 1st half of
    # the _final_ transcript hasn't been adjusted when the second half is adjusted/sliced. Results should be the same.
    sess = mk_session()
    d = TransspliceDemoDataSlice(sess)  # setup _d_ata
    d.make_all_handlers()
    new_coords_0 = geenuff.orm.Coordinates(seqid='a', start=0, end=915)
    new_coords_1 = geenuff.orm.Coordinates(seqid='a', start=915, end=2000)
    d.tiflip.modify4new_slice(new_coords=new_coords_1, is_plus_strand=False)
    d.tiflip.modify4new_slice(new_coords=new_coords_0, is_plus_strand=False)
    d.tiflip.modify4new_slice(new_coords=new_coords_0, is_plus_strand=True)
    # we expect 3 new pieces,
    #    1: TSS-start-DonorTranssplice-TTS via <2x status> to (6 features)
    #    2: <2x status>-TSS-AcceptorTranssplice-stop via <1x status> to (6 features)
    #    3: <1x status>-TTS (2 features)
    pieces = d.tiflip.transcript.data.transcribed_pieces
    assert len(pieces) == 3
    assert d.pieceA2D not in pieces  # pieces themselves should have been replaced
    sorted_pieces = d.tiflip.sort_pieces()

    assert [len(x.features) for x in sorted_pieces] == [6, 6, 2]
    ftypes_0 = set([(x.type.value, x.bearing.value) for x in sorted_pieces[0].features])
    assert ftypes_0 == {(geenuff.types.TRANSCRIBED, geenuff.types.START),
                        (geenuff.types.CODING, geenuff.types.START),
                        (geenuff.types.TRANS_INTRON, geenuff.types.START),
                        (geenuff.types.TRANSCRIBED, geenuff.types.END),
                        (geenuff.types.TRANS_INTRON, geenuff.types.CLOSE_STATUS),
                        (geenuff.types.CODING, geenuff.types.CLOSE_STATUS)}
    ftypes_2 = set([(x.type.value, x.bearing.value) for x in sorted_pieces[2].features])
    assert ftypes_2 == {(geenuff.types.TRANSCRIBED, geenuff.types.OPEN_STATUS),
                        (geenuff.types.TRANSCRIBED, geenuff.types.END)}

    for piece in sorted_pieces:
        for f in piece.features:
            assert f.coordinates in {new_coords_1, new_coords_0}


def test_slicing_multi_sl():
    # import standard testers
    destination = 'testdata/tmp.db'
    if os.path.exists(destination):
        os.remove(destination)
    source = 'testdata/dummyloci_annotations.sqlitedb'
    # TODO, add sliced sequences path
    controller = slicer.SliceController(db_path_in=source, db_path_sliced=destination,
                                        sequences_path='testdata/dummyloci.sequence.sliced.json')
    controller.mk_session()
    controller.load_annotations()
    controller.load_sliced_seqs()
    controller.fill_intervaltrees()
    slh = controller.super_loci[0]
    slh.make_all_handlers()
    # setup more
    more = SimplestDemoData(controller.session)
    controller.super_loci.append(more.slh)
    more.old_coor.seqid = '1'  # so it matches std dummyloci
    controller.session.commit()
    # and try and slice
    controller.slice_annotations(controller.get_one_annotated_genome())
    # todo, test if valid pass of final res.


def test_slicing_featureless_slice_inside_locus():
    destination = 'testdata/tmp.db'
    if os.path.exists(destination):
        os.remove(destination)
    source = 'testdata/dummyloci_annotations.sqlitedb'

    controller = slicer.SliceController(db_path_in=source, db_path_sliced=destination)
    controller.mk_session()
    controller.load_annotations()
    controller.fill_intervaltrees()
    ag = controller.get_one_annotated_genome()
    slh = controller.super_loci[0]
    transcript = [x for x in slh.data.transcribeds if x.given_id == 'y'][0]
    slices = (('1', 0, 40, '0-40'),
              ('1', 40, 80, '40-80'),
              ('1', 80, 120, '80-120'))
    slices = iter(slices)
    controller._slice_annotations_1way(slices, annotated_genome=ag, is_plus_strand=True)
    # todo, this is failing due coordinates issues. The status is closed for coding,transcribed prior to the
    #   end of exon/cds at same time error, and then we hit the slice, and then we open the error. AKA, no status
    #   at end of slice. Fix coordinates, get back to this.
    for piece in transcript.transcribed_pieces:
        print('got piece: {}\n-----------\n'.format(piece))
        for feature in piece.features:
            print('    {}'.format(feature))
    coordinate40 = controller.session.query(geenuff.orm.Coordinates).filter(
        geenuff.orm.Coordinates.start == 40
    ).first()
    features40 = coordinate40.features
    print(features40)

    # x & y -> 2 translated, 2 transcribed each, z -> 2 error
    assert len([x for x in features40 if x.type.value == geenuff.types.CODING]) == 4
    assert len([x for x in features40 if x.type.value == geenuff.types.TRANSCRIBED]) == 4
    assert len(features40) == 10
    assert set([x.type.value for x in features40]) == {geenuff.types.CODING, geenuff.types.TRANSCRIBED,
                                                       geenuff.types.ERROR}


def rm_transcript_and_children(transcript, sess):
    for piece in transcript.transcribed_pieces:
        for feature in piece.features:
            sess.delete(feature)
        sess.delete(piece)
    sess.delete(transcript)
    sess.commit()


def test_reslice_at_same_spot():
    destination = 'testdata/tmp.db'
    if os.path.exists(destination):
        os.remove(destination)
    source = 'testdata/dummyloci_annotations.sqlitedb'

    controller = slicer.SliceController(db_path_in=source, db_path_sliced=destination,
                                        sequences_path='testdata/dummyloci.sequence.sliced.json')
    controller.mk_session()
    controller.load_annotations()
    controller.load_sliced_seqs()

    slh = controller.super_loci[0]
    # simplify
    transcripty = [x for x in slh.data.transcribeds if x.given_id == 'y'][0]
    transcriptz = [x for x in slh.data.transcribeds if x.given_id == 'z'][0]
    rm_transcript_and_children(transcripty, controller.session)
    rm_transcript_and_children(transcriptz, controller.session)
    # slice
    controller.fill_intervaltrees()
    print('controller.sess', controller.session)
    slices = (('1', 1, 100, 'x01'), )
    controller._slice_annotations_1way(iter(slices), controller.get_one_annotated_genome(), is_plus_strand=True)
    controller.session.commit()
    old_len = len(controller.session.query(geenuff.orm.UpDownPair).all())
    print('used to be {} linkages'.format(old_len))
    controller._slice_annotations_1way(iter(slices), controller.get_one_annotated_genome(), is_plus_strand=True)
    controller.session.commit()
    assert old_len == len(controller.session.query(geenuff.orm.UpDownPair).all())


#### numerify ####
def test_sequence_numerify():
    sg = sequences.StructuredGenome()
    sg.from_json('testdata/tester.sequence.json')
    sequence = sg.sequences[0]
    slice0 = sequence.slices[0]
    numerifier = numerify.SequenceNumerifier()
    matrix = numerifier.slice_to_matrix(slice0, is_plus_strand=True)
    print(slice0.sequence)
    # ATATATAT, just btw
    x = [0., 1, 0, 0,
         0., 0, 1, 0]
    expect = np.array(x * 4).reshape([-1, 4])
    assert np.allclose(expect, matrix)


def setup4numerify():

    destination = 'testdata/tmp.db'
    if os.path.exists(destination):
        os.remove(destination)
    source = 'testdata/dummyloci_annotations.sqlitedb'

    controller = slicer.SliceController(db_path_in=source, db_path_sliced=destination,
                                        sequences_path='testdata/dummyloci.sequence.json')
    controller.mk_session()
    controller.load_annotations()
    # no need to modify, so can just load briefly
    sess = controller.session

    sinfo = sess.query(geenuff.orm.SequenceInfo).first()
    sinfo_h = slicer.SequenceInfoHandler()
    sinfo_h.add_data(sinfo)
    # simplify

    sinfo = sess.query(geenuff.orm.SequenceInfo).first()
    sinfo_h = slicer.SequenceInfoHandler()
    sinfo_h.add_data(sinfo)
    return sess, controller, sinfo_h


def test_base_level_annotation_numerify():
    sess, controller, sinfo_h = setup4numerify()

    numerifier = numerify.BasePairAnnotationNumerifier(data_slice=sinfo_h)
    with pytest.raises(numerify.DataInterpretationError):
        numerifier.slice_to_matrix(data_slice=sinfo_h, is_plus_strand=True)

    # simplify
    transcriptx = sess.query(geenuff.orm.Transcribed).filter(geenuff.orm.Transcribed.given_id == 'x').first()
    transcriptz = sess.query(geenuff.orm.Transcribed).filter(geenuff.orm.Transcribed.given_id == 'z').first()
    rm_transcript_and_children(transcriptx, sess)
    rm_transcript_and_children(transcriptz, sess)

    numerifier = numerify.BasePairAnnotationNumerifier(data_slice=sinfo_h)
    nums = numerifier.slice_to_matrix(data_slice=sinfo_h, is_plus_strand=True)
    expect = np.zeros([405, 3], dtype=float)
    expect[0:400, 0] = 1.  # set genic/in raw transcript
    expect[10:300, 1] = 1.  # set in transcribed
    expect[100:110, 2] = 1.  # both introns
    expect[120:200, 2] = 1.
    assert np.allclose(nums, expect)


def test_transition_annotation_numerify():
    sess, controller, sinfo_h = setup4numerify()

    numerifier = numerify.TransitionAnnotationNumerifier(data_slice=sinfo_h)
    with pytest.raises(numerify.DataInterpretationError):
        numerifier.slice_to_matrix(data_slice=sinfo_h, is_plus_strand=True)

    transcriptx = sess.query(geenuff.orm.Transcribed).filter(geenuff.orm.Transcribed.given_id == 'x').first()
    transcriptz = sess.query(geenuff.orm.Transcribed).filter(geenuff.orm.Transcribed.given_id == 'z').first()
    rm_transcript_and_children(transcriptx, sess)
    rm_transcript_and_children(transcriptz, sess)

    numerifier = numerify.TransitionAnnotationNumerifier(data_slice=sinfo_h)
    nums = numerifier.slice_to_matrix(data_slice=sinfo_h, is_plus_strand=True)
    expect = np.zeros([405, 12], dtype=float)
    expect[0, 0] = 1.  # TSS
    expect[400, 1] = 1.  # TTS
    expect[10, 4] = 1.  # start codon
    expect[300, 5] = 1.  # stop codon
    expect[(100, 120), 8] = 1.  # Don-splice
    expect[(110, 200), 9] = 1.  # Acc-splice
    assert np.allclose(nums, expect)


def test_numerify_from_gr0():
    sess, controller, sinfo_h = setup4numerify()
    transcriptx = sess.query(geenuff.orm.Transcribed).filter(geenuff.orm.Transcribed.given_id == 'x').first()
    transcriptz = sess.query(geenuff.orm.Transcribed).filter(geenuff.orm.Transcribed.given_id == 'z').first()
    rm_transcript_and_children(transcriptx, sess)
    rm_transcript_and_children(transcriptz, sess)

    x = geenuff.types.OnSequence(geenuff.types.TRANSCRIBED)
    print(x)
    tss = sess.query(geenuff.orm.Feature).filter(
        geenuff.orm.Feature.type == geenuff.types.OnSequence(geenuff.types.TRANSCRIBED)
    ).filter(
        geenuff.orm.Feature.bearing == geenuff.types.Bearings(geenuff.types.START)
    ).all()
    assert len(tss) == 1
    tss = tss[0]
    coords = sess.query(geenuff.orm.Coordinates).all()
    assert len(coords) == 1
    coords = coords[0]
    # move whole region back by 5 (was 0)
    tss.position = coords.start = 4
    # and now make sure it really starts form 4
    numerifier = numerify.TransitionAnnotationNumerifier(data_slice=sinfo_h)
    nums = numerifier.slice_to_matrix(data_slice=sinfo_h, is_plus_strand=True)
    # setup as above except for the change in position of TSS
    expect = np.zeros([405, 12], dtype=float)
    expect[4, 0] = 1.  # TSS
    expect[400, 1] = 1.  # TTS
    expect[10, 4] = 1.  # start codon
    expect[300, 5] = 1.  # stop codon
    expect[(100, 120), 8] = 1.  # Don-splice
    expect[(110, 200), 9] = 1.  # Acc-splice
    # truncate from start
    expect = expect[4:, :]
    assert np.allclose(nums, expect)

    # and now once for ranges
    numerifier = numerify.BasePairAnnotationNumerifier(data_slice=sinfo_h)
    nums = numerifier.slice_to_matrix(data_slice=sinfo_h, is_plus_strand=True)
    # as above (except TSS), then truncate
    expect = np.zeros([405, 3], dtype=float)
    expect[4:400, 0] = 1.  # set genic/in raw transcript
    expect[10:300, 1] = 1.  # set in transcribed
    expect[100:110, 2] = 1.  # both introns
    expect[120:200, 2] = 1.
    expect = expect[4:, :]
    assert np.allclose(nums, expect)


def setup_simpler_numerifier():
    sess = mk_session()
    ag = geenuff.orm.AnnotatedGenome()
    sinfo, sinfo_h = setup_data_handler(slicer.SequenceInfoHandler, geenuff.orm.SequenceInfo, annotated_genome=ag)
    coord = geenuff.orm.Coordinates(start=0, end=100, seqid='a', sequence_info=sinfo)
    sl = geenuff.orm.SuperLocus()
    transcript = geenuff.orm.Transcribed(super_locus=sl)
    piece = geenuff.orm.TranscribedPiece(transcribed=transcript)
    tss = geenuff.orm.Feature(position=40, is_plus_strand=False, type=geenuff.types.TRANSCRIBED,
                              bearing=geenuff.types.START, coordinates=coord, super_locus=sl)
    tts = geenuff.orm.Feature(position=9, is_plus_strand=False, type=geenuff.types.TRANSCRIBED,
                              bearing=geenuff.types.END, coordinates=coord, super_locus=sl)
    piece.features = [tss, tts]

    sess.add_all([ag, sinfo, coord, sl, transcript, piece, tss, tts])
    sess.commit()
    return sess, sinfo_h


def test_minus_strand_numerify():
    # setup a very basic -strand locus
    sess, sinfo_h = setup_simpler_numerifier()
    numerifier = numerify.BasePairAnnotationNumerifier(data_slice=sinfo_h)
    nums = numerifier.slice_to_matrix(data_slice=sinfo_h, is_plus_strand=True)
    # first, we should make sure the opposite strand is unmarked when empty
    expect = np.zeros([100, 3], dtype=float)
    assert np.allclose(nums, expect)

    # and now that we get the expect range on the minus strand, keeping in mind the 40 is inclusive, and the 9, not
    nums = numerifier.slice_to_matrix(data_slice=sinfo_h, is_plus_strand=False)
    expect[10:41, 0] = 1.
    assert np.allclose(nums, expect)


def test_live_slicing():
    sess, sinfo_h = setup_simpler_numerifier()
    # annotations by bp
    numerifier = numerify.BasePairAnnotationNumerifier(data_slice=sinfo_h)
    num_list = list(numerifier.slice_to_matrices(data_slice=sinfo_h, is_plus_strand=False, max_len=50))

    expect = np.zeros([100, 3], dtype=float)
    expect[10:41, 0] = 1.

    assert np.allclose(num_list[0], expect[0:50])
    assert np.allclose(num_list[1], expect[50:100])
    # sequences by bp
    sg = sequences.StructuredGenome()
    sg.from_json('testdata/dummyloci.sequence.json')
    sequence = sg.sequences[0]
    slice0 = sequence.slices[0]
    numerifier = numerify.SequenceNumerifier()
    num_list = list(numerifier.slice_to_matrices(data_slice=slice0, is_plus_strand=True, max_len=50))
    print([x.shape for x in num_list])
    # [(50, 4), (50, 4), (50, 4), (50, 4), (50, 4), (50, 4), (50, 4), (27, 4), (28, 4)]
    assert len(num_list) == 9
    for i in range(7):
        assert np.allclose(num_list[i], np.full([50, 4], 0.25, dtype=float))
    for i in [7, 8]:  # for the last two, just care that they're about the expected size...
        assert np.allclose(num_list[i][:27], np.full([27, 4], 0.25, dtype=float))


def test_example_gen():
    sess, controller, anno_slice = setup4numerify()
    transcriptx = sess.query(geenuff.orm.Transcribed).filter(geenuff.orm.Transcribed.given_id == 'x').first()
    transcriptz = sess.query(geenuff.orm.Transcribed).filter(geenuff.orm.Transcribed.given_id == 'z').first()
    rm_transcript_and_children(transcriptx, sess)
    rm_transcript_and_children(transcriptz, sess)

    controller.load_sliced_seqs()
    seq_slice = controller.structured_genome.sequences[0].slices[0]
    example_maker = numerify.ExampleMakerSeqMetaBP()
    egen = example_maker.examples_from_slice(anno_slice, seq_slice, controller.structured_genome, is_plus_strand=True,
                                             max_len=400)
    # prep anno
    expect = np.zeros([405, 3], dtype=float)
    expect[0:400, 0] = 1.  # set genic/in raw transcript
    expect[10:300, 1] = 1.  # set in transcribed
    expect[100:110, 2] = 1.  # both introns
    expect[120:200, 2] = 1.
    # prep seq
    seqexpect = np.full([405, 4], 0.25)

    step0 = next(egen)
    assert np.allclose(step0['input'].reshape([202, 4]), seqexpect[:202])
    assert np.allclose(step0['labels'].reshape([202, 3]), expect[:202])
    assert step0['meta_Gbp'] == [405 / 10**9]

    step1 = next(egen)
    assert np.allclose(step1['input'].reshape([203, 4]), seqexpect[202:])
    assert np.allclose(step1['labels'].reshape([203, 3]), expect[202:])
    assert step1['meta_Gbp'] == [405 / 10**9]


#### partitions
def test_stepper():
    # evenly divided
    s = partitions.Stepper(50, 10)
    strt_ends = list(s.step_to_end())
    assert len(strt_ends) == 5
    assert strt_ends[0] == (0, 10)
    assert strt_ends[-1] == (40, 50)
    # a bit short
    s = partitions.Stepper(49, 10)
    strt_ends = list(s.step_to_end())
    assert len(strt_ends) == 5
    assert strt_ends[-1] == (39, 49)
    # a bit long
    s = partitions.Stepper(52, 10)
    strt_ends = list(s.step_to_end())
    assert len(strt_ends) == 6
    assert strt_ends[-1] == (46, 52)
    # very short
    s = partitions.Stepper(9, 10)
    strt_ends = list(s.step_to_end())
    assert len(strt_ends) == 1
    assert strt_ends[-1] == (0, 9)


def test_id_maker():
    ider = helpers.IDMaker()
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
