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

from geenuff.tests.test_geenuff import (setup_data_handler,
                                        setup_testable_super_locus, TransspliceDemoData)


TMP_DB = 'testdata/tmp.db'
DUMMYLOCI_DB = 'testdata/dummyloci_annotations.sqlitedb'
SLICED_SEQ_PATH = 'testdata/dummyloci.sequence.sliced.json'


### helper functions ###
def construct_slice_controller(source=DUMMYLOCI_DB, dest=TMP_DB, sequences_path=None):
    if os.path.exists(dest):
        os.remove(dest)
    controller = slicer.SliceController(db_path_in=source,
                                        db_path_sliced=dest,
                                        sequences_path=sequences_path)
    controller.mk_session()
    controller.load_annotations()
    return controller


### preparation ###
@pytest.fixture(scope="session", autouse=True)
def setup_dummy_db(request):
    if os.path.exists(DUMMYLOCI_DB):
        os.remove(DUMMYLOCI_DB)
    sl, controller = setup_testable_super_locus('sqlite:///' + DUMMYLOCI_DB)
    coordinate = controller.genome_handler.data.coordinates[0]
    sl.check_and_fix_structure(coordinate=coordinate, controller=controller)
    controller.insertion_queues.execute_so_far()


def mk_session(db_path='sqlite:///:memory:'):
    engine = sqlalchemy.create_engine(db_path, echo=False)
    geenuff.orm.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session(), engine


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
    sess, engine = mk_session()
    genome = geenuff.orm.Genome()
    coordinate0, coordinate0_handler = setup_data_handler(slicer.CoordinateHandler, geenuff.orm.Coordinate,
                                                          genome=genome, start=0, end=100, seqid='a')
    coordinate0_set = slice_dbmods.CoordinateSet(processing_set='train', coordinate=coordinate0)

    coordinate1 = geenuff.orm.Coordinate(genome=genome, start=100, end=200, seqid='a')
    coordinate1_set = slice_dbmods.CoordinateSet(processing_set='train', coordinate=coordinate1)
    sess.add_all([genome, coordinate0_set, coordinate1_set])
    sess.commit()
    assert coordinate0_set.processing_set.value == 'train'
    assert coordinate1_set.processing_set.value == 'train'

    # make sure we can get the right info together back from the db
    maybe_join = sess.query(geenuff.orm.Coordinate, slice_dbmods.CoordinateSet).filter(
        geenuff.orm.Coordinate.id == slice_dbmods.CoordinateSet.id)
    for coord, coord_set in maybe_join.all():
        assert coord.id == coord_set.id
        assert coord_set.processing_set.value == 'train'

    # and make sure we can get the processing_set from the sequence_info
    coord_set = sess.query(slice_dbmods.CoordinateSet).filter(slice_dbmods.CoordinateSet.id == coordinate0.id).all()
    assert len(coord_set) == 1
    assert coord_set[0] == coordinate0_set

    # and over api
    coord_set = coordinate0_handler.processing_set(sess)
    assert coord_set is coordinate0_set
    assert coordinate0_handler.processing_set_val(sess) == 'train'
    # set over api
    coordinate0_handler.set_processing_set(sess, 'test')
    assert coordinate0_handler.processing_set_val(sess) == 'test'

    # confirm we can't have two processing sets per sequence_info
    with pytest.raises(sqlalchemy.orm.exc.FlushError):
        extra_set = slice_dbmods.CoordinateSet(processing_set='dev', coordinate=coordinate0)
        sess.add(extra_set)
        sess.commit()
    sess.rollback()
    assert coordinate0_handler.processing_set_val(sess) == 'test'

    # check that absence of entry, is handled with None
    coordinate2, coordinate2_handler = setup_data_handler(slicer.CoordinateHandler, geenuff.orm.Coordinate,
                                                          genome=genome, start=200, end=300, seqid='a')
    assert coordinate2_handler.processing_set(sess) is None
    assert coordinate2_handler.processing_set_val(sess) is None
    # finally setup complete new set via api
    coordinate2_handler.set_processing_set(sess, 'dev')
    assert coordinate2_handler.processing_set_val(sess) == 'dev'


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
    all_features = []
    for transcribed in sl.transcribeds:
        assert len(transcribed.transcribed_pieces) == 1
        piece = transcribed.transcribed_pieces[0]
        for feature in piece.features:
            all_features.append(feature)
        print('{}: {}'.format(transcribed.given_id, [x.type.value for x in piece.features]))
    for translated in sl.translateds:
        print('{}: {}'.format(translated.given_id, [x.type.value for x in translated.features]))

    assert len(all_features) == 12  # if I ever get to collapsing redundant features this will change


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
    for intv in controller.interval_trees["1"]:
        s = '{}:{}, {}'.format(intv.begin, intv.end, intv.data.data)
        print(s)
    # check that one known area has two errors, and one transcription termination site as expected
    intervals = controller.interval_trees['1'][399:406]
    assert len(intervals) == 3
    print(intervals, '...intervals')
    print([x.data.data.type.value for x in intervals])
    errors = [x for x in intervals if x.data.data.type.value == geenuff.types.ERROR]

    assert len(errors) == 2
    transcribeds = [x for x in intervals if x.data.data.type.value == geenuff.types.TRANSCRIBED]

    assert len(transcribeds) == 1
    # check that the major filter functions work
    sls = controller.get_super_loci_frm_slice(seqid='1', start=300, end=405, is_plus_strand=True)
    assert len(sls) == 1
    assert isinstance(list(sls)[0], slicer.SuperLocusHandler)

    features = controller.get_features_from_slice(seqid='1', start=0, end=1, is_plus_strand=True)
    assert len(features) == 3
    starts = [x for x in features if x.data.type.value == geenuff.types.TRANSCRIBED]

    assert len(starts) == 2
    errors = [x for x in features if x.data.type.value == geenuff.types.ERROR]
    assert len(errors) == 1


def test_order_features():
    controller = construct_slice_controller()
    sl = controller.super_loci[0]
    transcript = [x for x in sl.data.transcribeds if x.given_id == 'y'][0]
    transcripth = slicer.TranscribedHandler()
    transcripth.add_data(transcript)
    ti = slicer.TranscriptTrimmer(transcripth, super_locus=None, sess=controller.session,
                                  core_queue=controller.core_queue)
    assert len(transcript.transcribed_pieces) == 1
    piece = transcript.transcribed_pieces[0]
    # features expected to be ordered by increasing position (note: as they are in db)
    ordered_starts = [0, 10, 100, 120]
    features = ti.sorted_features(piece)
    for f in features:
        print(f)
    assert [x.start for x in features] == ordered_starts
    # force erroneous data
    for feature in piece.features:
        feature.is_plus_strand = False
    piece.features[0].is_plus_strand = True
    controller.session.add(piece.features[0])
    controller.session.commit()
    with pytest.raises(AssertionError):
        ti.sorted_features(piece)
    # todo, test s.t. on - strand? (or is this anyways redundant with geenuff?


# todo, we don't actually need slicer for this, mv to test_geenuff
def test_slicer_transition():
    controller = construct_slice_controller()
    sl = controller.super_loci[0]
    transcript = [x for x in sl.data.transcribeds if x.given_id == 'y'][0]
    transcripth = slicer.TranscribedHandler()
    transcripth.add_data(transcript)
    ti = slicer.TranscriptTrimmer(transcript=transcripth, super_locus=None, sess=controller.session,
                                  core_queue=controller.core_queue)
    transition_gen = ti.transition_5p_to_3p()
    transitions = list(transition_gen)
    assert len(transitions) == 4
    pieces = [x[1] for x in transitions]
    features = [x[0][0] for x in transitions]
    #ordered_starts = [0, 10, 100, 110, 120, 200, 300, 400]
    ordered_starts = [0, 10, 100, 120]
    assert [x.start for x in features] == ordered_starts
    expected_types = [geenuff.types.TRANSCRIBED, geenuff.types.CODING, geenuff.types.INTRON, geenuff.types.INTRON]
    assert [x.type.value for x in features] == expected_types
    assert all([x.position == 0 for x in pieces])


def test_split_features_downstream_border():
    sess, engine = mk_session()
    core_queue = slicer.CoreQueue(session=sess, engine=engine)
    genome = geenuff.orm.Genome()
    old_coor = geenuff.orm.Coordinate(genome=genome, seqid='a', start=1, end=1000)
    new_coord = geenuff.orm.Coordinate(genome=genome, seqid='a', start=100, end=200)
    sl, slh = setup_data_handler(slicer.SuperLocusHandler, geenuff.orm.SuperLocus)
    scribed, scribedh = setup_data_handler(slicer.TranscribedHandler, geenuff.orm.Transcribed, super_locus=sl)
    ti = slicer.TranscriptTrimmer(transcript=scribedh, super_locus=slh, sess=sess, core_queue=core_queue)
    # start   -> 5' [------piece0----------] 3'
    # expect  -> 5' [--piece0--][--piece1--] 3'
    piece0 = geenuff.orm.TranscribedPiece(position=0, transcribed=scribed)
    piece1 = geenuff.orm.TranscribedPiece(position=1, transcribed=scribed)
    # setup some paired features
    # new coords, is plus, template, status
    coding_feature = geenuff.orm.Feature(transcribed_pieces=[piece0], coordinate=old_coor, start=110, end=209,
                                         is_plus_strand=True, type=geenuff.types.CODING, start_is_biological_start=True,
                                         end_is_biological_end=True)
    transcribed_feature = geenuff.orm.Feature(transcribed_pieces=[piece0], coordinate=old_coor, start=100, end=300,
                                              is_plus_strand=True, type=geenuff.types.TRANSCRIBED,
                                              start_is_biological_start=True, end_is_biological_end=True)
    sess.add_all([scribed, piece0, piece1, old_coor, new_coord, sl])
    sess.commit()
    slh.make_all_handlers()

    ti.set_status_downstream_border(new_coords=new_coord, is_plus_strand=True, template_feature=coding_feature,
                                    old_piece=piece0, new_piece=piece1, old_coords=old_coor, trees={})
    ti.set_status_downstream_border(new_coords=new_coord, is_plus_strand=True, template_feature=transcribed_feature,
                                    old_piece=piece0, new_piece=piece1, old_coords=old_coor, trees={})

    sess.commit()
    assert len(piece0.features) == 2
    assert len(piece1.features) == 2

    assert set([x.type.value for x in piece0.features]) == {geenuff.types.CODING, geenuff.types.TRANSCRIBED}
    assert set([x.start_is_biological_start for x in piece0.features]) == {True}
    assert set([x.end_is_biological_end for x in piece0.features]) == {False}

    assert set([x.type.value for x in piece1.features]) == {geenuff.types.CODING, geenuff.types.TRANSCRIBED}
    assert set([x.start_is_biological_start for x in piece1.features]) == {False}
    assert set([x.end_is_biological_end for x in piece1.features]) == {True}

    translated0 = [x for x in piece0.features if x.type.value == geenuff.types.CODING][0]

    translated1 = [x for x in piece1.features if x.type.value == geenuff.types.CODING][0]
    assert translated0.end == 200
    assert translated1.start == 200
    # cleanup to try similar again
    for f in piece0.features:
        sess.delete(f)
    for f in piece1.features:
        sess.delete(f)
    sess.commit()

    # and now try backwards pass
    coding_feature = geenuff.orm.Feature(transcribed_pieces=[piece0], coordinate=old_coor, start=110, end=90,
                                         is_plus_strand=False, type=geenuff.types.CODING,
                                         start_is_biological_start=True, end_is_biological_end=True)
    transcribed_feature = geenuff.orm.Feature(transcribed_pieces=[piece0], coordinate=old_coor, start=130, end=80,
                                              is_plus_strand=False, type=geenuff.types.TRANSCRIBED,
                                              start_is_biological_start=True, end_is_biological_end=True)

    sess.add_all([coding_feature, transcribed_feature])
    sess.commit()
    slh.make_all_handlers()
    ti.set_status_downstream_border(new_coords=new_coord, is_plus_strand=False, template_feature=coding_feature,
                                    old_piece=piece0, new_piece=piece1, old_coords=old_coor, trees={})
    ti.set_status_downstream_border(new_coords=new_coord, is_plus_strand=False, template_feature=transcribed_feature,
                                    old_piece=piece0, new_piece=piece1, old_coords=old_coor, trees={})
    sess.commit()

    assert len(piece0.features) == 2
    assert len(piece1.features) == 2
    assert set([x.type.value for x in piece0.features]) == {geenuff.types.CODING, geenuff.types.TRANSCRIBED}
    assert set([(x.start_is_biological_start, x.end_is_biological_end) for x in piece0.features]) == \
        {(True, False)}

    assert set([x.type.value for x in piece1.features]) == {geenuff.types.CODING, geenuff.types.TRANSCRIBED}
    assert set([(x.start_is_biological_start, x.end_is_biological_end) for x in piece1.features]) == \
        {(False, True)}

    translated0 = [x for x in piece0.features if x.type.value == geenuff.types.CODING][0]
    translated1 = [x for x in piece1.features if x.type.value == geenuff.types.CODING][0]

    assert translated1.start == 99
    assert translated0.end == 99


def test_modify4slice():
    controller = construct_slice_controller()
    slh = controller.super_loci[0]
    transcript = [x for x in slh.data.transcribeds if x.given_id == 'y'][0]
    slh.make_all_handlers()
    genome = controller.get_one_annotated_genome()
    ti = slicer.TranscriptTrimmer(transcript=transcript.handler, super_locus=slh, sess=controller.session,
                                  core_queue=controller.core_queue)
    new_coords = geenuff.orm.Coordinate(genome=genome, seqid='1', start=0, end=100)
    newer_coords = geenuff.orm.Coordinate(genome=genome, seqid='1', start=100, end=200)
    controller.session.add_all([new_coords, newer_coords])
    controller.session.commit()
    ti.modify4new_slice(new_coords=new_coords, is_plus_strand=True)
    controller.core_queue.execute_so_far()
    assert len(transcript.transcribed_pieces) == 2
    print(transcript.transcribed_pieces[0])
    for feature in transcript.transcribed_pieces[1].features:
        print('-- {} --\n'.format(feature))
    print(transcript.transcribed_pieces[1])
    assert {len(transcript.transcribed_pieces[0].features), len(transcript.transcribed_pieces[1].features)} == {4, 2}
    upstream_piece = [x for x in transcript.transcribed_pieces if len(x.features) == 2][0]

    assert set([x.type.value for x in upstream_piece.features]) == {geenuff.types.TRANSCRIBED,
                                                                    geenuff.types.CODING}

    assert set([(x.start_is_biological_start, x.end_is_biological_end) for x in upstream_piece.features]) == \
        {(True, False)}

    print('starting second modify...')
    ti.modify4new_slice(new_coords=newer_coords, is_plus_strand=True)
    controller.core_queue.execute_so_far()
    for piece in transcript.transcribed_pieces:
        print(piece)
        for f in piece.features:
            print('::::', (f.type.value, f.start_is_biological_start, f.end_is_biological_end, f.start, f.end))

    lengths = sorted([len(x.features) for x in transcript.transcribed_pieces])
    assert lengths == [2, 2, 4]
    last_piece = ti.sort_pieces()[-1]
    assert set([x.type.value for x in last_piece.features]) == {geenuff.types.TRANSCRIBED,
                                                                geenuff.types.CODING}
    assert set([(x.start_is_biological_start, x.end_is_biological_end) for x in last_piece.features]) == \
        {(False, True)}


def test_modify4slice_directions():
    sess, engine = mk_session()

    core_queue = slicer.CoreQueue(session=sess, engine=engine)

    genome = geenuff.orm.Genome()
    old_coor = geenuff.orm.Coordinate(genome=genome, seqid='a', start=1, end=1000)
    # setup a transitions (where a transcript is split for technical reasons, e.g. mis assembly):
    # 2) scribedlong - [[DC<-],[->AB]] -> ABCD, -> two pieces forward, one backward
    sl, slh = setup_data_handler(slicer.SuperLocusHandler, geenuff.orm.SuperLocus)
    scribedlong, scribedlongh = setup_data_handler(slicer.TranscribedHandler, geenuff.orm.Transcribed,
                                                   super_locus=sl)

    tilong = slicer.TranscriptTrimmer(transcript=scribedlongh, super_locus=slh, sess=sess, core_queue=core_queue)

    pieceAB = geenuff.orm.TranscribedPiece(transcribed=scribedlong, position=0)
    pieceCD = geenuff.orm.TranscribedPiece(transcribed=scribedlong, position=1)

    fAB = geenuff.orm.Feature(transcribed_pieces=[pieceAB], coordinate=old_coor, start=190, end=210, given_id='AB',
                              start_is_biological_start=True, end_is_biological_end=False,
                              is_plus_strand=True, type=geenuff.types.TRANSCRIBED)

    fCD = geenuff.orm.Feature(transcribed_pieces=[pieceCD], coordinate=old_coor, start=110, end=90,
                              start_is_biological_start=False, end_is_biological_end=True,
                              is_plus_strand=False, type=geenuff.types.TRANSCRIBED, given_id='CD')

    half1_coords = geenuff.orm.Coordinate(genome=genome, seqid='a', start=1, end=200)
    half2_coords = geenuff.orm.Coordinate(genome=genome, seqid='a', start=200, end=400)
    sess.add_all([scribedlong, pieceAB, pieceCD, fAB, fCD, old_coor, sl, half1_coords, half2_coords])
    sess.commit()
    slh.make_all_handlers()

    tilong.modify4new_slice(new_coords=half1_coords, is_plus_strand=True)
    core_queue.execute_so_far()
    sess.commit()
    tilong.modify4new_slice(new_coords=half2_coords, is_plus_strand=True)
    core_queue.execute_so_far()
    tilong.modify4new_slice(new_coords=half1_coords, is_plus_strand=False)
    core_queue.execute_so_far()
    for f in sess.query(geenuff.orm.Feature).all():
        assert len(f.transcribed_pieces) == 1
    slices = scribedlong.transcribed_pieces
    slice0 = [x for x in slices if x.position == 0][0]
    slice1 = [x for x in slices if x.position == 1][0]
    slice2 = [x for x in slices if x.position == 2][0]
    assert [len(x.features) for x in tilong.transcript.data.transcribed_pieces] == [1, 1, 1]
    # expect slice 0: fAB, truncated
    assert len(slice0.features) == 1
    assert slice0.features[0] is fAB
    assert fAB.end == 200
    assert fAB.start_is_biological_start is True
    assert fAB.end_is_biological_end is False
    # expect slice1: new feature from fAB, second half
    assert len(slice1.features) == 1
    assert slice1.features[0].start == 200
    assert slice1.features[0].end == 210  # old end of fAB
    assert slice1.features[0].start_is_biological_start is False
    assert slice1.features[0].end_is_biological_end is False  # was False for fAB as well
    # expect slice2: same reverse piece as always, unchanged
    assert slice2.features == [fCD]
    assert slice2 is pieceCD


class TransspliceDemoDataSlice(TransspliceDemoData):
    def __init__(self, sess, engine):
        super().__init__(sess)
        self.core_queue = slicer.CoreQueue(session=sess, engine=engine)
        self.genome = geenuff.orm.Genome()
        self.old_coor = geenuff.orm.Coordinate(genome=self.genome, seqid='a', start=1, end=2000)
        # replace handlers with those from slicer
        self.slh = slicer.SuperLocusHandler()
        self.slh.add_data(self.sl)

        self.scribedh = slicer.TranscribedHandler()
        self.scribedh.add_data(self.scribed)

        self.scribedfliph = slicer.TranscribedHandler()
        self.scribedfliph.add_data(self.scribedflip)

        self.ti = slicer.TranscriptTrimmer(transcript=self.scribedh, super_locus=self.slh, sess=sess,
                                           core_queue=self.core_queue)
        self.tiflip = slicer.TranscriptTrimmer(transcript=self.scribedfliph, super_locus=self.slh, sess=sess,
                                               core_queue=self.core_queue)


class SimplestDemoData(object):
    def __init__(self, sess, engine, genome=None):
        self.core_queue = slicer.CoreQueue(session=sess, engine=engine)
        if genome is None:
            self.genome = geenuff.orm.Genome()
        else:
            self.genome = genome
        self.old_coor = geenuff.orm.Coordinate(seqid='a', start=0, end=1000, genome=self.genome)
        # setup 1 transition (encoding e.g. a transcript split across a miss assembled scaffold...):
        # 2) scribedlong - [[CD<-],[->AB]] -> ABCD, -> two transcribed pieces: one forward, one backward
        self.sl, self.slh = setup_data_handler(slicer.SuperLocusHandler, geenuff.orm.SuperLocus)
        self.scribedlong, self.scribedlongh = setup_data_handler(slicer.TranscribedHandler, geenuff.orm.Transcribed,
                                                                 super_locus=self.sl)

        self.tilong = slicer.TranscriptTrimmer(transcript=self.scribedlongh, super_locus=self.slh, sess=sess,
                                               core_queue=self.core_queue)

        self.pieceAB = geenuff.orm.TranscribedPiece(position=0)
        self.pieceCD = geenuff.orm.TranscribedPiece(position=1)
        self.scribedlong.transcribed_pieces = [self.pieceAB, self.pieceCD]

        self.fAB = geenuff.orm.Feature(transcribed_pieces=[self.pieceAB], coordinate=self.old_coor, start=190, end=210,
                                       start_is_biological_start=True, end_is_biological_end=False,
                                       given_id='AB', is_plus_strand=True, type=geenuff.types.TRANSCRIBED)

        self.fCD = geenuff.orm.Feature(transcribed_pieces=[self.pieceCD], coordinate=self.old_coor,
                                       is_plus_strand=False, start=110, end=90,
                                       start_is_biological_start=False, end_is_biological_end=True,
                                       type=geenuff.types.TRANSCRIBED, given_id='CD')

        self.pieceAB.features.append(self.fAB)
        self.pieceCD.features.append(self.fCD)

        sess.add_all([self.scribedlong, self.pieceAB, self.pieceCD, self.fAB, self.fCD,
                      self.old_coor, self.sl])
        sess.commit()
        self.slh.make_all_handlers()


def test_transition_unused_coordinates_detection():
    sess, engine = mk_session()
    d = SimplestDemoData(sess, engine)
    # modify to coordinates with complete contain, should work fine
    genome = d.genome
    new_coords = geenuff.orm.Coordinate(seqid='a', start=0, end=300, genome=genome)
    sess.add(new_coords)
    sess.commit()
    d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=True)
    d.core_queue.execute_so_far()
    d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=False)
    d.core_queue.execute_so_far()
    assert d.pieceCD in d.scribedlong.transcribed_pieces  # should now keep original at start
    assert d.pieceAB in d.scribedlong.transcribed_pieces
    # modify to coordinates across tiny slice, include those w/o original features, should work fine
    d = SimplestDemoData(sess, engine)
    new_coords_list = [geenuff.orm.Coordinate(genome=genome, seqid='a', start=185, end=195),
                       geenuff.orm.Coordinate(genome=genome, seqid='a', start=195, end=205),
                       geenuff.orm.Coordinate(genome=genome, seqid='a', start=205, end=215)]

    for new_coords in new_coords_list:
        sess.add(new_coords)
        sess.commit()
        print('fw {}, {}'.format(new_coords.id, new_coords))
        d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=True)
        d.core_queue.execute_so_far()
        print([x.id for x in d.tilong.transcript.data.transcribed_pieces])

    new_coords_list = [geenuff.orm.Coordinate(genome=genome, seqid='a', start=105, end=115),
                       geenuff.orm.Coordinate(genome=genome, seqid='a', start=95, end=105),
                       geenuff.orm.Coordinate(genome=genome, seqid='a', start=85, end=95)]
    for new_coords in new_coords_list:
        sess.add(new_coords)
        sess.commit()
        print('\nstart mod for coords, - strand', new_coords)
        d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=False)
        d.core_queue.execute_so_far()

    assert d.pieceCD in d.scribedlong.transcribed_pieces
    assert d.pieceAB in d.scribedlong.transcribed_pieces

    # try and slice before coordinates, should raise error
    d = SimplestDemoData(sess, engine)
    new_coords = geenuff.orm.Coordinate(genome=genome, seqid='a', start=0, end=10)
    sess.add(new_coords)
    sess.commit()
    with pytest.raises(slicer.NoFeaturesInSliceError):
        d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=True)
    # try and slice after coordinates, should raise error
    d = SimplestDemoData(sess, engine)
    new_coords = geenuff.orm.Coordinate(genome=genome, seqid='a', start=399, end=410)
    sess.add(new_coords)
    sess.commit()
    with pytest.raises(slicer.NoFeaturesInSliceError):
        d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=True)
    # try and slice between slices where there are no coordinates, should raise error
    d = SimplestDemoData(sess, engine)
    new_coords = geenuff.orm.Coordinate(genome=genome, seqid='a', start=149, end=160)
    sess.add(new_coords)
    sess.commit()
    with pytest.raises(slicer.NoFeaturesInSliceError):
        d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=True)
    d = SimplestDemoData(sess, engine)
    with pytest.raises(slicer.NoFeaturesInSliceError):
        d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=False)


def test_features_on_opposite_strand_are_not_modified():
    sess, engine = mk_session()
    d = SimplestDemoData(sess, engine)

    # forward pass only, back pass should be untouched (and + coordinates unaffected)
    new_coords = geenuff.orm.Coordinate(genome=d.genome, seqid='a', start=100, end=1000)
    d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=True)
    d.core_queue.execute_so_far()
    assert len(d.pieceAB.features) == 1
    assert len(d.pieceCD.features) == 1

    d = SimplestDemoData(sess, engine)
    # backward pass only (no coord hit), plus pass should be untouched
    new_coords = geenuff.orm.Coordinate(genome=d.genome, seqid='a', start=1, end=200)
    d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=False)
    d.core_queue.execute_so_far()

    assert len(d.pieceAB.features) == 1
    assert len(d.pieceCD.features) == 1


def test_modify4slice_transsplice():
    sess, engine = mk_session()
    d = TransspliceDemoDataSlice(sess, engine)  # setup _d_ata
    d.make_all_handlers()
    new_coords_0 = geenuff.orm.Coordinate(seqid='a', start=0, end=915, genome=d.genome)
    new_coords_1 = geenuff.orm.Coordinate(seqid='a', start=915, end=2000, genome=d.genome)
    sess.add_all([new_coords_1, new_coords_0])
    sess.commit()
    d.ti.modify4new_slice(new_coords=new_coords_0, is_plus_strand=True)
    d.core_queue.execute_so_far()
    d.ti.modify4new_slice(new_coords=new_coords_1, is_plus_strand=True)
    d.core_queue.execute_so_far()
    # we expect 3 new pieces,
    #    1: TSS-start-DonorTranssplice-TTS via <2x status> to (6 features)
    #    2: <2x status>-TSS- via <3x status> to (6 features)
    #    3: <3x status>-AcceptorTranssplice-stop-TTS (6 features)
    pieces = d.ti.transcript.data.transcribed_pieces
    assert len(pieces) == 3
    assert d.pieceA2C in pieces  # ori piece should not have been replaced
    sorted_pieces = d.ti.sort_pieces()

    assert [len(x.features) for x in sorted_pieces] == [3, 3, 3]
    ftypes_0 = set([(x.type.value, x.start_is_biological_start, x.end_is_biological_end)
                    for x in sorted_pieces[0].features])
    assert ftypes_0 == {(geenuff.types.TRANSCRIBED, True, True),
                        (geenuff.types.CODING, True, False),
                        (geenuff.types.TRANS_INTRON, True, False)}

    ftypes_2 = set([(x.type.value, x.start_is_biological_start, x.end_is_biological_end)
                    for x in sorted_pieces[2].features])
    assert ftypes_2 == {(geenuff.types.CODING, False, True),
                        (geenuff.types.TRANSCRIBED, False, True),
                        (geenuff.types.TRANS_INTRON, False, True)}

    # and now where second original piece is flipped and slice is thus between STOP and TTS
    print('moving on to flipped...')
    d.tiflip.modify4new_slice(new_coords=new_coords_0, is_plus_strand=True)
    d.core_queue.execute_so_far()
    assert len(d.tiflip.transcript.data.transcribed_pieces) == 2
    d.tiflip.modify4new_slice(new_coords=new_coords_1, is_plus_strand=False)
    d.core_queue.execute_so_far()
    d.tiflip.modify4new_slice(new_coords=new_coords_0, is_plus_strand=False)
    d.core_queue.execute_so_far()
    # we expect 3 new pieces,
    #    1: TSS-start-DonorTranssplice-TTS via <2x status> to (6 features)
    #    2: <2x status>-TSS-AcceptorTranssplice-stop via <1x status> to (6 features)
    #    3: <1x status>-TTS (2 features)
    pieces = d.tiflip.transcript.data.transcribed_pieces
    assert len(pieces) == 3
    assert d.pieceA2C_prime in pieces
    sorted_pieces = d.tiflip.sort_pieces()

    assert [len(x.features) for x in sorted_pieces] == [3, 3, 1]
    ftypes_0 = set([(x.type.value, x.start_is_biological_start, x.end_is_biological_end)
                    for x in sorted_pieces[0].features])
    assert ftypes_0 == {(geenuff.types.TRANSCRIBED, True, True),
                        (geenuff.types.CODING, True, False),
                        (geenuff.types.TRANS_INTRON, True, False)}
    ftypes_2 = set([(x.type.value, x.start_is_biological_start, x.end_is_biological_end)
                    for x in sorted_pieces[2].features])
    assert ftypes_2 == {(geenuff.types.TRANSCRIBED, False, True)}
    for piece in sorted_pieces:
        for f in piece.features:
            assert f.coordinate in {new_coords_1, new_coords_0}


def test_modify4slice_2nd_half_first():
    # because trans-splice occasions can theoretically hit transitions in the 'wrong' order where the 1st half of
    # the _final_ transcript hasn't been adjusted when the second half is adjusted/sliced. Results should be the same.
    sess, engine = mk_session()
    d = TransspliceDemoDataSlice(sess, engine)  # setup _d_ata
    d.make_all_handlers()
    new_coords_0 = geenuff.orm.Coordinate(genome=d.genome, seqid='a', start=0, end=915)
    new_coords_1 = geenuff.orm.Coordinate(genome=d.genome, seqid='a', start=915, end=2000)
    sess.add_all([new_coords_0, new_coords_1])
    sess.commit()
    d.tiflip.modify4new_slice(new_coords=new_coords_1, is_plus_strand=False)
    d.core_queue.execute_so_far()
    for piece in d.scribedflip.transcribed_pieces:
        print(">>> {}".format(piece))
        for feature in piece.features:
            print('     {}'.format(feature))
    d.tiflip.modify4new_slice(new_coords=new_coords_0, is_plus_strand=False)
    d.core_queue.execute_so_far()
    d.tiflip.modify4new_slice(new_coords=new_coords_0, is_plus_strand=True)
    d.core_queue.execute_so_far()
    # we expect to now have 3 pieces,
    #    1: TSS-start-DonorTranssplice-TTS via <2x status> to (3 features)
    #    2: <2x status>-TSS-AcceptorTranssplice-stop via <1x status> to (3 features)
    #    3: <1x status>-TTS (1 feature)
    pieces = d.tiflip.transcript.data.transcribed_pieces
    assert len(pieces) == 3
    assert d.pieceA2C_prime in pieces  # old pieces should be retained
    sorted_pieces = d.tiflip.sort_pieces()

    assert [len(x.features) for x in sorted_pieces] == [3, 3, 1]
    ftypes_0 = set([(x.type.value, x.start_is_biological_start, x.end_is_biological_end)
                    for x in sorted_pieces[0].features])
    assert ftypes_0 == {(geenuff.types.TRANSCRIBED, True, True),
                        (geenuff.types.CODING, True, False),
                        (geenuff.types.TRANS_INTRON, True, False)}
    ftypes_2 = [(x.type.value, x.start_is_biological_start, x.end_is_biological_end)
                    for x in sorted_pieces[2].features][0]
    assert ftypes_2 == (geenuff.types.TRANSCRIBED, False, True)

    for piece in sorted_pieces:
        for f in piece.features:
            assert f.coordinate in {new_coords_1, new_coords_0}


def test_slicing_multi_sl():
    # TODO, add sliced sequences path
    controller = construct_slice_controller(sequences_path=SLICED_SEQ_PATH)
    controller.load_sliced_seqs()
    controller.fill_intervaltrees()
    slh = controller.super_loci[0]
    slh.make_all_handlers()
    # setup more
    more = SimplestDemoData(controller.session, controller.engine, genome=controller.get_one_annotated_genome())
    controller.super_loci.append(more.slh)
    more.old_coor.seqid = '1'  # so it matches std dummyloci
    controller.session.commit()
    # and try and slice
    controller.slice_annotations(controller.get_one_annotated_genome())
    # todo, test if valid pass of final res.


def test_slicing_featureless_slice_inside_locus():
    controller = construct_slice_controller()
    controller.fill_intervaltrees()
    genome = controller.get_one_annotated_genome()
    slh = controller.super_loci[0]
    transcript = [x for x in slh.data.transcribeds if x.given_id == 'y'][0]
    slices = (('1', 0, 40, '0-40'),
              ('1', 40, 80, '40-80'),
              ('1', 80, 120, '80-120'))
    slices = iter(slices)
    controller._slice_annotations_1way(slices, genome=genome, is_plus_strand=True)

    for piece in transcript.transcribed_pieces:
        print('got piece: {}\n-----------\n'.format(piece))
        for feature in piece.features:
            print('    {}'.format(feature))
    coordinate40 = controller.session.query(geenuff.orm.Coordinate).filter(
        geenuff.orm.Coordinate.start == 40
    ).first()
    features40 = coordinate40.features
    print(features40)

    # x & y -> 2 translated, 2 transcribed each, z -> 2 error
    assert len([x for x in features40 if x.type.value == geenuff.types.CODING]) == 2
    assert len([x for x in features40 if x.type.value == geenuff.types.TRANSCRIBED]) == 2
    assert len(features40) == 5
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
    controller = construct_slice_controller(sequences_path=SLICED_SEQ_PATH)
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
    slices = (('1', 0, 100, 'x01'), )
    controller._slice_annotations_1way(iter(slices), controller.get_one_annotated_genome(), is_plus_strand=True)
    controller.session.commit()
    old_len = len(controller.session.query(geenuff.orm.TranscribedPiece).all())
    print('used to be {} linkages'.format(old_len))
    controller._slice_annotations_1way(iter(slices), controller.get_one_annotated_genome(), is_plus_strand=True)
    controller.session.commit()
    assert old_len == len(controller.session.query(geenuff.orm.TranscribedPiece).all())


#### numerify ####
def test_sequence_numerify():
    sg = sequences.StructuredGenome()
    sg.from_json('testdata/tester.sequence.json')
    sequence = sg.sequences[0]
    slice0 = sequence.slices[0]
    numerifier = numerify.SequenceNumerifier(is_plus_strand=True)
    matrix = numerifier.slice_to_matrix(slice0)
    print(slice0.sequence)
    # ATATATAT, just btw
    x = [0., 1, 0, 0,
         0., 0, 1, 0]
    expect = np.array(x * 4).reshape([-1, 4])
    assert np.allclose(expect, matrix)


def setup4numerify():
    controller = construct_slice_controller(sequences_path='testdata/dummyloci.sequence.json')
    # no need to modify, so can just load briefly
    sess = controller.session

    coordinate = sess.query(geenuff.orm.Coordinate).first()
    coordinate_handler = slicer.CoordinateHandler()
    coordinate_handler.add_data(coordinate)

    return sess, controller, coordinate_handler


def test_base_level_annotation_numerify():
    sess, controller, coordinate_handler = setup4numerify()

    numerifier = numerify.BasePairAnnotationNumerifier(data_slice=coordinate_handler, is_plus_strand=True)
    with pytest.raises(numerify.DataInterpretationError):
        numerifier.slice_to_matrix()

    # simplify
    transcriptx = sess.query(geenuff.orm.Transcribed).filter(geenuff.orm.Transcribed.given_id == 'x').first()
    transcriptz = sess.query(geenuff.orm.Transcribed).filter(geenuff.orm.Transcribed.given_id == 'z').first()
    rm_transcript_and_children(transcriptx, sess)
    rm_transcript_and_children(transcriptz, sess)

    transcribeds = sess.query(geenuff.orm.Transcribed).all()
    print(transcribeds, ' <- transcribeds')

    numerifier = numerify.BasePairAnnotationNumerifier(data_slice=coordinate_handler, is_plus_strand=True)
    nums = numerifier.slice_to_matrix()
    expect = np.zeros([405, 3], dtype=float)
    expect[0:400, 0] = 1.  # set genic/in raw transcript
    expect[10:300, 1] = 1.  # set in transcribed
    expect[100:110, 2] = 1.  # both introns
    expect[120:200, 2] = 1.
    assert np.allclose(nums, expect)


def test_transition_annotation_numerify():
    sess, controller, coordinate_handler = setup4numerify()

    numerifier = numerify.TransitionAnnotationNumerifier(data_slice=coordinate_handler, is_plus_strand=True)
    with pytest.raises(numerify.DataInterpretationError):
        numerifier.slice_to_matrix()

    transcriptx = sess.query(geenuff.orm.Transcribed).filter(geenuff.orm.Transcribed.given_id == 'x').first()
    transcriptz = sess.query(geenuff.orm.Transcribed).filter(geenuff.orm.Transcribed.given_id == 'z').first()
    rm_transcript_and_children(transcriptx, sess)
    rm_transcript_and_children(transcriptz, sess)

    numerifier = numerify.TransitionAnnotationNumerifier(data_slice=coordinate_handler, is_plus_strand=True)
    nums = numerifier.slice_to_matrix()
    expect = np.zeros([405, 12], dtype=float)
    expect[0, 0] = 1.  # TSS
    expect[400, 1] = 1.  # TTS
    expect[10, 4] = 1.  # start codon
    expect[300, 5] = 1.  # stop codon
    expect[(100, 120), 8] = 1.  # Don-splice
    expect[(110, 200), 9] = 1.  # Acc-splice
    assert np.allclose(nums, expect)


def test_numerify_from_gr0():
    sess, controller, coordinate_handler = setup4numerify()
    transcriptx = sess.query(geenuff.orm.Transcribed).filter(geenuff.orm.Transcribed.given_id == 'x').first()
    transcriptz = sess.query(geenuff.orm.Transcribed).filter(geenuff.orm.Transcribed.given_id == 'z').first()
    rm_transcript_and_children(transcriptx, sess)
    rm_transcript_and_children(transcriptz, sess)

    transcribed = sess.query(geenuff.orm.Feature).filter(
        geenuff.orm.Feature.type == geenuff.types.OnSequence(geenuff.types.TRANSCRIBED)
    ).all()
    assert len(transcribed) == 1
    transcribed = transcribed[0]
    coord = coordinate_handler.data
    # move whole region back by 5 (was 0)
    transcribed.start = coord.start = 4
    # and now make sure it really starts form 4
    numerifier = numerify.TransitionAnnotationNumerifier(data_slice=coordinate_handler, is_plus_strand=True)
    nums = numerifier.slice_to_matrix()
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
    numerifier = numerify.BasePairAnnotationNumerifier(data_slice=coordinate_handler, is_plus_strand=True)
    nums = numerifier.slice_to_matrix()
    # as above (except TSS), then truncate
    expect = np.zeros([405, 3], dtype=float)
    expect[4:400, 0] = 1.  # set genic/in raw transcript
    expect[10:300, 1] = 1.  # set in transcribed
    expect[100:110, 2] = 1.  # both introns
    expect[120:200, 2] = 1.
    expect = expect[4:, :]
    assert np.allclose(nums, expect)


def setup_simpler_numerifier():
    sess, engine = mk_session()
    genome = geenuff.orm.Genome()
    coord, coord_handler = setup_data_handler(slicer.CoordinateHandler, geenuff.orm.Coordinate, genome=genome,
                                              start=0, end=100, seqid='a')
    sl = geenuff.orm.SuperLocus()
    transcript = geenuff.orm.Transcribed(super_locus=sl)
    piece = geenuff.orm.TranscribedPiece(transcribed=transcript, position=0)
    transcribed_feature = geenuff.orm.Feature(start=40, end=9,  is_plus_strand=False, type=geenuff.types.TRANSCRIBED,
                                              start_is_biological_start=True, end_is_biological_end=True,
                                              coordinate=coord)
    piece.features = [transcribed_feature]

    sess.add_all([genome, coord, sl, transcript, piece, transcribed_feature])
    sess.commit()
    return sess, coord_handler


def test_minus_strand_numerify():
    # setup a very basic -strand locus
    sess, coordinate_handler = setup_simpler_numerifier()
    numerifier = numerify.BasePairAnnotationNumerifier(data_slice=coordinate_handler, is_plus_strand=True)
    nums = numerifier.slice_to_matrix()
    # first, we should make sure the opposite strand is unmarked when empty
    expect = np.zeros([100, 3], dtype=float)
    assert np.allclose(nums, expect)

    numerifier = numerify.BasePairAnnotationNumerifier(data_slice=coordinate_handler, is_plus_strand=False)
    # and now that we get the expect range on the minus strand, keeping in mind the 40 is inclusive, and the 9, not
    nums = numerifier.slice_to_matrix()

    expect[10:41, 0] = 1.
    expect = np.flip(expect, axis=1)
    assert np.allclose(nums, expect)
    # todo, test transitions as well...


def test_live_slicing():
    sess, coordinate_handler = setup_simpler_numerifier()
    # annotations by bp
    numerifier = numerify.BasePairAnnotationNumerifier(data_slice=coordinate_handler, is_plus_strand=False)
    num_list = list(numerifier.slice_to_matrices(max_len=50))

    expect = np.zeros([100, 3], dtype=float)
    expect[10:41, 0] = 1.

    assert np.allclose(num_list[0], np.flip(expect[50:100], axis=1))
    assert np.allclose(num_list[1], np.flip(expect[0:50], axis=1))
    # sequences by bp
    sg = sequences.StructuredGenome()
    sg.from_json('testdata/dummyloci.sequence.json')
    sequence = sg.sequences[0]
    slice0 = sequence.slices[0]
    numerifier = numerify.SequenceNumerifier(is_plus_strand=True)
    num_list = list(numerifier.slice_to_matrices(data_slice=slice0, max_len=50))
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
