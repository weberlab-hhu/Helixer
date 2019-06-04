import os
from shutil import copy
import numpy as np
import pytest
import deepdish as dd

import geenuff
from geenuff.tests.test_geenuff import setup_data_handler, mk_memory_session

from geenuff.applications.importer import ImportController
from geenuff.base.orm import SuperLocus, Genome, Coordinate
from geenuff.base.helpers import reverse_complement
from ..core import helpers
from ..core.mers import MerController
from ..core.orm import Mer
from ..export import numerify
from ..export.numerify import (SequenceNumerifier, BasePairAnnotationNumerifier, Stepper,
                               AMBIGUITY_DECODE)
from ..export.exporter import ExportController

TMP_DB = 'testdata/tmp.db'
DUMMYLOCI_DB = 'testdata/dummyloci.sqlite3'
H5_OUT = 'testdata/test.h5'


### preparation and breakdown ###
@pytest.fixture(scope="session", autouse=True)
def setup_dummy_db(request):
    if not os.getcwd().endswith('HelixerPrep/helixerprep'):
        pytest.exit('Tests need to be run from HelixerPrep/helixerprep directory')
    if os.path.exists(DUMMYLOCI_DB):
        os.remove(DUMMYLOCI_DB)

    # make sure we have the same test data that Geenuff has
    geenuff_test_folder = os.path.dirname(geenuff.__file__) + '/testdata/'
    files = ['dummyloci.fa', 'dummyloci.gff', 'basic_sequences.fa']
    for f in files:
        copy(geenuff_test_folder + f, 'testdata/')

    # setup dummyloci
    controller = ImportController(database_path='sqlite:///' + DUMMYLOCI_DB)
    controller.add_genome('testdata/dummyloci.fa', 'testdata/dummyloci.gff')

    # stuff after yield is going to be executed after all tests are run
    yield None

    # clean up tmp files
    for p in [TMP_DB, H5_OUT]:
        if os.path.exists(p):
            os.remove(p)


### helper functions ###
def mk_controllers(source_db, helixer_db=TMP_DB, h5_out=H5_OUT):
    if os.path.exists(helixer_db):
        os.remove(helixer_db)

    mer_controller = MerController(source_db, helixer_db)
    export_controller = ExportController(helixer_db, h5_out)
    return mer_controller, export_controller


def memory_import_fasta(fasta_path):
    controller = ImportController(database_path='sqlite:///:memory:')
    controller.add_sequences(fasta_path)
    coords = controller.session.query(Coordinate).order_by(Coordinate.id).all()
    return controller, coords


def setup_dummyloci():
    _, controller = mk_controllers(DUMMYLOCI_DB)
    session = controller.session
    coordinate = session.query(Coordinate).first()
    return session, controller, coordinate


def setup_simpler_numerifier():
    sess = mk_memory_session()
    genome = Genome()
    coord = Coordinate(genome=genome, sequence='A' * 100, start=0, end=100, seqid='a')
    sl = SuperLocus()
    transcript = geenuff.orm.Transcript(super_locus=sl)
    piece = geenuff.orm.TranscriptPiece(transcript=transcript, position=0)
    transcript_feature = geenuff.orm.Feature(start=40,
                                             end=9,
                                             is_plus_strand=False,
                                             type=geenuff.types.TRANSCRIPT_FEATURE,
                                             start_is_biological_start=True,
                                             end_is_biological_end=True,
                                             coordinate=coord)
    piece.features = [transcript_feature]

    sess.add_all([genome, coord, sl, transcript, piece, transcript_feature])
    sess.commit()
    return sess, coord


### db import from GeenuFF ###
def test_copy_n_import():
    _, controller = mk_controllers(source_db=DUMMYLOCI_DB)
    sl = controller.session.query(SuperLocus).filter(SuperLocus.given_name == 'gene0').one()
    assert len(sl.transcripts) == 3
    assert len(sl.proteins) == 3
    all_features = []
    for transcript in sl.transcripts:
        assert len(transcript.transcript_pieces) == 1
        piece = transcript.transcript_pieces[0]
        for feature in piece.features:
            all_features.append(feature)
        print('{}: {}'.format(transcript.given_name, [x.type.value for x in piece.features]))
    for protein in sl.proteins:
        print('{}: {}'.format(protein.given_name, [x.type.value for x in protein.features]))

    # if I ever get to collapsing redundant features this will change
    assert len(all_features) == 9


### sequences ###
def test_add_mers():
    mer_controller, _ = mk_controllers(source_db=DUMMYLOCI_DB)
    mer_controller.add_mers(1, 3)
    query = mer_controller.session.query

    coords = query(Coordinate).all()
    for coord in coords:
        mers = query(Mer).filter(Mer.coordinate==coord).filter(Mer.length==1).all()
        assert len(mers) == 3
        mers = query(Mer).filter(Mer.coordinate==coord).filter(Mer.length==2).all()
        assert len(mers) == 11  # some mers can be their own reverse complement here
        mers = query(Mer).filter(Mer.coordinate==coord).filter(Mer.length==3).all()
        assert len(mers) == ((4 ** 3) / 2) + 1


def test_count2mers():
    mc = helpers.MerCounter(2)
    sequence = 'AAAA'
    mc.add_sequence(sequence)
    counted = mc.export()
    print(counted)
    assert counted['AA'] == 3

    sequence = 'TTT'
    mc.add_sequence(sequence)
    counted = mc.export()
    assert counted['AA'] == 5

    mc2 = helpers.MerCounter(2)
    seq = 'AAATTT'
    mc2.add_sequence(seq)
    counted = mc2.export()
    non0 = [x for x in counted if counted[x] > 0]
    assert len(non0) == 2
    assert counted['AA'] == 4
    assert counted['AT'] == 1


def test_count_range_of_mers():
    seq = 'ATATAT'

    genome = Genome()
    coordinate = Coordinate(genome=genome, start=0, end=6, sequence=seq)
    all_mer_counters = MerController._count_mers(coordinate, 1, 6)[1]

    assert len(all_mer_counters) == 6

    # make sure total counts for any mer length, k, equal seq_length - k + 1
    for i in range(len(all_mer_counters)):
        counted = all_mer_counters[i].export()
        assert sum(counted.values()) == 6 - i

    # counting 1-mers, expect 6 x 'A'
    counted = all_mer_counters[0].export()
    assert counted['A'] == 6

    # counting 2-mers, expect (3, 'AT'; 2 'TA')
    counted = all_mer_counters[1].export()
    assert counted['AT'] == 3
    assert counted['TA'] == 2

    # counting 3-mers, expect (4, 'ATA')
    counted = all_mer_counters[2].export()
    assert counted['ATA'] == 4

    # counting 4-mers, expect (2, 'ATAT'; 1, 'TATA')
    counted = all_mer_counters[3].export()
    assert counted['ATAT'] == 2
    assert counted['TATA'] == 1

    # counting 5-mers, expect (2, 'ATATA')
    counted = all_mer_counters[4].export()
    assert counted['ATATA'] == 2

    # counting 6-mer, expect original sequence, uppercase
    counted = all_mer_counters[5].export()
    assert counted['ATATAT'] == 1


#### numerify ####
def test_stepper():
    # evenly divided
    s = Stepper(50, 10)
    strt_ends = list(s.step_to_end())
    assert len(strt_ends) == 5
    assert strt_ends[0] == (0, 10)
    assert strt_ends[-1] == (40, 50)
    # a bit short
    s = Stepper(49, 10)
    strt_ends = list(s.step_to_end())
    assert len(strt_ends) == 5
    assert strt_ends[-1] == (39, 49)
    # a bit long
    s = Stepper(52, 10)
    strt_ends = list(s.step_to_end())
    assert len(strt_ends) == 6
    assert strt_ends[-1] == (46, 52)
    # very short
    s = Stepper(9, 10)
    strt_ends = list(s.step_to_end())
    assert len(strt_ends) == 1
    assert strt_ends[-1] == (0, 9)


def test_short_sequence_numerify():
    _, coords = memory_import_fasta('testdata/basic_sequences.fa')
    numerifier = SequenceNumerifier(coord=coords[3], is_plus_strand=True, max_len=100)
    matrix = numerifier.coord_to_matrices()[0][0]
    # ATATATAT
    x = [0., 1, 0, 0, 0., 0, 1, 0]
    expect = np.array(x * 4).reshape((-1, 4))
    assert np.array_equal(expect, matrix)

    # on the minus strand
    numerifier = SequenceNumerifier(coord=coords[3], is_plus_strand=False, max_len=100)
    matrix = numerifier.coord_to_matrices()[0][0]

    seq_comp = reverse_complement(coords[3].sequence)
    expect = [numerify.AMBIGUITY_DECODE[bp] for bp in seq_comp]
    expect = np.vstack(expect)
    assert np.array_equal(expect, matrix)


def test_base_level_annotation_numerify():
    _, _, coord = setup_dummyloci()
    numerifier = BasePairAnnotationNumerifier(coord=coord,
                                              features=coord.features,
                                              is_plus_strand=True,
                                              max_len=5000)
    nums = numerifier.coord_to_matrices()[0][0][:405]
    expect = np.zeros([405, 3], dtype=np.float32)
    expect[0:400, 0] = 1.  # set genic/in raw transcript
    expect[10:301, 1] = 1.  # set in transcript
    expect[21:110, 2] = 1.  # both introns
    expect[120:200, 2] = 1.
    assert np.array_equal(nums, expect)


def test_sequence_slicing():
    _, coords = memory_import_fasta('testdata/basic_sequences.fa')
    seq_numerifier = SequenceNumerifier(coord=coords[0], is_plus_strand=True, max_len=50)
    num_list = seq_numerifier.coord_to_matrices()[0]
    print([x.shape for x in num_list])
    # [(50, 4), (50, 4), (50, 4), (50, 4), (50, 4), (50, 4), (50, 4), (27, 4), (28, 4)]
    assert len(num_list) == 9

    for i in range(7):
        assert np.array_equal(num_list[i], np.full([50, 4], 0.25, dtype=np.float32))
    for i in [7, 8]:  # for the last two, just care that they're about the expected size...
        assert np.array_equal(num_list[i][:27], np.full([27, 4], 0.25, dtype=np.float32))


def test_coherent_slicing():
    """Tests for coherent output when slicing the 405 bp long dummyloci.
    The correct divisions are already tested in the Stepper test.
    The array format of the individual matrices are tested in
    test_short_sequence_numerify().
    """
    _, _, coord = setup_dummyloci()
    seq_numerifier = SequenceNumerifier(coord=coord,
                                        is_plus_strand=True,
                                        max_len=100)
    anno_numerifier = BasePairAnnotationNumerifier(coord=coord,
                                                   features=coord.features,
                                                   is_plus_strand=True,
                                                   max_len=100)
    seq_slices = seq_numerifier.coord_to_matrices()[0]
    anno_slices = anno_numerifier.coord_to_matrices()[0]
    assert len(seq_slices) == len(anno_slices) == 19

    for s, a in zip(seq_slices, anno_slices):
        assert s.shape[0] == a.shape[0]

    # testing sequence error masks
    expect = np.zeros((1801, ), dtype=np.int8)
    # sequence error mask should be empty
    assert np.array_equal(expect, seq_numerifier.error_mask)
    # annotation error mask of test case 1 should reflect faulty exon/CDS ranges
    assert anno_numerifier.error_mask.shape == expect.shape
    expect[:110] = 1
    expect[120:499] = 1  # error from test case 1
    expect[499:1099] = 1  # error from test case 2
    # test equality for correct error ranges of first two test cases + some correct bases
    assert np.array_equal(expect[:1150], anno_numerifier.error_mask[:1150])


def test_minus_strand_numerify():
    # setup a very basic -strand locus
    _, coord = setup_simpler_numerifier()
    numerifier = BasePairAnnotationNumerifier(coord=coord,
                                              features=coord.features,
                                              is_plus_strand=True,
                                              max_len=1000)
    nums = numerifier.coord_to_matrices()[0][0]
    # first, we should make sure the opposite strand is unmarked when empty
    expect = np.zeros([100, 3], dtype=np.float32)
    assert np.array_equal(nums, expect)

    numerifier = BasePairAnnotationNumerifier(coord=coord,
                                              features=coord.features,
                                              is_plus_strand=False,
                                              max_len=1000)
    # and now that we get the expect range on the minus strand,
    # keeping in mind the 40 is inclusive, and the 9, not
    nums = numerifier.coord_to_matrices()[0][0]

    expect[10:41, 0] = 1.
    expect = np.flip(expect, axis=0)
    assert np.array_equal(nums, expect)

    # minus strand and actual cutting
    numerifier = BasePairAnnotationNumerifier(coord=coord,
                                              features=coord.features,
                                              is_plus_strand=False,
                                              max_len=50)
    num_list = numerifier.coord_to_matrices()[0]

    expect = np.zeros([100, 3], dtype=np.float32)
    expect[10:41, 0] = 1.

    assert np.array_equal(num_list[0], np.flip(expect[50:100], axis=0))
    assert np.array_equal(num_list[1], np.flip(expect[0:50], axis=0))


def test_coord_numerifier_and_h5_gen_plus_strand():
    _, controller, _ = setup_dummyloci()
    # dump the whole db in chunks into a .h5 file
    controller.export(chunk_size=400, shuffle=False, seed='puma')

    inputs = dd.io.load(H5_OUT, group='/inputs')
    labels = dd.io.load(H5_OUT, group='/labels')
    label_masks = dd.io.load(H5_OUT, group='/label_masks')
    config = dd.io.load(H5_OUT, group='/config')

    # five chunks for each the two coordinates and *2 for each strand
    # also tests if we ignore the third coordinate, that does not have any annotations
    assert len(inputs) == len(labels) == len(label_masks) == 20
    assert type(config) == dict

    # prep seq
    seq_expect = np.full((405, 4), 0.25)
    # set start codon
    seq_expect[10] = numerify.AMBIGUITY_DECODE['A']
    seq_expect[11] = numerify.AMBIGUITY_DECODE['T']
    seq_expect[12] = numerify.AMBIGUITY_DECODE['G']
    # stop codons
    seq_expect[117] = numerify.AMBIGUITY_DECODE['T']
    seq_expect[118] = numerify.AMBIGUITY_DECODE['A']
    seq_expect[119] = numerify.AMBIGUITY_DECODE['G']
    seq_expect[298] = numerify.AMBIGUITY_DECODE['T']
    seq_expect[299] = numerify.AMBIGUITY_DECODE['G']
    seq_expect[300] = numerify.AMBIGUITY_DECODE['A']
    assert np.array_equal(inputs[0], seq_expect[:400])
    assert np.array_equal(inputs[1][:5], seq_expect[400:])

    # prep anno
    label_expect = np.zeros((405, 3), dtype=np.float32)
    label_expect[0:400, 0] = 1.  # set genic/in raw transcript
    label_expect[10:301, 1] = 1.  # set in transcript
    label_expect[21:110, 2] = 1.  # both introns
    label_expect[120:200, 2] = 1.
    assert np.array_equal(labels[0], label_expect[:400])
    assert np.array_equal(labels[1][:5], label_expect[400:])

    # prep anno mask
    label_mask_expect = np.zeros((405, ), dtype=np.int8)
    label_mask_expect[:110] = 1
    label_mask_expect[120:] = 1
    assert np.array_equal(label_masks[0], label_mask_expect[:400])
    assert np.array_equal(label_masks[1][:5], label_mask_expect[400:])


def test_coord_numerifier_and_h5_gen_minus_strand():
    _, controller, _ = setup_dummyloci()
    # dump the whole db in chunks into a .h5 file
    controller.export(chunk_size=200, shuffle=False, seed='puma')

    inputs = dd.io.load(H5_OUT, group='/inputs')
    labels = dd.io.load(H5_OUT, group='/labels')
    label_masks = dd.io.load(H5_OUT, group='/label_masks')

    # coord 1: 10 per strand
    # coord 2: 9 per strand
    assert len(inputs) == len(labels) == len(label_masks) == 38

    # the last 9 inputs/labels should be for the 2nd coord and the minus strand
    inputs = inputs[-9:]
    labels = labels[-9:]
    label_masks = label_masks[-9:]

    seq_expect = np.full((1755, 4), 0.25)
    # start codon
    seq_expect[1729] = np.flip(AMBIGUITY_DECODE['T'])
    seq_expect[1728] = np.flip(AMBIGUITY_DECODE['A'])
    seq_expect[1727] = np.flip(AMBIGUITY_DECODE['C'])
    # stop codon (other stop codon is intentionally missing in the test case)
    seq_expect[1576] = np.flip(AMBIGUITY_DECODE['A'])
    seq_expect[1575] = np.flip(AMBIGUITY_DECODE['T'])
    seq_expect[1574] = np.flip(AMBIGUITY_DECODE['C'])
    seq_expect = np.flip(seq_expect, axis=0)
    assert np.array_equal(inputs[0], seq_expect[:178])
    assert np.array_equal(inputs[1][:50], seq_expect[178:228])

    label_expect = np.zeros((1755, 3), dtype=np.float32)
    label_expect[1549:1750, 0] = 1.  # genic region
    label_expect[1574:1730, 1] = 1.  # transcript (2 overlapping ones)
    label_expect[1650:1719, 2] = 1.  # intron
    label_expect = np.flip(label_expect, axis=0)
    assert np.array_equal(labels[0], label_expect[:178])
    assert np.array_equal(labels[1][:50], label_expect[178:228])

    label_mask_expect = np.zeros((1755, ), dtype=np.int8)
    label_mask_expect[1725:] = 1.
    label_mask_expect[1549:1650] = 1.
    label_mask_expect = np.flip(label_mask_expect)
    assert np.array_equal(label_masks[0], label_mask_expect[:178])
    assert np.array_equal(label_masks[1][:50], label_mask_expect[178:228])
