import os
from shutil import copy
from sklearn.metrics import precision_recall_fscore_support as f1_scores
from sklearn.metrics import accuracy_score
import numpy as np
import pytest
import h5py
from abc import ABC, abstractmethod

import geenuff
from geenuff.tests.test_geenuff import setup_data_handler, mk_memory_session
from geenuff.applications.importer import ImportController
from geenuff.base.orm import SuperLocus, Genome, Coordinate
from geenuff.base.helpers import reverse_complement
from geenuff.base import types
from ..core.controller import HelixerController
from ..core import helpers
from ..export import numerify
from ..export.numerify import SequenceNumerifier, AnnotationNumerifier, Stepper, AMBIGUITY_DECODE
from ..export.exporter import HelixerExportController
from ..prediction.ConfusionMatrix import ConfusionMatrix
from helixer.prediction.HelixerModel import HelixerModel, HelixerSequence
from helixer.prediction.LSTMModel import LSTMSequence
from ..evaluation import rnaseq

TMP_DB = 'testdata/tmp.db'
DUMMYLOCI_DB = 'testdata/dummyloci.sqlite3'
H5_OUT_FOLDER = 'testdata/numerify_test_out/'
H5_OUT_FILE = H5_OUT_FOLDER + 'test_data.h5'
EVAL_H5 = 'testdata/tmp.h5'


### preparation and breakdown ###
@pytest.fixture(scope="session", autouse=True)
def setup_dummy_db(request):
    if not os.getcwd().endswith('Helixer/helixer'):
        pytest.exit('Tests need to be run from Helixer/helixer directory')
    if os.path.exists(DUMMYLOCI_DB):
        os.remove(DUMMYLOCI_DB)

    # make sure we have the same test data that Geenuff has
    geenuff_test_folder = os.path.dirname(geenuff.__file__) + '/testdata/'
    files = ['dummyloci.fa', 'dummyloci.gff', 'basic_sequences.fa']
    for f in files:
        copy(geenuff_test_folder + f, 'testdata/')

    # setup dummyloci
    controller = ImportController(database_path='sqlite:///' + DUMMYLOCI_DB)
    controller.add_genome('testdata/dummyloci.fa', 'testdata/dummyloci.gff',
                          genome_args={"species": "dummy"})

    # make tmp folder
    if not os.path.exists(H5_OUT_FOLDER):
        os.mkdir(H5_OUT_FOLDER)

    # stuff after yield is going to be executed after all tests are run
    yield None

    # clean up tmp files
    for p in [TMP_DB] + [H5_OUT_FOLDER + f for f in os.listdir(H5_OUT_FOLDER)]:
        if os.path.exists(p):
            os.remove(p)
    os.rmdir(H5_OUT_FOLDER)


@pytest.fixture(scope="session", autouse=True)
def setup_dummy_evaluation_h5(request):
    start_ends = [[0, 20000],  # increasing 0
                  [20000, 40000],  # increasing 0
                  [60000, 80000],  # increasing 1
                  [100000, 120000],  # increasing 2
                  [120000, 133333],  # increasing 3 (non contig, bc special edge handling)
                  [133333, 120000],  # decreasing 0 (")
                  [120000, 100000],  # decreasing 1
                  [100000, 80000],  # decreasing 1
                  [60000, 40000],  # decreasing 2
                  [20000, 0]]  # decreasing 3
    seqids = [b'chr1'] * len(start_ends)
    h5path = EVAL_H5

    h5 = h5py.File(h5path, 'a')
    h5.create_group('data')
    h5.create_dataset('/data/start_ends', data=start_ends, dtype='int64', compression='lzf')
    h5.create_dataset('/data/seqids', data=seqids, dtype='S50', compression='lzf')

    h5.create_group('evaluation')
    h5.create_dataset('evaluation/coverage', shape=[10, 20000], dtype='int', fillvalue=-1, compression='lzf')
    h5.close()

    yield None  # all tests are run

    os.remove(h5path)


### helper functions ###
def mk_controllers(source_db, helixer_db=TMP_DB, h5_out=H5_OUT_FOLDER, only_test_set=True):
    for p in [helixer_db] + [h5_out + f for f in os.listdir(h5_out)]:
        if os.path.exists(p):
            os.remove(p)

    mer_controller = HelixerController(source_db, helixer_db, '', '')
    export_controller = HelixerExportController(helixer_db, h5_out, only_test_set=only_test_set)
    return mer_controller, export_controller


def memory_import_fasta(fasta_path):
    controller = ImportController(database_path='sqlite:///:memory:')
    controller.add_sequences(fasta_path)
    coords = controller.session.query(Coordinate).order_by(Coordinate.id).all()
    return controller, coords


def setup_dummyloci(only_test_set=True):
    _, export_controller = mk_controllers(DUMMYLOCI_DB, only_test_set=only_test_set)
    session = export_controller.geenuff_exporter.session
    coordinate = session.query(Coordinate).first()
    return session, export_controller, coordinate


def setup_simpler_numerifier():
    sess = mk_memory_session()
    genome = Genome()
    coord = Coordinate(genome=genome, sequence='A' * 100, length=100, seqid='a')
    sl = SuperLocus()
    transcript = geenuff.orm.Transcript(super_locus=sl)
    piece = geenuff.orm.TranscriptPiece(transcript=transcript, position=0)
    transcript_feature = geenuff.orm.Feature(start=40,
                                             end=9,
                                             is_plus_strand=False,
                                             type=geenuff.types.GEENUFF_TRANSCRIPT,
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
    session = controller.geenuff_exporter.session
    sl = session.query(SuperLocus).filter(SuperLocus.given_name == 'gene0').one()
    assert len(sl.transcripts) == 3
    assert len(sl.proteins) == 3
    all_features = []
    for transcript in sl.transcripts:
        assert len(transcript.transcript_pieces) == 1
        piece = transcript.transcript_pieces[0]
        for feature in piece.features:
            if feature.type.value not in types.geenuff_error_type_values:
                all_features.append(feature)
        print('{}: {}'.format(transcript.given_name, [x.type.value for x in piece.features]))
    for protein in sl.proteins:
        print('{}: {}'.format(protein.given_name, [x.type.value for x in protein.features]))

    # if I ever get to collapsing redundant features this will change
    assert len(all_features) == 9


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
    assert strt_ends[-1] == (40, 49)
    # a bit long
    s = Stepper(52, 10)
    strt_ends = list(s.step_to_end())
    assert len(strt_ends) == 6
    assert strt_ends[-1] == (50, 52)
    # very short
    s = Stepper(9, 10)
    strt_ends = list(s.step_to_end())
    assert len(strt_ends) == 1
    assert strt_ends[-1] == (0, 9)


def test_short_sequence_numerify():
    _, coords = memory_import_fasta('testdata/basic_sequences.fa')
    numerifier = SequenceNumerifier(coord=coords[3], max_len=100)
    matrix = numerifier.coord_to_matrices()['plus'][0]
    # ATATATAT
    x = [0., 1, 0, 0, 0., 0, 1, 0]
    expect = np.array(x * 4).reshape((-1, 4))
    assert np.array_equal(expect, matrix)

    # on the minus strand
    numerifier = SequenceNumerifier(coord=coords[3], max_len=100)
    matrix = numerifier.coord_to_matrices()['plus'][0]

    seq_comp = reverse_complement(coords[3].sequence)
    expect = [numerify.AMBIGUITY_DECODE[bp] for bp in seq_comp]
    expect = np.vstack(expect)
    assert np.array_equal(expect, matrix)


def test_base_level_annotation_numerify():
    _, _, coord = setup_dummyloci()
    numerifier = AnnotationNumerifier(coord=coord,
                                      features=coord.features,
                                      max_len=5000,
                                      one_hot=False)
    nums = numerifier.coord_to_matrices()[0]["plus"][0][:405]
    expect = np.zeros([405, 3], dtype=np.float32)
    expect[0:400, 0] = 1.  # set genic/in raw transcript
    expect[10:301, 1] = 1.  # set in transcript
    expect[21:110, 2] = 1.  # both introns
    expect[120:200, 2] = 1.
    assert np.array_equal(nums, expect)


def test_sequence_slicing():
    _, coords = memory_import_fasta('testdata/basic_sequences.fa')
    seq_numerifier = SequenceNumerifier(coord=coords[0], max_len=50)
    num_mats = seq_numerifier.coord_to_matrices()
    num_list = num_mats["plus"] + num_mats["minus"]
    print([x.shape for x in num_list])
    # [(50, 4), (50, 4), (50, 4), (50, 4), (50, 4), (50, 4), (50, 4), (50, 4), (5, 4)]
    assert len(num_list) == 9 * 2  # both strands

    for i in range(8):
        assert np.array_equal(num_list[i], np.full([50, 4], 0.25, dtype=np.float32))
    assert np.array_equal(num_list[8], np.full([5, 4], 0.25, dtype=np.float32))


def test_coherent_slicing():
    """Tests for coherent output when slicing the 405 bp long dummyloci.
    The correct divisions are already tested in the Stepper test.
    The array format of the individual matrices are tested in
    test_short_sequence_numerify().
    """
    _, _, coord = setup_dummyloci()
    seq_numerifier = SequenceNumerifier(coord=coord,
                                        max_len=100)
    anno_numerifier = AnnotationNumerifier(coord=coord,
                                           features=coord.features,
                                           max_len=100,
                                           one_hot=False)

    seq_mats = seq_numerifier.coord_to_matrices()
    # appending +&- is historical / avoiding re-writing the test...
    seq_slices = seq_mats['plus'] + seq_mats['minus']

    anno_mats = anno_numerifier.coord_to_matrices()
    anno_mats = [x["plus"] + x['minus'] for x in anno_mats]
    anno_slices, anno_error_masks, gene_lengths, transitions = anno_mats
    assert (len(seq_slices) == len(anno_slices) == len(gene_lengths) == len(transitions) ==
            len(anno_error_masks) == 19 * 2)

    for s, a, ae in zip(seq_slices, anno_slices, anno_error_masks):
        assert s.shape[0] == a.shape[0] == ae.shape[0]

    # testing sequence error masks
    expect = np.ones((1801 * 2,), dtype=np.int8)

    # annotation error mask of test case 1 should reflect faulty exon/CDS ranges
    expect[:110] = 0
    expect[120:499] = 0  # error from test case 1
    # expect[499:1099] = 0  # error from test case 2, which we do not mark atm
    # test equality for correct error ranges of first two test cases + some correct bases
    assert np.array_equal(expect[:1150], np.concatenate(anno_error_masks)[:1150])


def test_minus_strand_numerify():
    # setup a very basic -strand locus
    _, coord = setup_simpler_numerifier()
    numerifier = AnnotationNumerifier(coord=coord,
                                      features=coord.features,
                                      max_len=1000,
                                      one_hot=False)
    nums = numerifier.coord_to_matrices()[0]
    # first, we should make sure the opposite strand is unmarked when empty
    expect = np.zeros([100, 3], dtype=np.float32)
    assert np.array_equal(nums["plus"][0], expect)

    # and now that we get the expect range on the minus strand,
    # keeping in mind the 40 is inclusive, and the 9, not
    expect[10:41, 0] = 1.
    expect = np.flip(expect, axis=0)
    assert np.array_equal(nums["minus"][0], expect)

    # with cutting
    numerifier = AnnotationNumerifier(coord=coord,
                                      features=coord.features,
                                      max_len=50,
                                      one_hot=False)
    nums = numerifier.coord_to_matrices()[0]

    expect = np.zeros([100, 3], dtype=np.float32)
    expect[10:41, 0] = 1.

    assert np.array_equal(nums['minus'][0], np.flip(expect[50:100], axis=0))
    assert np.array_equal(nums['minus'][1], np.flip(expect[0:50], axis=0))


def test_coord_numerifier_and_h5_gen_plus_strand():
    _, controller, _ = setup_dummyloci()
    # dump the whole db in chunks into a .h5 file
    controller.export(chunk_size=400, genomes='', exclude='', val_size=0.2, one_hot=False,
                      all_transcripts=True)

    f = h5py.File(H5_OUT_FILE, 'r')
    not_erroneous = f['/data/err_samples'][:]
    x = f['/data/X'][:][not_erroneous]
    y = f['/data/y'][:][not_erroneous]
    sample_weights = f['/data/sample_weights'][:][not_erroneous]

    # five chunks for each the two annotated coordinates and one for the unannotated coord
    # then *2 for each strand and -2 for
    # completely erroneous sequences (at the end of the minus strand of the 2nd coord)
    # also tests if we ignore the third coordinate, that does not have any annotations
    assert len(x) == len(y) == len(sample_weights) == 18

    # prep seq
    x_expect = np.full((405, 4), 0.25)
    # set start codon
    x_expect[10] = numerify.AMBIGUITY_DECODE['A']
    x_expect[11] = numerify.AMBIGUITY_DECODE['T']
    x_expect[12] = numerify.AMBIGUITY_DECODE['G']
    # stop codons
    x_expect[117] = numerify.AMBIGUITY_DECODE['T']
    x_expect[118] = numerify.AMBIGUITY_DECODE['A']
    x_expect[119] = numerify.AMBIGUITY_DECODE['G']
    x_expect[298] = numerify.AMBIGUITY_DECODE['T']
    x_expect[299] = numerify.AMBIGUITY_DECODE['G']
    x_expect[300] = numerify.AMBIGUITY_DECODE['A']
    assert np.array_equal(x[0], x_expect[:400])
    assert np.array_equal(x[1][:5], x_expect[400:])

    # prep anno
    y_expect = np.zeros((405, 3), dtype=np.float16)
    y_expect[0:400, 0] = 1.  # set genic/in raw transcript
    y_expect[10:301, 1] = 1.  # set in transcript
    y_expect[21:110, 2] = 1.  # both introns
    y_expect[120:200, 2] = 1.
    assert np.array_equal(y[0], y_expect[:400])
    assert np.array_equal(y[1][:5], y_expect[400:])

    # prep anno mask
    sample_weight_expect = np.ones((405,), dtype=np.int8)
    sample_weight_expect[:110] = 0
    sample_weight_expect[120:] = 0
    assert np.array_equal(sample_weights[0], sample_weight_expect[:400])
    assert np.array_equal(sample_weights[1][:5], sample_weight_expect[400:])


def test_coord_numerifier_and_h5_gen_minus_strand():
    """Tests numerification of test case 8 on coordinate 2"""
    _, controller, _ = setup_dummyloci()
    # dump the whole db in chunks into a .h5 file
    controller.export(chunk_size=200, genomes='', exclude='', val_size=0.2, one_hot=False,
                      all_transcripts=True)

    f = h5py.File(H5_OUT_FILE, 'r')
    not_erroneous = f['/data/err_samples'][:]
    x = f['/data/X'][:][not_erroneous]
    y = f['/data/y'][:][not_erroneous]
    sample_weights = f['/data/sample_weights'][:][not_erroneous]

    assert len(x) == len(y) == len(sample_weights) == 33

    # the x/y selected below  should be for the 2nd coord and the minus strand
    # orginally there where 9 but 4 were tossed out because they were fully erroneous
    # all the sequences are also 0-padded
    a, b = 28, 33
    x = x[a:b]
    y = y[a:b]
    sample_weights = sample_weights[a:b]

    x_expect = np.full((955, 4), 0.25)
    # start codon
    x_expect[929] = np.flip(AMBIGUITY_DECODE['T'])
    x_expect[928] = np.flip(AMBIGUITY_DECODE['A'])
    x_expect[927] = np.flip(AMBIGUITY_DECODE['C'])
    # stop codon of second transcript
    x_expect[902] = np.flip(AMBIGUITY_DECODE['A'])
    x_expect[901] = np.flip(AMBIGUITY_DECODE['T'])
    x_expect[900] = np.flip(AMBIGUITY_DECODE['C'])
    # stop codon of first transcript
    x_expect[776] = np.flip(AMBIGUITY_DECODE['A'])
    x_expect[775] = np.flip(AMBIGUITY_DECODE['T'])
    x_expect[774] = np.flip(AMBIGUITY_DECODE['C'])
    # flip as the sequence is read 5p to 3p
    x_expect = np.flip(x_expect, axis=0)
    # insert 0-padding
    x_expect = np.insert(x_expect, 155, np.zeros((45, 4)), axis=0)
    assert np.array_equal(x[0], x_expect[:200])
    assert np.array_equal(x[1][:50], x_expect[200:250])

    y_expect = np.zeros((955, 3), dtype=np.float16)
    y_expect[749:950, 0] = 1.  # genic region
    y_expect[774:930, 1] = 1.  # transcript (2 overlapping ones)
    y_expect[850:919, 2] = 1.  # intron first transcript
    y_expect[800:879, 2] = 1.  # intron second transcript
    y_expect = np.flip(y_expect, axis=0)
    y_expect = np.insert(y_expect, 155, np.zeros((45, 3)), axis=0)
    assert np.array_equal(y[0], y_expect[:200])
    assert np.array_equal(y[1][:50], y_expect[200:250])

    sample_weight_expect = np.ones((955,), dtype=np.int8)
    sample_weight_expect[925:] = 0
    sample_weight_expect[749:850] = 0
    sample_weight_expect = np.flip(sample_weight_expect)
    sample_weight_expect = np.insert(sample_weight_expect, 155, np.zeros((45,)), axis=0)
    assert np.array_equal(sample_weights[0], sample_weight_expect[:200])
    assert np.array_equal(sample_weights[1][:50], sample_weight_expect[200:250])


def test_numerify_with_end_neg1():
    def check_one(coord, is_plus_strand, expect, maskexpect):
        numerifier = AnnotationNumerifier(coord=coord,
                                          features=coord.features,
                                          max_len=1000,
                                          one_hot=False)

        if is_plus_strand:
            nums, masks, _, _ = [x["plus"][0] for x in numerifier.coord_to_matrices()]
        else:
            nums, masks, _, _ = [x["minus"][0] for x in numerifier.coord_to_matrices()]

        if not np.array_equal(nums, expect):
            print(nums)
            for i in range(nums.shape[0]):
                if not np.all(nums[i] == expect[i]):
                    print("nums[i] != expect[i]: {} != {}, @ {}".format(nums[i], expect[i], i))
            assert False, "label arrays not equal, see above"
        if not np.array_equal(masks, maskexpect):
            for i in range(len(masks)):
                if masks[i] != maskexpect[i]:
                    print("masks[i] != maskexpect[i]: {} != {}, @ {}".format(masks[i], maskexpect[i], i))
            assert False, "mask arrays not equal, see above"

    def expect0():
        return np.zeros([1000, 3], dtype=np.float32)

    def masks1():
        return np.ones((1000,), dtype=np.int)

    controller = ImportController(database_path='sqlite:///:memory:')
    controller.add_genome('testdata/edges.fa', 'testdata/edges.gff',
                          genome_args={"species": "edges"})
    # test case: plus strand, start, features
    # + (each char represents ~ 50bp)
    # 1111 0000 0000 0000 0000 0000
    # 0110 0000 0000 0000 0000 0000
    # 0000 0000 0000 0000 0000 0000
    # err
    # 1111 1111 1111 1111 1111 1111
    coord = controller.session.query(Coordinate).filter(Coordinate.id == 1).first()
    expect = expect0()
    expect[0:200, 0] = 1.  # transcribed
    expect[50:149, 1] = 1.

    maskexpect = masks1()
    check_one(coord, True, expect, maskexpect)
    # - strand, as above, but expect all 0s no masking
    expect = expect0()
    check_one(coord, False, expect, maskexpect)

    # test case: plus strand, start, errors
    # + (each char represents ~ 50bp)
    # 0111 0000 0000 0000 0000 0000
    # 0110 0000 0000 0000 0000 0000
    # 0000 0000 0000 0000 0000 0000
    # err
    # 0111 1111 1111 1111 1111 1111
    coord = controller.session.query(Coordinate).filter(Coordinate.id == 2).first()
    expect = expect0()
    expect[50:200, 0] = 1.
    expect[50:149, 1] = 1.
    maskexpect = masks1()
    maskexpect[0:50] = 0
    check_one(coord, True, expect, maskexpect)
    # - strand
    expect = expect0()
    maskexpect = masks1()
    check_one(coord, False, expect, maskexpect)

    # test case: minus strand, end, features
    # -
    # 0000 0000 0000 0000 0000 1111
    # 0000 0000 0000 0000 0000 0110
    # 0000 0000 0000 0000 0000 0000
    # err
    # 1111 1111 1111 1111 1111 1111
    coord = controller.session.query(Coordinate).filter(Coordinate.id == 3).first()
    expect = expect0()
    expect[0:200, 0] = 1.  # transcribed
    expect[50:149, 1] = 1.
    expect = np.flip(expect, axis=0)

    maskexpect = masks1()
    check_one(coord, False, expect, maskexpect)
    # + strand, as above, but expect all 0s no masking
    expect = expect0()
    check_one(coord, True, expect, maskexpect)

    # test case: minus strand, end, errors
    # -
    # 0000 0000 0000 0000 0000 1110
    # 0000 0000 0000 0000 0000 0110
    # 0000 0000 0000 0000 0000 0000
    # err
    # 1111 1111 1111 1111 1111 1110
    coord = controller.session.query(Coordinate).filter(Coordinate.id == 4).first()
    expect = expect0()
    expect[50:200, 0] = 1.  # transcribed
    expect[50:149, 1] = 1.
    expect = np.flip(expect, axis=0)

    maskexpect = masks1()
    maskexpect[0:50] = 0
    maskexpect = np.flip(maskexpect, axis=0)
    check_one(coord, False, expect, maskexpect)
    # + strand, as above, but expect all 0s no masking
    expect = expect0()
    maskexpect = masks1()
    check_one(coord, True, expect, maskexpect)

    # test case: plus strand, end, features
    # +
    # 0000 0000 0000 0000 0000 1111
    # 0000 0000 0000 0000 0000 0110
    # 0000 0000 0000 0000 0000 0000
    # err
    # 1111 1111 1111 1111 1111 1111
    coord = controller.session.query(Coordinate).filter(Coordinate.id == 5).first()
    expect = expect0()
    expect[799:1000, 0] = 1.  # transcribed
    expect[851:950, 1] = 1.
    maskexpect = masks1()
    check_one(coord, True, expect, maskexpect)
    # - strand, as above, but expect all 0s no masking
    expect = expect0()
    maskexpect = masks1()
    check_one(coord, False, expect, maskexpect)

    # test case: plus strand, end, errors
    # +
    # 0000 0000 0000 0000 0000 1110
    # 0000 0000 0000 0000 0000 0110
    # 0000 0000 0000 0000 0000 0000
    # err
    # 1111 1111 1111 1111 1111 1110
    coord = controller.session.query(Coordinate).filter(Coordinate.id == 6).first()
    expect = expect0()
    expect[799:950, 0] = 1.  # transcribed
    expect[851:950, 1] = 1.
    maskexpect = masks1()
    maskexpect[950:1000] = 0
    check_one(coord, True, expect, maskexpect)
    # - strand, as above, but expect all 0s no masking
    expect = expect0()
    maskexpect = masks1()
    check_one(coord, False, expect, maskexpect)

    # test case: minus strand, start, features
    # -
    # 1111 0000 0000 0000 0000 0000
    # 0110 0000 0000 0000 0000 0000
    # 0000 0000 0000 0000 0000 0000
    # err
    # 1111 1111 1111 1111 1111 1111
    coord = controller.session.query(Coordinate).filter(Coordinate.id == 7).first()
    expect = expect0()
    expect[799:1000, 0] = 1.  # transcribed
    expect[851:950, 1] = 1.
    expect = np.flip(expect, axis=0)
    maskexpect = masks1()
    check_one(coord, False, expect, maskexpect)
    # + strand, as above, but expect all 0s no masking
    expect = expect0()
    maskexpect = masks1()
    check_one(coord, True, expect, maskexpect)

    # test case: minus strand, start, errors
    # -
    # 0111 0000 0000 0000 0000 0000
    # 0110 0000 0000 0000 0000 0000
    # 0000 0000 0000 0000 0000 0000
    # err
    # 0111 1111 1111 1111 1111 1111
    coord = controller.session.query(Coordinate).filter(Coordinate.id == 8).first()
    expect = expect0()
    expect[799:950, 0] = 1.  # transcribed
    expect[851:950, 1] = 1.
    expect = np.flip(expect, axis=0)
    maskexpect = masks1()
    maskexpect[950:1000] = 0
    maskexpect = np.flip(maskexpect, axis=0)
    check_one(coord, False, expect, maskexpect)
    # + strand, as above, but expect all 0s no masking
    expect = expect0()
    maskexpect = masks1()
    check_one(coord, True, expect, maskexpect)


def test_one_hot_encodings():
    classes_multi = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ]
    classes_4 = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]

    # make normal encoding (multi class)
    _, _, coord = setup_dummyloci()
    numerifier = AnnotationNumerifier(coord=coord,
                                      features=coord.features,
                                      max_len=5000,
                                      one_hot=False)

    y_multi = numerifier.coord_to_matrices()[0]["plus"][0]
    # count classes
    uniques_multi = np.unique(y_multi, return_counts=True, axis=0)

    # make one hot encoding
    numerifier = AnnotationNumerifier(coord=coord,
                                      features=coord.features,
                                      max_len=5000,
                                      one_hot=True)
    y_one_hot_4 = numerifier.coord_to_matrices()[0]["plus"][0]
    uniques_4 = np.unique(y_one_hot_4, return_counts=True, axis=0)
    # this loop has to be changed when using accounting for non-coding introns as well
    for i in range(len(classes_multi)):
        idx = (uniques_4[0] == classes_4[i]).all(axis=1).nonzero()[0][0]
        assert uniques_multi[1][i] == uniques_4[1][idx]

    # test if they are one-hot at all
    assert np.all(np.count_nonzero(y_one_hot_4, axis=1) == 1)


def test_confusion_matrix():
    # 10 bases Intergenic
    # 8 bases UTR
    # 11 bases exon
    # 2 bases intron
    y_true = np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],

        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],

        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],

        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],

        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],

        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],

        [1, 0, 0, 0],
        [0, 0, 0, 0],  # start 0-padding
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    y_pred = np.array([
        [0.97320538, 0.00241233, 0.00655741, 0.01782488],
        [0.97320538, 0.00241233, 0.00655741, 0.01782488],
        [0.11245721, 0.83095266, 0.0413707, 0.01521943],  # error IG -> UTR
        [0.11245721, 0.83095266, 0.0413707, 0.01521943],  # error IG -> UTR
        [0.11245721, 0.83095266, 0.0413707, 0.01521943],

        [0.11245721, 0.83095266, 0.0413707, 0.01521943],
        [0.0349529, 0.25826895, 0.70204779, 0.00473036],  # error UTR -> Exon
        [0.0349529, 0.25826895, 0.70204779, 0.00473036],
        [0.0349529, 0.25826895, 0.70204779, 0.00473036],
        [0.0349529, 0.25826895, 0.70204779, 0.00473036],

        [0.01203764, 0.08894682, 0.24178252, 0.65723302],  # error Exon -> Intron
        [0.01203764, 0.08894682, 0.24178252, 0.65723302],  # error Exon -> Intron
        [0.0349529, 0.25826895, 0.70204779, 0.00473036],
        [0.97320538, 0.00241233, 0.00655741, 0.01782488],  # error Exon -> IG
        [0.97320538, 0.00241233, 0.00655741, 0.01782488],  # error Exon -> IG

        [0.0349529, 0.25826895, 0.70204779, 0.00473036],  # error Intron -> Exon
        [0.01203764, 0.08894682, 0.24178252, 0.65723302],
        [0.0349529, 0.25826895, 0.70204779, 0.00473036],
        [0.0349529, 0.25826895, 0.70204779, 0.00473036],
        [0.11245721, 0.83095266, 0.0413707, 0.01521943],  # error Exon -> UTR

        [0.11245721, 0.83095266, 0.0413707, 0.01521943],
        [0.11245721, 0.83095266, 0.0413707, 0.01521943],
        [0.11245721, 0.83095266, 0.0413707, 0.01521943],
        [0.97320538, 0.00241233, 0.00655741, 0.01782488],  # error UTR -> IG
        [0.97320538, 0.00241233, 0.00655741, 0.01782488],  # error UTR -> IG

        [0.97320538, 0.00241233, 0.00655741, 0.01782488],
        [0.97320538, 0.00241233, 0.00655741, 0.01782488],
        [0.97320538, 0.00241233, 0.00655741, 0.01782488],
        [0.97320538, 0.00241233, 0.00655741, 0.01782488],
        [0.97320538, 0.00241233, 0.00655741, 0.01782488],

        [0.97320538, 0.00241233, 0.00655741, 0.01782488],
        [0.0320586, 0.08714432, 0.23688282, 0.64391426],  # start 0-padding
        [0.0320586, 0.08714432, 0.23688282, 0.64391426],
        [0.0320586, 0.08714432, 0.23688282, 0.64391426],
        [0.0320586, 0.08714432, 0.23688282, 0.64391426],
        [0.0320586, 0.08714432, 0.23688282, 0.64391426]
    ])

    sample_weights = np.sum(y_true, axis=1)  # works bc, y_true is padded with ones

    cm_true = np.array([
        [8, 2, 0, 0],
        [2, 5, 1, 0],
        [2, 1, 6, 2],
        [0, 0, 1, 1]
    ])

    cm = ConfusionMatrix(None)
    # add data in two parts
    cm.count_and_calculate_one_batch(y_true[:15], y_pred[:15], sample_weights[:15])
    cm.count_and_calculate_one_batch(y_true[15:], y_pred[15:], sample_weights[15:])
    print(cm.cm)
    assert np.array_equal(cm_true, cm.cm)

    # test normalization
    cm_true_normalized = np.array([
        [8 / 10, 2 / 10, 0, 0],
        [2 / 8, 5 / 8, 1 / 8, 0],
        [2 / 11, 1 / 11, 6 / 11, 2 / 11],
        [0, 0, 1 / 2, 1 / 2]
    ])

    assert np.allclose(cm_true_normalized, cm._get_normalized_cm())

    # argmax and filter y_true and y_pred
    y_pred, y_true = ConfusionMatrix._remove_masked_bases(y_true, y_pred, sample_weights)
    y_pred = ConfusionMatrix._reshape_data(y_pred)
    y_true = ConfusionMatrix._reshape_data(y_true)

    # test other metrics
    precision_true, recall_true, f1_true, _ = f1_scores(y_true, y_pred)
    scores = cm._get_composite_scores()

    one_col_values = list(scores.values())[:4]  # excluding composite metrics
    assert np.allclose(precision_true, np.array([s['precision'] for s in one_col_values]))
    assert np.allclose(recall_true, np.array([s['recall'] for s in one_col_values]))
    assert np.allclose(f1_true, np.array([s['f1'] for s in one_col_values]))

    # test legacy cds metrics
    # essentially done in the same way as in ConfusionMatrix.py but copied here in case
    # it changes
    tp_cds = cm_true[2, 2] + cm_true[3, 3] + cm_true[2, 3] + cm_true[3, 2]
    fp_cds = cm_true[0, 2] + cm_true[1, 2] + cm_true[0, 3] + cm_true[1, 3]
    fn_cds = cm_true[2, 0] + cm_true[2, 1] + cm_true[3, 0] + cm_true[3, 1]
    cds_true = ConfusionMatrix._precision_recall_f1(tp_cds, fp_cds, fn_cds)
    assert np.allclose(cds_true[0], scores['legacy_cds']['precision'])
    assert np.allclose(cds_true[1], scores['legacy_cds']['recall'])
    assert np.allclose(cds_true[2], scores['legacy_cds']['f1'])

    # test subgenic metrics
    tp_genic = cm_true[2, 2] + cm_true[3, 3]
    fp_genic = (cm_true[0, 2] + cm_true[1, 2] + cm_true[3, 2] +
                cm_true[0, 3] + cm_true[1, 3] + cm_true[2, 3])
    fn_genic = (cm_true[2, 0] + cm_true[2, 1] + cm_true[2, 3] +
                cm_true[3, 0] + cm_true[3, 1] + cm_true[3, 2])
    genic_true = ConfusionMatrix._precision_recall_f1(tp_genic, fp_genic, fn_genic)
    assert np.allclose(genic_true[0], scores['sub_genic']['precision'])
    assert np.allclose(genic_true[1], scores['sub_genic']['recall'])
    assert np.allclose(genic_true[2], scores['sub_genic']['f1'])

    # test genic metrics
    tp_genic = cm_true[1, 1] + cm_true[2, 2] + cm_true[3, 3]
    fp_genic = (cm_true[0, 1] + cm_true[2, 1] + cm_true[3, 1] +
                cm_true[0, 2] + cm_true[1, 2] + cm_true[3, 2] +
                cm_true[0, 3] + cm_true[1, 3] + cm_true[2, 3])
    fn_genic = (cm_true[1, 0] + cm_true[1, 2] + cm_true[1, 3] +
                cm_true[2, 0] + cm_true[2, 1] + cm_true[2, 3] +
                cm_true[3, 0] + cm_true[3, 1] + cm_true[3, 2])
    genic_true = ConfusionMatrix._precision_recall_f1(tp_genic, fp_genic, fn_genic)
    assert np.allclose(genic_true[0], scores['genic']['precision'])
    assert np.allclose(genic_true[1], scores['genic']['recall'])
    assert np.allclose(genic_true[2], scores['genic']['f1'])

    # test accuracy
    acc_true = accuracy_score(y_pred, y_true)
    assert np.allclose(acc_true, cm._total_accuracy())


def test_gene_lengths():
    """Tests the '/data/gene_lengths' array"""
    _, controller, _ = setup_dummyloci()
    # dump the whole db in chunks into a .h5 file
    controller.export(chunk_size=5000, genomes='', exclude='', val_size=0.2, one_hot=True,
                      all_transcripts=True)

    f = h5py.File(H5_OUT_FILE, 'r')
    gl = f['/data/gene_lengths']
    y = f['/data/y']

    assert len(gl) == 4  # one for each coord and strand

    # check if there is a value > 0 wherever there is something genic
    for i in range(len(gl)):
        genic_gl = gl[i] > 0
        utr_y = np.all(y[i] == [0, 1, 0, 0], axis=-1)
        exon_y = np.all(y[i] == [0, 0, 1, 0], axis=-1)
        intron_y = np.all(y[i] == [0, 0, 0, 1], axis=-1)
        genic_y = np.logical_or(np.logical_or(utr_y, exon_y), intron_y)
        assert np.array_equal(genic_gl, genic_y)

    # first coord plus strand (test cases 1-3)
    assert np.array_equal(gl[0][:400], np.full((400,), 400, dtype=np.uint32))
    assert np.array_equal(gl[0][400:1199], np.full((1199 - 400,), 0, dtype=np.uint32))
    assert np.array_equal(gl[0][1199:1400], np.full((1400 - 1199,), 201, dtype=np.uint32))

    # second coord plus strand (test cases 5-6)
    assert np.array_equal(gl[2][:300], np.full((300,), 300, dtype=np.uint32))
    assert np.array_equal(gl[2][300:549], np.full((549 - 300,), 0, dtype=np.uint32))
    assert np.array_equal(gl[2][549:750], np.full((750 - 549,), 201, dtype=np.uint32))

    # second coord minus strand (test cases 7-8)
    # check 0-padding
    assert np.array_equal(gl[3][-(5000 - 1755):], np.full((5000 - 1755,), 0, dtype=np.uint32))
    # check genic regions
    gl_3 = np.flip(gl[3])[5000 - 1755:]
    assert np.array_equal(gl_3[:949], np.full((949,), 0, dtype=np.uint32))
    assert np.array_equal(gl_3[949:1350], np.full((1350 - 949,), 401, dtype=np.uint32))
    assert np.array_equal(gl_3[1350:1549], np.full((1549 - 1350,), 0, dtype=np.uint32))
    assert np.array_equal(gl_3[1549:1750], np.full((1750 - 1549,), 201, dtype=np.uint32))
    f.close()


def test_featureless_filter():
    """Tests the exclusion/inclusion of chunks from featureless coordinates"""
    _, controller, _ = setup_dummyloci()
    # dump the whole db in chunks into a .h5 file
    controller.export(chunk_size=5000, genomes='', exclude='', val_size=0.2, one_hot=True,
                      all_transcripts=True)

    f = h5py.File(H5_OUT_FILE, 'r')
    y = f['/data/y']
    sw = f['/data/sample_weights'][:]
    print(sw.shape)
    print(sw)
    print(np.all(sw == 0, axis=1), 'wwwwwwwww')
    print(f['data/X'][:])
    assert len(y) == 4  # one for each coord and strand, without featureless coord 3 (filtered)
    f.close()
    # dump the whole db in chunks into a .h5 file
    _, controller, _ = setup_dummyloci()
    controller.export(chunk_size=5000, genomes='', exclude='', val_size=0.2, one_hot=True,
                      all_transcripts=True, keep_featureless=True)

    f = h5py.File(H5_OUT_FILE, 'r')
    y = f['/data/y']
    assert len(y) == 6  # one for each coord and strand, with featureless coord 3 (kept)
    f.close()


# Setup dummy sequence with different feature transitions
def setup_feature_transitions():
    sess = mk_memory_session()
    genome = Genome()
    coord = Coordinate(genome=genome, sequence='A' * 720, length=720, seqid='a')
    s1 = SuperLocus()
    transcript = geenuff.orm.Transcript(super_locus=s1)
    piece = geenuff.orm.TranscriptPiece(transcript=transcript, position=0)
    transcript_feature_tr = geenuff.orm.Feature(start=41,
                                                end=671,
                                                is_plus_strand=True,
                                                type=geenuff.types.GEENUFF_TRANSCRIPT,
                                                start_is_biological_start=True,
                                                end_is_biological_end=True,
                                                coordinate=coord)
    transcript_feature_cds = geenuff.orm.Feature(start=131,
                                                 end=401,
                                                 is_plus_strand=True,
                                                 type=geenuff.types.GEENUFF_CDS,
                                                 start_is_biological_start=True,
                                                 end_is_biological_end=True,
                                                 coordinate=coord)
    transcript_feature_intron1 = geenuff.orm.Feature(start=221,
                                                     end=311,
                                                     is_plus_strand=True,
                                                     type=geenuff.types.GEENUFF_INTRON,
                                                     start_is_biological_start=True,
                                                     end_is_biological_end=True,
                                                     coordinate=coord)
    transcript_feature_intron2 = geenuff.orm.Feature(start=491,
                                                     end=581,
                                                     is_plus_strand=True,
                                                     type=geenuff.types.GEENUFF_INTRON,
                                                     start_is_biological_start=True,
                                                     end_is_biological_end=True,
                                                     coordinate=coord)
    transcript_feature_tr2 = geenuff.orm.Feature(start=705,
                                                 end=720,
                                                 is_plus_strand=True,
                                                 type=geenuff.types.GEENUFF_TRANSCRIPT,
                                                 start_is_biological_start=True,
                                                 end_is_biological_end=True,
                                                 coordinate=coord)

    piece.features = [transcript_feature_tr, transcript_feature_cds, transcript_feature_intron1,
                      transcript_feature_intron2, transcript_feature_tr2]

    sess.add_all([genome, coord, s1, transcript, piece, transcript_feature_tr, transcript_feature_cds,
                  transcript_feature_intron1, transcript_feature_intron2, transcript_feature_tr2])
    sess.commit()
    return sess, coord


def test_transition_encoding_and_weights():
    """Tests encoding of feature transitions, usage of transition weights and stretched weights"""
    _, coord = setup_feature_transitions()
    numerifier = AnnotationNumerifier(coord=coord,
                                      features=coord.features,
                                      max_len=1000,
                                      one_hot=False)
    nums = numerifier.coord_to_matrices()[-1]

    # expected output of AnnotationNumerifier.coord_to_matrices()
    expect_plus_strand_encoding = np.zeros((720, 6)).astype(np.int8)
    expect_plus_strand_encoding[40:42, 0] = 1  # UTR 1 +
    expect_plus_strand_encoding[670:672, 3] = 1  # UTR 2 -
    expect_plus_strand_encoding[130:132, 1] = 1  # CDS +
    expect_plus_strand_encoding[400:402, 4] = 1  # CDS -
    expect_plus_strand_encoding[220:222, 2] = 1  # First Intron +
    expect_plus_strand_encoding[310:312, 5] = 1  # First Intron -
    expect_plus_strand_encoding[490:492, 2] = 1  # Second Intron +
    expect_plus_strand_encoding[580:582, 5] = 1  # Second Intron -
    expect_plus_strand_encoding[704:706, 0] = 1  # Start of 2. 5prime UTR

    expect_minus_strand_encoding = np.zeros((720, 6)).astype(np.int8)

    assert np.array_equal(nums['plus'][0], expect_plus_strand_encoding)
    assert np.array_equal(nums['minus'][0], expect_minus_strand_encoding)

    # initializing variables + reshape   
    transitions_plus_strand = np.array(nums['plus']).reshape((8, 9, 10, 6))
    transitions_minus_strand = np.array(nums['minus']).reshape((8, 9, 10, 6))
    transition_weights = [10, 20, 30, 40, 50, 60]
    stretch = 0  # if stretch is not called the default value is 0

    # tw = Transition weights; xS = xStretch
    applied_tw_no_stretch_plus = LSTMSequence._squish_tw_to_sw(transitions_plus_strand, transition_weights, stretch)
    applied_tw_no_stretch_minus = LSTMSequence._squish_tw_to_sw(transitions_minus_strand, transition_weights, stretch)
    expect_tw_minus = np.ones((8, 9))
    assert np.array_equal(expect_tw_minus, applied_tw_no_stretch_minus)

    expect_no_stretch = np.array([
        [1, 1, 1, 1, 10, 1, 1, 1, 1],
        [1, 1, 1, 1, 20, 1, 1, 1, 1],
        [1, 1, 1, 1, 30, 1, 1, 1, 1],
        [1, 1, 1, 1, 60, 1, 1, 1, 1],
        [1, 1, 1, 1, 50, 1, 1, 1, 1],
        [1, 1, 1, 1, 30, 1, 1, 1, 1],
        [1, 1, 1, 1, 60, 1, 1, 1, 1],
        [1, 1, 1, 1, 40, 1, 1, 10, 1]
    ])
    assert np.array_equal(applied_tw_no_stretch_plus, expect_no_stretch)

    # transition weights are spread over sample weights in each direction
    # amplifies area around the transition by: 
    # [ tw/2**3 [ tw/2**2 [ tw/2**1 [ tw ] tw/2**1 ] tw/2**2] .. 
    stretch = 3
    expect_3_stretch = np.array([
        [1, 1.25, 2.5, 5, 10, 5, 2.5, 1.25, 1],
        [1, 2.5, 5, 10, 20, 10, 5, 2.5, 1],
        [1, 3.75, 7.5, 15, 30, 15, 7.5, 3.75, 1],
        [1, 7.5, 15, 30, 60, 30, 15, 7.5, 1],
        [1, 6.25, 12.5, 25, 50, 25, 12.5, 6.25, 1],
        [1, 3.75, 7.5, 15, 30, 15, 7.5, 3.75, 1],
        [1, 7.5, 15, 30, 60, 30, 15, 7.5, 1],
        [1, 5, 10, 20, 40, 20, 5, 10, 5],  # works as intended,
        # but should be [.., 40, 20, 10, 10, 5]
        # should not be a problem for smaller s_tw values
        # due to feature transition frequency
    ])
    applied_tw_3_stretch_plus = LSTMSequence._squish_tw_to_sw(transitions_plus_strand, transition_weights, stretch)
    assert np.array_equal(applied_tw_3_stretch_plus, expect_3_stretch)


### RNAseq / coverage or scoring related (evaluation)
def test_contiguous_bits():
    """confirm correct splitting at sequence breaks or after filtering when data is chunked for mem efficiency"""

    h5 = h5py.File(EVAL_H5, 'r')
    bits_plus, bits_minus = rnaseq.find_contiguous_segments(h5, start_i=0, end_i=h5['data/start_ends'].shape[0],
                                                            chunk_size=h5['evaluation/coverage'].shape[1])

    assert [len(x.start_ends) for x in bits_plus] == [2, 1, 1, 1]
    assert [len(x.start_ends) for x in bits_minus] == [1, 2, 1, 1]
    assert [x.start_i_h5 for x in bits_plus] == [0, 2, 3, 4]
    assert [x.end_i_h5 for x in bits_plus] == [2, 3, 4, 5]
    assert [x.start_i_h5 for x in bits_minus] == [5, 6, 8, 9]
    assert [x.end_i_h5 for x in bits_minus] == [6, 8, 9, 10]

    for b in bits_plus:
        print(b)
    print('---- and now minus ----')
    for b in bits_minus:
        print(b)
    h5.close()


def test_coverage_in_bits():
    # coverage arrays have the total sequence length [0, 133333) and data for every point
    # just needs to be divvied up to match the bits of sequence that exist in the h5 start_ends
    length = 133333
    coverage = np.arange(length)
    rev_coverage = np.arange(10 ** 6, 10 ** 6 + length, 1)
    print(coverage, rev_coverage)
    h5 = h5py.File(EVAL_H5, 'a')
    start_ends = h5['data/start_ends'][:]
    print(start_ends)
    chunk_size = h5['evaluation/coverage'].shape[1]
    bits_plus, bits_minus = rnaseq.find_contiguous_segments(h5, start_i=0, end_i=h5['data/start_ends'].shape[0],
                                                            chunk_size=chunk_size)
    rnaseq.write_in_bits(coverage, bits_plus, h5['evaluation/coverage'], chunk_size)
    rnaseq.write_in_bits(rev_coverage, bits_minus, h5['evaluation/coverage'], chunk_size)
    for i, (start, end) in enumerate(start_ends):
        cov_chunk = h5['evaluation/coverage'][i]
        assert end != start
        print(start, end, h5['evaluation/coverage'][i])
        if start < end:
            pystart, pyend = start, end
            is_plus = True
        else:
            pystart, pyend = end, start
            is_plus = False
        # remember, forward coverage was set to be the index, and rev coverage 1mil + index before (forward dir)
        # padding stays -1
        if pyend == length:
            # edge case, +strand, end of seq
            if is_plus:
                assert cov_chunk[0] == pystart
                assert cov_chunk[length % chunk_size - 1] == length - 1
                assert cov_chunk[length % chunk_size] == -1
                assert cov_chunk[-1] == -1
            # edge case, -strand, end of seq (or maybe start?... but it's handled more like an end)
            else:
                # this is flipped, and then padded right... not contiguous and not like the others... -_-
                # padding separates the end piece, from the next otherwise contiguous segment
                assert cov_chunk[-1] == -1
                assert cov_chunk[length % chunk_size] == -1
                assert cov_chunk[length % chunk_size - 1] == pystart + 10 ** 6
                assert cov_chunk[0] == length - 1 + 10 ** 6
        # normal, +strand
        elif is_plus:
            assert cov_chunk[0] == pystart
            assert cov_chunk[-1] == pyend - 1
        # normal, -strand
        else:
            assert cov_chunk[-1] == pystart + 10 ** 6
            assert cov_chunk[0] == pyend + 10 ** 6 - 1
    h5.close()


def test_super_chunking4write():
    """Tests that the exact same h5 is produced, regardless of how many super-chunks it is written in"""
    _, controller, _ = setup_dummyloci()
    # dump the whole db in chunks into a .h5 file
    n_writing_chunks = controller.export(chunk_size=500, genomes='', exclude='', val_size=0.2, one_hot=True,
                                         all_transcripts=True,
                                         write_by=10_000_000_000)  # write by left large enough to yield just once

    f = h5py.File(H5_OUT_FILE, 'r')
    y0 = np.array(f['/data/y'][:])
    se0 = np.array(f['/data/start_ends'][:])
    seqids0 = np.array(f['/data/seqids'][:])
    species0 = np.array(f['/data/species'][:])
    gl0 = np.array(f['/data/gene_lengths'][:])
    x0 = np.array(f['/data/X'][:])

    assert n_writing_chunks == 4  # 2 per feature-containing coordinate
    _, controller, _ = setup_dummyloci()
    # dump the whole db in chunks into a .h5 file
    n_writing_chunks = controller.export(chunk_size=500, genomes='', exclude='', val_size=0.2, one_hot=True,
                                         all_transcripts=True,
                                         write_by=1000)  # write by should result in multiple super-chunks

    f = h5py.File(H5_OUT_FILE, 'r')
    y1 = np.array(f['/data/y'][:])
    se1 = np.array(f['/data/start_ends'][:])
    seqids1 = np.array(f['/data/seqids'][:])
    species1 = np.array(f['/data/species'][:])
    gl1 = np.array(f['/data/gene_lengths'][:])
    x1 = np.array(f['/data/X'][:])
    assert np.all(se0 == se1)
    assert np.all(seqids0 == seqids1)
    assert np.all(species0 == species1)
    assert np.all(x0 == x1)
    assert np.all(y0 == y1)
    assert np.all(gl0 == gl1)
    # this makes sure it's being writen in multiple pieces at all
    assert n_writing_chunks == 8  # 4 per feature-containing coordinate (each is 1000 < 2000 in length)

    # finally, make sure it fails on invalid write_by val
    _, controller, _ = setup_dummyloci()
    # dump the whole db in chunks into a .h5 file
    with pytest.raises(ValueError):
        controller.export(chunk_size=500, genomes='', exclude='', val_size=0.2, one_hot=True,
                          all_transcripts=True,
                          write_by=1001)  # write by should result in multiple super-chunks


def test_rangefinder():
    _, controller, _ = setup_dummyloci()
    # dump the whole db in chunks into a .h5 file
    n_writing_chunks = controller.export(chunk_size=500, genomes='', exclude='', val_size=0.2, one_hot=True,
                                         all_transcripts=True,
                                         write_by=10_000_000_000)
    f = h5py.File(H5_OUT_FILE, 'r')
    sp_seqid_ranges = helpers.get_sp_seq_ranges(f)
    assert sp_seqid_ranges == {b'dummy': {'start': 0, 'seqids': {b'1': [0, 8], b'2': [8, 16]}, 'end': 16}}
