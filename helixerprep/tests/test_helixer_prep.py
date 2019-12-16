import os
from shutil import copy
from sklearn.metrics import precision_recall_fscore_support as f1_scores
from sklearn.metrics import accuracy_score
import numpy as np
import pytest
import h5py

import geenuff
from geenuff.tests.test_geenuff import setup_data_handler, mk_memory_session
from geenuff.applications.importer import ImportController
from geenuff.base.orm import SuperLocus, Genome, Coordinate
from geenuff.base.helpers import reverse_complement
from geenuff.base import types
from ..core.controller import HelixerController
from ..core.orm import Mer
from ..export import numerify
from ..export.numerify import SequenceNumerifier, AnnotationNumerifier, Stepper, AMBIGUITY_DECODE
from ..export.exporter import HelixerExportController
from ..prediction.ConfusionMatrix import ConfusionMatrix

TMP_DB = 'testdata/tmp.db'
DUMMYLOCI_DB = 'testdata/dummyloci.sqlite3'
H5_OUT_FOLDER = 'testdata/numerify_test_out/'
H5_OUT_FILE = H5_OUT_FOLDER + 'test_data.h5'


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
    matrix = numerifier.coord_to_matrices()[0][0]
    # ATATATAT
    x = [0., 1, 0, 0, 0., 0, 1, 0]
    expect = np.array(x * 4).reshape((-1, 4))
    assert np.array_equal(expect, matrix)

    # on the minus strand
    numerifier = SequenceNumerifier(coord=coords[3], max_len=100)
    matrix = numerifier.coord_to_matrices()[0][0]

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
    nums = numerifier.coord_to_matrices()[0][0][:405]
    expect = np.zeros([405, 3], dtype=np.float32)
    expect[0:400, 0] = 1.  # set genic/in raw transcript
    expect[10:301, 1] = 1.  # set in transcript
    expect[21:110, 2] = 1.  # both introns
    expect[120:200, 2] = 1.
    assert np.array_equal(nums, expect)


def test_sequence_slicing():
    _, coords = memory_import_fasta('testdata/basic_sequences.fa')
    seq_numerifier = SequenceNumerifier(coord=coords[0], max_len=50)
    num_list = seq_numerifier.coord_to_matrices()[0]
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
    seq_slices, seq_error_masks = seq_numerifier.coord_to_matrices()
    anno_slices, anno_error_masks, gene_lengths, transitions = anno_numerifier.coord_to_matrices()
    assert (len(seq_slices) == len(anno_slices) == len(gene_lengths) == len(transitions) ==
            len(anno_error_masks) == len(seq_error_masks) == 19 * 2)

    for s, a, se, ae in zip(seq_slices, anno_slices, seq_error_masks, anno_error_masks):
        assert s.shape[0] == a.shape[0] == se.shape[0] == ae.shape[0]

    # testing sequence error masks
    expect = np.ones((1801 * 2, ), dtype=np.int8)
    # sequence error mask should be empty
    assert np.array_equal(expect, np.concatenate(seq_error_masks))
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
    assert np.array_equal(nums[0], expect)

    # and now that we get the expect range on the minus strand,
    # keeping in mind the 40 is inclusive, and the 9, not
    expect[10:41, 0] = 1.
    expect = np.flip(expect, axis=0)
    assert np.array_equal(nums[1], expect)  # nums[1] is now from the minus strand

    # with cutting
    numerifier = AnnotationNumerifier(coord=coord,
                                      features=coord.features,
                                      max_len=50,
                                      one_hot=False)
    nums = numerifier.coord_to_matrices()[0]

    expect = np.zeros([100, 3], dtype=np.float32)
    expect[10:41, 0] = 1.

    assert np.array_equal(nums[2], np.flip(expect[50:100], axis=0))
    assert np.array_equal(nums[3], np.flip(expect[0:50], axis=0))


def test_coord_numerifier_and_h5_gen_plus_strand():
    _, controller, _ = setup_dummyloci()
    # dump the whole db in chunks into a .h5 file
    controller.export(chunk_size=400, genomes='', exclude='', val_size=0.2, one_hot=False,
                      keep_errors=False, all_transcripts=True)

    f = h5py.File(H5_OUT_FILE, 'r')
    inputs = f['/data/X']
    labels = f['/data/y']
    label_masks = f['/data/sample_weights']

    # five chunks for each the two coordinates and *2 for each strand and -2 for
    # completely erroneous sequences (at the end of the minus strand of the 2nd coord)
    # also tests if we ignore the third coordinate, that does not have any annotations
    assert len(inputs) == len(labels) == len(label_masks) == 18

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
    label_expect = np.zeros((405, 3), dtype=np.float16)
    label_expect[0:400, 0] = 1.  # set genic/in raw transcript
    label_expect[10:301, 1] = 1.  # set in transcript
    label_expect[21:110, 2] = 1.  # both introns
    label_expect[120:200, 2] = 1.
    assert np.array_equal(labels[0], label_expect[:400])
    assert np.array_equal(labels[1][:5], label_expect[400:])

    # prep anno mask
    label_mask_expect = np.ones((405, ), dtype=np.int8)
    label_mask_expect[:110] = 0
    label_mask_expect[120:] = 0
    assert np.array_equal(label_masks[0], label_mask_expect[:400])
    assert np.array_equal(label_masks[1][:5], label_mask_expect[400:])


def test_coord_numerifier_and_h5_gen_minus_strand():
    """Tests numerification of test case 8 on coordinate 2"""
    _, controller, _ = setup_dummyloci()
    # dump the whole db in chunks into a .h5 file
    controller.export(chunk_size=200, genomes='', exclude='', val_size=0.2, one_hot=False,
                      keep_errors=False, all_transcripts=True)

    f = h5py.File(H5_OUT_FILE, 'r')
    inputs = f['/data/X']
    labels = f['/data/y']
    label_masks = f['/data/sample_weights']

    assert len(inputs) == len(labels) == len(label_masks) == 33
    # the last 5 inputs/labels should be for the 2nd coord and the minus strand
    # orginally there where 9 but 4 were tossed out due to be fully erroneous
    # all the sequences are also 0-padded
    inputs = inputs[-5:]
    labels = labels[-5:]
    label_masks = label_masks[-5:]

    seq_expect = np.full((955, 4), 0.25)
    # start codon
    seq_expect[929] = np.flip(AMBIGUITY_DECODE['T'])
    seq_expect[928] = np.flip(AMBIGUITY_DECODE['A'])
    seq_expect[927] = np.flip(AMBIGUITY_DECODE['C'])
    # stop codon of second transcript
    seq_expect[902] = np.flip(AMBIGUITY_DECODE['A'])
    seq_expect[901] = np.flip(AMBIGUITY_DECODE['T'])
    seq_expect[900] = np.flip(AMBIGUITY_DECODE['C'])
    # stop codon of first transcript
    seq_expect[776] = np.flip(AMBIGUITY_DECODE['A'])
    seq_expect[775] = np.flip(AMBIGUITY_DECODE['T'])
    seq_expect[774] = np.flip(AMBIGUITY_DECODE['C'])
    # flip as the sequence is read 5p to 3p
    seq_expect = np.flip(seq_expect, axis=0)
    # insert 0-padding
    seq_expect = np.insert(seq_expect, 155, np.zeros((45, 4)), axis=0)
    assert np.array_equal(inputs[0], seq_expect[:200])
    assert np.array_equal(inputs[1][:50], seq_expect[200:250])

    label_expect = np.zeros((955, 3), dtype=np.float16)
    label_expect[749:950, 0] = 1.  # genic region
    label_expect[774:930, 1] = 1.  # transcript (2 overlapping ones)
    label_expect[850:919, 2] = 1.  # intron first transcript
    label_expect[800:879, 2] = 1.  # intron second transcript
    label_expect = np.flip(label_expect, axis=0)
    label_expect = np.insert(label_expect, 155, np.zeros((45, 3)), axis=0)
    assert np.array_equal(labels[0], label_expect[:200])
    assert np.array_equal(labels[1][:50], label_expect[200:250])

    label_mask_expect = np.ones((955, ), dtype=np.int8)
    label_mask_expect[925:] = 0
    label_mask_expect[749:850] = 0
    label_mask_expect = np.flip(label_mask_expect)
    label_mask_expect = np.insert(label_mask_expect, 155, np.zeros((45,)), axis=0)
    assert np.array_equal(label_masks[0], label_mask_expect[:200])
    assert np.array_equal(label_masks[1][:50], label_mask_expect[200:250])


def test_numerify_with_end_neg1():
    def check_one(coord, is_plus_strand, expect, maskexpect):
        numerifier = AnnotationNumerifier(coord=coord,
                                          features=coord.features,
                                          max_len=1000,
                                          one_hot=False)

        if is_plus_strand:
            nums, masks, _, _ = [x[0] for x in numerifier.coord_to_matrices()]
        else:
            nums, masks, _, _ = [x[1] for x in numerifier.coord_to_matrices()]

        if not np.array_equal(nums, expect):
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

    y_multi = numerifier.coord_to_matrices()[0][0]
    # count classes
    uniques_multi = np.unique(y_multi, return_counts=True, axis=0)

    # make one hot encoding
    numerifier = AnnotationNumerifier(coord=coord,
                                      features=coord.features,
                                      max_len=5000,
                                      one_hot=True)
    y_one_hot_4 = numerifier.coord_to_matrices()[0][0]
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
                      keep_errors=False, all_transcripts=True)

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
