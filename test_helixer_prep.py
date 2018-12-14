import sequences
import structure
import annotations
import annotations_orm
import helpers
import type_enums
import pytest
import partitions
import copy
from dustdas import gffhelper
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sqlalchemy
import gff_2_annotations


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


### annotations_orm ###
def mk_session():
    engine = create_engine('sqlite:///:memory:', echo=False)
    annotations_orm.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_annogenome2sequence_infos_relation():
    sess = mk_session()
    ag = annotations_orm.AnnotatedGenome(species='Athaliana', version='1.2', acquired_from='Phytozome12')
    sequence_info = annotations_orm.SequenceInfo(annotated_genome=ag)
    assert ag is sequence_info.annotated_genome
    # actually put everything in db
    sess.add(sequence_info)
    sess.commit()
    # check primary keys were assigned
    assert ag.id == 1
    assert sequence_info.id == 1
    # check we can access sequence_info from ag
    sequence_info_q = ag.sequence_infos[0]
    assert sequence_info is sequence_info_q
    assert ag is sequence_info.annotated_genome
    assert ag.id == sequence_info_q.annotated_genome_id
    # check we get logical behavior on deletion
    sess.delete(sequence_info)
    sess.commit()
    assert len(ag.sequence_infos) == 0
    print(sequence_info.annotated_genome)
    sess.delete(ag)
    sess.commit()
    with pytest.raises(sqlalchemy.exc.InvalidRequestError):
        sess.add(sequence_info)


def test_processing_set_enum():
    sess = mk_session()
    # valid numbers can be setup
    ps = annotations_orm.ProcessingSet(annotations_orm.ProcessingSet.train)
    ps2 = annotations_orm.ProcessingSet('train')
    assert ps == ps2
    # other numbers can't
    with pytest.raises(ValueError):
        annotations_orm.ProcessingSet('training')
    with pytest.raises(ValueError):
        annotations_orm.ProcessingSet(1.3)
    with pytest.raises(ValueError):
        annotations_orm.ProcessingSet('Dev')
    ag = annotations_orm.AnnotatedGenome()
    sequence_info = annotations_orm.SequenceInfo(processing_set=ps, annotated_genome=ag)
    sequence_info2 = annotations_orm.SequenceInfo(processing_set=ps2, annotated_genome=ag)
    assert sequence_info.processing_set.value == 'train'
    assert sequence_info2.processing_set.value == 'train'

    sess.add_all([sequence_info, sequence_info2])
    sess.commit()
    # make sure we get the exact same handling when we come back out of the database
    for s in ag.sequence_infos:
        assert s.processing_set.value == 'train'
    # null value works
    sequence_info3 = annotations_orm.SequenceInfo(annotated_genome=ag)
    assert sequence_info3.processing_set is None


def test_coordinate_contraints():
    sess = mk_session()
    coors = annotations_orm.Coordinates(start=1, end=30, seqid='abc')
    coors2 = annotations_orm.Coordinates(start=1, end=1, seqid='abc')
    coors_bad1 = annotations_orm.Coordinates(start=0, end=30, seqid='abc')
    coors_bad2 = annotations_orm.Coordinates(start=100, end=30, seqid='abc')
    coors_bad3 = annotations_orm.Coordinates(start=1, end=30)
    # should be ok
    sess.add_all([coors, coors2])
    sess.commit()
    # should cause trouble
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        sess.add(coors_bad1)  # start below 1
        sess.commit()
    sess.rollback()
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        sess.add(coors_bad2)  # end below start
        sess.commit()
    sess.rollback()
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        sess.add(coors_bad3)
        sess.commit()


def test_coordinate_seqinfo_query():
    sess = mk_session()
    ag = annotations_orm.AnnotatedGenome()
    si_model = annotations_orm.SequenceInfo(annotated_genome=ag)
    slic = annotations.SequenceInfoHandler()
    slic.add_data(si_model)
    coors = annotations_orm.Coordinates(start=1, end=30, seqid='abc', sequence_info=si_model)
    coors2 = annotations_orm.Coordinates(start=11, end=330, seqid='def', sequence_info=si_model)
    seq_info = slic.seq_info
    # should be ok
    sess.add_all([coors, coors2])
    assert slic.seq_info['abc'].start == 1
    assert slic.seq_info['def'].end == 330
    assert seq_info is slic.seq_info


def test_many2many_scribed2slated():
    ag = annotations_orm.AnnotatedGenome()
    sinfo = annotations_orm.SequenceInfo(annotated_genome=ag)
    # test transcript to multi proteins
    sl = annotations_orm.SuperLocus(sequence_info=sinfo)
    scribed0 = annotations_orm.Transcribed(super_locus=sl)
    slated0 = annotations_orm.Translated(super_locus=sl, transcribeds=[scribed0])
    slated1 = annotations_orm.Translated(super_locus=sl, transcribeds=[scribed0])
    # test protein to multi transcripts
    assert set(scribed0.translateds) == {slated0, slated1}
    slated2 = annotations_orm.Translated(super_locus=sl)
    scribed1 = annotations_orm.Transcribed(super_locus=sl)
    scribed2 = annotations_orm.Transcribed(super_locus=sl)
    slated2.transcribeds = [scribed1, scribed2]
    assert set(slated2.transcribeds) == {scribed1, scribed2}
    scribed3 = annotations_orm.Transcribed(super_locus=sl)
    slated2.transcribeds.append(scribed3)
    assert set(slated2.transcribeds) == {scribed1, scribed2, scribed3}


def test_many2many_with_features():
    ag = annotations_orm.AnnotatedGenome()
    sinfo = annotations_orm.SequenceInfo(annotated_genome=ag)
    sl = annotations_orm.SuperLocus(sequence_info=sinfo)
    # one transcript, multiple proteins
    scribed0 = annotations_orm.Transcribed(super_locus=sl)
    slated0 = annotations_orm.Translated(super_locus=sl, transcribeds=[scribed0])
    slated1 = annotations_orm.Translated(super_locus=sl, transcribeds=[scribed0])
    # features representing alternative start codon for proteins on one transcript
    feat0_tss = annotations_orm.Feature(super_locus=sl, transcribeds=[scribed0])
    feat1_tss = annotations_orm.Feature(super_locus=sl, transcribeds=[scribed0])
    feat2_stop = annotations_orm.Feature(super_locus=sl, translateds=[slated0, slated1])
    feat3_start = annotations_orm.Feature(super_locus=sl, translateds=[slated0])
    feat4_start = annotations_orm.Feature(super_locus=sl, translateds=[slated1])
    # test they all made it to super locus
    assert len(sl.features) == 5
    # test multi features per translated worked
    assert len(slated0.features) == 2
    # test mutli translated per feature worked
    assert len(feat2_stop.translateds) == 2
    assert len(feat3_start.translateds) == 1
    assert len(feat0_tss.translateds) == 0
    # test we can get to all of this from transcribed
    indirect_features = set()
    for slated in scribed0.translateds:
        for f in slated.features:
            indirect_features.add(f)
    assert len(indirect_features) == 3


def test_feature_has_its_things():
    sess = mk_session()
    # should be ok

    ag = annotations_orm.AnnotatedGenome()
    sinfo = annotations_orm.SequenceInfo(annotated_genome=ag)
    sl = annotations_orm.SuperLocus(sequence_info=sinfo)
    # test feature with nothing much set
    f = annotations_orm.Feature(super_locus=sl)
    sess.add(f)
    sess.commit()

    assert f.is_plus_strand is None
    assert f.source is None
    assert f.seqid is None
    assert f.score is None
    # test feature with
    f1 = annotations_orm.Feature(super_locus=sl, is_plus_strand=False, seqid='abc', start=3, end=6)
    assert not f1.is_plus_strand
    assert f1.seqid == 'abc'
    assert f1.start == 3
    assert f1.end == 6
    # test bad input
    with pytest.raises(KeyError):
        f2 = annotations_orm.Feature(super_locus=f)

    f2 = annotations_orm.Feature(start=3, end=1)
    sess.add(f2)
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        sess.commit()
    sess.rollback()

    f2 = annotations_orm.Feature(is_plus_strand=-1)  # note that 0, and 1 are accepted
    sess.add(f2)
    with pytest.raises(sqlalchemy.exc.StatementError):
        sess.commit()
    sess.rollback()


def test_feature_streamlinks():
    sess = mk_session()
    f = annotations_orm.Feature(start=1)
    sfA0 = annotations_orm.UpstreamFeature(start=2)
    sfA1 = annotations_orm.DownstreamFeature(start=3, upstream=sfA0)
    sess.add_all([f, sfA0, sfA1])
    sess.commit()
    sfA0back = sess.query(annotations_orm.UpstreamFeature).first()
    assert sfA0back is sfA0
    sfA1back = sfA0back.downstream
    assert sfA1 is sfA1back
    assert sfA1.upstream is sfA0
    sf_friendless = annotations_orm.DownstreamFeature(start=4)
    sess.add(sf_friendless)
    sess.commit()
    downstreams = sess.query(annotations_orm.DownstreamFeature).all()
    downlinked = sess.query(annotations_orm.DownstreamFeature).filter(
        annotations_orm.DownstreamFeature.upstream != None  # todo, isn't there a 'right' way to do this?
    ).all()
    print([(x.start, x.upstream) for x in downstreams])
    print([(x.start, x.upstream) for x in downlinked])
    assert len(downstreams) == 2
    assert len(downlinked) == 1


def test_linking_via_fkey():
    sess = mk_session()
    sfA0 = annotations_orm.UpstreamFeature(start=2)
    sess.add(sfA0)
    sess.commit()
    sfA1 = annotations_orm.DownstreamFeature(start=3, upstream_id=sfA0.id)
    sess.add(sfA1)
    sess.commit()
    assert sfA1.upstream is sfA0
    assert sfA0.downstream is sfA1


def test_delinking_from_oneside():
    sess = mk_session()
    ag = annotations_orm.AnnotatedGenome()
    place_holder = annotations_orm.AnnotatedGenome()
    si0 = annotations_orm.SequenceInfo(annotated_genome=ag)
    si1 = annotations_orm.SequenceInfo(annotated_genome=ag)
    sess.add_all([ag, si0, si1])
    sess.commit()
    assert len(ag.sequence_infos) == 2
    ag.sequence_infos.remove(si0)
    si0.annotated_genome = place_holder  # else we'll fail the not NULL constraint
    sess.commit()
    # removed from ag
    assert len(ag.sequence_infos) == 1
    # but still in table
    assert len(sess.query(annotations_orm.SequenceInfo).all()) == 2


### annotations ###
def test_copy_over_attr():
    sess = mk_session()
    data_ag, dummy_ag = setup_data_handler(annotations.AnnotatedGenomeHandler, annotations_orm.AnnotatedGenome,
                                           species='mammoth', version='v1.0.3', acquired_from='nowhere')
    odata_ag, other_ag = setup_data_handler(annotations.AnnotatedGenomeHandler, annotations_orm.AnnotatedGenome)

    sess.add_all([data_ag, other_ag.data])
    # make sure copy only copies what it says and nothing else
    dummy_ag.copy_data_attr_to_other(other_ag, copy_only='species')
    assert other_ag.get_data_attribute('species') == 'mammoth'
    assert other_ag.get_data_attribute('version') is None
    sess.add_all([odata_ag, data_ag])
    # make sure do_not_copy excludes what is says and copies the rest
    dummy_ag.copy_data_attr_to_other(other_ag, do_not_copy='acquired_from')
    assert other_ag.get_data_attribute('version') == 'v1.0.3'
    assert other_ag.get_data_attribute('acquired_from') is None
    # make sure everything is copied
    dummy_ag.copy_data_attr_to_other(other_ag)
    assert other_ag.get_data_attribute('acquired_from') == 'nowhere'
    # make sure commit/actual entry works
    # sess.add_all([data_ag, other_ag.data])
    sess.commit()
    assert dummy_ag.get_data_attribute('species') == 'mammoth'
    assert other_ag.get_data_attribute('species') == 'mammoth'
    assert other_ag.get_data_attribute('acquired_from') == 'nowhere'
    assert other_ag.data.id is not None


def test_swap_link_annogenome2seqinfo():
    sess = mk_session()

    ag, agh = setup_data_handler(annotations.AnnotatedGenomeHandler, annotations_orm.AnnotatedGenome)
    ag2, ag2h = setup_data_handler(annotations.AnnotatedGenomeHandler, annotations_orm.AnnotatedGenome)

    si, sih = setup_data_handler(annotations.SequenceInfoHandler, annotations_orm.SequenceInfo, annotated_genome=ag)

    sess.add_all([ag, ag2, si])
    sess.commit()
    assert agh.data.sequence_infos == [sih.data]
    agh.de_link(sih)
    ag2h.link_to(sih)
    sess.commit()
    assert agh.data.sequence_infos == []
    assert ag2h.data.sequence_infos == [sih.data]
    # swap back from sequence info interface
    sih.de_link(ag2h)
    sih.link_to(agh)
    assert agh.data.sequence_infos == [sih.data]
    assert ag2h.data.sequence_infos == []


def test_swap_link_seqinfo2superlocus():
    sess = mk_session()
    ag = annotations_orm.AnnotatedGenome()

    si, sih = setup_data_handler(annotations.SequenceInfoHandler, annotations_orm.SequenceInfo, annotated_genome=ag)

    si2, si2h = setup_data_handler(annotations.SequenceInfoHandler, annotations_orm.SequenceInfo, annotated_genome=ag)

    slc, slch = setup_data_handler(annotations.SuperLocusHandler, annotations_orm.SuperLocus, sequence_info=si)
    sess.add_all([si, si2, slc])
    sess.commit()

    sih.de_link(slch)
    si2h.link_to(slch)
    assert sih.data.super_loci == []
    assert si2h.data.super_loci == [slch.data]
    # and back from super locus side
    slch.de_link(si2h)
    slch.link_to(sih)
    assert si2h.data.super_loci == []
    assert sih.data.super_loci == [slch.data]


def test_swap_links_superlocus2ttfs():
    sess = mk_session()

    slc, slch = setup_data_handler(annotations.SuperLocusHandler, annotations_orm.SuperLocus)

    slc2, slc2h = setup_data_handler(annotations.SuperLocusHandler, annotations_orm.SuperLocus)

    scribed, scribedh = setup_data_handler(annotations.TranscribedHandler, annotations_orm.Transcribed,
                                           super_locus=slc)
    slated, slatedh = setup_data_handler(annotations.TranslatedHandler, annotations_orm.Translated,
                                         super_locus=slc, transcribeds=[scribed])
    feature, featureh = setup_data_handler(annotations.FeatureHandler, annotations_orm.Feature,
                                           super_locus=slc)

    sess.add_all([slc, slc2, scribed, slated])
    sess.commit()
    # swapping super locus
    slch.de_link(slatedh)
    slch.de_link(scribedh)
    slch.de_link(featureh)
    slc2h.link_to(slatedh)
    slc2h.link_to(scribedh)
    slc2h.link_to(featureh)
    assert scribed.super_locus is slc2
    assert slated.super_locus is slc2
    assert feature.super_locus is slc2
    assert slc.translateds == []
    assert slc.transcribeds == []
    assert slc.features == []
    # swapping back from transcribed, translated, feature side
    scribedh.de_link(slc2h)
    slatedh.de_link(slc2h)
    featureh.de_link(slc2h)
    scribedh.link_to(slch)
    slatedh.link_to(slch)
    featureh.link_to(slch)
    assert slated.super_locus is slc
    assert scribed.super_locus is slc
    assert feature.super_locus is slc
    assert slc2.translateds == []
    assert slc2.transcribeds == []
    assert slc2.features == []
    sess.commit()


def test_swap_links_t2t2f():
    sess = mk_session()

    slc = annotations_orm.SuperLocus()

    scribed, scribedh = setup_data_handler(annotations.TranscribedHandler, annotations_orm.Transcribed, super_locus=slc)
    slated, slatedh = setup_data_handler(annotations.TranslatedHandler, annotations_orm.Translated, super_locus=slc,
                                         transcribeds=[scribed])
    feature, featureh = setup_data_handler(annotations.FeatureHandler, annotations_orm.Feature, super_locus=slc,
                                           transcribeds=[scribed])

    sess.add_all([slc, scribed, slated, feature])
    sess.commit()

    assert scribed.translateds == [slated]
    assert slated.transcribeds == [scribed]
    assert feature.transcribeds == [scribed]
    # de_link / link_to from scribed side
    scribedh.de_link(slatedh)
    scribedh.de_link(featureh)
    assert slated.transcribeds == []
    assert scribed.translateds == []
    assert feature.transcribeds == []
    scribedh.link_to(slatedh)
    assert scribed.translateds == [slated]
    assert slated.transcribeds == [scribed]
    # de_link / link_to from slated side
    slatedh.de_link(scribedh)
    assert slated.transcribeds == []
    assert scribed.translateds == []
    slatedh.link_to(scribedh)
    slatedh.link_to(featureh)
    assert scribed.translateds == [slated]
    assert slated.transcribeds == [scribed]
    assert feature.translateds == [slated]
    # mod links from feature side
    featureh.de_link(slatedh)
    featureh.link_to(scribedh)
    assert slated.features == []
    assert scribed.features == [feature]
    sess.commit()


def setup_data_handler(handler_type, data_type, **kwargs):
    data = data_type(**kwargs)
    handler = handler_type()
    handler.add_data(data)
    return data, handler


def test_replacelinks():
    sess = mk_session()
    slc = annotations_orm.SuperLocus()

    scribed, scribedh = setup_data_handler(annotations.TranscribedHandler, annotations_orm.Transcribed, super_locus=slc)
    assert scribed.super_locus is slc
    slated, slatedh = setup_data_handler(annotations.TranslatedHandler, annotations_orm.Translated, super_locus=slc)
    f0, f0h = setup_data_handler(annotations.FeatureHandler, annotations_orm.Feature, super_locus=slc,
                                 translateds=[slated])

    f1, f1h = setup_data_handler(annotations.FeatureHandler, annotations_orm.Feature, super_locus=slc,
                                 translateds=[slated])

    f2, f2h = setup_data_handler(annotations.FeatureHandler, annotations_orm.Feature, super_locus=slc,
                                 translateds=[slated])

    sess.add_all([slc, scribed, slated, f0, f1, f2])
    sess.commit()
    assert len(slated.features) == 3
    slatedh.replace_selflinks_w_replacementlinks(replacement=scribedh, to_replace=['features'])

    assert len(slated.features) == 0
    assert len(scribed.features) == 3


def test_data_frm_gffentry():
    #sess = mk_session()
    controller = gff_2_annotations.ImportControl(database_path=None, err_path=None)
    sess = controller.session
    gene_string = 'NC_015438.2\tGnomon\tgene\t4343\t5685\t.\t+\t.\tID=gene0;Dbxref=GeneID:104645797;Name=LOC10'
    mrna_string = 'NC_015438.2\tBestRefSeq\tmRNA\t13024\t15024\t.\t+\t.\tID=rna0;Parent=gene0;Dbxref=GeneID:'
    exon_string = 'NC_015438.2\tGnomon\texon\t4343\t4809\t.\t+\t.\tID=id1;Parent=rna0;Dbxref=GeneID:104645797'
    gene_entry = gffhelper.GFFObject(gene_string)
    handler = gff_2_annotations.SuperLocusHandler()
    handler.gen_data_from_gffentry(gene_entry)
    sess.add(handler.data)
    sess.commit()
    assert handler.data.given_id == 'gene0'
    assert handler.data.type.value == 'gene'

    mrna_entry = gffhelper.GFFObject(mrna_string)
    mandler = gff_2_annotations.TranscribedHandler()
    mandler.gen_data_from_gffentry(mrna_entry, super_locus=handler.data)
    sess.add(mandler.data)
    sess.commit()
    assert mandler.data.given_id == 'rna0'
    assert mandler.data.type.value == 'mRNA'
    assert mandler.data.super_locus is handler.data

    exon_entry = gffhelper.GFFObject(exon_string)
    controller.clean_entry(exon_entry)
    hendler = gff_2_annotations.FeatureHandler()
    hendler.gen_data_from_gffentry(exon_entry, super_locus=handler.data, transcribeds=[mandler.data])

    d = hendler.data
    s = """
    seqid {} {}
    start {} {}
    end {} {}
    is_plus_strand {} {}
    score {} {}
    source {} {}
    phase {} {}
    given_id {} {}""".format(d.seqid, type(d.seqid),
                          d.start, type(d.start),
                          d.end, type(d.end),
                          d.is_plus_strand, type(d.is_plus_strand),
                          d.score, type(d.score),
                          d.source, type(d.source),
                          d.phase, type(d.phase),
                          d.given_id, type(d.given_id))
    print(s)
    sess.add(hendler.data)
    sess.commit()

    assert hendler.data.start == 4343
    assert hendler.data.is_plus_strand
    assert hendler.data.score is None
    assert hendler.data.seqid == 'NC_015438.2'
    assert hendler.data.type.value == 'exon'
    assert hendler.data.super_locus is handler.data
    assert mandler.data in hendler.data.transcribeds
    assert hendler.data.translateds == []


def test_data_from_cds_gffentry():
    s = "NC_015447.2\tGnomon\tCDS\t5748\t5840\t.\t-\t0\tID=cds28210;Parent=rna33721;Dbxref=GeneID:101263940,Genbank:" \
        "XP_004248424.1;Name=XP_004248424.1;gbkey=CDS;gene=LOC101263940;product=protein IQ-DOMAIN 14-like;" \
        "protein_id=XP_004248424.1"
    cds_entry = gffhelper.GFFObject(s)
    controller = gff_2_annotations.ImportControl(database_path=None, err_path=None)
    controller.clean_entry(cds_entry)
    handler = gff_2_annotations.FeatureHandler()
    handler.gen_data_from_gffentry(cds_entry)
    print([x.value for x in type_enums.OnSequence])
    controller.session.add(handler.data)
    controller.session.commit()
    assert not handler.data.is_plus_strand
    assert handler.data.type.value == 'CDS'
    assert handler.data.phase == 0
    assert handler.data.score is None


def setup_testable_super_loci():
    controller = gff_2_annotations.ImportControl(err_path='/dev/null')
    controller.mk_session()
    controller.add_sequences('testdata/dummyloci.sequence.json')
    controller.add_gff('testdata/dummyloci.gff3')
    return controller.super_loci[0], controller


def test_organize_and_split_features():
    sl, _ = setup_testable_super_loci()
    transcript_full = [x for x in sl.transcribed_handlers if x.data.given_id == 'y']
    assert len(transcript_full) == 1
    transcript_full = transcript_full[0]
    transcript_interpreter = gff_2_annotations.TranscriptInterpreter(transcript_full)
    ordered_features = transcript_interpreter.organize_and_split_features()
    ordered_features = list(ordered_features)
    for i in [0, 4]:
        assert len(ordered_features[i]) == 1
        assert 'CDS' not in [x.data.data.type.value for x in ordered_features[i]]
    for i in [1, 2, 3]:
        assert len(ordered_features[i]) == 2
        assert 'CDS' in [x.data.data.type.value for x in ordered_features[i]]

    transcript_short = [x for x in sl.transcribed_handlers if x.data.given_id == 'z'][0]
    transcript_interpreter = gff_2_annotations.TranscriptInterpreter(transcript_short)
    ordered_features = transcript_interpreter.organize_and_split_features()
    ordered_features = list(ordered_features)
    assert len(ordered_features) == 1
    assert len(ordered_features[0]) == 2


def test_possible_types():
    cds = type_enums.OnSequence.CDS.name
    five_prime = type_enums.OnSequence.five_prime_UTR.name
    three_prime = type_enums.OnSequence.three_prime_UTR.name

    sl, _ = setup_testable_super_loci()
    transcript_full = [x for x in sl.transcribed_handlers if x.data.given_id == 'y']
    transcript_full = transcript_full[0]
    transcript_interpreter = gff_2_annotations.TranscriptInterpreter(transcript_full)
    ordered_features = transcript_interpreter.intervals_5to3(plus_strand=True)
    ordered_features = list(ordered_features)
    pt = transcript_interpreter.possible_types(ordered_features[0])
    assert set(pt) == {five_prime, three_prime}
    pt = transcript_interpreter.possible_types(ordered_features[1])
    assert set(pt) == {cds}
    pt = transcript_interpreter.possible_types(ordered_features[-1])
    assert set(pt) == {five_prime, three_prime}


def test_seqinfo_from_transcript_interpreter():
    sess = mk_session()
    ag = annotations_orm.AnnotatedGenome()
    si, sih = setup_data_handler(annotations.SequenceInfoHandler, annotations_orm.SequenceInfo,
                                 annotated_genome=ag)
    annotations_orm.Coordinates(sequence_info=si, seqid='chr1', start=1, end=100)
    annotations_orm.Coordinates(sequence_info=si, seqid='chr2', start=900, end=1222)

    sl = annotations_orm.SuperLocus(sequence_info=si)
    transcript, transcripth = setup_data_handler(gff_2_annotations.TranscribedHandler,
                                                 annotations_orm.Transcribed,
                                                 super_locus=sl)
    sess.add(sl)
    sess.commit()
    assert ag.id is not None
    transcript_interp = gff_2_annotations.TranscriptInterpreter(transcripth)
    assert transcript_interp.get_seq_end('chr2') == 1222
    assert transcript_interp.get_seq_start('chr1') == 1


def test_import_seqinfo():
    controller = gff_2_annotations.ImportControl()
    controller.mk_session()
    #genome = sequences.StructuredGenome()
    #genome.add_fasta('testdata/dummyloci.fa')
    json_path = 'testdata/dummyloci.sequence.json'
    #genome.to_json(json_path)
    controller.add_sequences(json_path)
    coors = controller.sequence_info.data.coordinates
    assert len(coors) == 1
    assert coors[0].seqid == '1'
    assert coors[0].start == 1
    assert coors[0].end == 405


def test_fullcopy():
    sess = mk_session()
    sl, slh = setup_data_handler(annotations.SuperLocusHandler, annotations_orm.SuperLocus)
    scribed, scribedh = setup_data_handler(annotations.TranscribedHandler, annotations_orm.Transcribed,
                                           super_locus=sl, given_id='soup')
    f, fh = setup_data_handler(annotations.FeatureHandler, annotations_orm.Feature, super_locus=sl,
                               transcribeds=[scribed], seqid='1', start=13, end=33)
    sess.add_all([scribed, f])
    sess.commit()
    tdict = annotations_orm.Transcribed.__dict__
    print(tdict.keys())
    for key in tdict.keys():
        print('{} {}'.format(key, type(tdict[key])))

    # try copying feature
    new, newh = setup_data_handler(annotations.FeatureHandler, annotations_orm.Feature)
    fh.fax_all_attrs_to_another(newh)
    sess.commit()
    assert new.start == 13
    assert set(scribed.features) == {f, new}
    assert new.seqid == '1'
    assert new.super_locus == sl
    assert new is not f
    assert new.id != f.id

    # try copying most of transcribed things to translated
    slated, slatedh = setup_data_handler(annotations.TranslatedHandler, annotations_orm.Translated)
    scribedh.fax_all_attrs_to_another(slatedh, skip_copying='type', skip_linking='translateds')
    sess.commit()
    assert slated.given_id == 'soup'
    assert set(slated.features) == {f, new}
    assert f.translateds == [slated]
    assert new.transcribeds == [scribed]


#def test_feature_overlap_detection():
#    sl = setup_testable_super_loci()
#    assert sl.features['ftr000000'].fully_overlaps(sl.features['ftr000004'])
#    assert sl.features['ftr000005'].fully_overlaps(sl.features['ftr000001'])
#    # a few that should not overlap
#    assert not sl.features['ftr000000'].fully_overlaps(sl.features['ftr000001'])
#    assert not sl.features['ftr000000'].fully_overlaps(sl.features['ftr000002'])
#
#
def test_transcript_interpreter():
    sl, controller = setup_testable_super_loci()
    transcript = [x for x in sl.data.transcribeds if x.given_id == 'y'][0]
    # change so that there are implicit UTRs
    t_interp = gff_2_annotations.TranscriptInterpreter(transcript.handler)
    t_interp.decode_raw_features()
    controller.session.commit()
    # has all standard features
    types_out = set([x.data.type.value for x in t_interp.clean_features])
    assert types_out == {type_enums.START_CODON,
                         type_enums.STOP_CODON,
                         type_enums.TRANSCRIPTION_START_SITE,
                         type_enums.TRANSCRIPTION_TERMINATION_SITE,
                         type_enums.DONOR_SPLICE_SITE,
                         type_enums.ACCEPTOR_SPLICE_SITE}

    assert t_interp.clean_features[-1].data.end == 400
    assert t_interp.clean_features[0].data.start == 1


def test_transcript_get_first():
    # plus strand
    sl, controller = setup_testable_super_loci()
    print(sl.data.sequence_info)
    transcript = [x for x in sl.data.transcribeds if x.given_id == 'y'][0]
    t_interp = gff_2_annotations.TranscriptInterpreter(transcript.handler)
    i0 = t_interp.intervals_5to3(plus_strand=True)[0]
    t_interp.interpret_first_pos(i0)
    features = t_interp.clean_features
    status = t_interp.status
    assert len(features) == 1
    f0 = features[0]
    print(f0)
    print(status.__dict__)
    print(i0[0].data.data.is_plus_strand)
    assert f0.data.start == 1
    assert status.is_5p_utr()
    assert f0.data.phase is None
    assert f0.data.is_plus_strand

    # minus strand
    sl, controller = setup_testable_super_loci()
    transcript = [x for x in sl.data.transcribeds if x.given_id == 'y'][0]
    for feature in sl.data.features:  # force minus strand
        feature.is_plus_strand = False

    # new transcript interpreter so the clean features reset
    t_interp = gff_2_annotations.TranscriptInterpreter(transcript.handler)
    i0 = t_interp.intervals_5to3(plus_strand=False)[0]
    t_interp.interpret_first_pos(i0, plus_strand=False)
    features = t_interp.clean_features
    status = t_interp.status
    assert len(features) == 1
    f0 = features[0]
    print(f0)
    print(status)
    print(i0[0].data.data.is_plus_strand)
    print(f0.data.type)
    assert f0.data.start == 400
    assert status.is_5p_utr()
    assert f0.data.phase is None
    assert not f0.data.is_plus_strand

    # test without UTR (x doesn't have last exon, and therefore will end in CDS)
    transcript = [x for x in sl.data.transcribeds if x.given_id == 'x'][0]
    t_interp = gff_2_annotations.TranscriptInterpreter(transcript.handler)
    i0 = t_interp.intervals_5to3(plus_strand=False)[0]
    t_interp.interpret_first_pos(i0, plus_strand=False)
    features = t_interp.clean_features
    status = t_interp.status
    assert len(features) == 3
    f_err = features[0]
    f_status_coding = features[1]
    f_status_transcribed = features[2]
    print(f_err)
    print(status)
    print(i0)
    # should get in_translated_region instead of a start codon
    assert f_status_coding.data.start == 120
    assert f_status_coding.data.type == type_enums.IN_TRANSLATED_REGION
    assert not f_status_coding.data.is_plus_strand
    # and should get accompanying in raw transcript
    assert f_status_transcribed.data.type == type_enums.IN_RAW_TRANSCRIPT
    # region beyond exon should be marked erroneous
    assert f_err.data.start == 121
    assert f_err.data.end == 405
    assert f_err.data.type == type_enums.ERROR
    assert status.is_coding()
    assert status.seen_start
    assert status.genic


def test_transcript_transition_from_5p_to_end():
    sl, controller = setup_testable_super_loci()
    transcript = [x for x in sl.data.transcribeds if x.given_id == 'y'][0]
    t_interp = gff_2_annotations.TranscriptInterpreter(transcript.handler)
    ivals_sets = t_interp.intervals_5to3(plus_strand=True)
    t_interp.interpret_first_pos(ivals_sets[0])
    # hit start codon
    t_interp.interpret_transition(ivals_before=ivals_sets[0], ivals_after=ivals_sets[1], plus_strand=True)
    features = t_interp.clean_features
    assert features[-1].data.type == type_enums.START_CODON
    assert features[-1].data.start == 11
    assert features[-1].data.end == 13
    # hit splice site
    t_interp.interpret_transition(ivals_before=ivals_sets[1], ivals_after=ivals_sets[2], plus_strand=True)
    assert features[-1].data.type == type_enums.ACCEPTOR_SPLICE_SITE
    assert features[-2].data.type == type_enums.DONOR_SPLICE_SITE
    assert features[-2].data.start == 101  # splice from
    assert features[-1].data.start == 110  # splice to
    assert t_interp.status.is_coding()
    # hit splice site
    t_interp.interpret_transition(ivals_before=ivals_sets[2], ivals_after=ivals_sets[3], plus_strand=True)
    assert features[-1].data.type == type_enums.ACCEPTOR_SPLICE_SITE
    assert features[-2].data.type == type_enums.DONOR_SPLICE_SITE
    assert features[-2].data.start == 121  # splice from
    assert features[-1].data.start == 200  # splice to
    # hit stop codon
    t_interp.interpret_transition(ivals_before=ivals_sets[3], ivals_after=ivals_sets[4], plus_strand=True)
    assert features[-1].data.type == type_enums.STOP_CODON
    assert features[-1].data.start == 298
    # hit transcription termination site
    t_interp.interpret_last_pos(ivals_sets[4], plus_strand=True)
    assert features[-1].data.type == type_enums.TRANSCRIPTION_TERMINATION_SITE
    assert features[-1].data.start == 400


def test_non_coding_transitions():
    sl, controller = setup_testable_super_loci()
    transcript = [x for x in sl.data.transcribeds if x.given_id == 'z'][0]
    t_interp = gff_2_annotations.TranscriptInterpreter(transcript.handler)
    # get single-exon no-CDS transcript
    cds = [x for x in transcript.features if x.type.value == type_enums.CDS][0]
    transcript.handler.de_link(cds.handler)
    print(transcript)
    ivals_sets = t_interp.intervals_5to3(plus_strand=True)
    assert len(ivals_sets) == 1
    t_interp.interpret_first_pos(ivals_sets[0])
    features = t_interp.clean_features
    assert features[-1].data.type == type_enums.TRANSCRIPTION_START_SITE
    assert features[-1].data.start == 111
    t_interp.interpret_last_pos(ivals_sets[0], plus_strand=True)
    assert features[-1].data.type == type_enums.TRANSCRIPTION_TERMINATION_SITE
    assert features[-1].data.start == 120
    assert len(features) == 2


def test_errors_not_lost():
    sl, controller = setup_testable_super_loci()
    feature_e, feature_eh = setup_data_handler(annotations.FeatureHandler, annotations_orm.Feature, start=40, end=80,
                                               super_locus=sl.data, type=type_enums.ERROR)
    #feature_e.id = sl.genome.feature_ider.next_unique_id()
    #feature_e.super_locus = sl.data
    #sl.features[feature_e.id] = feature_e
    #feature_e.start, feature_e.end = 40, 80
    #feature_e.change_to_error()
    print('what features did we start with::?')
    for feature in sl.data.features:
        print(feature)
    sl.check_and_fix_structure(entries=None)
    print('---and what features did we leave?---')
    for feature in sl.data.features:
        print(feature)
    assert feature_e in sl.features.values()


def test_setup_proteins():
    sl, controller = setup_testable_super_loci()
    transcript = [x for x in sl.data.transcribeds if x.given_id == 'y'][0]
    t_interp = gff_2_annotations.TranscriptInterpreter(transcript.handler)
    print(t_interp.proteins)
    assert len(t_interp.proteins.keys()) == 1
    protein = t_interp.proteins['y.p'].data
    assert transcript in protein.transcribeds
    assert protein in transcript.translateds
    assert protein.given_id == 'y.p'
    controller.session.commit()


def test_mv_features_to_prot():
    sl, controller = setup_testable_super_loci()
    transcript = [x for x in sl.data.transcribeds if x.given_id == 'y'][0]
    t_interp = gff_2_annotations.TranscriptInterpreter(transcript.handler)
    protein = t_interp.proteins['y.p'].data
    t_interp.decode_raw_features()
    t_interp.mv_coding_features_to_proteins()
    controller.session.commit()
    assert len(protein.features) == 2  # start and stop codon
    assert set([x.type.value for x in protein.features]) == {type_enums.START_CODON, type_enums.STOP_CODON}
#
#def test_anno2json_and_back():
#    # setup the sequence file
#    json_path = 'testdata/testerSl.sequence.json'
#    # can uncomment the following 4 lines if one intentionally changed the format, but check
#    # fa_path = 'testdata/testerSl.fa'
#    # sd_fa = sequences.StructuredGenome()
#    # sd_fa.add_fasta(fa_path)
#    # sd_fa.to_json(json_path)
#    sd_fa = sequences.StructuredGenome()
#    sd_fa.from_json(json_path)
#
#    gfffile = 'testdata/testerSl.gff3'
#    json_anno = 'testdata/testerSl.annotation.json'
#    ag = annotations.AnnotatedGenome()
#    ag.add_gff(gfffile, sd_fa, 'testdata/deletable')
#    ag.to_json(json_anno)
#    ag_json = annotations.AnnotatedGenome()
#    print('i do not get it')
#    ag_json.from_json(json_anno)
#    print('is it the call back to jsonable')
#    assert ag.to_jsonable() == ag_json.to_jsonable()
#    # check recursive load has worked out
#    print(ag_json.super_loci_slices[0].super_loci[0].slice, 'slice')
#    assert ag_json.super_loci_slices[0].super_loci[0].slice is ag_json.super_loci_slices[0]
#    assert ag_json.super_loci_slices[0].super_loci[0].genome is ag_json
#    # and same for super_loci
#    sl = ag_json.super_loci_slices[0].super_loci[0]
#    fkey = sorted(sl.features.keys())[0]
#    feature = sl.features[fkey]
#    assert feature.super_locus is sl
#    assert sl.slice.seq_info[feature.seqid].end == 16000
#
#
#def test_to_intervaltree():
#    sl = setup_loci_with_utr()
#    print(sl.slice.seq_info, 'seq_info')
#    print(sl.slice._seq_info, '_seq_info')
#    print(sl.slice.coordinates, 'coords')
#    trees = sl.slice.load_to_interval_tree()
#    print(sl.slice.super_loci, 'loci')
#    # get a single tree
#    assert len(trees.keys()) == 1
#    tree = trees['1']
#    for itvl in tree:
#        print(itvl)
#    assert len(tree) == len(sl.features)
#    minf = min([sl.features[f].py_start for f in sl.features])
#    maxf = max([sl.features[f].py_end for f in sl.features])
#    assert minf == min(tree).begin
#    assert maxf == max(tree).end
#
#
#def test_deepcopies():
#    sl = setup_loci_with_utr()
#    sl2 = copy.deepcopy(sl)
#    assert sl is not sl2
#    for key in sl.__dict__:
#        val = sl.__getattribute__(key)
#        # most GenericData pieces should be objects, AKA, not is
#        if isinstance(val, structure.GenericData):
#            if key is not 'slice':  # slice points up, aka, should be the same as we only copied from super locus lev.
#                assert val is not sl2.__getattribute__(key)
#        elif isinstance(val, dict) or isinstance(val, list):
#            pass  # skipping as __eq__ etc not implemented
#        else:
#            # normal values should be identical
#            if key is not 'gff_entry':  # no equality implemented for this class
#                assert val == sl2.__getattribute__(key)
#
#    # same idea for one of the sub pieces skipped above
#    f = sl.features['ftr000010']
#    f2 = sl2.features['ftr000010']
#    for key in f.__dict__:
#        val = f.__getattribute__(key)
#        if not isinstance(val, structure.GenericData):
#            if key is not 'gff_entry':  # no equality implemented for this class
#                assert val == f2.__getattribute__(key)
#        else:
#            assert val is not f2.__getattribute__(key)

#def test_swap_type():
#    sl = setup_loci_with_utr()
#    holder = sl.generic_holders['x']
#    old_n_feature_holders = len(sl.generic_holders)
#    ori_fholder_features = copy.deepcopy(holder.features)
#    protein = holder.swap_type('proteins')
#    assert len(sl.generic_holders) == old_n_feature_holders - 1
#    assert len(sl.proteins) == 1
#    assert holder.id not in sl.generic_holders
#    assert holder.id == protein.id
#    assert ori_fholder_features == protein.features
#    assert holder.super_locus is protein.super_locus
#
#
#def test_entries_are_imported():
#    sl = setup_loci_with_utr()
#    pass # todo, finish
#
#
#def test_renamer():
#    staticsl = setup_loci_with_utr()
#    sl = setup_loci_with_utr()
#    y = sl.generic_holders['y']
#    newy = y.replace_id_everywhere('newy')
#    print(sl.generic_holders.keys())
#    assert 'newy' in sl.generic_holders.keys()
#    assert 'y' not in sl.generic_holders.keys()
#    for key in sl.features.keys():
#        val_old = staticsl.features[key]
#        val = sl.features[key]
#        if 'y' in val_old.generic_holders:
#            assert 'y' not in val.generic_holders
#            assert 'newy' in val.generic_holders

#### gff_2_annotations ####
def test_gff_gen():
    controller = gff_2_annotations.ImportControl()
    x = list(controller.gff_gen('testdata/testerSl.gff3'))
    assert len(x) == 103
    assert x[0].type == 'region'
    assert x[-1].type == 'CDS'


def test_gff_useful_gen():
    controller = gff_2_annotations.ImportControl()
    x = list(controller.useful_gff_entries('testdata/testerSl.gff3'))
    assert len(x) == 100  # should drop the region entry
    assert x[0].type == 'gene'
    assert x[-1].type == 'CDS'


def test_gff_grouper():
    controller = gff_2_annotations.ImportControl()
    x = list(controller.group_gff_by_gene('testdata/testerSl.gff3'))
    assert len(x) == 5
    for group in x:
        assert group[0].type == 'gene'


#### type_enumss ####
def test_enum_non_inheritance():
    allknown = [x.name for x in list(type_enums.AllKnown)]
    allnice = [x.name for x in list(type_enums.AllKeepable)]
    # check that some random bits made it in to all
    assert 'error' in allknown
    assert 'region' in allknown

    # check that some annoying bits are not in nice set
    for not_nice in ['transcript', 'primary_transcript', 'exon', 'five_prime_UTR', 'CDS']:
        assert not_nice not in allnice
        assert not_nice in allknown

    # check nothing is there twice
    assert len(set(allknown)) == len(allknown)


def test_enums_name_val_match():
    for x in type_enums.AllKnown:
        assert x.name == x.value


#### helpers ####
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
