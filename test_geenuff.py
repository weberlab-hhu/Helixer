from dustdas import gffhelper
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sqlalchemy
import pytest
import os

import annotations_orm
import annotations
import gff_2_annotations
import type_enums


# section: annotations_orm
def mk_session(db_path='sqlite:///:memory:'):
    engine = create_engine(db_path, echo=False)
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


def test_coordinate_constraints():
    sess = mk_session()
    coors = annotations_orm.Coordinates(start=0, end=30, seqid='abc')
    coors2 = annotations_orm.Coordinates(start=0, end=1, seqid='abc')
    coors_bad1 = annotations_orm.Coordinates(start=-12, end=30, seqid='abc')
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
    sl = annotations_orm.SuperLocus()
    f0 = annotations_orm.Feature(super_locus=sl, coordinates=coors)
    f1 = annotations_orm.Feature(super_locus=sl, coordinates=coors2)
    # should be ok
    sess.add_all([coors, coors2, sl, f0, f1])
    assert f0.coordinates.start == 1
    assert f1.coordinates.end == 330
    #assert seq_info is slic.seq_info


def test_many2many_scribed2slated():
    # test transcript to multi proteins
    sl = annotations_orm.SuperLocus()
    scribed0 = annotations_orm.Transcribed(super_locus=sl)
    slated0 = annotations_orm.Translated(super_locus=sl, transcribeds=[scribed0])
    slated1 = annotations_orm.Translated(super_locus=sl, transcribeds=[scribed0])
    # test scribed 2 scribed_piece works
    piece0 = annotations_orm.TranscribedPiece(transcribed=scribed0)
    assert piece0 == slated0.transcribeds[0].transcribed_pieces[0]
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
    sl = annotations_orm.SuperLocus()
    # one transcript, multiple proteins
    piece0 = annotations_orm.TranscribedPiece(super_locus=sl)
    scribed0 = annotations_orm.Transcribed(super_locus=sl, transcribed_pieces=[piece0])
    slated0 = annotations_orm.Translated(super_locus=sl, transcribeds=[scribed0])
    slated1 = annotations_orm.Translated(super_locus=sl, transcribeds=[scribed0])
    # features representing alternative start codon for proteins on one transcript
    feat0_tss = annotations_orm.Feature(super_locus=sl, transcribed_pieces=[piece0])
    feat1_tss = annotations_orm.Feature(super_locus=sl, transcribed_pieces=[piece0])
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
    sl = annotations_orm.SuperLocus()
    # test feature with nothing much set
    f = annotations_orm.Feature(super_locus=sl)
    sess.add(f)
    sess.commit()

    assert f.is_plus_strand is None
    assert f.source is None
    assert f.coordinates is None
    assert f.score is None
    # test feature with
    f1 = annotations_orm.Feature(super_locus=sl, is_plus_strand=False, start=3)
    assert not f1.is_plus_strand
    assert f1.start == 3
    # test bad input
    with pytest.raises(KeyError):
        f2 = annotations_orm.Feature(super_locus=f)

    f2 = annotations_orm.Feature(is_plus_strand=-1)  # note that 0, and 1 are accepted
    sess.add(f2)
    with pytest.raises(sqlalchemy.exc.StatementError):
        sess.commit()
    sess.rollback()


def test_feature_streamlinks():
    sess = mk_session()
    f = annotations_orm.Feature(start=1)
    pair = annotations_orm.UpDownPair()
    sfA0 = annotations_orm.UpstreamFeature(start=2, pairs=[pair])
    sfA1 = annotations_orm.DownstreamFeature(start=3, pairs=[pair])
    sess.add_all([f, sfA0, sfA1, pair])
    sess.commit()
    sfA0back = sess.query(annotations_orm.UpstreamFeature).first()
    assert sfA0back is sfA0
    sfA1back = sfA0back.pairs[0].downstream
    assert sfA1 is sfA1back
    assert sfA1.pairs[0].upstream is sfA0
    sf_friendless = annotations_orm.DownstreamFeature(start=4)
    sess.add(sf_friendless)
    sess.commit()
    downstreams = sess.query(annotations_orm.DownstreamFeature).all()
    pre_downlinked = sess.query(annotations_orm.UpDownPair).filter(
        annotations_orm.DownstreamFeature != None  # todo, isn't there a 'right' way to do this?
    ).all()
    downlinked = [x.downstream for x in pre_downlinked]
    print([(x.start, x.pairs) for x in downstreams])
    print([(x.start, x.pairs[0].upstream) for x in downlinked])
    assert len(downstreams) == 2
    assert len(downlinked) == 1


def test_linking_via_fkey():
    sess = mk_session()
    sfA0 = annotations_orm.UpstreamFeature(start=2)
    sfA1 = annotations_orm.DownstreamFeature(start=3)
    sess.add_all([sfA0, sfA1])
    sess.commit()
    pair = annotations_orm.UpDownPair(upstream_id=sfA0.id, downstream_id=sfA1.id)
    sess.add_all([pair])
    sess.commit()
    assert sfA1.pairs[0].upstream is sfA0
    assert sfA0.pairs[0].downstream is sfA1


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


# section: annotations
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

    scribedpiece, scribedpieceh = setup_data_handler(annotations.TranscribedPieceHandler,
                                                     annotations_orm.TranscribedPiece, super_locus=slc)
    scribed, scribedh = setup_data_handler(annotations.TranscribedHandler, annotations_orm.Transcribed,
                                           super_locus=slc, transcribed_pieces=[scribedpiece])
    slated, slatedh = setup_data_handler(annotations.TranslatedHandler, annotations_orm.Translated, super_locus=slc,
                                         transcribeds=[scribed])
    feature, featureh = setup_data_handler(annotations.FeatureHandler, annotations_orm.Feature, super_locus=slc,
                                           transcribed_pieces=[scribedpiece])

    sess.add_all([slc, scribed, scribedpiece, slated, feature])
    sess.commit()

    assert scribed.translateds == [slated]
    assert slated.transcribeds == [scribed]
    assert scribedpiece.transcribed == scribed
    assert feature.transcribed_pieces == [scribedpiece]
    # de_link / link_to from scribed side
    scribedh.de_link(slatedh)
    scribedpieceh.de_link(featureh)
    assert slated.transcribeds == []
    assert scribed.translateds == []
    assert feature.transcribed_pieces == []

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
    featureh.link_to(scribedpieceh)
    assert slated.features == []
    assert scribed.transcribed_pieces[0].features == [feature]
    sess.commit()


def test_updownhandler_links():
    sess = mk_session()
    coor_old = annotations_orm.Coordinates(start=1, end=1000, seqid='a')
    coor_new = annotations_orm.Coordinates(start=1, end=100, seqid='a')
    slc = annotations_orm.SuperLocus()
    scribedpiece, scribedpieceh = setup_data_handler(annotations.TranscribedPieceHandler,
                                                     annotations_orm.TranscribedPiece, super_locus=slc)
    scribed, scribedh = setup_data_handler(annotations.TranscribedHandler, annotations_orm.Transcribed,
                                           super_locus=slc, transcribed_pieces=[scribedpiece])
    up, uph = setup_data_handler(annotations.UpstreamFeatureHandler, annotations_orm.UpstreamFeature, super_locus=slc,
                                 transcribed_pieces=[scribedpiece], coordinates=coor_old)
    up2, up2h = setup_data_handler(annotations.UpstreamFeatureHandler, annotations_orm.UpstreamFeature, super_locus=slc,
                                   transcribed_pieces=[scribedpiece], coordinates=coor_new)

    down, downh = setup_data_handler(annotations.DownstreamFeatureHandler, annotations_orm.DownstreamFeature,
                                     coordinates=coor_old)

    pair, pairh = setup_data_handler(annotations.UpDownPairHandler, annotations_orm.UpDownPair, transcribed=scribed,
                                     upstream=up, downstream=down)
    sess.add_all([up, up2, down, slc, coor_old, coor_new, pair])
    sess.commit()
    assert up2.pairs == []
    assert up.pairs[0].downstream == down
    pairh.de_link(uph)
    pairh.link_to(up2h)
    assert up2.pairs[0].downstream == down
    assert up.pairs == []


def setup_data_handler(handler_type, data_type, **kwargs):
    data = data_type(**kwargs)
    handler = handler_type()
    handler.add_data(data)
    return data, handler


def test_replacelinks():
    sess = mk_session()
    slc = annotations_orm.SuperLocus()

    scribedpiece, scribedpieceh = setup_data_handler(annotations.TranscribedPieceHandler,
                                                     annotations_orm.TranscribedPiece, super_locus=slc)
    assert scribedpiece.super_locus is slc
    slated, slatedh = setup_data_handler(annotations.TranslatedHandler, annotations_orm.Translated, super_locus=slc)
    f0, f0h = setup_data_handler(annotations.FeatureHandler, annotations_orm.Feature, super_locus=slc,
                                 translateds=[slated])

    f1, f1h = setup_data_handler(annotations.FeatureHandler, annotations_orm.Feature, super_locus=slc,
                                 translateds=[slated])

    f2, f2h = setup_data_handler(annotations.FeatureHandler, annotations_orm.Feature, super_locus=slc,
                                 translateds=[slated])

    sess.add_all([slc, scribedpiece, slated, f0, f1, f2])
    sess.commit()
    assert len(slated.features) == 3
    slatedh.replace_selflinks_w_replacementlinks(replacement=scribedpieceh, to_replace=['features'])

    assert len(slated.features) == 0
    assert len(scribedpiece.features) == 3


def test_order_pieces():
    sess = mk_session()
    ag = annotations_orm.AnnotatedGenome(species='Athaliana', version='1.2', acquired_from='Phytozome12')
    sequence_info = annotations_orm.SequenceInfo(annotated_genome=ag)
    coor = annotations_orm.Coordinates(seqid='a', start=1, end=1000, sequence_info=sequence_info)
    sess.add_all([ag, sequence_info, coor])
    sess.commit()
    # setup one transcribed handler with pieces
    scribed, scribedh = setup_data_handler(annotations.TranscribedHandler, annotations_orm.Transcribed)
    ti = annotations.TranscriptInterpBase(transcript=scribedh, session=sess)
    piece1 = annotations_orm.TranscribedPiece()
    piece0 = annotations_orm.TranscribedPiece()
    piece2 = annotations_orm.TranscribedPiece()
    scribed.transcribed_pieces = [piece0, piece1, piece2]
    sess.add_all([scribed, piece0, piece1, piece2])
    sess.commit()
    # setup some paired features
    feature0u = annotations_orm.UpstreamFeature(transcribed_pieces=[piece0], coordinates=coor, start=100, given_id='0u',
                                                is_plus_strand=True)
    feature1d = annotations_orm.DownstreamFeature(transcribed_pieces=[piece1], coordinates=coor, start=1, given_id='1d',
                                                  is_plus_strand=True)
    feature1u = annotations_orm.UpstreamFeature(transcribed_pieces=[piece1], coordinates=coor, start=100, given_id='1u',
                                                is_plus_strand=True)
    feature2d = annotations_orm.DownstreamFeature(transcribed_pieces=[piece2], coordinates=coor, start=1, given_id='2d',
                                                  is_plus_strand=True)
    pair01 = annotations_orm.UpDownPair(upstream=feature0u, downstream=feature1d, transcribed=scribed)
    pair12 = annotations_orm.UpDownPair(upstream=feature1u, downstream=feature2d, transcribed=scribed)
    # check getting upstream link
    upstream_link = ti.get_upstream_link(piece1)
    assert upstream_link is pair01
    upstream = upstream_link.upstream
    assert upstream is feature0u
    assert upstream.transcribed_pieces == [piece0]
    # check getting downstream link
    downstream_link = ti.get_downstream_link(piece1)
    assert downstream_link is pair12
    downstream = downstream_link.downstream
    assert downstream is feature2d
    assert downstream.transcribed_pieces == [piece2]
    # and see if they can be ordered as expected overall
    op = ti.sort_pieces()
    print([piece0, piece1, piece2], 'expected')
    print(op, 'sorted')
    assert op == [piece0, piece1, piece2]
    # and finally, order features by piece
    fully_sorted = ti.sort_all()
    expected = [[feature0u],
                [feature1d, feature1u],
                [feature2d]]
    assert fully_sorted == expected
    # finally make it circular, and make sure it throws an error
    feature2u = annotations_orm.UpstreamFeature(transcribed_pieces=[piece2], coordinates=coor, start=100, given_id='2u',
                                                is_plus_strand=True)
    feature0d = annotations_orm.DownstreamFeature(transcribed_pieces=[piece0], coordinates=coor, start=1, given_id='0d',
                                                  is_plus_strand=True)
    pair20 = annotations_orm.UpDownPair(upstream=feature2u, downstream=feature0d, transcribed=scribed)
    sess.add(pair20)
    sess.commit()
    with pytest.raises(annotations.IndecipherableLinkageError):
        ti.sort_pieces()


class TransspliceDemoData(object):
    def __init__(self, sess):
        # setup two transitions:
        # 1) scribed - [->[TSS(A),START(B),TDSS(C{->F}),TTS(D)], ->[TSS(E), <<slice>>> TASS(F),STOP(G),TTS(H)]]
        # 2) scribedflip - [->[TSS(A),START(B),TDSS(C{->F'}),TTS(D)], <-[TTS(H'), <<slice>> STOP(G'),TASS(F'),TSS(E')]]
        self.old_coor = annotations_orm.Coordinates(seqid='a', start=1, end=2000)
        self.sl, self.slh = setup_data_handler(annotations.SuperLocusHandler, annotations_orm.SuperLocus)
        self.scribed, self.scribedh = setup_data_handler(annotations.TranscribedHandler, annotations_orm.Transcribed,
                                                         super_locus=self.sl)
        self.scribedflip, self.scribedfliph = setup_data_handler(annotations.TranscribedHandler,
                                                                 annotations_orm.Transcribed,
                                                                 super_locus=self.sl)

        self.ti = annotations.TranscriptInterpBase(transcript=self.scribedh, session=sess)
        self.tiflip = annotations.TranscriptInterpBase(transcript=self.scribedfliph, session=sess)

        self.pieceA2D = annotations_orm.TranscribedPiece(super_locus=self.sl)
        self.pieceA2Dp = annotations_orm.TranscribedPiece(super_locus=self.sl)
        self.pieceE2H = annotations_orm.TranscribedPiece(super_locus=self.sl)
        self.pieceEp2Hp = annotations_orm.TranscribedPiece(super_locus=self.sl)
        self.scribed.transcribed_pieces = [self.pieceA2D, self.pieceE2H]
        self.scribedflip.transcribed_pieces = [self.pieceA2Dp, self.pieceEp2Hp]
        # pieceA2D features

        self.fA = annotations_orm.Feature(coordinates=self.old_coor, start=10, end=10, given_id='A',
                                          is_plus_strand=True, super_locus=self.sl,
                                          type=type_enums.TRANSCRIBED, bearing=type_enums.START)
        self.fB = annotations_orm.Feature(coordinates=self.old_coor, start=20, end=20, given_id='B',
                                          is_plus_strand=True, super_locus=self.sl, type=type_enums.CODING,
                                          bearing=type_enums.START)

        self.fC = annotations_orm.Feature(coordinates=self.old_coor, start=30, end=30, given_id='C',
                                          is_plus_strand=True, super_locus=self.sl,
                                          type=type_enums.TRANS_INTRON, bearing=type_enums.START)
        self.fD = annotations_orm.Feature(coordinates=self.old_coor, start=40, end=40, given_id='D',
                                          is_plus_strand=True, super_locus=self.sl,
                                          type=type_enums.TRANSCRIBED, bearing=type_enums.END)
        self.fADs0 = annotations_orm.UpstreamFeature(coordinates=self.old_coor, start=40, end=40, given_id='ADs0',
                                                     is_plus_strand=True, super_locus=self.sl,
                                                     type=type_enums.TRANS_INTRON, bearing=type_enums.CLOSE_STATUS)
        self.fADs1 = annotations_orm.UpstreamFeature(coordinates=self.old_coor, start=40, end=40, given_id='ADs1',
                                                     is_plus_strand=True, super_locus=self.sl,
                                                     type=type_enums.CODING, bearing=type_enums.CLOSE_STATUS)
        # pieceE2H features
        self.fEHs0 = annotations_orm.DownstreamFeature(coordinates=self.old_coor, start=910, end=910, given_id='EHs0',
                                                       is_plus_strand=True, super_locus=self.sl,
                                                       type=type_enums.TRANS_INTRON, bearing=type_enums.OPEN_STATUS)
        self.fEHs1 = annotations_orm.DownstreamFeature(coordinates=self.old_coor, start=910, end=910, given_id='EHs1',
                                                       is_plus_strand=True, super_locus=self.sl,
                                                       type=type_enums.CODING, bearing=type_enums.OPEN_STATUS)
        self.fE = annotations_orm.Feature(coordinates=self.old_coor, start=910, end=910, given_id='E',
                                          is_plus_strand=True, super_locus=self.sl,
                                          type=type_enums.TRANSCRIBED, bearing=type_enums.START)
        self.fF = annotations_orm.Feature(coordinates=self.old_coor, start=920, end=920, given_id='F',
                                          super_locus=self.sl, is_plus_strand=True,
                                          type=type_enums.TRANS_INTRON, bearing=type_enums.END)
        self.fG = annotations_orm.Feature(coordinates=self.old_coor, start=930, end=930, given_id='G',
                                          is_plus_strand=True, super_locus=self.sl, type=type_enums.CODING,
                                          bearing=type_enums.END)
        self.fH = annotations_orm.Feature(coordinates=self.old_coor, start=940, end=940, given_id='H',
                                          is_plus_strand=True, super_locus=self.sl,
                                          type=type_enums.TRANSCRIBED, bearing=type_enums.END)
        # pieceEp2Hp features
        self.fEHps0 = annotations_orm.DownstreamFeature(coordinates=self.old_coor, start=940, end=940, given_id='EHsp0',
                                                        is_plus_strand=False, super_locus=self.sl,
                                                        type=type_enums.TRANS_INTRON, bearing=type_enums.OPEN_STATUS)
        self.fEHps1 = annotations_orm.DownstreamFeature(coordinates=self.old_coor, start=940, end=940, given_id='EHsp1',
                                                        is_plus_strand=False, super_locus=self.sl,
                                                        type=type_enums.CODING, bearing=type_enums.OPEN_STATUS)
        self.fEp = annotations_orm.Feature(coordinates=self.old_coor, start=940, end=940, given_id='Ep',
                                           is_plus_strand=False, super_locus=self.sl,
                                           type=type_enums.TRANSCRIBED, bearing=type_enums.START)
        self.fFp = annotations_orm.Feature(coordinates=self.old_coor, start=930, end=930, given_id='Fp',
                                           super_locus=self.sl, bearing=type_enums.END,
                                           is_plus_strand=False, type=type_enums.TRANS_INTRON)
        self.fGp = annotations_orm.Feature(coordinates=self.old_coor, start=920, end=920, given_id='Gp',
                                           is_plus_strand=False, super_locus=self.sl, type=type_enums.CODING,
                                           bearing=type_enums.END)
        self.fHp = annotations_orm.Feature(coordinates=self.old_coor, start=910, end=910, given_id='Hp',
                                           is_plus_strand=False, super_locus=self.sl,
                                           type=type_enums.TRANSCRIBED, bearing=type_enums.END)

        self.pieceA2D.features = [self.fA, self.fB, self.fC, self.fD, self.fADs0, self.fADs1]
        self.pieceA2Dp.features = [self.fA, self.fB, self.fC, self.fD, self.fADs0, self.fADs1]
        self.pieceE2H.features = [self.fE, self.fF, self.fG, self.fH, self.fEHs0, self.fEHs1]
        self.pieceEp2Hp.features = [self.fEp, self.fFp, self.fGp, self.fHp, self.fEHps0, self.fEHps1]
        self.pairADEH0 = annotations_orm.UpDownPair(upstream=self.fADs0, downstream=self.fEHs0,
                                                    transcribed=self.scribed)
        self.pairADEH1 = annotations_orm.UpDownPair(upstream=self.fADs1, downstream=self.fEHs1,
                                                    transcribed=self.scribed)
        self.pairADEHp0 = annotations_orm.UpDownPair(upstream=self.fADs0, downstream=self.fEHps0,
                                                     transcribed=self.scribedflip)
        self.pairADEHp1 = annotations_orm.UpDownPair(upstream=self.fADs1, downstream=self.fEHps1,
                                                     transcribed=self.scribedflip)
        sess.add_all([self.sl, self.pairADEH0, self.pairADEH1, self.pairADEHp0, self.pairADEHp1])
        sess.commit()

    def make_all_handlers(self):
        self.slh.make_all_handlers()


def test_transition_transsplice():
    sess = mk_session()
    d = TransspliceDemoData(sess)  # setup _d_ata
    d.make_all_handlers()
    # forward pass, same sequence, two pieces
    ti_transitions = list(d.ti.transition_5p_to_3p())
    # from transition gen: 0 -> aligned_Features, 1 -> status copy
    assert [set(x[0]) for x in ti_transitions] == [{d.fA}, {d.fB}, {d.fC}, {d.fD}, {d.fADs1, d.fADs0},
                                                   {d.fEHs0, d.fEHs1}, {d.fE}, {d.fF}, {d.fG}, {d.fH}]
    print([x[1].genic for x in ti_transitions])
    assert [x[1].genic for x in ti_transitions] == [bool(x) for x in [1, 1, 1, 0, 0, 0, 1, 1, 1, 0]]
    assert [x[1].in_translated_region for x in ti_transitions] == [bool(x) for x in [0, 1, 1, 1, 0, 1, 1, 1, 0, 0]]
    assert [x[1].in_trans_intron for x in ti_transitions] == [bool(x) for x in [0, 0, 1, 1, 0, 1, 1, 0, 0, 0]]
    # forward, then backward pass, same sequence, two pieces
    ti_transitions = list(d.tiflip.transition_5p_to_3p())
    assert [set(x[0]) for x in ti_transitions] == [{d.fA}, {d.fB}, {d.fC}, {d.fD}, {d.fADs1, d.fADs0},
                                                   {d.fEHps0, d.fEHps1}, {d.fEp}, {d.fFp}, {d.fGp}, {d.fHp}]
    print([x[1].genic for x in ti_transitions])
    assert [x[1].genic for x in ti_transitions] == [bool(x) for x in [1, 1, 1, 0, 0, 0, 1, 1, 1, 0]]
    assert [x[1].in_translated_region for x in ti_transitions] == [bool(x) for x in [0, 1, 1, 1, 0, 1, 1, 1, 0, 0]]
    assert [x[1].in_trans_intron for x in ti_transitions] == [bool(x) for x in [0, 0, 1, 1, 0, 1, 1, 0, 0, 0]]


# section: gff_2_annotations
def test_data_frm_gffentry():
    #sess = mk_session()
    controller = gff_2_annotations.ImportControl(database_path='sqlite:///:memory:', err_path=None)

    sess = controller.session
    ag = annotations_orm.AnnotatedGenome()
    slice, sliceh = setup_data_handler(gff_2_annotations.SequenceInfoHandler, annotations_orm.SequenceInfo,
                                       annotated_genome=ag)
    coors = annotations_orm.Coordinates(seqid='NC_015438.2', start=1, end=100000, sequence_info=slice)
    sess.add_all([ag, slice, coors])
    sess.commit()
    sliceh.mk_mapper()  # todo, why doesn't this work WAS HERE
    gene_string = 'NC_015438.2\tGnomon\tgene\t4343\t5685\t.\t+\t.\tID=gene0;Dbxref=GeneID:104645797;Name=LOC10'
    mrna_string = 'NC_015438.2\tBestRefSeq\tmRNA\t13024\t15024\t.\t+\t.\tID=rna0;Parent=gene0;Dbxref=GeneID:'
    exon_string = 'NC_015438.2\tGnomon\texon\t4343\t4809\t.\t+\t.\tID=id1;Parent=rna0;Dbxref=GeneID:104645797'
    gene_entry = gffhelper.GFFObject(gene_string)
    handler = gff_2_annotations.SuperLocusHandler()
    handler.gen_data_from_gffentry(gene_entry)
    handler.data.sequence_info = slice
    print(sliceh.gffid_to_coords.keys())
    print(sliceh._gff_seq_ids)
    sess.add(handler.data)
    sess.commit()
    assert handler.data.given_id == 'gene0'
    assert handler.data.type.value == 'gene'

    mrna_entry = gffhelper.GFFObject(mrna_string)
    mrna_handler = gff_2_annotations.TranscribedHandler()
    mrna_handler.gen_data_from_gffentry(mrna_entry, super_locus=handler.data)
    piece_handler = gff_2_annotations.TranscribedPieceHandler()
    piece_handler.gen_data_from_gffentry(mrna_entry, super_locus=handler.data, transcribed=mrna_handler.data)

    sess.add_all([mrna_handler.data, piece_handler.data])
    sess.commit()
    assert mrna_handler.data.given_id == 'rna0'
    assert mrna_handler.data.type.value == 'mRNA'
    assert mrna_handler.data.super_locus is handler.data

    exon_entry = gffhelper.GFFObject(exon_string)
    controller.clean_entry(exon_entry)
    exon_handler = gff_2_annotations.FeatureHandler()
    exon_handler.process_gffentry(exon_entry, super_locus=handler.data, transcribed_pieces=[piece_handler.data],
                                  coordinates=coors)

    d = exon_handler.data
    s = """
    seqid {} {}
    start {} {}
    end {} {}
    is_plus_strand {} {}
    score {} {}
    source {} {}
    phase {} {}
    given_id {} {}""".format(d.coordinates.seqid, type(d.coordinates.seqid),
                             d.start, type(d.start),
                             d.end, type(d.end),
                             d.is_plus_strand, type(d.is_plus_strand),
                             d.score, type(d.score),
                             d.source, type(d.source),
                             d.phase, type(d.phase),
                             d.given_id, type(d.given_id))
    print(s)
    sess.add(exon_handler.data)
    sess.commit()

    assert exon_handler.gffentry.start == 4343
    assert exon_handler.data.is_plus_strand
    assert exon_handler.data.score is None
    assert exon_handler.data.coordinates.seqid == 'NC_015438.2'
    assert exon_handler.data.type.value == 'exon'
    assert exon_handler.data.super_locus is handler.data
    assert piece_handler.data in exon_handler.data.transcribed_pieces
    assert exon_handler.data.translateds == []
    assert exon_handler.data.transcribed_pieces[0].transcribed == mrna_handler.data


def test_data_from_cds_gffentry():
    s = "NC_015447.2\tGnomon\tCDS\t5748\t5840\t.\t-\t0\tID=cds28210;Parent=rna33721;Dbxref=GeneID:101263940,Genbank:" \
        "XP_004248424.1;Name=XP_004248424.1;gbkey=CDS;gene=LOC101263940;product=protein IQ-DOMAIN 14-like;" \
        "protein_id=XP_004248424.1"
    cds_entry = gffhelper.GFFObject(s)
    controller = gff_2_annotations.ImportControl(database_path='sqlite:///:memory:', err_path=None)
    slic, slich = setup_data_handler(gff_2_annotations.SequenceInfoHandler, annotations_orm.SequenceInfo)
    coords = annotations_orm.Coordinates(sequence_info=slic, seqid='dummy')
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


def setup_testable_super_loci(db_path='sqlite:///:memory:'):
    controller = gff_2_annotations.ImportControl(err_path='/dev/null', database_path=db_path)
    controller.mk_session()
    controller.add_sequences('testdata/dummyloci.sequence.json')
    controller.add_gff('testdata/dummyloci.gff3')
    return controller.super_loci[0], controller


def test_organize_and_split_features():
    sl, _ = setup_testable_super_loci()
    transcript_full = [x for x in sl.transcribed_handlers if x.data.given_id == 'y']
    print([x.data.given_id for x in sl.transcribed_handlers])
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


def test_import_seqinfo():
    controller = gff_2_annotations.ImportControl(database_path='sqlite:///:memory:')
    controller.mk_session()
    json_path = 'testdata/dummyloci.sequence.json'
    controller.add_sequences(json_path)
    coors = controller.sequence_info.data.coordinates
    assert len(coors) == 1
    assert coors[0].seqid == '1'
    assert coors[0].start == 0
    assert coors[0].end == 405


def test_fullcopy():
    sess = mk_session()
    sl, slh = setup_data_handler(annotations.SuperLocusHandler, annotations_orm.SuperLocus)
    scribed, scribedh = setup_data_handler(annotations.TranscribedHandler, annotations_orm.Transcribed, super_locus=sl)
    scribedpiece, scribedpieceh = setup_data_handler(annotations.TranscribedPieceHandler,
                                                     annotations_orm.TranscribedPiece, transcribed=scribed,
                                                     super_locus=sl, given_id='soup',)
    f, fh = setup_data_handler(annotations.FeatureHandler, annotations_orm.Feature, super_locus=sl,
                               transcribed_pieces=[scribedpiece], start=13, end=33)
    sess.add_all([scribedpiece, f])
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
    assert set(scribedpiece.features) == {f, new}
    assert new.super_locus == sl
    assert new is not f
    assert new.id != f.id

    # try copying most of transcribed things to translated
    slated, slatedh = setup_data_handler(annotations.TranslatedHandler, annotations_orm.Translated)
    print(scribedpiece.transcribed, 'piece.transcriebd')
    scribedpieceh.fax_all_attrs_to_another(slatedh, skip_copying=None, skip_linking=None)
    sess.commit()
    assert slated.given_id == 'soup'
    assert set(slated.features) == {f, new}
    assert f.translateds == [slated]
    assert new.transcribed_pieces == [scribedpiece]



def test_transcript_interpreter():
    sl, controller = setup_testable_super_loci()
    transcript = [x for x in sl.data.transcribeds if x.given_id == 'y'][0]
    # change so that there are implicit UTRs
    t_interp = gff_2_annotations.TranscriptInterpreter(transcript.handler)
    t_interp.decode_raw_features()
    controller.session.commit()
    # has all standard features
    types_out = set([x.data.type.value for x in t_interp.clean_features])
    assert types_out == {type_enums.CODING,
                         type_enums.TRANSCRIBED,
                         type_enums.INTRON}
    bearings_out = set([x.data.bearing.value for x in t_interp.clean_features])
    assert bearings_out == {type_enums.START, type_enums.END}

    assert t_interp.clean_features[-1].data.start == 400
    assert t_interp.clean_features[0].data.start == 0


def test_transcript_get_first():
    # plus strand
    sl, controller = setup_testable_super_loci()
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
    assert f0.data.start == 0
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
    assert f0.data.start == 399
    assert status.is_5p_utr()
    assert f0.data.phase is None
    assert not f0.data.is_plus_strand

    # test without UTR (x doesn't have last exon, and therefore will end in CDS); remember, flipped it to minus strand
    transcript = [x for x in sl.data.transcribeds if x.given_id == 'x'][0]
    t_interp = gff_2_annotations.TranscriptInterpreter(transcript.handler)
    i0 = t_interp.intervals_5to3(plus_strand=False)[0]
    t_interp.interpret_first_pos(i0, plus_strand=False)
    features = t_interp.clean_features
    status = t_interp.status
    assert len(features) == 4
    f_err_open = features[0]
    f_err_close = features[1]
    f_status_coding = features[2]
    f_status_transcribed = features[3]
    print(f_err_open, f_err_open.data)
    print(status)
    print(i0)
    # should get in_translated_region instead of a start codon
    assert f_status_coding.data.start == 119
    assert f_status_coding.data.type == type_enums.CODING
    assert f_status_coding.data.bearing == type_enums.OPEN_STATUS
    assert not f_status_coding.data.is_plus_strand
    # and should get accompanying in raw transcript
    assert f_status_transcribed.data.type == type_enums.TRANSCRIBED
    assert f_status_coding.data.bearing == type_enums.OPEN_STATUS
    # region beyond exon should be marked erroneous
    assert not f_err_close.data.is_plus_strand and not f_err_open.data.is_plus_strand
    assert f_err_close.data.start == 118  # so that err overlaps 1bp with the coding status checked above
    assert f_err_open.data.start == 404
    assert f_err_open.data.type == type_enums.ERROR
    assert f_err_open.data.bearing == type_enums.START
    assert f_err_close.data.type == type_enums.ERROR
    assert f_err_close.data.bearing == type_enums.END
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
    assert features[-1].data.type == type_enums.CODING
    assert features[-1].data.bearing == type_enums.START
    assert features[-1].data.start == 10
    # hit splice site
    t_interp.interpret_transition(ivals_before=ivals_sets[1], ivals_after=ivals_sets[2], plus_strand=True)
    assert features[-1].data.type == type_enums.INTRON
    assert features[-1].data.bearing == type_enums.END
    assert features[-2].data.type == type_enums.INTRON
    assert features[-2].data.bearing == type_enums.START
    assert features[-2].data.start == 100  # splice from
    assert features[-1].data.start == 110  # splice to
    assert t_interp.status.is_coding()
    # hit splice site
    t_interp.interpret_transition(ivals_before=ivals_sets[2], ivals_after=ivals_sets[3], plus_strand=True)
    assert features[-1].data.type == type_enums.INTRON
    assert features[-1].data.bearing == type_enums.END
    assert features[-2].data.type == type_enums.INTRON
    assert features[-2].data.bearing == type_enums.START
    assert features[-2].data.start == 120  # splice from
    assert features[-1].data.start == 200  # splice to
    # hit stop codon
    t_interp.interpret_transition(ivals_before=ivals_sets[3], ivals_after=ivals_sets[4], plus_strand=True)
    assert features[-1].data.type == type_enums.CODING
    assert features[-1].data.bearing == type_enums.END
    assert features[-1].data.start == 300
    # hit transcription termination site
    t_interp.interpret_last_pos(ivals_sets[4], plus_strand=True)
    assert features[-1].data.type == type_enums.TRANSCRIBED
    assert features[-1].data.bearing == type_enums.END
    assert features[-1].data.start == 400


def test_non_coding_transitions():
    sl, controller = setup_testable_super_loci()
    transcript = [x for x in sl.data.transcribeds if x.given_id == 'z'][0]
    piece = transcript.transcribed_pieces[0]
    t_interp = gff_2_annotations.TranscriptInterpreter(transcript.handler)
    # get single-exon no-CDS transcript
    cds = [x for x in transcript.transcribed_pieces[0].features if x.type.value == type_enums.CDS][0]
    piece.handler.de_link(cds.handler)
    print(transcript)
    ivals_sets = t_interp.intervals_5to3(plus_strand=True)
    assert len(ivals_sets) == 1
    t_interp.interpret_first_pos(ivals_sets[0])
    features = t_interp.clean_features
    assert features[-1].data.type == type_enums.TRANSCRIBED
    assert features[-1].data.bearing == type_enums.START
    assert features[-1].data.start == 110
    t_interp.interpret_last_pos(ivals_sets[0], plus_strand=True)
    assert features[-1].data.type == type_enums.TRANSCRIBED
    assert features[-1].data.bearing == type_enums.END
    assert features[-1].data.start == 120
    assert len(features) == 2


def test_errors_not_lost():
    sl, controller = setup_testable_super_loci()
    s = "1\tGnomon\tgene\t20\t405\t0.\t-\t0\tID=eg_missing_children"
    gene_entry = gffhelper.GFFObject(s)

    coordinates = controller.sequence_info.data.coordinates[0]
    controller.session.add(coordinates)
    sl._mark_erroneous(gene_entry, coordinates=coordinates)
    print(sl.data.transcribeds, len(sl.data.transcribeds), '...sl transcribeds')
    feature_eh, feature_e2h = sl.feature_handlers[-2:]

    print('what features did we start with::?')
    for feature in sl.data.features:
        print(feature)
        controller.session.add(feature)
    controller.session.commit()

    sl.check_and_fix_structure(sess=controller.session, coordinates=coordinates)
    for feature in sl.data.features:
        controller.session.add(feature)
    controller.session.commit()
    print('---and what features did we leave?---')
    for feature in sl.data.features:
        print(feature)
    print(feature_eh.delete_me)
    print(str(feature_eh.data), 'hello...')
    # note, you probably get sqlalchemy.orm.exc.DetachedInstanceError before failing on AssertionError below
    assert feature_eh.data in sl.data.features
    assert feature_e2h.data in sl.data.features


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
    assert set([x.type.value for x in protein.features]) == {type_enums.CODING}
    assert set([x.bearing.value for x in protein.features]) == {type_enums.START, type_enums.END}


def test_check_and_fix_structure():
    rel_path = 'testdata/dummyloci_annotations.sqlitedb'  # so we save a copy of the cleaned up loci once
    if os.path.exists(rel_path):
        #os.remove(rel_path)
        db_path = 'sqlite:///:memory:'
    else:
        db_path = 'sqlite:///{}'.format(rel_path)
    sl, controller = setup_testable_super_loci(db_path)
    coordinates = controller.sequence_info.data.coordinates[0]
    sl.check_and_fix_structure(controller.session, coordinates=coordinates)
    # check handling of nice transcript
    transcript = [x for x in sl.data.transcribeds if x.given_id == 'y'][0]
    protein = [x for x in sl.data.translateds if x.given_id == 'y.p'][0]
    # check we get a protein with start and stop codon for the nice transcript
    assert len(protein.features) == 2  # start and stop codon
    assert set([x.type.value for x in protein.features]) == {type_enums.CODING}
    assert set([x.bearing.value for x in protein.features]) == {type_enums.START, type_enums.END}
    # check we get a transcript with tss, 2x(dss, ass), and tts (+ start & stop codons)
    piece = transcript.handler.one_piece().data
    assert len(piece.features) == 8
    assert set([x.type.value for x in piece.features]) == {type_enums.TRANSCRIBED,
                                                           type_enums.INTRON,
                                                           type_enums.CODING,
                                                           }
    assert set([x.bearing.value for x in piece.features]) == {type_enums.START, type_enums.END}
    # check handling of truncated transcript
    transcript = [x for x in sl.data.transcribeds if x.given_id == 'x'][0]
    piece = transcript.handler.one_piece().data
    protein = [x for x in sl.data.translateds if x.given_id == 'x.p'][0]
    print(protein.features)
    assert len(protein.features) == 2
    assert set([x.type.value for x in protein.features]) == {type_enums.CODING}
    assert set([x.bearing.value for x in protein.features]) == {type_enums.START, type_enums.CLOSE_STATUS}

    assert len(piece.features) == 8
    assert set([x.type.value for x in piece.features]) == {type_enums.TRANSCRIBED, type_enums.INTRON,
                                                           type_enums.ERROR, type_enums.CODING}
    coding_fs = [x for x in piece.features if x.type.value == type_enums.CODING]
    assert len(coding_fs) == 2
    assert set([x.bearing.value for x in coding_fs]) == {type_enums.START, type_enums.CLOSE_STATUS}

    transcribed_fs = [x for x in piece.features if x.type.value == type_enums.TRANSCRIBED]
    assert len(transcribed_fs) == 2
    assert set([x.bearing.value for x in transcribed_fs]) == {type_enums.START, type_enums.CLOSE_STATUS}

    assert len(sl.data.translateds) == 3
    controller.session.commit()



def test_erroneous_splice():
    db_path = 'sqlite:///:memory:'

    sl, controller = setup_testable_super_loci(db_path)
    sess = controller.session
    # get target transcript
    transcript = [x for x in sl.data.transcribeds if x.given_id == 'x'][0]
    # fish out "first exon" features and extend so intron is of -length
    f0 = sess.query(annotations_orm.Feature).filter(annotations_orm.Feature.given_id == 'ftr000000').first()
    f1 = sess.query(annotations_orm.Feature).filter(annotations_orm.Feature.given_id == 'ftr000001').first()
    f0.handler.gffentry.end = f1.handler.gffentry.end = 115

    ti = gff_2_annotations.TranscriptInterpreter(transcript.handler)
    ti.decode_raw_features()
    clean_datas = [x.data for x in ti.clean_features]
    # TSS, start codon, 2x error splice, 2x error splice, 2x error no stop
    print('---\n'.join([str(x) for x in clean_datas]))
    assert len(clean_datas) == 10

    assert len([x for x in clean_datas if x.type == type_enums.ERROR]) == 6
    # make sure splice error covers whole exon-intron-exon region
    assert clean_datas[2].type == type_enums.ERROR
    assert clean_datas[2].bearing == type_enums.START
    assert clean_datas[2].start == 10
    assert clean_datas[3].type == type_enums.ERROR
    assert clean_datas[3].bearing == type_enums.END
    assert clean_datas[3].start == 120


def test_gff_gen():
    controller = gff_2_annotations.ImportControl(database_path='sqlite:///:memory:')
    x = list(controller.gff_gen('testdata/testerSl.gff3'))
    assert len(x) == 103
    assert x[0].type == 'region'
    assert x[-1].type == 'CDS'


def test_gff_useful_gen():
    controller = gff_2_annotations.ImportControl(database_path='sqlite:///:memory:')
    x = list(controller.useful_gff_entries('testdata/testerSl.gff3'))
    assert len(x) == 100  # should drop the region entry
    assert x[0].type == 'gene'
    assert x[-1].type == 'CDS'


def test_gff_grouper():
    controller = gff_2_annotations.ImportControl(database_path='sqlite:///:memory:')
    x = list(controller.group_gff_by_gene('testdata/testerSl.gff3'))
    assert len(x) == 5
    for group in x:
        assert group[0].type == 'gene'

