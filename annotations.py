import copy
from types import GeneratorType
import annotations_orm


def convert2list(obj):
    if isinstance(obj, list):
        out = obj
    elif isinstance(obj, set) or isinstance(obj, GeneratorType) or isinstance(obj, tuple):
        out = list(obj)
    else:
        out = [obj]
    return out


class Handler(object):

    def __init__(self):
        self.data = None

    def add_data(self, data):
        assert isinstance(data, self.data_type)
        self.data = data
        data.handler = self  # terrible form, but I need some sort of efficient point back

    def copy_data_attr_to_other(self, other, copy_only=None, do_not_copy=None):
        if not isinstance(other, Handler):
            raise ValueError('other must be an instance of Handler, "{}" found'.format(type(other)))
        # everything that could be copied
        if copy_only is None:
            to_copy = list(type(self.data).__dict__.keys())
            to_copy = [x for x in to_copy if not x.startswith('_')]
            to_copy = set(copy.deepcopy(to_copy))
        else:
            copy_only = convert2list(copy_only)
            to_copy = set(copy_only)

        if do_not_copy is not None:
            do_not_copy = convert2list(do_not_copy)
            for item in do_not_copy:
                to_copy.remove(item)
        to_copy = copy.deepcopy(to_copy)
        for never_copy in ['id', 'handler']:
            try:
                to_copy.remove(never_copy)  # todo, confirm this is the primary key
            except KeyError:
                pass
        # acctually copy
        for item in to_copy:
            val = self.get_data_attribute(item)
            other.set_data_attribute(item, val)

    def set_data_attribute(self, attr, val):
        self.data.__setattr__(attr, val)

    def get_data_attribute(self, attr):
        return self.data.__getattribute__(attr)

    def replace_selflinks_w_replacementlinks(self, replacement, to_replace):
        to_replace = copy.deepcopy(to_replace)
        for item in ['id', 'handler']:
            assert item not in to_replace
        for attr in to_replace:
            val = self.get_data_attribute(attr)
            if isinstance(val, list):
                n = len(val)
                for i in reversed(list(range(n))):  # go through backwards to hit every item even though we're removing
                    #for data in val:
                    data = val[i]
                    self._replace_selflink_with_replacementlink(replacement, data)
            elif isinstance(val, annotations_orm.Base):
                self._replace_selflink_with_replacementlink(replacement, val)
            else:
                raise ValueError("replace_selflinks_w_replacementlinks only implemented for {} types".format(
                    [list, annotations_orm.Base]
                ))

    def _replace_selflink_with_replacementlink(self, replacement, data):
        other = data.handler
        self.de_link(other)
        replacement.link_to(other)

    def copy_selflinks_to_another(self, another, to_copy):
        to_copy = copy.deepcopy(to_copy)

        for item in ['id', 'handler']:
            assert item not in to_copy

        for attr in to_copy:
            val = self.get_data_attribute(attr)
            if isinstance(val, list):
                n = len(val)
                for i in reversed(list(range(n))):  # go through backwards to hit every item even though we're removing
                    #for data in val:
                    data = val[i]
                    self._copy_selflinks_to_another(another, data)
            elif isinstance(val, annotations_orm.Base):
                self._copy_selflinks_to_another(another, val)
            else:
                raise ValueError("copy_selflinks_to_another only implemented for {} types".format(
                    [list, annotations_orm.Base]
                ))

    def _copy_selflinks_to_another(self, another, data):
        other = data.handler
        another.link_to(other)

    def link_to(self, other):
        raise NotImplementedError

    def de_link(self, other):
        raise NotImplementedError

    @property
    def data_type(self):
        raise NotImplementedError

    @property
    def _valid_links(self):
        raise NotImplementedError

    def _link_value_error(self, other):
        link_error = "from {} can only link / de_link to {}; found {}".format(type(self), self._valid_links,
                                                                              type(other))
        return ValueError(link_error)


class AnnotatedGenomeHandler(Handler):
    @property
    def data_type(self):
        return annotations_orm.AnnotatedGenome

    @property
    def _valid_links(self):
        return [SequenceInfoHandler]

    def link_to(self, other):
        if isinstance(other, SequenceInfoHandler):
            other.data.annotated_genome = self.data
            # switched as below maybe checks other.data integrity, and fails on NULL anno genome?
            # self.data.sequence_infos.append(other.data)
        else:
            raise self._link_value_error(other)

    def de_link(self, other):
        if isinstance(other, SequenceInfoHandler):
            self.data.sequence_infos.remove(other.data)
        else:
            raise self._link_value_error(other)


    #gffkey = FeatureDecoder()
    # todo, figure out if one wants to keep the id makers like this...
#    transcript_ider = helpers.IDMaker(prefix='trx')
#    protein_ider = helpers.IDMaker(prefix='prt')
#    feature_ider = helpers.IDMaker(prefix='ftr')
#
#    def add_gff(self, gff_file, genome, err_file='trans_splicing.txt'):
#        sls = SuperLociSlice()
#        sls.genome = self
#        sls.add_gff(gff_file, genome, err_file=err_file)
#        self.super_loci_slices.append(sls)
#
#    def divvy_up_super_loci(self, divvied_sequences):
#        # todo: load to interval tree
#        # todo: represent partial super_loci
#        # todo: code split super_loci
#        # todo: put it together
#        pass
#
#    def clean_post_load(self):
#        for sl in self.super_loci_slices:
#            sl.genome = self


class SequenceInfoHandler(Handler):
    def __init__(self):
        super().__init__()
        self._seq_info = None
#        self.mapper = helpers.Mapper()
#
#    @property
#    def gffkey(self):
#        return self.genome.gffkey
#

    @property
    def data_type(self):
        return annotations_orm.SequenceInfo

    @property
    def _valid_links(self):
        return [AnnotatedGenomeHandler, SuperLocusHandler]

    def link_to(self, other):
        if isinstance(other, AnnotatedGenomeHandler):
            self.data.annotated_genome = other.data
            # switched as below maybe checks other.data integrity, and fails on NULL anno genome?
            # self.data.sequence_infos.append(other.data)
        elif isinstance(other, SuperLocusHandler):
            other.data.sequence_info = self.data
        else:
            raise self._link_value_error(other)

    def de_link(self, other):
        if isinstance(other, AnnotatedGenomeHandler):
            other.data.sequence_infos.remove(self.data)
        elif isinstance(other, SuperLocusHandler):
            self.data.super_loci.remove(other.data)
        else:
            raise self._link_value_error(other)

    @property
    def seq_info(self):
        if self._seq_info is not None:
            pass
        else:
            seq_info = {}
            for x in self.data.coordinates:
                seq_info[x.seqid] = x
            self._seq_info = seq_info
        return self._seq_info
#
#    def _add_sequences(self, genome):
#        for seq in genome.sequences:
#            mi = CoordinateInfo()
#            mi.seqid = seq.meta_info.seqid
#            mi.start = 1
#            mi.end = seq.meta_info.total_bp
#            self.coordinates.append(mi)
#
#    def add_gff(self, gff_file, genome, err_file='trans_splicing.txt'):
#        err_handle = open(err_file, 'w')
#        self._add_sequences(genome)
#
#        gff_seq_ids = helpers.get_seqids_from_gff(gff_file)
#        mapper, is_forward = helpers.two_way_key_match(self.seq_info.keys(), gff_seq_ids)
#        self.mapper = mapper
#
#        if not is_forward:
#            raise NotImplementedError("Still need to implement backward match if fasta IDs are subset of gff IDs")
#
#        for entry_group in self.group_gff_by_gene(gff_file):
#            new_sl = SuperLocus()
#            new_sl.slice = self
#            new_sl.add_gff_entry_group(entry_group, err_handle)
#
#            self.super_loci.append(new_sl)
#            if not new_sl.transcripts and not new_sl.features:
#                print('{} from {} with {} transcripts and {} features'.format(new_sl.id,
#                                                                              entry_group[0].source,
#                                                                              len(new_sl.transcripts),
#                                                                              len(new_sl.features)))
#        err_handle.close()
#
#    def useful_gff_entries(self, gff_file): >> gff_2_annotations
#    def group_gff_by_gene(self, gff_file): >> gff_2_annotations

#    def load_to_interval_tree(self):
#        trees = {}
#        for seqid in self.seq_info:
#            trees[seqid] = intervaltree.IntervalTree()
#        for sl in self.super_loci:
#            for fkey in sl.features:
#                feature = sl.features[fkey]
#                trees[feature.seqid][feature.py_start:feature.py_end] = feature
#        return trees
#
#    def slice_further(self, seqid, slice_id, start, end, processing_set, trees):
#        # setup new slice
#        new = SuperLociSlice()
#        mi = CoordinateInfo()
#        mi.seqid = seqid
#        mi.start = start
#        mi.end = end
#        new.coordinates = mi
#        new.slice_id = slice_id
#        new.processing_set = processing_set
#        # and get all features
#        tree = trees[seqid]
#        branch = tree[start - 1:end]  # back to python coordinates  # todo, double check this gets overlaps not contains
#        features_by_sl = {}
#        for intvl in branch:
#            sl_id = intvl.data.super_locus.id
#            if sl_id in features_by_sl:
#                features_by_sl[sl_id].append(intvl.data)
#            else:
#                features_by_sl[sl_id] = [intvl.data]
#        for sl_id in features_by_sl:
#            super_locus = features_by_sl[sl_id][0].super_locus
#            for transcript in super_locus.transcripts:
#                trimmed_transcript = transcript.reconcile_with_slice(seqid, start, end)  # todo
#                # todo, add transcript & features to new slice
#            # todo add sl
#
#    def add_slice(self, seqid, slice_id, start, end, processing_set):
#        pass #todo
#
#    def to_example(self):
#        raise NotImplementedError
#
#    def clean_post_load(self):
#        for sl in self.super_loci:
#            sl.slice = self
#
#    def __deepcopy__(self, memodict={}):
#        raise NotImplementedError  # todo


class GFFDerivedHandler(Handler):
    # todo, move this & all gen_data_from_gffentry to gff_2_annotations (multi inheritance?)
    def __init__(self):
        super().__init__()
        self.gffentry = None

    def add_gffentry(self, gffentry, gen_data=True):
        self.gffentry = gffentry
        if gen_data:
            data = self.gen_data_from_gffentry(gffentry)
            self.add_data(data)

    def gen_data_from_gffentry(self, gffentry, **kwargs):
        raise NotImplementedError


class SuperLocusHandler(GFFDerivedHandler):

    @property
    def data_type(self):
        return annotations_orm.SuperLocus

    @property
    def _valid_links(self):
        return [SequenceInfoHandler, TranscribedHandler, TranslatedHandler, FeatureHandler]

    def link_to(self, other):
        if isinstance(other, SequenceInfoHandler):
            self.data.sequence_info = other.data
        elif type(other) in [TranscribedHandler, TranslatedHandler, FeatureHandler]:
            other.data.super_locus = self.data
        else:
            raise self._link_value_error(other)

    def de_link(self, other):
        if isinstance(other, SequenceInfoHandler):
            other.data.super_loci.remove(self.data)
        elif type(other) in [TranscribedHandler, TranslatedHandler, FeatureHandler]:
            other.data.super_locus = None
        else:
            raise self._link_value_error(other)

    def gen_data_from_gffentry(self, gffentry, sequence_info=None, **kwargs):
        data = self.data_type(type=gffentry.type,
                              given_id=gffentry.get_ID(),
                              sequence_info=sequence_info)
        self.add_data(data)
        # todo, grab more aliases from gff attribute


class FeatureHolderHandler(GFFDerivedHandler):

    def gen_data_from_gffentry(self, gffentry, super_locus=None, **kwargs):
        parents = gffentry.get_Parent()
        # the simple case
        if len(parents) == 1:
            assert super_locus.given_id == parents[0]
            data = self.data_type(type=gffentry.type,
                                  given_id=gffentry.get_ID(),
                                  super_locus=super_locus)
            self.add_data(data)
        else:
            raise NotImplementedError  # todo handle multi inheritance, etc...
#    holder_type = SuperLocus.t_feature_holders
#
#    def __init__(self):
#        super().__init__()
#        self.spec += [('super_locus', False, SuperLocus, None),
#                      ('features', True, list, None),
#                      ('next_feature_5p', True, str, None),
#                      ('next_feature_3p', True, str, None)]
#
#        self.features = []
#        self.next_feature_5p = None
#        self.next_feature_3p = None
#
#    def add_data(self, super_locus, gff_entry):
#        self.super_locus = super_locus
#        self.id = gff_entry.get_ID()
#        self.type = gff_entry.type
#        self.gff_entry = gff_entry
#
#    def link_to_feature(self, feature_id, at=None):
#        assert feature_id not in self.features, "{} already in features {} for {} {} in loci {}".format(
#            feature_id, self.features, self.type, self.id, self.super_locus.id)
#        if at is None:
#            self.features.append(feature_id)
#        else:
#            self.features.insert(at, feature_id)
#
#    def remove_feature(self, feature_id):
#        at = self.features.index(feature_id)
#        self.features.pop(at)
#        return at
#
#    def short_str(self):
#        return '{}. --> {}'.format(self.id, self.features)
#
#    def delink_features(self, features):
#        for feature in features:
#            holder_type = type(self).holder_type
#            try:
#                self.super_locus.features[feature].de_link_from_feature_holder(self.id, holder_type)
#            except ValueError:
#                feature_fholder_list = self.super_locus.features[feature].__getattribute__(holder_type)
#                raise ValueError("{} not in feature's {}: {}".format(self.id, holder_type,
#                                                                     feature_fholder_list))
#
#    def reconcile_with_slice(self, seqid, start, end):
#        pass  #todo, WAS HERE, make valid (partial) transcript within slice
#
#    def __deepcopy__(self, memodict={}):
#        new = type(self)()
#        copy_over = copy.deepcopy(list(new.__dict__.keys()))
#
#        for to_skip in ['super_locus']:
#            copy_over.pop(copy_over.index(to_skip))
#
#        # copy everything
#        for item in copy_over:
#            new.__setattr__(item, copy.deepcopy(self.__getattribute__(item)))
#
#        new.super_locus = self.super_locus  # fix super_locus
#        if new.gff_entry is None:
#            print('warning, gff_entry none at {}'.format(new.id))
#        return new
#
#    def clone_but_swap_type(self, new_holder_type):
#        # todo, should this actually go into some sort of generic ordered subclass?
#        old_holder_type = type(self).holder_type
#        assert new_holder_type in SuperLocus.types_feature_holders  # todo, stop retyping this
#        assert new_holder_type != old_holder_type  # Shouldn't call swap_type if one has the right type already
#
#        # setup new type with all transferable attributes
#        to_transfer = self.__deepcopy__()  # can copy from
#        holder_type = [x for x in [Transcribed, Translated, FeatureHolder] if x.holder_type == new_holder_type][0]
#        new = holder_type()
#        transferable = set(to_transfer.__dict__.keys())
#        transferable.remove('spec')
#        transferable = transferable.intersection(set(new.__dict__.keys()))
#        for item in transferable:
#            new.__setattr__(item, to_transfer.__getattribute__(item))
#
#        add_to = self.super_locus.__getattribute__(new_holder_type)
#
#
#        # put new in requested place
#        add_to[new.id] = new
#        # swap feature links from old to new
#        for fkey in copy.deepcopy(self.features):
#            feature = self.super_locus.features[fkey]
#            feature.link_to_feature_holder(new.id, new_holder_type)
#        # remove old from
#        return new
#
#    def swap_type(self, new_holder_type):
#        new = self.clone_but_swap_type(new_holder_type)
#        old_holder_type = type(self).holder_type
#        for fkey in copy.deepcopy(self.features):
#            feature = self.super_locus.features[fkey]
#            feature.de_link_from_feature_holder(self.id, old_holder_type)
#            #feature.link_to_feature_holder(new.id, new_holder_type)
#
#        remove_from = self.super_locus.__getattribute__(old_holder_type)
#        remove_from.pop(self.id)
#        return new
#
#    def feature_obj(self, feature_id):
#        return self.super_locus.features[feature_id]
#
#    def feature_objs(self):
#        return [self.feature_obj(x) for x in self.features]
#
#    def replace_id_everywhere(self, new_id):
#        holder_type = type(self).holder_type
#        # new object with new id
#        new = self.__deepcopy__()
#        new.id = new_id
#        # add to super_locus
#        add_to = self.super_locus.__getattribute__(holder_type)
#        add_to[new_id] = new
#
#        old_id = self.id
#        # replace all references to features
#        for fobj in self.feature_objs():
#            at = fobj.de_link_from_feature_holder(old_id, holder_type)  # todo, keep the position!
#            fobj.link_to_feature_holder(new_id, holder_type, at=at)
#        # replace any other by-id references
#        self.replace_subclass_only_ids(new_id)
#        # remove self from superlocus
#        add_to.pop(self.id)
#        # convenience
#        return new
#
#    def replace_subclass_only_ids(self, new_id):
#        pass
#


class TranscribedHandler(FeatureHolderHandler):

    @property
    def data_type(self):
        return annotations_orm.Transcribed

    @property
    def _valid_links(self):
        return [TranslatedHandler, SuperLocusHandler, FeatureHandler]

    def link_to(self, other):
        if isinstance(other, SuperLocusHandler):
            self.data.super_locus = other.data
        elif type(other) in [TranslatedHandler, FeatureHandler]:
            other.data.transcribeds.append(self.data)
        else:
            raise self._link_value_error(other)

    def de_link(self, other):
        if type(other) in self._valid_links:
            other.data.transcribeds.remove(self.data)
        else:
            raise self._link_value_error(other)
#    holder_type = 'transcripts'
#
#        self.proteins = []  # list of protein IDs, matching subset of keys in self.super_locus.proteins
#
#    def replace_subclass_only_ids(self, new_id):
#        for prot in self.protein_objs():
#            i = prot.transcripts.index(self.id)
#            prot.transcripts.pop(i)
#            prot.transcripts.insert(i, new_id)
#
#    def protein_obj(self, prot_id):
#        return self.super_locus.proteins[prot_id]
#
#    def protein_objs(self):
#        return [self.protein_obj(x) for x in self.proteins]
#


class TranslatedHandler(FeatureHolderHandler):
    @property
    def data_type(self):
        return annotations_orm.Translated

    @property
    def _valid_links(self):
        return [TranscribedHandler, SuperLocusHandler, FeatureHandler]

    def link_to(self, other):
        if isinstance(other, SuperLocusHandler):
            self.data.super_locus = other.data
        elif type(other) in [TranscribedHandler, FeatureHandler]:
            other.data.translateds.append(self.data)
        else:
            raise self._link_value_error(other)

    def de_link(self, other):
        if type(other) in self._valid_links:
            other.data.translateds.remove(self.data)
        else:
            raise self._link_value_error(other)
#    holder_type = 'proteins'
#
#        self.transcripts = []  # list of transcript IDs, matching subset of keys in self.super_locus.transcripts
#
#    def replace_subclass_only_ids(self, new_id):
#        for transcript in self.transcript_objs():
#            i = transcript.transcripts.index(self.id)
#            transcript.transcripts.pop(i)
#            transcript.transcripts.insert(i, new_id)
#
#    def transcript_obj(self, prot_id):
#        return self.super_locus.transcripts[prot_id]
#
#    def transcript_objs(self):
#        return [self.transcript_obj(x) for x in self.transcripts]
#
#


class FeatureHandler(GFFDerivedHandler):
    @property
    def data_type(self):
        return annotations_orm.Feature

    @property
    def _valid_links(self):
        return [TranscribedHandler, SuperLocusHandler, TranslatedHandler]

    def link_to(self, other):
        if isinstance(other, SuperLocusHandler):
            self.data.super_locus = other.data
        elif type(other) in [TranscribedHandler, TranslatedHandler]:
            other.data.features.append(self.data)
        else:
            raise self._link_value_error(other)

    def de_link(self, other):
        if type(other) in self._valid_links:
            other.data.features.remove(self.data)
        else:
            raise self._link_value_error(other)

    def gen_data_from_gffentry(self, gffentry, super_locus=None, transcribeds=None, translateds=None, **kwargs):
        if transcribeds is None:
            transcribeds = []
        if translateds is None:
            translateds = []
        given_id = gffentry.get_ID()  # todo, None on missing
        is_plus_strand = gffentry.strand == '+'

        data = self.data_type(
            given_id=given_id,
            type=gffentry.type,
            seqid=gffentry.seqid,
            start=gffentry.start,
            end=gffentry.end,
            is_plus_strand=is_plus_strand,
            score=gffentry.score,
            source=gffentry.source,
            phase=gffentry.phase,
            super_locus=super_locus,
            transcribeds=transcribeds,
            translateds=translateds
        )
        self.add_data(data)

    @property
    def py_start(self):
        return self.data.start - 1

    @property
    def py_end(self):
        return self.data.end
#
#    def short_str(self):
#        return '{} is {}: {}-{} on {}. --> {}|{}|{}'.format(self.id, self.type, self.start, self.end, self.seqid,
#                                                            self.transcripts, self.proteins, self.generic_holders)
#
#    def add_data(self, super_locus, gff_entry):
#        gffkey = super_locus.genome.gffkey
#        try:
#            fid = gff_entry.get_ID()
#        except TypeError:
#            fid = None
#            logging.debug('no ID in attr {} in {}, making new unique ID'.format(gff_entry.attribute, super_locus.id))
#        self.gff_entry = gff_entry
#        self.super_locus = super_locus
#        self.id = super_locus.genome.feature_ider.next_unique_id(fid)
#        self.type = gff_entry.type
#        self.start = int(gff_entry.start)
#        self.end = int(gff_entry.end)
#        self.strand = gff_entry.strand
#        self.seqid = self.super_locus.slice.mapper(gff_entry.seqid)
#        if gff_entry.phase == '.':
#            self.phase = None
#        else:
#            self.phase = int(gff_entry.phase)
#        try:
#            self.score = float(gff_entry.score)
#        except ValueError:
#            pass
#        new_transcripts = gff_entry.get_Parent()
#        if not new_transcripts:
#            self.type = gffkey.error
#            logging.warning('{species}:{seqid}:{fid}:{new_id} - No Parents listed'.format(
#                species=super_locus.genome.meta_info.species, seqid=self.seqid, fid=fid, new_id=self.id
#            ))
#        for transcript_id in new_transcripts:
#            new_t_id = transcript_id
#            if new_t_id not in super_locus.generic_holders:
#                if transcript_id == super_locus.id:
#                    # if we just skipped the transcript, and linked to gene, use dummy transcript in between
#                    transcript = super_locus.dummy_transcript()
#                    logging.info(
#                        '{species}:{seqid}:{fid}:{new_id} - Parent gene instead of transcript, recreating'.format(
#                            species=super_locus.genome.meta_info.species, seqid=self.seqid, fid=fid, new_id=self.id
#                        ))
#                    new_t_id = transcript.id
#                else:
#                    self.type = gffkey.error
#                    new_t_id = None
#                    logging.warning(
#                        '{species}:{seqid}:{fid}:{new_id} - Parent: "{parent}" not found at loci'.format(
#                            species=super_locus.genome.meta_info.species, seqid=self.seqid, fid=fid, new_id=self.id,
#                            parent=transcript_id
#                        ))
#            self.link_to_feature_holder_and_back(new_t_id, SuperLocus.t_feature_holders)
#
#    def add_erroneous_data(self, super_locus, gff_entry):
#        self.super_locus = super_locus
#        feature_e = self.clone()
#        feature_e.start = int(gff_entry.start)
#        feature_e.end = int(gff_entry.end)
#        feature_e.strand = gff_entry.strand
#        feature_e.seqid = gff_entry.seqid
#        feature_e.change_to_error()
#        return feature_e
#
#    def change_to_error(self):
#        self.type = self.super_locus.genome.gffkey.error
#
#    def link_to_feature_holder_and_back(self, holder_id, holder_type=None, at=None):
#        #print('link_to_feature_holder_and_back ({}) {} {}'.format(self.short_str(), holder_id, holder_type))
#        if holder_type is None:
#            holder_type = SuperLocus.t_feature_holders
#
#        sl_holders = self.super_locus.__getattribute__(holder_type)
#        holder = sl_holders[holder_id]  # get feature holder
#        holder.link_to_feature(self.id, at)  # link to and from self
#        # get ordered feature holder (transcripts / proteins / feature_holders)
#        self.link_to_feature_holder(holder_id, holder_type)
#
#    def link_to_feature_holder(self, holder_id, holder_type=None, at=None):
#        #print('link_fo_feature_holder ({}) {} {}'.format(self.short_str(), holder_id, holder_type))
#        if holder_type is None:
#            holder_type = SuperLocus.t_feature_holders
#        holder = self.__getattribute__(holder_type)
#        assert holder_type in SuperLocus.types_feature_holders
#        e = "{} already in {}: {}".format(
#            holder_id, holder_type, holder
#        )
#        assert holder_id not in holder, e
#        if at is None:
#            holder.append(holder_id)
#        else:
#            holder.insert(at, holder_id)
#
#    def de_link_from_feature_holder(self, holder_id, holder_type=None):
#        if holder_type is None:
#            holder_type = SuperLocus.t_feature_holders
#        assert holder_type in SuperLocus.types_feature_holders
#        sl_holders = self.super_locus.__getattribute__(holder_type)
#        holder = sl_holders[holder_id]  # get transcript
#
#        at = holder.remove_feature(self.id)  # drop other
#        # and drop from local ordered feature holder set
#        holders = self.__getattribute__(holder_type)
#        holders.pop(holders.index(holder_id))
#        return at
#
#    def fully_overlaps(self, other):
#        should_match = ['type', 'start', 'end', 'seqid', 'strand', 'phase']
#        does_it_match = [self.__getattribute__(x) == other.__getattribute__(x) for x in should_match]
#        same_gene = self.super_locus is other.super_locus
#        out = False
#        if all(does_it_match + [same_gene]):
#            out = True
#        return out
#
#    def is_contained_in(self, other):
#        should_match = ['seqid', 'strand', 'phase']
#        does_it_match = [self.__getattribute__(x) == other.__getattribute__(x) for x in should_match]
#        same_gene = self.super_locus is other.super_locus
#        coordinates_within = self.start >= other.start and self.end <= other.end
#        return all(does_it_match + [coordinates_within, same_gene])
#
#    def reconstruct_exon(self):
#        """creates an exon exactly containing this feature"""
#        exon = self.clone()
#        exon.type = self.super_locus.genome.gffkey.exon
#        return exon
#
#    def clone(self, copy_feature_holders=True):
#        """makes valid, independent clone/copy of this feature"""
#        new = StructuredFeature()
#        copy_over = copy.deepcopy(list(new.__dict__.keys()))
#
#        for to_skip in ['super_locus', 'id', 'transcripts']:
#            copy_over.pop(copy_over.index(to_skip))
#
#        # handle can't just be copied things
#        new.super_locus = self.super_locus
#        new.id = self.super_locus.genome.feature_ider.next_unique_id()
#        if copy_feature_holders:
#            for transcript in self.transcripts:
#                new.link_to_feature_holder_and_back(transcript, SuperLocus.t_transcripts)
#            for protein in self.proteins:
#                new.link_to_feature_holder_and_back(protein, SuperLocus.t_proteins)
#            for ordf in self.generic_holders:
#                new.link_to_feature_holder_and_back(ordf, SuperLocus.t_feature_holders)
#
#        # copy the rest
#        for item in copy_over:
#            new.__setattr__(item, copy.deepcopy(self.__getattribute__(item)))
#        if new.gff_entry is None:
#            raise ValueError('want real gff entry')
#        return new
#
#    def __deepcopy__(self, memodict={}):
#        new = StructuredFeature()
#        copy_over = copy.deepcopy(list(new.__dict__.keys()))
#
#        for to_skip in ['super_locus']:
#            copy_over.pop(copy_over.index(to_skip))
#
#        # copy everything
#        for item in copy_over:
#            new.__setattr__(item, copy.deepcopy(self.__getattribute__(item)))
#
#        new.super_locus = self.super_locus  # fix super_locus
#
#        return new
#
#    def merge(self, other):
#        assert self is not other
#        # move transcript reference from other to self
#        for fset in SuperLocus.types_feature_holders:
#            for ordf in copy.deepcopy(other.__getattribute__(fset)):
#                self.link_to_feature_holder_and_back(ordf, fset)
#                other.de_link_from_transcript(ordf)
#
#    def is_plus_strand(self):
#        if self.strand == '+':
#            return True
#        elif self.strand == '-':
#            return False
#        else:
#            raise ValueError('strand should be +- {}'.format(self.strand))
#
#    def upstream(self):
#        if self.is_plus_strand():
#            return self.start
#        else:
#            return self.end
#
#    def downstream(self):
#        if self.is_plus_strand():
#            return self.end
#        else:
#            return self.start
#
#    # inclusive and from 1 coordinates
#    def upstream_from_interval(self, interval):
#        if self.is_plus_strand():
#            return interval.begin + 1
#        else:
#            return interval.end
#
#    def downstream_from_interval(self, interval):
#        if self.is_plus_strand():
#            return interval.end
#        else:
#            return interval.begin + 1
#
#    def reconcile_with_slice(self, seqid, start, end, status, last_before_slice):
#        #overlap_status = OverlapStatus()
#        #overlap_status.set_status(self, seqid, start, end)
#        #status = overlap_status.status
#        if status == OverlapStatus.contained:
#            pass  # leave it alone
#        elif status == OverlapStatus.no_overlap:
#            # todo, if it is the last feature before the slice (aka, if the next one is contained)
#            if last_before_slice:
#                self.shift_phase(start, end)
#                pass  # todo, change to 1bp status_at (w/ phase if appropriate)
#            pass  # todo, delete (and from transcripts / super_locus)
#        elif status == OverlapStatus.overlaps_upstream:
#            self.shift_phase(start, end)
#            self.crop(start, end)
#        elif status == OverlapStatus.overlaps_downstream:
#            # just crop
#            self.crop(start, end)

#    def length_outside_slice(self, start, end):
#        if self.is_plus_strand():
#            length_outside_slice = start - self.start
#        else:
#            length_outside_slice = self.end - end
#        return length_outside_slice
#
#    def crop(self, start, end):
#        if self.start < start:
#            self.start = start
#        if self.end > end:
#            self.end = end
#
#    def shift_phase(self, start, end):
#        if self.phase is not None:
#            l_out = self.length_outside_slice(start, end)
#            self.phase = (l_out - self.phase) % 3
