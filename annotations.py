import copy
from types import GeneratorType
import annotations_orm
from helpers import as_py_end, as_py_start


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
        self.delete_me = False
        self.copyable = []
        self.linkable = []

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

    def fax_all_attrs_to_another(self, another, skip_copying=None, skip_linking=None):
        linkable = copy.deepcopy(self.linkable)
        if skip_linking is not None:
            skip_linking = convert2list(skip_linking)
            for item in skip_linking:
                linkable.pop(linkable.index(item))
        copyable = copy.deepcopy(self.copyable)
        if skip_copying is not None:
            skip_copying = convert2list(skip_copying)
            for item in skip_copying:
                copyable.pop(copyable.index(item))
        self.copy_data_attr_to_other(another, copy_only=copyable)
        self.copy_selflinks_to_another(another, to_copy=linkable)

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
                    self.replace_selflink_with_replacementlink(replacement, data)
            elif isinstance(val, annotations_orm.Base):
                self.replace_selflink_with_replacementlink(replacement, val)
            else:
                raise ValueError("replace_selflinks_w_replacementlinks only implemented for {} types".format(
                    [list, annotations_orm.Base]
                ))

    def replace_selflink_with_replacementlink(self, replacement, data):
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

    def mark_for_deletion(self):
        self.delete_me = True


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


#    def divvy_up_super_loci(self, divvied_sequences):
#        # todo: load to interval tree
#        # todo: represent partial super_loci
#        # todo: code split super_loci
#        # todo: put it together
#        pass


class SequenceInfoHandler(Handler):

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

    def add_sequences(self, genome):
        for seq in genome.sequences:
            # todo, parallelize sequence & annotation format, then import directly from sequence_info (~Slice)
            annotations_orm.Coordinates(seqid=seq.meta_info.seqid, start=1,
                                        end=seq.meta_info.total_bp, sequence_info=self.data)


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


class SuperLocusHandler(Handler):

    @property
    def data_type(self):
        return annotations_orm.SuperLocus

    @property
    def _valid_links(self):
        return [SequenceInfoHandler, TranscribedHandler, TranslatedHandler, FeatureHandler]

    def link_to(self, other):
        if isinstance(other, SequenceInfoHandler):
            self.data.sequence_info = other.data
        elif any([isinstance(other, x) for x in [TranscribedHandler, TranslatedHandler, FeatureHandler]]):
            other.data.super_locus = self.data
        else:
            raise self._link_value_error(other)

    def de_link(self, other):
        if isinstance(other, SequenceInfoHandler):
            other.data.super_loci.remove(self.data)
        elif any([isinstance(other, x) for x in [TranscribedHandler, TranslatedHandler, FeatureHandler]]):
            other.data.super_locus = None
        else:
            raise self._link_value_error(other)

    def delete_marked_underlings(self, sess):
        for data in self.data.features + self.data.transcribeds + self.data.translateds:
            if data.handler.delete_me:
                sess.delete(data)
        sess.commit()


class TranscribedHandler(Handler):
    def __init__(self):
        super().__init__()
        self.copyable += ['given_id', 'type']
        self.linkable += ['super_locus', 'transcribed_pieces', 'translateds']

    @property
    def data_type(self):
        return annotations_orm.Transcribed

    @property
    def _valid_links(self):
        return [TranslatedHandler, SuperLocusHandler, TranscribedPieceHandler]

    def link_to(self, other):
        if isinstance(other, SuperLocusHandler):
            self.data.super_locus = other.data
        elif any([isinstance(other, x) for x in [TranslatedHandler, TranscribedPieceHandler]]):
            other.data.transcribeds.append(self.data)
        else:
            raise self._link_value_error(other)

    def de_link(self, other):
        if any([isinstance(other, x) for x in self._valid_links]):
            other.data.transcribeds.remove(self.data)
        else:
            raise self._link_value_error(other)


class TranscribedPieceHandler(Handler):
    def __init__(self):
        super().__init__()
        self.copyable += ['given_id']
        self.linkable += ['super_locus', 'features', 'transcribeds']

    @property
    def data_type(self):
        return annotations_orm.TranscribedPiece

    @property
    def _valid_links(self):
        return [TranscribedHandler, SuperLocusHandler, FeatureHandler]

    def link_to(self, other):
        if isinstance(other, SuperLocusHandler):
            self.data.super_locus = other.data
        elif any([isinstance(other, x) for x in [TranscribedHandler, FeatureHandler]]):
            other.data.transcribed_pieces.append(self.data)
        else:
            raise self._link_value_error(other)

    def de_link(self, other):
        if any([isinstance(other, x) for x in self._valid_links]):
            other.data.transcribed_pieces.remove(self.data)
        else:
            raise self._link_value_error(other)


class TranslatedHandler(Handler):
    def __init__(self):
        super().__init__()
        self.copyable += ['given_id']
        self.linkable += ['super_locus', 'features', 'transcribeds']

    @property
    def data_type(self):
        return annotations_orm.Translated

    @property
    def _valid_links(self):
        return [TranscribedHandler, SuperLocusHandler, FeatureHandler]

    def link_to(self, other):
        if isinstance(other, SuperLocusHandler):
            self.data.super_locus = other.data
        elif any([isinstance(other, x) for x in [TranscribedHandler, FeatureHandler]]):
            other.data.translateds.append(self.data)
        else:
            raise self._link_value_error(other)

    def de_link(self, other):
        if any([isinstance(other, x) for x in self._valid_links]):
            other.data.translateds.remove(self.data)
        else:
            raise self._link_value_error(other)


class FeatureHandler(Handler):
    def __init__(self):
        super().__init__()
        self.copyable += ['given_id', 'type', 'start', 'end', 'coordinates', 'is_plus_strand', 'score', 'source', 'phase']
        self.linkable += ['super_locus', 'transcribed_pieces', 'translateds']

    @property
    def data_type(self):
        return annotations_orm.Feature

    @property
    def _valid_links(self):
        return [TranscribedPieceHandler, SuperLocusHandler, TranslatedHandler]

    def link_to(self, other):
        if isinstance(other, SuperLocusHandler):
            self.data.super_locus = other.data
        elif any([isinstance(other, x) for x in [TranslatedHandler, TranscribedPieceHandler]]):
            other.data.features.append(self.data)
        else:
            raise self._link_value_error(other)

    def de_link(self, other):
        if any([isinstance(other, x) for x in self._valid_links]):
            other.data.features.remove(self.data)
        else:
            raise self._link_value_error(other)

    @property
    def py_start(self):
        return as_py_start(self.data.start)

    @property
    def py_end(self):
        return as_py_end(self.data.end)

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
