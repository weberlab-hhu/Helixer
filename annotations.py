import copy
from types import GeneratorType


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
        self.data = data
        data.handler = self  # terrible form, but I need some sort of efficient point back

    def copy_data_attr_to_other(self, other, copy_only=None, do_not_copy=None):
        if not isinstance(other, Handler):
            raise ValueError('other must be an instance of Handler, "{}" found'.format(type(other)))
        # everything that could be copied
        if copy_only is None:
            to_copy = list(self.data.__dict__.keys())
            to_copy = set(copy.deepcopy(to_copy))
        else:
            copy_only = convert2list(copy_only)
            to_copy = set(copy_only)

        if do_not_copy is not None:
            do_not_copy = convert2list(do_not_copy)
            for item in do_not_copy:
                to_copy.remove(item)
        to_copy = copy.deepcopy(to_copy)
        for never_copy in ['id', '_sa_instance_state', 'handler']:
            try:
                to_copy.remove(never_copy)  # todo, confirm this is the primary key
            except KeyError:
                pass
        # acctually copy
        print(to_copy)
        for item in to_copy:
            val = self.get_data_attribute(item)
            other.set_data_attribute(item, val)

    def set_data_attribute(self, attr, val):
        self.data.__setattr__(attr, val)

    def get_data_attribute(self, attr):
        return self.data.__getattribute__(attr)


class AnnotatedGenomeHandler(Handler):
    pass

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
    pass

#        self.mapper = helpers.Mapper()
#
#    @property
#    def gffkey(self):
#        return self.genome.gffkey
#

    @property
    def seq_info(self):
        try:
            _ = self._seq_info
        except AttributeError:
            seq_info = {}
            for x in self.coordinates:
                seq_info[x.seqid] = x
            self._seq_info = seq_info  # todo, is there a sqlalchemy Base compatible way to add attr in init funciton?
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
#    def useful_gff_entries(self, gff_file):
#        skipable = self.gffkey.regions + self.gffkey.ignorable
#        reader = gffhelper.read_gff_file(gff_file)
#        for entry in reader:
#            if entry.type not in self.gffkey.known:
#                raise ValueError("unrecognized feature type fr:qom gff: {}".format(entry.type))
#            if entry.type not in skipable:
#                yield entry
#
#    def group_gff_by_gene(self, gff_file):
#        reader = self.useful_gff_entries(gff_file)
#        gene_group = [next(reader)]
#        for entry in reader:
#            if entry.type in self.gffkey.gene_level:
#                yield gene_group
#                gene_group = [entry]
#            else:
#                gene_group.append(entry)
#        yield gene_group
#
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
    def add_gffentry(self, gffentry, gen_data=True):
        self.gffentry = gffentry
        if gen_data:
            data = self.gen_data_from_gffentry(gffentry)
            self.add_data(data)

    def gen_data_from_gffentry(self, gffentry):
        raise NotImplementedError


class SuperLocusHandler(object):
    pass
#    t_transcripts = 'transcripts'
#    t_proteins = 'proteins'
#    t_feature_holders = 'generic_holders'
#    types_feature_holders = [t_transcripts, t_proteins, t_feature_holders]
#
#        self._dummy_transcript = None
#
#    def short_str(self):
#        string = "{}\ntranscripts: {}\nproteins: {}\ngeneric holders: {}".format(self.id, list(self.transcripts.keys()),
#                                                                                 list(self.proteins.keys()),
#                                                                                 list(self.generic_holders.keys()))
#        return string
#
#    @property
#    def genome(self):
#        return self.slice.genome
#
#    def dummy_transcript(self):
#        if self._dummy_transcript is not None:
#            return self._dummy_transcript
#        else:
#            # setup new blank transcript
#            transcript = FeatureHolder()
#            transcript.id = self.genome.transcript_ider.next_unique_id()  # add an id
#            transcript.super_locus = self
#            self._dummy_transcript = transcript  # save to be returned by next call of dummy_transcript
#            self.generic_holders[transcript.id] = transcript  # save into main dict of transcripts
#            return transcript
#
#    def add_gff_entry(self, entry):
#        exceptions = entry.attrib_filter(tag="exception")
#        for exception in [x.value for x in exceptions]:
#            if 'trans-splicing' in exception:
#                raise TransSplicingError('trans-splice in attribute {} {}'.format(entry.get_ID(), entry.attribute))
#        gffkey = self.genome.gffkey
#        if entry.type in gffkey.gene_level:
#            self.type = entry.type
#            gene_id = entry.get_ID()
#            self.id = gene_id
#            self.ids.append(gene_id)
#            self.gff_entry = entry
#        elif entry.type in gffkey.transcribed:
#            parent = self.one_parent(entry)
#            assert parent == self.id, "not True :( [{} == {}]".format(parent, self.id)
#            transcript = FeatureHolder()
#            transcript.add_data(self, entry)
#            self.generic_holders[transcript.id] = transcript
#        elif entry.type in gffkey.on_sequence:
#            feature = StructuredFeature()
#            feature.add_data(self, entry)
#            self.features[feature.id] = feature
#
#    def _add_gff_entry_group(self, entries):
#        entries = list(entries)
#        for entry in entries:
#            self.add_gff_entry(entry)
#
#    def add_gff_entry_group(self, entries, ts_err_handle):
#        try:
#            self._add_gff_entry_group(entries)
#            self.check_and_fix_structure(entries)
#        except TransSplicingError as e:
#            self._mark_erroneous(entries[0])
#            logging.warning('skipping but noting trans-splicing: {}'.format(str(e)))
#            ts_err_handle.writelines([x.to_json() for x in entries])
#            # todo, log to file
#
#    @staticmethod
#    def one_parent(entry):
#        parents = entry.get_Parent()
#        assert len(parents) == 1
#        return parents[0]
#
#    def _mark_erroneous(self, entry):
#        assert entry.type in self.genome.gffkey.gene_level
#        logging.warning(
#            '{species}:{seqid}, {start}-{end}:{gene_id} by {src}, No valid features found - marking erroneous'.format(
#                src=entry.source, species=self.genome.meta_info.species, seqid=entry.seqid, start=entry.start,
#                end=entry.end, gene_id=self.id
#            ))
#        sf = StructuredFeature()
#        feature = sf.add_erroneous_data(self, entry)
#        self.features[feature.id] = feature
#
#    def check_and_fix_structure(self, entries):
#        # if it's empty (no bottom level features at all) mark as erroneous
#        if not self.features:
#            self._mark_erroneous(entries[0])
#
#        # todo, but with reconstructed flag (also check for and mark pseudogenes)
#        to_remove = []
#        for key in copy.deepcopy(list(self.generic_holders.keys())):
#            transcript = self.generic_holders[key]
#            old_features = copy.deepcopy(transcript.features)
#
#            t_interpreter = TranscriptInterpreter(transcript)
#            transcript = t_interpreter.transcript  # because of non-inplace shift from ofs -> transcripts, todo, fix
#            t_interpreter.decode_raw_features()
#            # no transcript, as they're already linked
#            self.add_features(t_interpreter.clean_features, feature_holders=None)
#            transcript.delink_features(old_features)
#            to_remove += old_features
#        self.remove_features(to_remove)
#
#    def add_features(self, features, feature_holders=None, holder_type=None):
#        if holder_type is None:
#            holder_type = SuperLocus.t_feature_holders
#
#        feature_holders = none_to_list(feature_holders)
#
#        for feature in features:
#            self.features[feature.id] = feature
#            for holder in feature_holders:
#                feature.link_to_feature_holder_and_back(holder.id, holder_type)
#
#    def remove_features(self, to_remove):
#        for f_key in to_remove:
#            self.features.pop(f_key)
#
#    def exons(self):
#        return [self.features[x] for x in self.features if self.features[x].type == self.genome.gffkey.exon]
#
#    def coding_info_features(self):
#        return [self.features[x] for x in self.features if self.features[x].type in self.genome.gffkey.coding_info]
#
#    def check_sequence_assumptions(self):
#        pass
#
#    def clean_post_load(self):
#        for key in self.transcripts:
#            self.transcripts[key].super_locus = self
#
#        for key in self.features:
#            self.features[key].super_locus = self

#    def __deepcopy__(self, memodict={}):
#        new = SuperLocus()
#        copy_over = copy.deepcopy(list(new.__dict__.keys()))
#
#        for to_skip in ['slice']:
#            copy_over.pop(copy_over.index(to_skip))
#
#        # copy everything
#        for item in copy_over:
#            new.__setattr__(item, copy.deepcopy(self.__getattribute__(item)))
#
#        new.slice = self.slice
#
#        # fix point back references to point to new
#        for val in new.transcripts.values():
#            val.super_locus = new
#
#        for val in new.features.values():
#            val.super_locus = new
#
#        return new
#
#


class FeatureHolderHandler(object):
    pass

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


class GenericHolderHandler(FeatureHolderHandler):
    pass


class TranscribedHandler(FeatureHolderHandler):
    pass
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
    pass
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


class FeatureHandler(object):
    pass

#        self.phase = None

#    @property
#    def py_start(self):
#        return self.start - 1
#
#    @property
#    def py_end(self):
#        return self.end
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
