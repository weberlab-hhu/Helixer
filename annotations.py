from dustdas import gffhelper
import logging
import copy
import intervaltree
import helpers



class FeatureDecoder(object):
    def __init__(self):
        self.error_buffer = 2000
        # gene like, generally having a collection of transcripts
        self.gene = 'gene'
        self.super_gene = 'super_gene'
        self.ncRNA_gene = 'ncRNA_gene'
        self.pseudogene = 'pseudogene'
        self.gene_level = [self.gene, self.super_gene, self.ncRNA_gene, self.pseudogene]

        # transcript like, generally having a collection of exons, indicating how they are spliced
        # also ultimately, if not explicitly having a transcription start and termination site
        self.mRNA = 'mRNA'
        self.tRNA = 'tRNA'
        self.rRNA = 'rRNA'
        self.miRNA = 'miRNA'
        self.snoRNA = 'snoRNA'
        self.snRNA = 'snRNA'
        self.SRP_RNA = 'SRP_RNA'
        self.lnc_RNA = 'lnc_RNA'
        self.pre_miRNA = 'pre_miRNA'
        self.RNase_MRP_RNA = 'RNase_MRP_RNA'
        self.transcript = 'transcript'
        self.primary_transcript = 'primary_transcript'
        self.pseudogenic_transcript = 'pseudogenic_transcript'  # which may or may not be transcribed, hard to say
        self.transcribed = [self.mRNA, self.transcript, self.tRNA, self.primary_transcript, self.rRNA, self.miRNA,
                            self.snoRNA, self.snRNA, self.SRP_RNA, self.lnc_RNA, self.pre_miRNA, self.RNase_MRP_RNA,
                            self.pseudogenic_transcript]

        # regions of original (both) or processed (exon) transcripts
        self.exon = 'exon'
        self.intron = 'intron'
        self.sub_transcribed = [self.exon, self.intron]

        # sub-exon-level categorization (but should have transcribed as parent)
        self.cds = 'CDS'
        self.five_prime_UTR = 'five_prime_UTR'
        self.three_prime_UTR = 'three_prime_UTR'
        self.coding_info = [self.cds, self.five_prime_UTR, self.three_prime_UTR]

        # point annotations
        self.TSS = 'TSS'  # transcription start site
        self.TTS = 'TTS'  # transcription termination site
        self.start_codon = 'start_codon'
        self.stop_codon = 'stop_codon'
        self.donor_splice_site = 'donor_splice_site'
        self.acceptor_splice_site = 'acceptor_splice_site'
        # use the following when the far side of a splice site is on a different strand and/or sequence
        # does not imply an intron!
        # e.g. for trans-splicing
        self.trans_donor_splice_site = 'trans_donor_splice_site'
        self.trans_acceptor_splice_site = 'trans_acceptor_splice_site'
        self.point_annotations = [self.TSS, self.TTS, self.start_codon, self.stop_codon, self.donor_splice_site,
                                  self.acceptor_splice_site, self.trans_donor_splice_site,
                                  self.trans_acceptor_splice_site]

        # regions (often but not always included so one knows the size of the chromosomes / contigs / whatever
        self.region = 'region'
        self.chromosome = 'chromosome'
        self.supercontig = 'supercontig'
        self.regions = [self.region, self.chromosome, self.supercontig]

        # things that don't appear to really be annotations
        self.match = 'match'
        self.cDNA_match = 'cDNA_match'
        self.ignorable = [self.match, self.cDNA_match]

        # for mistakes or near-mistakes / marking partials
        self.error = 'error'
        self.status_coding = 'status_coding'
        self.status_intron = 'status_intron'
        self.status_five_prime_UTR = 'status_five_prime_UTR'
        self.status_three_prime_UTR = 'status_three_prime_UTR'
        self.status_intergenic = 'status_intergenic'
        self.statuses = [self.status_coding, self.status_intron, self.status_five_prime_UTR,
                         self.status_three_prime_UTR, self.status_intergenic]
        # and putting them together
        self.on_sequence = self.sub_transcribed + self.coding_info + self.point_annotations
        self.known = self.gene_level + self.transcribed + self.sub_transcribed + self.coding_info + \
                     self.point_annotations + self.regions + self.ignorable + [self.error] + self.statuses




class OverlapStatus(object):
    contained = 'contained'
    contains = 'contains'
    no_overlap = 'no_overlap'
    overlaps_upstream = 'overlaps_upstream'
    overlaps_downstream = 'overlaps_downstream'
    accepted_stati = (contained, no_overlap, overlaps_upstream, overlaps_downstream)

    def __init__(self):
        self._status = None

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        assert status in OverlapStatus.accepted_stati
        self._status = status

    def set_status(self, feature, seqid, start, end):
        err_str = 'Non handled overlap feature({}, {}, {}) vs slice({}, {}, {})'.format(
                feature.seqid, feature.start, feature.end,
                seqid, start, end
            )
        overlaps_at_start = False
        overlaps_at_end = False
        if feature.seqid != seqid:
            out = OverlapStatus.no_overlap
        elif feature.start >= start and feature.end <= end:
            out = OverlapStatus.contained
        elif feature.start < start and feature.end > end:
            out = OverlapStatus.contains
        elif feature.end < start or feature.start > end:
            out = OverlapStatus.no_overlap
        elif feature.start < start and feature.end >= start:
            overlaps_at_start = True
        elif feature.end > end and feature.start <= end:
            overlaps_at_end = True
        else:
            raise ValueError(err_str)

        plus_strand = feature.is_plus_strand()
        if overlaps_at_start and overlaps_at_end:
            raise ValueError(err_str + ' Overlaps both ends???')  # todo, test this properly and remove run time check

        if (overlaps_at_start and plus_strand) or (overlaps_at_end and not plus_strand):
            out = OverlapStatus.overlaps_upstream
        if (overlaps_at_end and plus_strand) or (overlaps_at_start and not plus_strand):
            out = OverlapStatus.overlaps_downstream
        self.status = out




#### section TranscriptInterpreter, might end up in a separate file later
class TranscriptStatus(object):
    """can hold and manipulate all the info on current status of a transcript"""
    def __init__(self):
        # initializes to intergenic
        self.genic = False  # todo, have some thoughts about how trans-splicing will fit in
        self.in_intron = False
        self.seen_start = False
        self.seen_stop = False
        self.phase = None  # todo, proper tracking / handling

    def saw_tss(self):
        self.genic = True

    def saw_start(self, phase):
        self.genic = True
        self.seen_start = True
        self.phase = phase

    def saw_stop(self):
        self.seen_stop = True
        self.phase = None

    def saw_tts(self):
        self.genic = False

    def splice_open(self):
        self.in_intron = True

    def splice_close(self):
        self.in_intron = False

    def is_5p_utr(self):
        return self.genic and not any([self.in_intron, self.seen_start, self.seen_stop])

    def is_3p_utr(self):
        return all([self.genic, self.seen_stop, self.seen_start]) and not self.in_intron

    def is_coding(self):
        return self.genic and self.seen_start and not any([self.in_intron, self.seen_stop])

    def is_intronic(self):
        return self.in_intron and self.genic

    def is_intergenic(self):
        return not self.genic


class TranscriptInterpBase(object):
    def __init__(self, transcript):
        self.status = TranscriptStatus()
        self.transcript = transcript

    @property
    def super_locus(self):
        return self.transcript.super_locus

    @property
    def gffkey(self):
        return self.transcript.super_locus.genome.gffkey


class TranscriptTrimmer(TranscriptInterpBase):
    """takes pre-cleaned/explicit transcripts and crops to what fits in a slice"""
    def __init__(self, transcript):
        super().__init__(transcript)

    def crop_to_slice(self, seqid, start, end):
        """crops transcript in place"""
        pass

    def transition_5p_to_3p(self):
        pass


class TranscriptInterpreter(TranscriptInterpBase):
    """takes raw/from-gff transcript, and makes totally explicit"""
    def __init__(self, transcript):
        super().__init__(transcript)
        self.clean_features = []  # will hold all the 'fixed' features
        self.transcript = transcript.swap_type('transcripts')
        self.protein_id_key = self._get_raw_protein_ids()
        self.proteins = self._setup_proteins()

    # todo, divvy features to transcript or proteins
    # todo, get_protein_id function (protein_id, Parent of CDS, None to IDMAker)
    # todo, make new protein when ID changes / if we've hit stop codon?

    def new_feature(self, template, **kwargs):
        try:
            new = template.clone()
        except KeyError as e:
            print(self.transcript.super_locus.short_str())
            print(template.short_str())
            raise e
        for key in kwargs:
            new.__setattr__(key, kwargs[key])
        return new

    @staticmethod
    def pick_one_interval(interval_set, target_type=None):
        if target_type is None:
            return interval_set[0]
        else:
            return [x for x in interval_set if x.data.type == target_type][0]

    def _get_protein_id_from_cds(self, cds_feature):
        assert cds_feature.gff_entry.type == self.gffkey.cds, "{} != {}".format(cds_feature.gff_entry.type,
                                                                                self.gffkey.cds)
        # check if anything is labeled as protein_id
        protein_id = cds_feature.gff_entry.attrib_filter(tag='protein_id')
        # failing that, try and get parent ID (presumably transcript, maybe gene)
        if not protein_id:
            protein_id = cds_feature.gff_entry.get_Parent()
        # hopefully take single hit
        if len(protein_id) == 1:
            protein_id = protein_id[0]
            if isinstance(protein_id, gffhelper.GFFAttribute):
                protein_id = protein_id.value
                assert len(protein_id) == 1
                protein_id = protein_id[0]
        # or handle other cases
        elif len(protein_id) == 0:
            protein_id = None
        else:
            raise ValueError('indeterminate single protein id {}'.format(protein_id))
        print(protein_id, type(protein_id), 'pid, type')
        return protein_id

    def _get_raw_protein_ids(self):
        # only meant for use before feature interpretation
        protein_ids = set()
        for fkey in self.transcript.features:
            feature = self.super_locus.features[fkey]
            if feature.type == self.gffkey.cds:
                protein_id = self._get_protein_id_from_cds(feature)
                protein_ids.add(protein_id)
        # map if necessary to unique / not-None IDs
        prot_id_key = {}
        for pkey in protein_ids:
            prot_id_key[pkey] = self.super_locus.genome.protein_ider.next_unique_id(pkey)
        return prot_id_key

    def _setup_proteins(self):
        proteins = {}
        for key in self.protein_id_key:
            val = self.protein_id_key[key]
            print('making protein {} (ori was {})'.format(val, key))
            protein = self.transcript.clone_but_swap_type(SuperLocus.t_proteins)
            protein = protein.replace_id_everywhere(val)
            proteins[val] = protein
        return proteins

    def _mv_coding_features_to_proteins(self):
        print('proteins to move to', self.proteins)
        print('known proteins in sl', self.super_locus.proteins.keys())
        print('proteins in key', self.protein_id_key)
        for protein in self.proteins:
            print(protein, self.super_locus.proteins[protein].features, 'pid, p features')
        # only meant for use after feature interpretation
        for feature in self.clean_features:
            if feature.type in [self.gffkey.status_coding, self.gffkey.stop_codon, self.gffkey.start_codon]:
                assert len(feature.transcripts) == 1
                feature.de_link_from_feature_holder(
                    holder_id=feature.transcripts[0],
                    holder_type=SuperLocus.t_transcripts
                )
                protein_id = self.protein_id_key[self._get_protein_id_from_cds(feature)]
                feature.link_to_feature_holder(protein_id, SuperLocus.t_proteins)

    def is_plus_strand(self):
        features = [self.super_locus.features[f] for f in self.transcript.features]
        seqids = [x.seqid for x in features]
        if not all([x == seqids[0] for x in seqids]):
            raise TransSplicingError("non matching seqids {}, for {}".format(seqids, self.super_locus.id))
        if all([x.strand == '+' for x in features]):
            return True
        elif all([x.strand == '-' for x in features]):
            return False
        else:
            raise TransSplicingError("Mixed strands at {} with {}".format(self.super_locus.id,
                                                                          [(x.seqid, x.strand) for x in features]))

    def interpret_transition(self, ivals_before, ivals_after, plus_strand=True):
        sign = 1
        if not plus_strand:
            sign = -1
        before_types = self.possible_types(ivals_before)
        after_types = self.possible_types(ivals_after)
        # 5' UTR can hit either start codon or splice site
        if self.status.is_5p_utr():
            # start codon
            self.handle_from_5p_utr(ivals_before, ivals_after, before_types, after_types, sign)
        elif self.status.is_coding():
            self.handle_from_coding(ivals_before, ivals_after, before_types, after_types, sign)
        elif self.status.is_3p_utr():
            self.handle_from_3p_utr(ivals_before, ivals_after, before_types, after_types, sign)
        elif self.status.is_intronic():
            self.handle_from_intron()
        elif self.status.is_intergenic():
            self.handle_from_intergenic()
        else:
            raise ValueError('unknown status {}'.format(self.status.__dict__))

    def handle_from_coding(self, ivals_before, ivals_after, before_types, after_types, sign):
        assert self.gffkey.cds in before_types
        # stop codon
        if self.gffkey.three_prime_UTR in after_types:
            self.handle_control_codon(ivals_before, ivals_after, sign, is_start=False)
        # splice site
        elif self.gffkey.cds in after_types:
            self.handle_splice(ivals_before, ivals_after, sign)

    def handle_from_intron(self):
        raise NotImplementedError  # todo later

    def handle_from_3p_utr(self, ivals_before, ivals_after, before_types, after_types, sign):
        assert self.gffkey.three_prime_UTR in before_types
        # the only thing we should encounter is a splice site
        if self.gffkey.three_prime_UTR in after_types:
            self.handle_splice(ivals_before, ivals_after, sign)
        else:
            raise ValueError('wrong feature types after three prime: b: {}, a: {}'.format(
                [x.data.type for x in ivals_before], [x.data.type for x in ivals_after]))

    def handle_from_5p_utr(self, ivals_before, ivals_after, before_types, after_types, sign):
        assert self.gffkey.five_prime_UTR in before_types
        # start codon
        if self.gffkey.cds in after_types:
            self.handle_control_codon(ivals_before, ivals_after, sign, is_start=True)
        # intron
        elif self.gffkey.five_prime_UTR in after_types:
            self.handle_splice(ivals_before, ivals_after, sign)
        else:
            raise ValueError('wrong feature types after five prime: b: {}, a: {}'.format(
                [x.data.type for x in ivals_before], [x.data.type for x in ivals_after]))

    def handle_from_intergenic(self):
        raise NotImplementedError  # todo later

    def is_gap(self, ivals_before, ivals_after, sign):
        """checks for a gap between intervals, and validates it's a positive one on strand of interest"""
        after0 = self.pick_one_interval(ivals_after)
        before0 = self.pick_one_interval(ivals_before)
        before_downstream = before0.data.downstream_from_interval(before0)
        after_upstream = after0.data.upstream_from_interval(after0)
        is_gap = before_downstream + 1 * sign != after_upstream
        if is_gap:
            # if there's a gap, confirm it's in the right direction
            gap_len = (after_upstream - (before_downstream + 1 * sign)) * sign
            assert gap_len > 0, "inverse gap between {} and {} at putative control codon seq {}, gene {}, " \
                                "features {} {}".format(
                before_downstream, after_upstream, after0.data.seqid, self.super_locus.id, before0.data.id,
                after0.data.id
            )
        return is_gap

    def handle_control_codon(self, ivals_before, ivals_after, sign, is_start=True):
        target_after_type = None
        target_before_type = None
        if is_start:
            target_after_type = self.gffkey.cds
        else:
            target_before_type = self.gffkey.cds

        after0 = self.pick_one_interval(ivals_after, target_after_type)
        before0 = self.pick_one_interval(ivals_before, target_before_type)
        # make sure there is no gap
        is_gap = self.is_gap(ivals_before, ivals_after, sign)

        if is_start:
            if is_gap:
                self.handle_splice(ivals_before, ivals_after, sign)

            template = after0.data
            # it better be std phase if it's a start codon
            at = template.upstream_from_interval(after0)
            if template.phase == 0:  # "non-0 phase @ {} in {}".format(template.id, template.super_locus.id)
                start, end = min_max(at, at + 2 * sign)
                start_codon = self.new_feature(template=template, start=start, end=end, type=self.gffkey.start_codon)
                self.status.saw_start(phase=0)
                self.clean_features.append(start_codon)
            else:
                upstream_buffered = before0.data.upstream_from_interval(before0) - sign * self.gffkey.error_buffer
                err_start, err_end = min_max(at - 1 * sign, upstream_buffered)
                feature_e = self.new_feature(template=template, type=self.gffkey.error,
                                             start=err_start, end=err_end, phase=None)
                coding_status = self.new_feature(template=template, type=self.gffkey.status_coding, start=at, end=at)
                self.status.saw_start(template.phase)
                self.clean_features += [feature_e, coding_status]
        else:
            # todo, confirm phase for stop codon
            template = before0.data
            at = template.downstream_from_interval(before0)
            start, end = min_max(at, at - 2 * sign)
            stop_codon = self.new_feature(template=template, start=start, end=end, type=self.gffkey.stop_codon)
            self.status.saw_stop()
            self.clean_features.append(stop_codon)
            if is_gap:
                self.handle_splice(ivals_before, ivals_after, sign)

    def handle_splice(self, ivals_before, ivals_after, sign):
        target_type = None
        if self.status.is_coding():
            target_type = self.gffkey.cds

        before0 = self.pick_one_interval(ivals_before, target_type)
        after0 = self.pick_one_interval(ivals_after, target_type)
        donor_tmplt = before0.data
        acceptor_tmplt = after0.data
        donor_at = donor_tmplt.downstream_from_interval(before0) + (1 * sign)
        acceptor_at = acceptor_tmplt.upstream_from_interval(after0) - (1 * sign)
        # add splice sites if there's a gap
        between_splice_sites = (acceptor_at - donor_at) * sign
        min_intron_len = 3  # todo, maybe get something small but not entirely impossible?
        if between_splice_sites > min_intron_len - 1:  # -1 because the splice sites are _within_ the intron
            donor = self.new_feature(template=donor_tmplt, start=donor_at, end=donor_at, phase=None,
                                     type=self.gffkey.donor_splice_site)
            # todo, check position of DSS/ASS to be consistent with Augustus, hopefully
            acceptor = self.new_feature(template=acceptor_tmplt, start=acceptor_at, end=acceptor_at,
                                        type=self.gffkey.acceptor_splice_site)
            self.clean_features += [donor, acceptor]
        # do nothing if there is just no gap between exons for a techinical / reporting error
        elif between_splice_sites == -1:
            pass
        # everything else is invalid
        else:
            feature_e = before0.data.clone()
            all_coords = [before0.data.start, before0.data.end, after0.data.start, after0.data.end]
            feature_e.start = sorted(all_coords)[0]
            feature_e.end = sorted(all_coords)[-1]
            feature_e.type = self.gffkey.error
            self.clean_features.append(feature_e)

    def interpret_first_pos(self, intervals, plus_strand=True):
        i0 = self.pick_one_interval(intervals)
        at = i0.data.upstream_from_interval(i0)
        possible_types = self.possible_types(intervals)
        if self.gffkey.five_prime_UTR in possible_types:
            # this should indicate we're good to go and have a transcription start site
            tss = self.new_feature(template=i0.data, type=self.gffkey.TSS, start=at, end=at, phase=None)
            self.clean_features.append(tss)
            self.status.saw_tss()
        elif self.gffkey.cds in possible_types:
            # this could be first exon detected or start codon, ultimately, indeterminate
            cds_feature = self.pick_one_interval(intervals, target_type=self.gffkey.cds).data
            coding = self.new_feature(template=cds_feature, type=self.gffkey.status_coding, start=at, end=at)
            self.clean_features.append(coding)
            self.status.saw_start(phase=coding.phase)
            # mask a dummy region up-stream as it's very unclear whether it should be intergenic/intronic/utr
            if plus_strand:
                # unless we're at the start of the sequence
                start_of_sequence = self.get_seq_start(cds_feature.seqid)
                if at != start_of_sequence:
                    feature_e = self.new_feature(template=cds_feature, type=self.gffkey.error,
                                                 start=max(start_of_sequence, at - self.gffkey.error_buffer - 1),
                                                 end=at - 1, phase=None)
                    self.clean_features.insert(0, feature_e)
            else:
                end_of_sequence = self.get_seq_end(cds_feature.seqid)
                if at != end_of_sequence:
                    feature_e = self.new_feature(template=cds_feature, type=self.gffkey.error, start=at + 1,
                                                 end=min(end_of_sequence, at + self.gffkey.error_buffer + 1),
                                                 phase=None)
                    feature_e.type = self.gffkey.error
                    self.clean_features.insert(0, feature_e)
        else:
            raise ValueError("why's this gene not start with 5' utr nor cds? types: {}, interpretations: {}".format(
                [x.data.type for x in intervals], possible_types))

    def interpret_last_pos(self, intervals, plus_strand=True):
        i0 = self.pick_one_interval(intervals)
        at = i0.data.downstream_from_interval(i0)
        possible_types = self.possible_types(intervals)
        if self.gffkey.three_prime_UTR in possible_types:
            # this should be transcription termination site
            tts = self.new_feature(template=i0.data, type=self.gffkey.TTS, start=at, end=at, phase=None)
            self.clean_features.append(tts)
            self.status.saw_tts()
        elif self.gffkey.cds in possible_types:
            # may or may not be stop codon, but will just mark as error (unless at edge of sequence)
            start_of_sequence = self.get_seq_start(i0.data.seqid)
            end_of_sequence = self.get_seq_end(i0.data.seqid)
            if plus_strand:
                if at != end_of_sequence:
                    feature_e = self.new_feature(template=i0.data, type=self.gffkey.error, start=at + 1, phase=None,
                                                 end=min(at + 1 + self.gffkey.error_buffer, end_of_sequence))
                    self.clean_features.append(feature_e)
            else:
                if at != start_of_sequence:
                    feature_e = self.new_feature(template=i0.data, type=self.gffkey.error, end=at - 1, phase=None,
                                                 start=max(start_of_sequence, at - self.gffkey.error_buffer - 1))
                    self.clean_features.append(feature_e)
        else:
            raise ValueError("why's this gene not end with 3' utr/exon nor cds? types: {}, interpretations: {}".format(
                [x.data.type for x in intervals], possible_types)
            )

    def intervals_5to3(self, plus_strand=False):
        interval_sets = list(self.organize_and_split_features())
        if not plus_strand:
            interval_sets.reverse()
        return interval_sets

    def decode_raw_features(self):
        plus_strand = self.is_plus_strand()
        interval_sets = self.intervals_5to3(plus_strand)
        self.interpret_first_pos(interval_sets[0], plus_strand)
        for i in range(len(interval_sets) - 1):
            ivals_before = interval_sets[i]
            ivals_after = interval_sets[i + 1]
            self.interpret_transition(ivals_before, ivals_after, plus_strand)

        self.interpret_last_pos(intervals=interval_sets[-1])
        self._mv_coding_features_to_proteins()

    def possible_types(self, intervals):
        # shortcuts
        cds = self.gffkey.cds
        five_prime = self.gffkey.five_prime_UTR
        exon = self.gffkey.exon
        three_prime = self.gffkey.three_prime_UTR

        # what we see
        observed_types = [x.data.type for x in intervals]
        set_o_types = set(observed_types)
        # check length
        if len(intervals) not in [1, 2]:
            raise ValueError('check interpretation by hand for transcript start with {}, {}'.format(
                '\n'.join([ival.data.short_str() for ival in intervals]), observed_types
            ))
        # interpret type combination
        if set_o_types == {exon, five_prime} or set_o_types == {five_prime}:
            out = [five_prime]
        elif set_o_types == {exon, three_prime} or set_o_types == {three_prime}:
            out = [three_prime]
        elif set_o_types == {exon}:
            out = [five_prime, three_prime]
        elif set_o_types == {cds, exon} or set_o_types == {cds}:
            out = [cds]
        else:
            raise ValueError('check interpretation of combination for transcript start with {}, {}'.format(
                intervals, observed_types
            ))
        return out

    def organize_and_split_features(self):
        # todo, handle non-single seqid loci
        tree = intervaltree.IntervalTree()
        features = [self.super_locus.features[f] for f in self.transcript.features]
        for f in features:
            tree[f.py_start:f.py_end] = f
        tree.split_overlaps()
        # todo, minus strand
        intervals = iter(sorted(tree))
        out = [next(intervals)]
        for interval in intervals:
            if out[-1].begin == interval.begin:
                out.append(interval)
            else:
                yield out
                out = [interval]
        yield out

    def get_seq_end(self, seqid):
        return self.super_locus.slice.seq_info[seqid].end

    def get_seq_start(self, seqid):
        return self.super_locus.slice.seq_info[seqid].start


def min_max(x, y):
    return min(x, y), max(x, y)


def none_to_list(x):
    if x is None:
        return []
    else:
        assert isinstance(x, list)
        return x


def upstream(x, y, sign):
    if (y - x) * sign >= 0:
        return x
    else:
        return y


def downstream(x, y, sign):
    if (x - y) * sign >= 0:
        return x
    else:
        return y
