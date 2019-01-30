"""convert cleaned-db schema to numeric values describing gene structure"""

import numpy as np
import copy

import annotations_orm
import gff_2_annotations
import slicer


# for now collapse everything to one vector (with or without pre-selection of primary transcript)
# 1x coding, utr, intron, intergenic (precedence on collapse and/or multi label)
# 1x TSS, TTS, status-transcribed, start, stop, status-translated, don-splice, acc-splice, status intron (")
# both of the above + trans-splice separate from splicing

# general structuring
# class defining data manipulation functions (Numerifier)
#   takes a slice & returns a matrix of values;
#   and can transform matrix <-> flat;
#   provides name
#
# class defining examples (ExampleMaker)
#   makes x, y pairs of data (as .dict)
#   handles processing of said data via calls to appropriate Numerifier


class Numerifier(object):
    def __init__(self, shape, dtype=np.float):
        self.shape = shape
        self.dtype = dtype
        self.matrix = None

    def slice_to_matrix(self, data_slice, *args, **kwargs):
        raise NotImplementedError

    def flatten_matrix(self):
        assert isinstance(self.matrix, np.ndarray)
        return self.matrix.flatten()

    def deflatten_matrix(self, flattened):
        return np.reshape(flattened, self.shape)

    def _zeros(self):
        return np.zeros(self.shape, self.dtype)


class AnnotationFoo(object):

    def __init__(self, data_slice):
        assert isinstance(data_slice, slicer.SequenceInfoHandler)
        self.data_slice = data_slice
        self.coordinates = self._get_coordinates()
        self.super_loci = self._get_super_loci()

    def _get_coordinates(self):
        coords = self.data_slice.data.coordinates
        assert len(coords) == 1
        coords = coords[0]
        return coords

    def _get_super_loci(self):
        super_loci = set()
        for feature in self.coordinates.features:
            super_loci.add(feature.super_locus)
        out = []
        for sl in super_loci:
            handler = slicer.SuperLocusHandler()
            handler.add_data(sl)
            out.append(handler)
        return out


class BasePairAnnotationNumerifier(Numerifier, AnnotationFoo):
    def __init__(self, shape, data_slice, *args, **kwargs):
        super(Numerifier).__init__(shape)
        super(AnnotationFoo).__init__(data_slice)

    def slice_to_matrix(self, data_slice, *args, **kwargs):
        matrix = self._zeros()
        # todo, setup transcript local reader for each transcript.
        #   maybe grab primary transcript only
        #   transition setting numbers into matrix


class TranscriptLocalReader(gff_2_annotations.TranscriptInterpBase):
    def sort_features(self, coords, is_plus_strand):
        features = [x for x in coords.features if x.coordinates is coords and x.is_plus_strand == is_plus_strand]
        features = sorted(features, key=lambda x: x.pos_cmp_key())
        if not features[0].is_plus_strand:
            features.reverse()

    def transition_5p_to_3p(self, coords, is_plus_strand):
        status = gff_2_annotations.TranscriptStatus()
        for aligned_features in self.stack_matches(self.sort_features(coords, is_plus_strand)):
            self.update_status(status, aligned_features)
            yield aligned_features, copy.deepcopy(status)

    def transition_5p_to_3p_with_ranges(self):
        pass
        # todo, track previous position to laydown range to change
