"""reopen and slice the new annotation.sqlitedb and divvy superloci to train/dev/test processing sets"""
import annotations
import annotations_orm

from gff_2_annotations import TranscriptStatus  # todo, move to helpers?

class SliceController(object):
    
