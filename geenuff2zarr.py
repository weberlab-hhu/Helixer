#! /usr/bin/env python3
import click

from helixer.cli.main_cli import geenuff2zarr_parameters, universal_export_parameters, universal_parameters
from helixer.cli.cli_formatter import HelpGroupCommand
from helixer.export.exporter import HelixerExportController


@click.command(cls=HelpGroupCommand, context_settings={'show_default': True})
@geenuff2zarr_parameters
@universal_parameters
@universal_export_parameters
def main(input_db_path, zarr_output_path, subsequence_length, modes, write_by, no_multiprocess, additional):
    """Convert GeenuFF's sqlite database into a zarr file Helixer can use for training and testing"""
    if modes == 'all':
        modes = ('X', 'y', 'anno_meta', 'transitions')
    else:
        modes = tuple(modes.split(','))

    if additional:
        match_existing = True
        zarr_group = '/alternative/' + additional + '/'
    else:
        match_existing = False
        zarr_group = '/data/'

    write_by = round(write_by / subsequence_length) * subsequence_length
    controller = HelixerExportController(input_db_path, zarr_output_path, match_existing=match_existing,
                                         zarr_group=zarr_group)
    controller.export(chunk_size=subsequence_length, write_by=write_by, modes=modes,
                      multiprocess=not no_multiprocess)


if __name__ == '__main__':
    main()
