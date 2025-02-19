#! /usr/bin/env python3
import click

from helixer.cli.main_cli import main_and_fasta_export_parameters, universal_export_parameters, universal_parameters
from helixer.cli.cli_formatter import HelpGroupCommand
from helixer.export.exporter import HelixerFastaToZarrController


@click.command(cls=HelpGroupCommand, context_settings={'show_default': True})
@main_and_fasta_export_parameters
@universal_parameters
@universal_export_parameters
def main(fasta_path, zarr_output_path, subsequence_length, species, no_multiprocess, write_by):
    """Convert a fasta file into a zarr file Helixer can use for predicting"""
    controller = HelixerFastaToZarrController(fasta_path, zarr_output_path)
    controller.export_fasta_to_zarr(chunk_size=subsequence_length, multiprocess=not no_multiprocess,
                                    species=species, write_by=write_by)


if __name__ == '__main__':
    main()
