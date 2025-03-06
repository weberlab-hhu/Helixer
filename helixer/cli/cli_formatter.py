import click
import textwrap

from collections import OrderedDict


class HelpGroups:
    """Class to bundle all help groups used by the cli module."""
    def __init__(self):
        self.io = 'I/O options'
        self.data = 'Data options'
        self.train = 'Training options'
        self.test = 'Test options'
        self.pred = 'Prediction options'
        self.proc = 'Processing options'
        self.post = 'Postprocessing options'
        self.resource = 'Resource options'
        self.misc = 'Miscellaneous options'
        self.fine_tune = 'Fine-tuning options'
        self.model = 'Model options'


class HelpGroupOption(click.Option):
    def __init__(self, *args, **kwargs):
        """Adds the custom help_group option to click's options.
        This makes it possible to sort options into their own help group for visual separation.
        """
        self.help_group = kwargs.pop('help_group', None)
        super(HelpGroupOption, self).__init__(*args, **kwargs)


class ColumnHelpFormatter(click.HelpFormatter):
    def write_dl(self, rows, col_widths, **kwargs):
        """Overrides click's default method to align 4-column help text dynamically.
            rows: list of help text to write
        """
        row_width = 100
        # Determine the max length of every column including spacing before the help message
        padding = sum(col_widths[:2]) + (2 * 2)  # spacing between all 3 columns

        formatted_rows = []
        for row in rows:
            # Adjust each row to have all columns align the same way
            aligned_row = '  '.join(value.ljust(col_widths[i]) for i, value in enumerate(row))
            # If the row exceeds the max row length (click's default: 78, default here: 100) wrap the
            #  help message so it can continue on the next line while still aligning with its column
            aligned_row = textwrap.wrap(aligned_row,
                                        width=row_width,
                                        subsequent_indent=' ' * padding  # align wrapped text under the help message
                                        )
            # Append the aligned rows
            formatted_rows.append(aligned_row[0])
            formatted_rows.extend(aligned_row[1:])

        # Add indent at the start of each row
        formatted_rows = ['  ' + row for row in formatted_rows]
        text = '\n'.join(formatted_rows)  # print rows on different lines
        # Use click's built-in write instead of write_text method to print aligned rows and avoid the
        #   extra wrapping function in write_text messing up the custom formatting above
        super().write('-' * row_width + '\n')  # add separator between options header and options list
        super().write(text)
        super().write('\n')


# WARNING: no custom help message is supposed to end on ']', as this will break the custom formatting options
class HelpGroupCommand(click.Command):
    def format_options(self, ctx, formatter):
        """Writes all options into the formatter if they exist and then
        sorts them into different help groups if they are part of one.
        Otherwise, they get sorted into the category 'Other'.
        """
        options = OrderedDict()
        all_rows = []
        for param in self.get_params(ctx):
            # todo: add how rv is structured
            rv = param.get_help_record(ctx)
            if rv is not None:
                # Extract option flags (e.g., "--cnn-layers")
                opt_str = ', '.join(param.opts)

                # Extract type (e.g., "INTEGER RANGE")
                type_str = param.type.name.upper() if param.type else ''

                # Extract help text and split off constraints
                #  constraint examples: [x>=1], [x>=2, required], etc.
                help_text = rv[1]
                if type_str == "CHOICE":
                    # adapted from click's way to format the choices, as we want to display them
                    # in the help text instead of in the type column which is the default
                    choices = f"  [one of: {', '.join(map(str, param.type.choices))}]"
                    help_text += choices
                # Create row in custom order
                row = (opt_str, type_str, help_text.strip())
                # If a help_group was set when defining the option, add the row to the specified options group
                if hasattr(param, 'help_group') and param.help_group:
                    options.setdefault(str(param.help_group), []).append(row)
                else:
                    options.setdefault('Other', []).append(row)
                all_rows.append(row)
        # Calculate width of each column for all help groups
        #  columns: option, type, range and if required, help message
        #  example: '--cnn-layers', 'INTEGER RANGE', '[x>=1]', 'Number of convolutional layers'
        col_widths = click.formatting.measure_table(all_rows)

        # Write every options group and their options in order to the CLI help message
        # The group "Other" is always at the end.
        options.move_to_end('Other')
        for name, opts_group in options.items():
            with formatter.section(name):
                if isinstance(formatter, ColumnHelpFormatter):
                    formatter.write_dl(opts_group, col_widths)
                else:
                    formatter.write_dl(opts_group)
