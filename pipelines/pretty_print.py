from pandas import DataFrame
from polars.dependencies import numpy
from tabulate import tabulate
import matplotlib.pyplot as plt


def pretty_print(df, drop_columns=None, float_format='.3f', tablefmt='rounded_grid'):
    print(prettify(
        df=df,
        drop_columns=drop_columns,
        float_format=float_format,
        tablefmt=tablefmt,
    ))


def ascii_as_image(ascii, filename):
    fig = plt.figure(figsize=(6, 2))
    plt.text(0.5, 0.5, ascii, fontsize=12, va='center', ha='center', family='monospace')
    plt.axis('off')

    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def prettify(df, drop_columns=None, float_format='.3f', tablefmt='rounded_grid'):
    def is_sequence(arg):
        return (
                not hasattr(arg, "strip") and
                (hasattr(arg, "__getitem__") or hasattr(arg, "__iter__"))
        ) and not isinstance(arg, str) and not isinstance(arg, numpy.number)

    def format_list_as_table(element):
        if isinstance(element, dict):
            formatted_dict = {k: format_list_as_table(v) for k, v in element.items()}
            return tabulate(
                tabular_data=[formatted_dict],
                tablefmt=tablefmt,
                showindex=False,
                floatfmt=float_format,
                headers='keys'
            )
        if isinstance(element, DataFrame):
            return prettify(
                df=element,
                drop_columns=drop_columns,
                float_format=float_format,
                tablefmt=tablefmt,
            )
        if is_sequence(element):
            return tabulate(
                tabular_data=[[format_list_as_table(e)] for e in element],
                tablefmt=tablefmt,
                showindex=False,
                floatfmt=float_format
            )
        if isinstance(element, numpy.int64):
            return format(element, 'd')
        else:
            return element

    if drop_columns is not None:
        df = df.drop(drop_columns, axis=1, inplace=False, errors='ignore')

    for column in df.columns:
        df[column] = df[column].apply(format_list_as_table)

    return tabulate(
        tabular_data=df,
        headers='keys',
        tablefmt=tablefmt,
        showindex=False,
        floatfmt=float_format
    )
