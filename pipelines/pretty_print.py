from polars.dependencies import numpy
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
        if isinstance(element, pd.DataFrame):
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


def plot_model_metrics(dataframe):
    # Extract unique metrics from the 'report' dataframes
    unique_metrics = set()
    for row in dataframe:
        unique_metrics.update(row['report'].columns.drop('class').drop('instance_count'))

    # Number of classes for positioning bars
    num_classes = len(dataframe[0]['report']['class'])

    # Initialize a figure
    fig, axes = plt.subplots(len(dataframe), 1, figsize=(12, 6 * len(dataframe)), squeeze=False)

    # Width of each bar
    bar_width = 0.15

    # Iterate over each model
    for idx, row in enumerate(dataframe):
        ax = axes[idx, 0]
        model_name = row['model']['classifier'] + ' + ' + ', '.join(row['model']['steps'])
        report_df = row['report']

        # Base positions for the bars
        base_positions = np.arange(num_classes)

        for metric_idx, metric in enumerate(unique_metrics):
            # Position for bars of this metric
            positions = base_positions + (metric_idx - len(unique_metrics) / 2) * bar_width
            ax.bar(positions, report_df[metric], width=bar_width, label=metric)

        ax.set_xticks(base_positions)
        ax.set_xticklabels(report_df['class'])
        ax.set_xlabel('Class')
        ax.set_ylabel('Metric Values')
        ax.set_title(f'Metrics for Model: {model_name}')
        ax.legend()

    plt.tight_layout()
    plt.show()
