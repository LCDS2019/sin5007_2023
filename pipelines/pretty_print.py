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


def plot_model_metrics(models, key, from_size, to_size):
    # Width of each bar
    bar_width = 0.15

    # Total number of metrics (assuming each model report has the same metrics)
    num_metrics = len(models[0][key]['mean'])

    # Generate a color palette
    colors = plt.cm.get_cmap('viridis', len(models))
    def model_name(row):
        return row['model']['classifier'] + ' + ' + ', '.join(row['model']['steps'])

    # Create a model to color mapping
    model_colors = {model_name(model): colors(i) for i, model in enumerate(models)}

    # Initialize a figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Iterate over each metric
    for metric_idx, metric_name in enumerate(models[0][key]['mean'].keys()):
        # Iterate over each model for this metric
        for model_idx, model in enumerate(models):
            # Extract metric value and std
            metric_value = model[key]['mean'][metric_name]
            metric_std = model[key]['std'][metric_name]
            model_name_ = model_name(model)

            # Calculate bar position
            position = metric_idx + (model_idx - len(models) / 2) * bar_width

            # Plot bar with error bar
            ax.bar(position, metric_value, width=bar_width, label=model_name_ if metric_idx == 0 else "",
                   yerr=metric_std, color=model_colors[model_name_], capsize=5)

    # Setting the x-axis labels
    ax.set_xticks(np.arange(num_metrics))
    ax.set_xticklabels(models[0][key]['mean'].keys())

    # Adding labels and title
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title('Comparison of Model Metrics with Standard Deviation')

    ax.legend()

    plt.ylim(from_size, to_size)
    plt.tight_layout()
    plt.show()
