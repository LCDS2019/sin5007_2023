import tempfile

import seaborn as sns
from keras.src.utils import plot_model
from matplotlib import pyplot as plt, image as mpimg
from sklearn.tree import plot_tree


def plot_keras_model(model):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as temp_file:
        # Use plot_model to save the plot to the temporary file
        plot_model(model, to_file=temp_file.name, show_shapes=True, show_layer_names=True)
        img = mpimg.imread(temp_file.name)
        fig, ax = plt.subplots()
        ax.axis('off')
        plt.imshow(img)
        plt.show()


def plot_decision_tree(tree, feature_names):
    plt.figure(figsize=(10, 10), dpi=650)
    plot_tree(tree, filled=True, feature_names=feature_names, proportion=True, rounded=True)
    plt.show()


def plot_pair(df, title=None, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    plt.title(title)
    sns.pairplot(data=df)
    plt.show()


def plot_heatmap(df, title=None,  figsize=(10, 8), cmap='coolwarm'):
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(df, cmap=cmap, interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(df)), df.columns, rotation=45)
    plt.yticks(range(len(df)), df.columns)
    for i in range(len(df)):
        for j in range(len(df)):
            plt.text(j, i, f"{df.iloc[i, j]:.2f}", ha='center', va='center', color='white')
    plt.show()


def plot_table(df, title=None, ):
    fig, ax = plt.subplots(figsize=(12, 3))
    plt.title(title)
    ax.axis('off')
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     rowLabels=df.index,
                     loc='center',
                     cellLoc='left')
    table.scale(1.2, 1.2)
    header_row_color = '#007BFF'
    row_colors = ['#F2F2F2', '#FFFFFF']

    for position, cell in table.get_celld().items():
        row, col = position

        cell.set_edgecolor('white')

        if row == 0 or col == -1:
            cell.set_facecolor(header_row_color)
            cell.set_text_props(color='white')
        else:
            cell.set_facecolor(row_colors[row % 2])
            cell.set_text_props(color='black')
    plt.tight_layout()
    plt.show()
