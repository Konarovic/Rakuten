from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np


def classification_results(y_true, y_pred, index=None, title=None):
    # Print evaluation metrics
    print(classification_report(y_true, y_pred, target_names=index))
    print(f1_score(y_true, y_pred, average='weighted'))

    # Build confusion matrix
    conf_mat = round(pd.crosstab(y_true, y_pred, rownames=[
                     'Classes reelles'], colnames=['Classes predites'], normalize='columns')*100)

    # in case labels are encoded, update with the original lables provided
    if index is not None:
        conf_mat.index = index
        conf_mat.columns = index

    # hierarchical clustering to find optimal order of labels
    Z = linkage(conf_mat, 'ward')
    order = leaves_list(Z)
    conf_mat = conf_mat.iloc[order, order]

    # plot confusion matrix as heatmap
    mask_other = np.eye(conf_mat.shape[0], dtype=bool) | conf_mat.apply(
        lambda x: x == 0).values
    mask_diag = ~np.eye(conf_mat.shape[0], dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.gca().patch.set_facecolor('#f8e9db')
    sns.heatmap(conf_mat, mask=mask_diag, cmap="rocket", alpha=0.5, annot=round(
        conf_mat, 2), cbar=False, ax=ax)
    sns.heatmap(conf_mat, mask=mask_other, cmap="rocket_r", alpha=0.5, annot=round(
        conf_mat, 2), cbar=False, ax=ax)
    if title is not None:
        plt.title(title)
    plt.show()

    return plt


def plot_training_history(history):
    f, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    axs[0].plot(history.history['loss'], label='Train')
    if 'val_loss' in history.history.keys():
        axs[0].plot(history.history['val_loss'], label='Test')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(history.epoch)
    axs[0].legend()
    axs[0].set_title('Loss')

    axs[1].plot(history.history['accuracy'], label='Train')
    if 'val_accuracy' in history.history.keys():
        axs[1].plot(history.history['val_accuracy'], label='Test')
    axs[1].set_xticks(history.epoch)
    axs[1].legend()
    axs[1].set_title('Accuracy')

    plt.show()

    return plt


def plot_bench_results(data, x_column, y_column, x_label, y_label, color_column=None, title=None):

    custom_categories_order = data[x_column].tolist()
    fig = px.bar(
        data,
        y=x_column,
        x=y_column,
        color=color_column,
        color_discrete_sequence=px.colors.qualitative.Plotly,
        category_orders={x_column: custom_categories_order},
    )

    fig.update_traces(
        width=0.8,

    )
    # Update layout to remove legend and adjust xaxis title
    fig.update_layout(
        legend=None,
        xaxis_title=y_label,
        yaxis_title=x_label,
        bargap=0.3,
        bargroupgap=0.2,
        barmode='group',
        width=1200,
        height=600,
        title=title

    )

    # Show the plot
    fig.show()
    return plt
