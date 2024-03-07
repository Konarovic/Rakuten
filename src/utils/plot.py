from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.metrics import f1_score
import pandas as pd

def classification_results(y_true, y_pred, index=None):
    #Print evaluation metrics
    print(classification_report(y_true, y_pred))
    print(f1_score(y_true, y_pred))
    
    #Build confusion matrix
    conf_mat = round(pd.crosstab(y_true, y_pred, rownames=['Classes reelles'], colnames=['Classes predites'], normalize='columns')*100)

    #in case labels are encoded, update with the original lables provided
    if index is not None:
        conf_mat.index = index
        conf_mat.columns = index
        
    #hierarchical clustering to find optimal order of labels
    Z = linkage(conf_mat, 'ward')
    order = leaves_list(Z)
    conf_mat = conf_mat.iloc[order, order]

    #plot confusion matrix as heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_mat, annot=round(conf_mat,12), center=50, cmap=sns.color_palette('rocket',  as_cmap=True))
    plt.show()
    
    return plt


def plot_training_history(history):
    f, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    axs[0].plot(history.history['loss'], label='Train')
    axs[0].plot(history.history['val_loss'], label='Test')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(history.epoch)
    axs[0].legend()
    axs[0].set_title('Loss')

    axs[1].plot(history.history['accuracy'], label='Train')
    axs[1].plot(history.history['val_accuracy'], label='Test')
    axs[1].set_xticks(history.epoch)
    axs[1].legend()
    axs[1].set_title('Accuracy')

    plt.show()
    
    return plt